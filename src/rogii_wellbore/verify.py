"""Checkpoint replay: does the package give the notebook's numbers from the notebook's own weights?

    python -m rogii_wellbore.verify --config configs/members/v2l.yaml --data-root <data> \
        --run-dir <member run folder> [--ckpt-dir <folder with ckpt_f*_s*.pt>] [--fold 0] [--seeds 34] \
        [--max-wells N]

Retraining cannot reproduce a run (seed and GPU noise, and hours of synthetic pretraining), but
inference can. For each (fold, seed) this loads ``ckpt_f{fold}_s{seed}.pt``, renders the fold's
validation wells with the package, predicts them at the member's ``checkpoint_k``, and compares
with ``oof_f{fold}_s{seed}.npz``, which the training notebook wrote from the same checkpoint.
It also checks the package's fold split against the run folder's ``fold_splits.json`` (or
``ft39l_fold_splits.json``) when one is there.

What it exercises: reading the real competition files, the fold split, image rendering at the
member's geometry, the real encoder, loading a notebook checkpoint, refeed, decode and scoring.
Expected: the same RMSE to about 0.01 ft and per-row differences of hundredths of a foot or less
(the notebook decoded in fp32 too; what remains is GPU arithmetic order).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from . import config as C
from .cv import load_folds, make_folds
from .data import TruthCache, WellStore
from .image import arrays_from_frames, build_image
from .oof import decode_seed_oof, load_seed_oof

RMSE_TOL = 0.01      # ft
ROW_TOL = 0.10       # ft, largest single-row difference


def find_ckpt(ckpt_dir: str, fold: int, seed: int) -> str:
    p = os.path.join(ckpt_dir, f'ckpt_f{fold}_s{seed}.pt')
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(ckpt_dir, f'ckpt_f{fold}_s{seed}_*.pt')))
    if not hits:
        raise FileNotFoundError(f'no ckpt_f{fold}_s{seed}[_*].pt in {ckpt_dir}')
    return hits[0]


def recorded_folds(run_dir: str):
    for name in ('fold_splits.json', 'ft39l_fold_splits.json'):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return load_folds(p), name
    return None, None


def check_folds(ours: dict, theirs: dict) -> list[str]:
    """Differences between two fold splits, as readable lines (empty = identical)."""
    bad = []
    for f in sorted(set(ours) | set(theirs)):
        a, b = ours.get(f, {}).get('val', []), theirs.get(f, {}).get('val', [])
        if sorted(a) != sorted(b):
            bad.append(f'fold {f}: {len(set(a) - set(b))} wells only in ours, {len(set(b) - set(a))} only in the record')
    return bad


def compare(replay: dict, record: dict) -> dict:
    """Per-row comparison of two per-well OOF dicts on the wells they share."""
    wids = [w for w in record if w != '_meta' and w in replay]
    d_pred = np.concatenate([np.abs(replay[w]['pred_post'].astype(np.float64)
                                    - record[w]['pred_post'].astype(np.float64)) for w in wids])
    d_er = max(float(np.max(np.abs(replay[w]['er_row'].astype(np.float64)
                                   - record[w]['er_row'].astype(np.float64)))) for w in wids)
    def score(d):
        return float(np.sqrt(np.concatenate([d[w]['resid_post'].astype(np.float64) ** 2 for w in wids]).mean()))
    r_replay, r_record = score(replay), score(record)
    return dict(n_wells=len(wids), rmse_record=r_record, rmse_replay=r_replay,
                d_rmse=abs(r_replay - r_record), max_d_pred=float(d_pred.max()),
                p99_d_pred=float(np.percentile(d_pred, 99)), max_d_er_row=d_er,
                ok=abs(r_replay - r_record) <= RMSE_TOL and float(d_pred.max()) <= ROW_TOL)


def replay(cfg: C.RunConfig, data_root: str, run_dir: str, ckpt_dir: str | None = None, fold: int = 0,
           seeds=None, max_wells: int | None = None, device=None, log=print) -> dict:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = ckpt_dir or run_dir
    seeds = list(seeds) if seeds else [cfg.train.seeds[0]]
    store = WellStore.train(data_root)
    truths = TruthCache(store)

    folds = make_folds(store, seed=cfg.cv.seed, n_folds=cfg.cv.n_folds)
    rec_folds, rec_name = recorded_folds(run_dir)
    fold_diff = check_folds(folds, rec_folds) if rec_folds else None
    if rec_folds is None:
        log('fold split: no fold_splits.json in the run folder, not compared')
    elif fold_diff:
        log('fold split: DIFFERENT from ' + rec_name + '\n  ' + '\n  '.join(fold_diff))
    else:
        log(f'fold split: identical to {rec_name} ({", ".join(str(len(folds[f]["val"])) for f in sorted(folds))})')

    out = {'device': str(device), 'fold_split_ok': None if rec_folds is None else not fold_diff, 'seeds': {}}
    for seed in seeds:
        record = load_seed_oof(os.path.join(run_dir, f'oof_f{fold}_s{seed}.npz'))
        meta = record['_meta']
        if 'checkpoint_k' in meta and int(meta['checkpoint_k']) != int(cfg.train.checkpoint_k):
            log(f'  note: the OOF file was decoded at k={int(meta["checkpoint_k"])}, '
                f'the config says {cfg.train.checkpoint_k}; using the file\'s')
        k = int(meta.get('checkpoint_k', cfg.train.checkpoint_k))
        wids = [w for w in record if w != '_meta']
        if rec_folds is None and sorted(wids) != sorted(folds[fold]['val']):
            log(f'  fold split: the OOF file\'s wells differ from our fold {fold}')
            out['fold_split_ok'] = False
        wids = wids[:max_wells] if max_wells else wids
        recs = {w: build_image(**arrays_from_frames(store.read_hw(w), store.read_tw(w)), geom=cfg.geometry)
                for w in wids}
        ckpt = find_ckpt(ckpt_dir, fold, seed)
        log(f'[f{fold}_s{seed}] {len(wids)} wells, k={k}, {os.path.basename(ckpt)} on {device}')
        ours = decode_seed_oof(cfg, ckpt, wids, recs, truths, device, k=k)
        r = compare(ours, {w: record[w] for w in wids})
        log(f'  RMSE record {r["rmse_record"]:.4f}  replay {r["rmse_replay"]:.4f}  (diff {r["d_rmse"]:.4f})  '
            f'row diff max {r["max_d_pred"]:.4f} ft, p99 {r["p99_d_pred"]:.4f} ft  -> '
            f'{"PASS" if r["ok"] else "FAIL"}')
        out['seeds'][seed] = r
    out['ok'] = all(r['ok'] for r in out['seeds'].values()) and out['fold_split_ok'] is not False
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--run-dir', required=True, help='member run folder holding oof_f*_s*.npz')
    ap.add_argument('--ckpt-dir', default=None, help='folder holding ckpt_f*_s*.pt (default: --run-dir)')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--max-wells', type=int, default=None)
    a = ap.parse_args(argv)
    res = replay(C.load(a.config), a.data_root, a.run_dir, a.ckpt_dir, a.fold, a.seeds, a.max_wells)
    print('PASS' if res['ok'] else 'FAIL')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
