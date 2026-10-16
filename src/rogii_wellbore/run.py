"""A whole member: folds, pools, every (fold, seed) job, OOF files, summary and manifest.

    python -m rogii_wellbore.run --config configs/members/v2l.yaml --data-root <dir> --out <dir> \
        [--pretrained-dir <dir>] [--folds 0 1] [--seeds 34 68]

Output folder (named ``cfg.experiment_name()`` under ``--out`` unless ``--exact-out``):

* ``ckpt_f{fold}_s{seed}.pt``   best EMA state per job;  ``result_f{fold}_s{seed}.npz``  its scalars
* ``pretrain_f{fold}_s{seed}_ep{4,5}.pt``  when pretraining ran
* ``oof_f{fold}_s{seed}.npz``  per-seed out-of-fold predictions at ``checkpoint_k``
* ``summary.json``  config, per-fold and full-CV ensemble scores, diagnostics
* ``<name>_kaggle_inference_manifest.json``  what stage 03 reads
* ``fold_splits.json``,  ``config.yaml``

Finished jobs are skipped on re-run, so a Colab session that dies can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from . import config as C
from .cv import make_folds, save_folds
from .data import TruthCache, WellStore, build_hint_real_dist, constant_baseline
from .image import arrays_from_frames, build_image, build_image_synth
from .oof import decode_seed_oof, ensemble_fold, save_seed_oof
from .metrics import diagnostics, format_diagnostics, rmse
from .synth import build_pool, leakage_audit
from .train import train_one_run


def build_real_recs(store: WellStore, geom, log=print) -> dict:
    t0 = time.time(); recs = {}
    for wid in sorted(store.well_ids()):
        recs[wid] = build_image(**arrays_from_frames(store.read_hw(wid), store.read_tw(wid)), geom=geom)
    log(f'rendered {len(recs)} real wells in {time.time() - t0:.0f}s')
    return recs


def build_synth_recs(pool, geom) -> dict:
    out = {}
    for syn in pool:
        try:
            out[syn['_synth_id']] = build_image_synth(syn, geom)
        except Exception:      # the notebook skipped wells that fail to render
            pass
    return out


def _pretrained(pretrained_dir, fold, seed, epochs):
    if not pretrained_dir:
        return None
    p = os.path.join(pretrained_dir, f'pretrain_f{fold}_s{seed}_ep{epochs}.pt')
    return torch.load(p, map_location='cpu') if os.path.exists(p) else None


def run(cfg: C.RunConfig, data_root: str, out_dir: str, pretrained_dir: str | None = None,
        folds_subset=None, seeds_subset=None, device=None, encoder_weights='imagenet', log=print) -> dict:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    store = WellStore.train(data_root)
    truths = TruthCache(store)
    folds = make_folds(store, seed=cfg.cv.seed, n_folds=cfg.cv.n_folds)
    run_folds = list(folds_subset if folds_subset is not None else cfg.train.folds)
    seeds = list(seeds_subset if seeds_subset is not None else cfg.train.seeds)
    fold_const = {f: constant_baseline([truths(w) for w in folds[f]['val']]) for f in folds}
    save_folds(folds, os.path.join(out_dir, 'fold_splits.json'), fold_const)
    for f in run_folds:
        log(f'fold {f}: {len(folds[f]["train"])} train / {len(folds[f]["val"])} val, const {fold_const[f]:.3f}')
    real_recs = build_real_recs(store, cfg.geometry, log)

    results = {}
    for fold in run_folds:
        synth_recs = None
        for seed in seeds:
            label = f'f{fold}_s{seed}'
            ckpt = os.path.join(out_dir, f'ckpt_{label}.pt'); resf = os.path.join(out_dir, f'result_{label}.npz')
            if os.path.exists(ckpt) and os.path.exists(resf):
                d = np.load(resf, allow_pickle=True)
                results[(fold, seed)] = dict(best_rmse=float(d['best_rmse']), best_epoch=int(d['best_epoch']),
                                             final_epoch=int(d['final_epoch']), ckpt=ckpt)
                log(f'[{label}] resumed: {results[(fold, seed)]["best_rmse"]:.3f}')
                continue
            pt = _pretrained(pretrained_dir, fold, seed, cfg.train.pretrain_epochs)
            if pt is None and synth_recs is None:
                hint, _ = build_hint_real_dist(store, folds[fold]['train'])
                pool = build_pool(store, folds[fold]['train'], hint, fold, cfg.synth, cv_seed=cfg.cv.seed, log=log)
                leakage_audit(pool, folds[fold])
                synth_recs = build_synth_recs(pool, cfg.geometry)
                del pool
            log(f'--- [{label}] ---')
            r = train_one_run(cfg, fold, seed, folds, real_recs, synth_recs, truths, device, out_dir,
                              pretrained_state=pt, encoder_weights=encoder_weights, log=log)
            torch.save(r['best_state'], ckpt)
            np.savez(resf, best_rmse=r['best_rmse'], best_epoch=r['best_epoch'], final_epoch=r['final_epoch'],
                     val_traj=np.asarray(r['val_traj'], dtype=np.float32))
            results[(fold, seed)] = dict(best_rmse=r['best_rmse'], best_epoch=r['best_epoch'],
                                         final_epoch=r['final_epoch'], ckpt=ckpt)
            log(f'[{label}] best {r["best_rmse"]:.3f} at epoch {r["best_epoch"]}/{r["final_epoch"]}')

    # --- per-seed OOF, then the seed ensemble per fold --------------------------------------
    per_seed = {}
    for (fold, seed), r in sorted(results.items()):
        path = os.path.join(out_dir, f'oof_f{fold}_s{seed}.npz')
        if not os.path.exists(path):
            per_well = decode_seed_oof(cfg, r['ckpt'], folds[fold]['val'], real_recs, truths, device)
            save_seed_oof(path, per_well, fold, seed, cfg)
        per_seed[(fold, seed)] = path
    summary = summarize(cfg, folds, run_folds, seeds, per_seed, results, truths, fold_const, log)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, f'{cfg.name}_kaggle_inference_manifest.json'), 'w') as fh:
        json.dump(manifest(cfg, summary, results, out_dir), fh, indent=2)
    return summary


def summarize(cfg, folds, run_folds, seeds, per_seed, results, truths, fold_const, log=print) -> dict:
    residuals, fold_rmse, per_fold = {}, {}, {}
    for fold in run_folds:
        paths = [per_seed[(fold, s)] for s in seeds if (fold, s) in per_seed]
        res_f = ensemble_fold(paths, truths)
        residuals.update(res_f)
        fold_rmse[fold] = rmse(res_f.values())
        ss = np.array([fold_const[fold] - results[(fold, s)]['best_rmse'] for s in seeds if (fold, s) in results])
        per_fold[f'fold{fold}'] = {'ss_mean_margin': float(ss.mean()), 'ss_std_margin': float(ss.std(ddof=1)) if len(ss) > 1 else 0.0,
                                   'ens_rmse': fold_rmse[fold], 'ens_margin': fold_const[fold] - fold_rmse[fold],
                                   'fold_const': fold_const[fold], 'n_val_wells': len(folds[fold]['val'])}
        log(f'fold {fold}: ensemble of {len(paths)} seeds = {fold_rmse[fold]:.4f}')
    full = rmse(residuals.values())
    const = constant_baseline([truths(w) for f in run_folds for w in folds[f]['val']])
    diag = diagnostics(residuals, folds, fold_rmse)
    log(f'full CV ({len(run_folds)} folds): {full:.4f}  (const {const:.3f})\n' + format_diagnostics(diag))
    return {'name': cfg.name, 'experiment': cfg.experiment_name(), 'config': cfg.to_dict(),
            'folds': run_folds, 'seeds': seeds,
            'full_cv': {'ens_rmse': full, 'ens_margin': const - full, 'const_rmse': const,
                        'n_wells': len(residuals), 'n_post_ps_rows': int(sum(len(r) for r in residuals.values()))},
            'results_per_fold': per_fold, 'diagnostics': diag,
            'cells': [{'fold': f, 'seed': s, **{k: v for k, v in r.items() if k != 'ckpt'}} for (f, s), r in sorted(results.items())]}


def manifest(cfg: C.RunConfig, summary: dict, results: dict, out_dir: str) -> dict:
    """What the Kaggle notebook reads. Keeps the notebooks' field names (``recipe.encoder_name``,
    ``image_constants``, ``checkpoints[].filename``) and adds the temperatures as fields, which
    the old manifests only carried in free text."""
    g = cfg.geometry
    ckpts = [{'fold': f, 'seed': s, 'filename': os.path.basename(r['ckpt']),
              'size_bytes': os.path.getsize(r['ckpt']) if os.path.exists(r['ckpt']) else -1,
              'best_epoch': int(r['best_epoch']), 'best_rmse': float(r['best_rmse'])}
             for (f, s), r in sorted(results.items())]
    return {'format_version': 2, 'experiment_id': cfg.experiment_name(), 'name': cfg.name,
            'recipe': {'encoder_name': cfg.model.encoder, 'in_ch': cfg.model.in_ch, 'stack_n': 2,
                       'stack_handoff': 'softmax_posterior_plane',
                       'decoder_channels': list(cfg.model.decoder_channels), 'decoder_interp': cfg.model.decoder_interp,
                       'handoff_T': cfg.model.handoff_t, 'refeed_T': cfg.model.refeed_t,
                       'checkpoint_k': cfg.train.checkpoint_k},
            'image_constants': {'H': g.H, 'W': g.W, 'BIN_SIZE': g.bin_size, 'K_MAX': g.k_max},
            'input_channels': ['TW_GR_norm', 'HW_GR_norm', 'sparse_hint', 'cost_volume_norm', 'dZ_per_col_norm'],
            'full_cv_metric': summary['full_cv'], 'checkpoints': ckpts, 'n_checkpoints': len(ckpts)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exact-out', action='store_true', help='write into --out itself, not a named subfolder')
    ap.add_argument('--pretrained-dir', default=None)
    ap.add_argument('--folds', type=int, nargs='*')
    ap.add_argument('--seeds', type=int, nargs='*')
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    out = a.out if a.exact_out else os.path.join(a.out, cfg.experiment_name())
    print(f'{cfg.name}: {cfg.experiment_name()} -> {out}')
    run(cfg, a.data_root, out, pretrained_dir=a.pretrained_dir, folds_subset=a.folds, seeds_subset=a.seeds)


if __name__ == '__main__':
    main()
