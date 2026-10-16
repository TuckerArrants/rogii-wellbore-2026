"""Scoring members and blends from their OOF files (no GPU, no checkpoints).

    python -m rogii_wellbore.eval --config configs/eval/final_blend.yaml [--root <dir>] [--data-root <dir>]

The config lists members (run folders holding ``oof_f*_s*.npz``) and blend weights. For each
member: its seed-ensemble CV per fold and pooled. For the blend: the weighted per-row average
of the members' predictions, scored like a member, with the full diagnostics block, plus the
member-to-member error correlations (all rows and worst-decile wells).

Truth comes from ``--data-root`` (exact) or, without it, from ``pred_post + resid_post`` in the
OOF files, which works when every post-PS row has truth. Without the data, per-row predictions
are the mean of the seeds' stored float32 predictions rather than one decode of the mean
expected row, so scores can differ from the training-time numbers in the 4th decimal.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import yaml

from .data import TruthCache, WellStore
from .metrics import diagnostics, format_diagnostics, rmse
from .oof import ensemble_fold, member_predictions, truth_from_oof


def oof_paths(run_dir: str, folds=None) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(run_dir, 'oof_f*_s*.npz'))):
        f = int(re.search(r'oof_f(\d+)_s', os.path.basename(p)).group(1))
        if folds is None or f in folds:
            out.setdefault(f, []).append(p)
    if not out:
        raise FileNotFoundError(f'no oof_f*_s*.npz under {run_dir}')
    return out


def _residuals(preds: dict, truth: dict) -> dict:
    out = {}
    for w, p in preds.items():
        t = truth[w]
        ok = np.isfinite(t)
        out[w] = t[ok] - p[ok]
    return out


def evaluate_blend(members: dict, weights: dict, data_root: str | None = None, folds=None, log=print) -> dict:
    """``members``: name -> run dir. ``weights``: name -> weight (must sum to 1)."""
    total = float(sum(weights.values()))
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f'blend weights sum to {total}, not 1')
    paths = {n: oof_paths(d, folds) for n, d in members.items()}
    fold_ids = sorted(set.intersection(*[set(p) for p in paths.values()]))
    split = {}
    for n, by_fold in paths.items():
        for f in fold_ids:
            wids = [str(w) for w in np.load(by_fold[f][0], allow_pickle=True)['well_ids']]
            split.setdefault(f, set()).update(wids)
    folds_val = {f: {'val': sorted(split[f])} for f in fold_ids}

    if data_root:
        cache = TruthCache(WellStore.train(data_root))
        truth = {w: _full_post(cache(w)) for f in fold_ids for w in folds_val[f]['val']}
    else:
        truth = {}
        for by_fold in paths.values():
            for f in fold_ids:
                for w, t in truth_from_oof(by_fold[f]).items():
                    truth.setdefault(w, t)
        missing = [w for f in fold_ids for w in folds_val[f]['val'] if w not in truth]
        if missing:
            raise ValueError(f'{len(missing)} wells have post-PS rows without truth; pass --data-root')

    out = {'members': {}, 'weights': weights, 'folds': fold_ids}
    member_preds = {}
    for n, by_fold in paths.items():
        if data_root:
            res = {}
            for f in fold_ids:
                res.update(ensemble_fold(by_fold[f], cache))
            preds = None
        else:
            preds = {}
            for f in fold_ids:
                preds.update(member_predictions(by_fold[f]))
            res = _residuals(preds, truth)
        if preds is None:
            preds = {}
            for f in fold_ids:
                preds.update(member_predictions(by_fold[f]))
        member_preds[n] = preds
        per_fold = {f: rmse([res[w] for w in folds_val[f]['val']]) for f in fold_ids}
        out['members'][n] = {'full_cv': rmse(res.values()), 'per_fold': per_fold,
                             'n_seed_files': sum(len(v) for v in by_fold.values())}
        log(f'{n:>20}: full CV {out["members"][n]["full_cv"]:.4f}  per fold '
            + ' '.join(f'{per_fold[f]:.3f}' for f in fold_ids))

    blend = {w: sum(weights[n] * member_preds[n][w] for n in members) for w in truth}
    res_b = _residuals(blend, truth)
    fold_rmse = {f: rmse([res_b[w] for w in folds_val[f]['val']]) for f in fold_ids}
    diag = diagnostics(res_b, folds_val, fold_rmse)
    out['blend'] = {'full_cv': rmse(res_b.values()), 'per_fold': fold_rmse, 'diagnostics': diag}
    out['correlation'] = _correlations(member_preds, truth, res_b)
    log(f'{"blend":>20}: full CV {out["blend"]["full_cv"]:.4f}\n' + format_diagnostics(diag))
    log('error correlation (all rows | worst-decile wells):')
    for (a, b), (c_all, c_w) in out['correlation'].items():
        log(f'  {a} x {b}: {c_all:.3f} | {c_w:.3f}')
    out['correlation'] = {f'{a}|{b}': v for (a, b), v in out['correlation'].items()}
    return out


def _full_post(t) -> np.ndarray:
    v = np.full(len(t.true_post), np.nan)
    v[t.ok] = t.true_post[t.ok]
    return v


def _correlations(member_preds, truth, res_blend):
    names = list(member_preds)
    per_well_b = {w: float(np.sqrt(np.mean(r ** 2))) for w, r in res_blend.items() if len(r)}
    worst = sorted(per_well_b, key=per_well_b.get, reverse=True)[:max(1, len(per_well_b) // 10)]
    resid = {n: _residuals(member_preds[n], truth) for n in names}
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra = np.concatenate([resid[a][w] for w in truth]); rb = np.concatenate([resid[b][w] for w in truth])
            wa = np.concatenate([resid[a][w] for w in worst]); wb = np.concatenate([resid[b][w] for w in worst])
            out[(a, b)] = (float(np.corrcoef(ra, rb)[0, 1]), float(np.corrcoef(wa, wb)[0, 1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--root', default=None, help='folder the member paths are relative to')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--folds', type=int, nargs='*')
    ap.add_argument('--out', default=None, help='write the result as JSON here')
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    root = a.root or cfg.get('root', '.')
    members = {n: os.path.join(root, m['dir']) for n, m in cfg['members'].items()}
    weights = {n: float(m['weight']) for n, m in cfg['members'].items()}
    res = evaluate_blend(members, weights, data_root=a.data_root, folds=a.folds)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=2, default=float)


if __name__ == '__main__':
    main()
