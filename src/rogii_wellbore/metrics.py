"""Decoding predictions back to the well's rows, and the error read-outs every run reports.

Decode: column c's expected row ``er`` becomes TVT ``tvt_lo + (er + 0.5) * bin_size`` and is
copied to every native row in that column's bin. The score is RMSE over all post-PS rows with
finite truth, pooled over wells. Read-outs (all on per-row residuals ``truth - pred``):

* offset / drift: each well's mean residual vs what is left (``datum`` in the notebooks);
* tail: per-well RMSE percentiles, counts above 8/10/12/15 ft, worst-k% share of squared error;
* position: RMSE in 20 equal bins from heel (PS point) to toe, and heel/toe means of 4 bins;
* residual modes: share of energy in the first 1 and 8 half-sine modes.
"""
from __future__ import annotations

import numpy as np

from .data import WellTruth

POSITION_BINS = 20
TAIL_THRESH = (8, 10, 12, 15)
WORST_FRACS = (0.05, 0.10, 0.20)
PCTLS = (50, 75, 90, 95, 99)


def decode_rows(er_row: np.ndarray, tvt_lo: float, bin_size: float, n_hw: int, W: int) -> tuple:
    """Per-column TVT and the per-native-row prediction (float32, as the notebooks stored it)."""
    pred_col = tvt_lo + (er_row + 0.5) * bin_size
    bin_edges = np.linspace(0, n_hw, W + 1).astype(int)
    pred_full = np.zeros(n_hw, dtype=np.float32)
    for c in range(W):
        pred_full[bin_edges[c]:bin_edges[c + 1]] = pred_col[c]
    return pred_col, pred_full


def well_residual(pred_full: np.ndarray, truth: WellTruth) -> tuple:
    pred_post = pred_full[truth.post]
    return pred_post, truth.true_post[truth.ok] - pred_post[truth.ok]


def rmse(residuals) -> float:
    r = np.concatenate([np.asarray(x, dtype=np.float64) for x in residuals])
    return float(np.sqrt((r ** 2).mean()))


def position_sq(resid: np.ndarray, bins: int = POSITION_BINS):
    """(sum of squares, count) per heel-to-toe bin for one well."""
    s = np.zeros(bins); n = np.zeros(bins, dtype=np.int64)
    if len(resid) == 0:
        return s, n
    pos = np.linspace(0, 1, len(resid), endpoint=False) + 0.5 / len(resid)
    idx = np.clip((pos * bins).astype(int), 0, bins - 1)
    for k in range(bins):
        sel = idx == k
        if sel.any():
            s[k] += float(np.sum(resid[sel] ** 2)); n[k] += int(sel.sum())
    return s, n


def position_rmse(residuals: dict, bins: int = POSITION_BINS) -> np.ndarray:
    s = np.zeros(bins); n = np.zeros(bins, dtype=np.int64)
    for r in residuals.values():
        a, b = position_sq(np.asarray(r, dtype=np.float64), bins); s += a; n += b
    return np.sqrt(s / np.maximum(n, 1))


def offset_drift(residuals: dict, wids=None) -> dict | None:
    b, w, pwb = [], [], []
    for wid in (residuals if wids is None else wids):
        r = np.asarray(residuals.get(wid, []), dtype=np.float64)
        if r.size == 0:
            continue
        mu = r.mean(); b.append(np.full(r.size, mu)); w.append(r - mu); pwb.append(mu)
    if not b:
        return None
    b = np.concatenate(b); w = np.concatenate(w); pwb = np.asarray(pwb)
    return {'datum_rms': float(np.sqrt((b ** 2).mean())), 'drift_rms': float(np.sqrt((w ** 2).mean())),
            'datum_share_of_var': float((b ** 2).mean() / max((b ** 2).mean() + (w ** 2).mean(), 1e-12)),
            'per_well_bias_mean': float(pwb.mean()), 'per_well_bias_std': float(pwb.std()),
            'n_wells': int(pwb.size)}


def tail_stats(residuals: dict, wids=None) -> dict | None:
    rmse_w, sse_w, n_w = [], [], []
    for wid in (residuals if wids is None else wids):
        r = np.asarray(residuals.get(wid, []), dtype=np.float64)
        if r.size == 0:
            continue
        rmse_w.append(float(np.sqrt((r ** 2).mean()))); sse_w.append(float((r ** 2).sum())); n_w.append(r.size)
    if not rmse_w:
        return None
    rmse_w = np.asarray(rmse_w); sse_w = np.asarray(sse_w); n_w = np.asarray(n_w)
    order = np.argsort(rmse_w)[::-1]; tot = float(sse_w.sum())
    return {'n_wells': int(rmse_w.size),
            'rmse_pct': {f'p{p}': float(np.percentile(rmse_w, p)) for p in PCTLS},
            'rmse_max': float(rmse_w.max()), 'rmse_mean': float(rmse_w.mean()),
            'catastrophic_counts': {f'gt_{t}ft': int((rmse_w > t).sum()) for t in TAIL_THRESH},
            'catastrophic_frac': {f'gt_{t}ft': float((rmse_w > t).mean()) for t in TAIL_THRESH},
            'worst_frac_sse_share': {f'worst_{int(fr * 100)}pct':
                                     float(sse_w[order[:max(1, int(fr * rmse_w.size))]].sum() / max(tot, 1e-12))
                                     for fr in WORST_FRACS},
            'pooled_rmse': float(np.sqrt(tot / max(int(n_w.sum()), 1)))}


def heel_toe(pos_rmse) -> dict:
    pr = np.asarray(pos_rmse, dtype=np.float64)
    return {'heel': float(pr[:4].mean()), 'toe': float(pr[-4:].mean()), 'gap': float(pr[-4:].mean() - pr[:4].mean())}


def residual_modes(residuals: dict, n_modes: int = 8, n_resample: int = 200) -> dict:
    wids = [w for w, r in residuals.items() if len(r) >= 10]
    x_new = np.linspace(0, 1, n_resample, endpoint=False) + 0.5 / n_resample
    modes = np.stack([np.sqrt(2) * np.sin((2 * k - 1) * np.pi * x_new / 2) for k in range(1, n_modes + 1)])
    c_sq = np.zeros((len(wids), n_modes)); energies = np.zeros(len(wids))
    for i, w in enumerate(wids):
        r = residuals[w]
        x_orig = np.linspace(0, 1, len(r), endpoint=False) + 0.5 / len(r)
        rr = np.interp(x_new, x_orig, r)
        c_sq[i] = ((modes @ rr) / n_resample) ** 2
        energies[i] = (rr ** 2).sum() / n_resample
    return {'pool_mode1': float(c_sq[:, 0].sum() / energies.sum()),
            'pool_modes_1_8': float(c_sq.sum() / energies.sum())}


def diagnostics(residuals: dict, folds: dict, fold_rmse: dict) -> dict:
    """The ``diagnostics`` block of a run summary: pooled and per fold."""
    pos_fold = {f: position_rmse({w: residuals[w] for w in folds[f]['val'] if w in residuals}) for f in fold_rmse}
    pos_pooled = np.sqrt(np.mean([pos_fold[f] ** 2 for f in fold_rmse], axis=0))
    out = {'pooled': {'full_cv_rmse': rmse(residuals.values()), 'datum_drift': offset_drift(residuals),
                      'tail': tail_stats(residuals), 'heel_toe': heel_toe(pos_pooled)},
           'per_fold': {f'fold{f}': {'ens_rmse': fold_rmse[f],
                                     'datum_drift': offset_drift(residuals, [w for w in folds[f]['val'] if w in residuals]),
                                     'tail': tail_stats(residuals, [w for w in folds[f]['val'] if w in residuals]),
                                     'heel_toe': heel_toe(pos_fold[f])} for f in fold_rmse},
           'per_position_rmse': {'bins': POSITION_BINS, 'pooled': [float(v) for v in pos_pooled],
                                 **{f'fold{f}': [float(v) for v in pos_fold[f]] for f in fold_rmse}},
           'residual_modes': residual_modes(residuals)}
    return out


def format_diagnostics(d: dict) -> str:
    lines = [f'  {"fold":<6}{"rmse":>7}{"offset":>8}{"drift":>8}{"w10%SSE":>9}{">12ft":>7}{"toe-heel":>9}']
    rows = [(k.replace('fold', ''), v) for k, v in d['per_fold'].items()]
    rows.append(('POOL', {'ens_rmse': d['pooled']['full_cv_rmse'], **d['pooled']}))
    for name, v in rows:
        dd, tl, ht = v['datum_drift'], v['tail'], v['heel_toe']
        lines.append(f'  {name:<6}{v["ens_rmse"]:>7.3f}{dd["datum_rms"]:>8.2f}{dd["drift_rms"]:>8.2f}'
                     f'{100 * tl["worst_frac_sse_share"]["worst_10pct"]:>8.0f}%'
                     f'{tl["catastrophic_counts"]["gt_12ft"]:>7d}{ht["gap"]:>9.2f}')
    return '\n'.join(lines)
