"""Per-seed out-of-fold predictions, and the seed / member ensembles built from them.

File format (``oof_f{fold}_s{seed}.npz``, as the training notebook wrote it; object arrays hold
one ragged vector per well, in validation order):

    well_ids (N,)  er_row  pred_col  pred_post  resid_post   rmse fold seed checkpoint_k refeed_T handoff_T

``er_row`` is the expected row per column (W,), ``pred_col`` the TVT per column, ``pred_post``
the prediction on every post-PS native row, ``resid_post`` truth minus prediction on the post-PS
rows with finite truth. Files written by this package add ``tvt_lo`` (N,) and ``bin_size``.

The seed ensemble averages ``er_row`` over seeds and decodes once, as the notebook's
``_eval_model_pool`` did; because the decode is linear this equals averaging the per-seed
predictions up to float32 rounding. A member blend averages members' per-row predictions
with weights, as the Kaggle kernel did.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import ValDataset
from .metrics import decode_rows, well_residual
from .model import StackedUNet, expected_row, iter_logits, load_checkpoint


def decode_seed_oof(cfg, ckpt_path, val_ids, recs, truths, device, k=None, batch_size=8) -> dict:
    k = int(cfg.train.checkpoint_k if k is None else k)
    geom = cfg.geometry
    m = load_checkpoint(StackedUNet(cfg.model, encoder_weights=None, train_mode=False), ckpt_path).to(device).eval()
    loader = DataLoader(ValDataset(recs, val_ids), batch_size=batch_size, shuffle=False)
    row_idx = torch.arange(geom.H, device=device, dtype=torch.float32)
    out = {}
    with torch.no_grad():
        for batch in loader:
            kl, _ = iter_logits(m, batch['image'].to(device), k)
            er = expected_row(kl[k], row_idx).cpu().numpy()
            for b, wid in enumerate(batch['well_id']):
                rec, t = recs[wid], truths(wid)
                pred_col, pred_full = decode_rows(er[b], rec['tvt_lo'], geom.bin_size, t.n_hw, geom.W)
                pred_post, resid = well_residual(pred_full, t)
                out[wid] = dict(er_row=er[b].astype(np.float32), pred_col=pred_col.astype(np.float32),
                                pred_post=pred_post.astype(np.float32), resid_post=resid.astype(np.float32),
                                tvt_lo=float(rec['tvt_lo']))
    return out


def save_seed_oof(path, per_well: dict, fold, seed, cfg) -> float:
    wids = list(per_well)

    def obj(key):
        a = np.empty(len(wids), dtype=object)
        for i, w in enumerate(wids):
            a[i] = per_well[w][key]
        return a

    resid = [per_well[w]['resid_post'] ** 2 for w in wids]
    r = float(np.sqrt(np.concatenate(resid).mean())) if resid else float('nan')
    np.savez(path, well_ids=np.array(wids), er_row=obj('er_row'), pred_col=obj('pred_col'),
             pred_post=obj('pred_post'), resid_post=obj('resid_post'),
             tvt_lo=np.array([per_well[w]['tvt_lo'] for w in wids]), bin_size=cfg.geometry.bin_size,
             rmse=r, fold=fold, seed=seed, checkpoint_k=int(cfg.train.checkpoint_k),
             refeed_T=float(cfg.model.refeed_t), handoff_T=float(cfg.model.handoff_t))
    return r


def load_seed_oof(path) -> dict:
    """{well_id: dict(er_row, pred_col, pred_post, resid_post[, tvt_lo])} plus '_meta'."""
    d = np.load(path, allow_pickle=True)
    wids = [str(w) for w in d['well_ids']]
    out = {}
    for i, w in enumerate(wids):
        out[w] = {k: np.asarray(d[k][i]) for k in ('er_row', 'pred_col', 'pred_post', 'resid_post')}
        if 'tvt_lo' in d:
            out[w]['tvt_lo'] = float(d['tvt_lo'][i])
    out['_meta'] = {k: d[k].item() for k in ('rmse', 'fold', 'seed', 'checkpoint_k', 'refeed_T', 'handoff_T') if k in d}
    return out


def ensemble_fold(paths, truths, bin_size=None, W=None) -> dict:
    """Seed ensemble for one fold from its per-seed OOF files: mean ``er_row``, one decode.
    Returns {well_id: residuals}. ``tvt_lo`` comes from the files when stored, else it is
    recovered from ``pred_col`` (float32, so the result can differ from the notebook's by ~1e-3 ft)."""
    seeds = [load_seed_oof(p) for p in paths]
    wids = [w for w in seeds[0] if w != '_meta']
    out = {}
    for w in wids:
        ers = np.stack([s[w]['er_row'] for s in seeds], axis=0)
        er_mean = ers.mean(axis=0)
        W_ = W or ers.shape[1]
        if 'tvt_lo' in seeds[0][w]:
            tvt_lo = seeds[0][w]['tvt_lo']
            bs = bin_size if bin_size is not None else _bin_from(seeds[0][w])
        else:
            bs = bin_size if bin_size is not None else _bin_from(seeds[0][w])
            tvt_lo = float(np.median(seeds[0][w]['pred_col'].astype(np.float64)
                                     - (seeds[0][w]['er_row'].astype(np.float64) + 0.5) * bs))
        t = truths(w)
        _, pred_full = decode_rows(er_mean, tvt_lo, bs, t.n_hw, W_)
        out[w] = well_residual(pred_full, t)[1]
    return out


def _bin_from(rec) -> float:
    """Bin size from two columns of one well: d(pred_col) / d(er_row)."""
    er = rec['er_row'].astype(np.float64); pc = rec['pred_col'].astype(np.float64)
    i, j = int(np.argmin(er)), int(np.argmax(er))
    return float(round((pc[j] - pc[i]) / (er[j] - er[i]), 4)) if er[j] > er[i] else 1.0


def member_predictions(paths) -> dict:
    """Per-row prediction of one member on the post-PS rows: the mean of its seeds' ``pred_post``,
    per well, over all folds in ``paths``. Needs no truth or geometry."""
    acc, cnt = {}, {}
    for p in paths:
        d = load_seed_oof(p)
        for w, v in d.items():
            if w == '_meta':
                continue
            acc[w] = acc.get(w, 0.0) + v['pred_post'].astype(np.float64)
            cnt[w] = cnt.get(w, 0) + 1
    return {w: acc[w] / cnt[w] for w in acc}


def truth_from_oof(paths) -> dict:
    """Post-PS truth recovered as pred_post + resid_post, for wells whose post-PS rows all have
    truth (the OOF files do not store which rows were dropped)."""
    out = {}
    for p in paths:
        d = load_seed_oof(p)
        for w, v in d.items():
            if w == '_meta' or w in out:
                continue
            if len(v['pred_post']) == len(v['resid_post']):
                out[w] = v['pred_post'].astype(np.float64) + v['resid_post'].astype(np.float64)
    return out
