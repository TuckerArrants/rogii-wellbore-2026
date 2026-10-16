"""Turning one well into the 5-channel image the U-Net reads.

Columns: W equal-width bins along MD (``np.linspace(0, n_rows, W + 1)`` edges). Rows: H bins
of ``bin_size`` feet in TVT, centred on the last known (pre-PS) TVT. Channels:

0. ``TW_GR``  the typewell GR at each row's TVT (constant across columns), z-scored;
1. ``HW_GR``  the well's mean GR in each column (constant down rows), z-scored;
2. ``hint``   1 at the known TVT row of each pre-PS column, else 0;
3. ``cost``   (HW_GR - TW_GR)^2 per row and column, z-scored and clipped to +-3;
4. ``dZ``     the well's mean dZ/dMD per column, z-scored, clipped to +-3.

The typewell is sampled ``k_max`` rows past both edges so anchor jitter can shift the window
without re-reading the log. Shared by training, validation and inference (the notebooks kept
two copies; see docs/PORTING.md for the one degenerate-case difference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Geometry


def dz_per_col(hw_z, hw_md, bin_edges, W):
    """Mean dZ/dMD per column; NaN where a column has no rows."""
    n = len(hw_z)
    if n < 2:
        return np.full(W, 0.0, dtype=np.float32)
    dz_per_row = np.gradient(np.asarray(hw_z, dtype=np.float64), np.asarray(hw_md, dtype=np.float64))
    dz = np.full(W, np.nan, dtype=np.float32)
    for c in range(W):
        lo, hi = bin_edges[c], bin_edges[c + 1]
        if hi <= lo:
            continue
        dz[c] = float(np.mean(dz_per_row[lo:hi]))
    return dz


def dz_channel(dz_col, hw_mask, H, W):
    """(H, W) dZ channel: columns without data take the well mean, then z-score, clip, broadcast."""
    bad = (~hw_mask) | (~np.isfinite(dz_col))
    fill = float(np.nanmean(dz_col[~bad])) if (~bad).any() else 0.0
    dz = np.where(bad, fill, dz_col).astype(np.float32)
    m, s = dz.mean(), dz.std()
    dz_norm = (dz - m) / s if s > 1e-6 else np.zeros_like(dz)
    dz_norm = np.clip(dz_norm, -3.0, 3.0)
    return np.broadcast_to(dz_norm[None, :], (H, W)).copy().astype(np.float32)


def _zscore(a):
    m, s = a.mean(), a.std()
    return (a - m) / s if s > 1e-6 else a


def _cost_channel(ch_hw_raw, ch_tw_raw, hw_mask):
    cost = (ch_hw_raw - ch_tw_raw) ** 2
    mean_cost = float(cost[:, hw_mask].mean()) if hw_mask.any() else 0.0
    cost[:, ~hw_mask] = mean_cost
    cm, cs = cost.mean(), cost.std()
    norm = (cost - cm) / cs if cs > 1e-6 else np.zeros_like(cost)
    return np.clip(norm, -3.0, 3.0)


def build_image(hw_md, hw_gr, hw_z, hw_tvt, hw_tvti, tw_tvt, tw_gr, geom: Geometry) -> dict:
    """Render one well. ``hw_tvt`` may be all-NaN (test wells); it only sets ``truth_row``."""
    H, W, BIN, K = geom.H, geom.W, geom.bin_size, geom.k_max
    n_hw = len(hw_md)
    known_tvt = pd.Series(hw_tvti).dropna().values
    if len(known_tvt) == 0:
        raise ValueError('No pre-PS rows')
    depth_center = float(known_tvt[-1])
    tvt_lo = depth_center - (H / 2) * BIN

    def tvt_to_row(t):
        if np.isnan(t):
            return -1
        r = int((t - tvt_lo) / BIN)
        return r if 0 <= r < H else -1

    bin_edges = np.linspace(0, n_hw, W + 1).astype(int)
    hw_gr_col_raw = np.full(W, np.nan, dtype=np.float32)
    hw_mask = np.zeros(W, dtype=bool)
    truth_row = np.full(W, -1, dtype=np.int32)
    hint_row = np.full(W, -1, dtype=np.int32)
    post_ps_mask = np.zeros(W, dtype=bool)
    for c in range(W):
        lo, hi = bin_edges[c], bin_edges[c + 1]
        if hi <= lo:
            continue
        gr_slc = hw_gr[lo:hi]
        if np.any(~np.isnan(gr_slc)):
            hw_gr_col_raw[c] = np.nanmean(gr_slc)
            hw_mask[c] = True
        tvt_slc = hw_tvt[lo:hi]
        if np.any(~np.isnan(tvt_slc)):
            truth_row[c] = tvt_to_row(np.nanmedian(tvt_slc))
        tvti_slc = hw_tvti[lo:hi]
        is_known = ~np.isnan(tvti_slc)
        if is_known.any():
            hint_row[c] = tvt_to_row(np.nanmedian(tvti_slc[is_known]))
        else:
            post_ps_mask[c] = True
    if np.all(np.isnan(hw_gr_col_raw)):
        hw_gr_col = np.zeros(W, dtype=np.float32)
    else:
        hw_gr_col = hw_gr_col_raw.copy()
        hw_gr_col[np.isnan(hw_gr_col)] = np.nanmean(hw_gr_col_raw)

    ext_row_tvts = (tvt_lo - K * BIN) + (np.arange(H + 2 * K) + 0.5) * BIN
    sort_idx = np.argsort(tw_tvt)
    tw_tvt_s = tw_tvt[sort_idx]; tw_gr_s = tw_gr[sort_idx]
    ok = ~np.isnan(tw_gr_s)
    if ok.sum() < 2:
        tw_gr_row_ext = np.zeros(H + 2 * K, dtype=np.float32)
    else:
        tw_gr_row_ext = np.interp(ext_row_tvts, tw_tvt_s[ok], tw_gr_s[ok],
                                  left=tw_gr_s[ok][0], right=tw_gr_s[ok][-1]).astype(np.float32)

    rec = dict(truth_row=truth_row, hint_row=hint_row, post_ps_mask=post_ps_mask,
               depth_center=depth_center, tvt_lo=tvt_lo, hw_mask=hw_mask, bin_edges=bin_edges,
               dz_per_col=dz_per_col(hw_z, hw_md, bin_edges, W),
               tw_gr_per_row_ext=tw_gr_row_ext, hw_gr_per_col=hw_gr_col)
    image, _, _ = render(rec, 0, geom)
    rec['image_in'] = image
    return rec


def render(rec: dict, k: int, geom: Geometry):
    """The image at anchor-jitter offset ``k`` (rows), plus the shifted truth and loss mask.
    ``k = 0`` is the canonical image; training draws k uniformly from [-k_max, k_max]."""
    H, W, K = geom.H, geom.W, geom.k_max
    tw_row = rec['tw_gr_per_row_ext'][K + k: K + k + H]
    ch_tw_raw = np.broadcast_to(tw_row[:, None], (H, W)).copy()
    ch_hw_raw = np.broadcast_to(rec['hw_gr_per_col'][None, :], (H, W)).copy()
    ch_hint = np.zeros((H, W), dtype=np.float32)
    hint = rec['hint_row'] - k
    for c in range(W):
        if not rec['post_ps_mask'][c] and 0 <= hint[c] < H:
            ch_hint[hint[c], c] = 1.0
    cost = _cost_channel(ch_hw_raw, ch_tw_raw, rec['hw_mask'])
    ch_dz = dz_channel(rec['dz_per_col'], rec['hw_mask'], H, W)
    image = np.stack([_zscore(ch_tw_raw.copy()), _zscore(ch_hw_raw.copy()), ch_hint, cost, ch_dz],
                     axis=0).astype(np.float32)
    truth = rec['truth_row'] - k
    mask = rec['post_ps_mask'] & (rec['truth_row'] >= 0) & (truth >= 0) & (truth < H)
    return image, np.where(mask, truth, 0).astype(np.int64), mask


def arrays_from_frames(hw: pd.DataFrame, tw: pd.DataFrame, with_truth: bool = True) -> dict:
    """The raw arrays ``build_image`` takes, from the two CSV frames of one well."""
    return dict(hw_md=hw['MD'].values, hw_gr=hw['GR'].values, hw_z=hw['Z'].values,
                hw_tvt=hw['TVT'].values if with_truth else np.full(len(hw), np.nan),
                hw_tvti=hw['TVT_input'].values, tw_tvt=tw['TVT'].values, tw_gr=tw['GR'].values)


def build_image_synth(syn: dict, geom: Geometry) -> dict:
    return build_image(syn['MD'], syn['GR'], syn['Z'], syn['TVT'], syn['TVT_input'],
                       syn['_tw_tvt'], syn['_tw_gr'], geom)


def val_example(rec: dict):
    """Canonical image, truth and mask for scoring (no jitter)."""
    truth = rec['truth_row']
    mask = rec['post_ps_mask'] & (truth >= 0)
    return rec['image_in'], np.where(mask, truth, 0).astype(np.int64), mask
