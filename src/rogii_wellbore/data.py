"""Reading the competition files.

Layout (``<root>/train`` and ``<root>/test``): per well, ``<id>__horizontal_well.csv``
(MD, GR, Z, X, Y, TVT_input, and TVT in train) and ``<id>__typewell.csv`` (TVT, GR).
Post-PS rows are the rows where ``TVT_input`` is NaN.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HW_SUFFIX = '__horizontal_well.csv'
TW_SUFFIX = '__typewell.csv'


def list_wells(wells_dir: str) -> list[str]:
    """Sorted well ids that have both a horizontal-well and a typewell file."""
    files = set(os.listdir(wells_dir))
    return [f[:-len(HW_SUFFIX)] for f in sorted(files)
            if f.endswith(HW_SUFFIX) and (f[:-len(HW_SUFFIX)] + TW_SUFFIX) in files]


@dataclass(frozen=True)
class WellStore:
    """Paths into one split directory (train or test)."""
    wells_dir: str

    @classmethod
    def train(cls, data_root: str) -> 'WellStore':
        return cls(os.path.join(data_root, 'train'))

    @classmethod
    def test(cls, data_root: str) -> 'WellStore':
        return cls(os.path.join(data_root, 'test'))

    def hw_path(self, wid: str) -> str:
        return os.path.join(self.wells_dir, wid + HW_SUFFIX)

    def tw_path(self, wid: str) -> str:
        return os.path.join(self.wells_dir, wid + TW_SUFFIX)

    def well_ids(self) -> list[str]:
        return list_wells(self.wells_dir)

    def read_hw(self, wid: str, usecols=None) -> pd.DataFrame:
        return pd.read_csv(self.hw_path(wid), usecols=usecols)

    def read_tw(self, wid: str, usecols=None) -> pd.DataFrame:
        return pd.read_csv(self.tw_path(wid), usecols=usecols)


@dataclass(frozen=True)
class WellTruth:
    """What the scorer needs for one training well: native row count, post-PS row mask,
    post-PS truth, finite-truth mask and the last known TVT."""
    n_hw: int
    post: np.ndarray        # (n_hw,) bool
    true_post: np.ndarray   # (n_post,) float
    ok: np.ndarray          # (n_post,) bool, finite truth
    last_known: float


def read_truth(store: WellStore, wid: str) -> WellTruth:
    hw = store.read_hw(wid, usecols=['TVT', 'TVT_input'])
    post = hw['TVT_input'].isna().values
    true_post = hw['TVT'].values[post]
    known = hw.loc[~hw['TVT_input'].isna(), 'TVT_input'].values
    return WellTruth(n_hw=len(hw), post=post, true_post=true_post, ok=~np.isnan(true_post),
                     last_known=float(known[-1]) if len(known) else float('nan'))


class TruthCache:
    """Memoised ``read_truth``: the notebook re-read every validation CSV every epoch."""
    def __init__(self, store: WellStore):
        self.store = store
        self._get = functools.lru_cache(maxsize=None)(lambda wid: read_truth(store, wid))

    def __call__(self, wid: str) -> WellTruth:
        return self._get(wid)


def build_hint_real_dist(store: WellStore, train_wids) -> tuple[dict, pd.DataFrame]:
    """Empirical distributions the synthetic generator samples from: well length, PS position
    (percent of MD) and post-PS TVT range, over one fold's TRAINING wells only."""
    rows = []
    for wid in train_wids:
        hw = store.read_hw(wid, usecols=['MD', 'TVT', 'TVT_input'])
        n_md = len(hw)
        known = ~hw['TVT_input'].isna()
        if known.sum() < 1 or known.sum() >= n_md:
            continue
        ps_md_idx = int(known.sum())
        ps_md_pct = 100.0 * ps_md_idx / n_md
        post_tvt = hw['TVT'].values[ps_md_idx:]
        post_ok = ~np.isnan(post_tvt)
        if post_ok.sum() < 10:
            continue
        tvt_range = float(np.nanmax(post_tvt) - np.nanmin(post_tvt))
        rows.append({'well_id': wid, 'n_md': n_md, 'ps_md_pct': ps_md_pct, 'tvt_range': tvt_range})
    df = pd.DataFrame(rows)
    return {'n_md': df['n_md'].values, 'ps_md_pct': df['ps_md_pct'].values,
            'tvt_range': df['tvt_range'].values}, df


def constant_baseline(truths) -> float:
    """RMSE of predicting the last known TVT on every post-PS row (the 'const' column)."""
    sq = [(t.true_post[t.ok] - t.last_known) ** 2 for t in truths if t.post.sum() > 0]
    return float(np.sqrt(np.concatenate(sq).mean()))
