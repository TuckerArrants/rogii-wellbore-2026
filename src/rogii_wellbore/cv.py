"""The 5-fold split used by every run in the project.

Wells are shuffled with ``RandomState(seed)``, ranked into quintiles of whole-well TVT range,
and each quintile is shuffled again (``RandomState(seed + stratum)``) and dealt into 5 equal
chunks; the remainder of a quintile goes to the last fold. With seed 34 on the 773 training
wells this gives 153/153/153/153/161 validation wells.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import WellStore


def make_folds(store: WellStore, seed: int = 34, n_folds: int = 5,
               well_ids: list[str] | None = None) -> dict[int, dict[str, list[str]]]:
    well_ids = store.well_ids() if well_ids is None else list(well_ids)
    rng = np.random.RandomState(seed)
    well_pool = list(well_ids)
    rng.shuffle(well_pool)
    meta = []
    for wid in well_pool:
        hw = store.read_hw(wid, usecols=['MD', 'TVT', 'TVT_input'])
        meta.append({'well_id': wid, 'n_rows': len(hw), 'tvt_range': hw['TVT'].max() - hw['TVT'].min()})
    meta_df = pd.DataFrame(meta)
    n_strata = n_folds
    meta_df['stratum'] = pd.qcut(meta_df['tvt_range'], q=n_strata, labels=list(range(n_strata)))
    assignment = {}
    for s in range(n_strata):
        pool = meta_df[meta_df['stratum'] == s]['well_id'].tolist()
        np.random.RandomState(seed + s).shuffle(pool)
        chunk = len(pool) // n_folds
        for f in range(n_folds):
            lo = f * chunk
            hi = (f + 1) * chunk if f < n_folds - 1 else len(pool)
            for wid in pool[lo:hi]:
                assignment[wid] = f
    return {f: {'val': [w for w, fa in assignment.items() if fa == f],
                'train': [w for w, fa in assignment.items() if fa != f]}
            for f in range(n_folds)}


def save_folds(folds: dict, path: str, fold_const: dict | None = None) -> None:
    """The notebook's ``ft39l_fold_splits.json`` layout."""
    out = {f'fold{f}': {'val': v['val'], 'train': v['train'], 'n_val': len(v['val']),
                        'n_train': len(v['train']),
                        **({'fold_const': float(fold_const[f])} if fold_const else {})}
           for f, v in folds.items()}
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)


def load_folds(path: str) -> dict[int, dict[str, list[str]]]:
    with open(path) as fh:
        raw = json.load(fh)
    return {int(k.replace('fold', '')): {'val': v['val'], 'train': v['train']} for k, v in raw.items()}
