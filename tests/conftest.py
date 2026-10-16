"""Shared test fixtures.

``nb`` loads functions from the notebooks of record in ``pipeline/`` so the port can be compared
with them directly: only top-level ``def`` / ``class`` nodes of a cell are executed, never its
top-level statements (which mount Drive, read the data and train), and the notebook's module
globals are supplied explicitly.

``fake_data`` writes a small competition-shaped data folder (train + test) built from
synthetic wells, so nothing here needs the real data.
"""
from __future__ import annotations

import ast
import json
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / 'pipeline/01_train/2SUNET_MASTER.ipynb'
INFER = REPO / 'pipeline/03_submit/final-submission-equal-weight-no-tta.ipynb'


def cell_source(path: Path, idx: int) -> str:
    nb = json.loads(path.read_text())
    src = ''.join(nb['cells'][idx]['source'])
    return '\n'.join(l for l in src.splitlines() if not l.lstrip().startswith(('!', '%')))


def load_defs(path: Path, cells, namespace: dict, names=None) -> dict:
    """Exec the def/class nodes of ``cells`` (in order) into ``namespace``."""
    for idx in cells:
        tree = ast.parse(cell_source(path, idx))
        keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and (names is None or n.name in names)]
        exec(compile(ast.Module(body=keep, type_ignores=[]), f'{path.name}[{idx}]', 'exec'), namespace)
    return namespace


# the notebook's module globals of record (cells 2-3 and 24 of 2SUNET_MASTER.ipynb, launch values)
MASTER_GLOBALS = dict(
    V3_WOBBLE_TOTAL_PCT=0.15, V3_WOBBLE_SPLIT=(0.50, 0.30, 0.20), V3_WAVELENGTH_DIVISORS=(1.0, 5.0, 20.0),
    DZ_SYNTH_DIP_FRAC=0.10, DZ_SYNTH_DIP_AUTOCORR_FRAC=0.20, DZ_SYNTH_NOISE_STD=0.5,
    EVENT_PACK='v3_expA', DIPBREAK_PROB=0.35, DIPBREAK_SECOND_P=0.30, DIPBREAK_FAULT_COUPLE_P=0.30,
    DIPBREAK_DSLOPE_MIN=0.008, DIPBREAK_DSLOPE_MAX=0.035, PINCH_PROB=0.00, PINCH_H_MIN=10.0, PINCH_H_MAX=30.0,
    PINCH_C_MIN=0.05, PINCH_C_MAX=0.20, REPEAT_PROB=0.12, REPEAT_SHIFT_MIN=10.0, REPEAT_SHIFT_MAX=40.0,
    REPEAT_DAMP_MIN=0.2, REPEAT_DAMP_MAX=0.5, REPEAT_L_MIN=150, REPEAT_L_MAX=400,
    EXPA_MAX_ABS_DIP=0.050, EXPA_EXC_CAP_FT=320.0, EXPA_HASH=3714159261,
    FZ_ENABLE=False, FZ_HASH=1414213562,
    FAULT_PROB=0.33, FAULT_MAGNITUDE_MIN=5.0, FAULT_MAGNITUDE_MAX=50.0,
    UNCONF_PROB=0.33, UNCONF_MAX_CUMULATIVE_SHIFT=30.0,
    VAR_BED_PROB=0.50, VAR_BED_AMP_STD=0.40, VAR_BED_AUTOCORR_LENGTH=200, VAR_BED_AMP_MIN=0.30, VAR_BED_AMP_MAX=2.00,
    SIGMA_DRIFT=0.5, SIGMA_TVT_OFFSET=0.2, SIGMA_MEAS=0.5, SIGMA_CALIB=0.06,
)


def master_namespace(H=256, W=512, BIN_SIZE=1.0, K_MAX=12, **extra) -> dict:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, F=F, os=os, Dataset=Dataset, DataLoader=DataLoader)
    ns.update(MASTER_GLOBALS)
    ns.update(H=H, W=W, BIN_SIZE=BIN_SIZE, K_MAX=K_MAX)
    ns.update(extra)
    return ns


def offline_smp():
    """``segmentation_models_pytorch`` with ``encoder_weights`` forced to None, so the notebook's
    ``smp.Unet(..., encoder_weights='imagenet')`` builds without a download."""
    import segmentation_models_pytorch as smp

    def Unet(*a, **kw):
        kw['encoder_weights'] = None
        return smp.Unet(*a, **kw)
    return types.SimpleNamespace(Unet=Unet)


# ---------------------------------------------------------------------------------------------
# fake data
# ---------------------------------------------------------------------------------------------
def _typewell(rng, n=1200):
    t = np.arange(0.0, n * 0.5, 0.5) + 11000.0
    gr = (80 + 35 * np.sin(t / rng.uniform(5, 9)) + 20 * np.sin(t / rng.uniform(18, 30) + rng.uniform(0, 6))
          + 8 * np.sin(t / rng.uniform(1.5, 3)) + rng.normal(0, 2, len(t)))
    return pd.DataFrame({'TVT': t, 'GR': np.clip(gr, 10, 250)})


def write_fake_data(root: Path, n_train=30, n_test=3, seed=0) -> Path:
    from rogii_wellbore.config import SynthConfig
    from rogii_wellbore.synth import generate_well
    rng = np.random.RandomState(seed)
    hint = {'n_md': np.array([1400, 1700, 2100]), 'ps_md_pct': np.array([30.0, 40.0, 45.0]),
            'tvt_range': np.array([15.0, 25.0, 35.0])}
    cfg = SynthConfig(corruption='reduced_B')
    for split, n in (('train', n_train), ('test', n_test)):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        i = 0; s = 0
        while i < n:
            tw = _typewell(rng)
            w = generate_well(tw, hint, seed=1000 * (split == 'test') + s, cfg=cfg); s += 1
            if w is None:
                continue
            wid = f'{split[:2]}{i:03d}ab'
            hw = pd.DataFrame({'MD': w['MD'].astype(np.float64) + 9000.0, 'GR': w['GR'].astype(np.float64),
                               'Z': w['Z'].astype(np.float64) - 3000.0, 'X': 0.0, 'Y': 0.0,
                               'TVT_input': w['TVT_input'].astype(np.float64)})
            if split == 'train':
                hw['TVT'] = w['TVT'].astype(np.float64)
            hw.to_csv(d / f'{wid}__horizontal_well.csv', index=False)
            tw.to_csv(d / f'{wid}__typewell.csv', index=False)
            i += 1
    test_ids = sorted(p.name.split('__')[0] for p in (root / 'test').glob('*__horizontal_well.csv'))
    rows = []
    for wid in test_ids:
        hw = pd.read_csv(root / 'test' / f'{wid}__horizontal_well.csv')
        rows += [f'{wid}_{r}' for r in np.where(hw['TVT_input'].isna())[0]]
    pd.DataFrame({'id': rows, 'tvt': 0.0}).to_csv(root / 'sample_submission.csv', index=False)
    return root


@pytest.fixture(scope='session')
def fake_data(tmp_path_factory) -> Path:
    return write_fake_data(tmp_path_factory.mktemp('rogii'))
