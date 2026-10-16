"""Folds, synthetic wells and images: the port against 2SUNET_MASTER.ipynb, byte for byte."""
import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

from conftest import MASTER, cell_source, load_defs, master_namespace, _typewell
from rogii_wellbore.config import Geometry, SynthConfig
from rogii_wellbore.cv import make_folds
from rogii_wellbore.data import WellStore, build_hint_real_dist
from rogii_wellbore.image import arrays_from_frames, build_image, build_image_synth, render
from rogii_wellbore.synth import generate_well

ARRK = ['MD', 'TVT', 'TVT_input', 'GR', 'Z', '_tw_tvt', '_tw_gr']
HINT = {'n_md': np.array([2200, 2600, 3000]), 'ps_md_pct': np.array([25.0, 30.0]),
        'tvt_range': np.array([25.0, 30.0, 40.0])}


def test_folds_match_notebook(fake_data):
    ns = master_namespace(DATA_ROOT=str(fake_data), SEED=34, tqdm=lambda x, **k: x)
    import os
    ns['os'] = os
    exec(compile(cell_source(MASTER, 5), 'cell5', 'exec'), ns)
    port = make_folds(WellStore.train(str(fake_data)), seed=34)
    assert {f: v for f, v in ns['fold_splits'].items()} == port


def _nb_generate(ns, seed, tw, hint, **over):
    for k, v in over.items():
        ns[k] = v
    g = ns
    return g['generate_synthetic_well_method_a_v3'](
        tw, hint, seed=seed, sigma_drift=g['SIGMA_DRIFT'], sigma_tvt_offset=g['SIGMA_TVT_OFFSET'],
        sigma_meas=g['SIGMA_MEAS'], sigma_calib=g['SIGMA_CALIB'], wobble_total_pct=g['V3_WOBBLE_TOTAL_PCT'],
        wobble_split=g['V3_WOBBLE_SPLIT'], wavelength_divisors=g['V3_WAVELENGTH_DIVISORS'],
        fault_prob=g['FAULT_PROB'], fault_magnitude_min=g['FAULT_MAGNITUDE_MIN'],
        fault_magnitude_max=g['FAULT_MAGNITUDE_MAX'], unconf_prob=g['UNCONF_PROB'],
        unconf_max_cumulative_shift=g['UNCONF_MAX_CUMULATIVE_SHIFT'], var_bed_prob=g['VAR_BED_PROB'],
        var_bed_amp_std=g['VAR_BED_AMP_STD'], var_bed_autocorr_length=g['VAR_BED_AUTOCORR_LENGTH'],
        var_bed_amp_min=g['VAR_BED_AMP_MIN'], var_bed_amp_max=g['VAR_BED_AMP_MAX'])


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return all(np.array_equal(a[k], b[k], equal_nan=True) for k in ARRK) and a['ps_md_idx'] == b['ps_md_idx'] \
        and a['_fault_info'] == b['_fault_info'] and a['_unconf_info'] == b['_unconf_info'] \
        and a['_pack_info'] == b['_pack_info']


@pytest.mark.parametrize('pack,corruption,extra', [
    ('v3_expA', 'clean', {}),                                 # the final members' pool
    ('v3', 'clean', {}),                                      # pack off
    ('v3_expA', 'baseline', {'PINCH_PROB': 0.5}),             # noisy corruption + the dormant pinch armed
])
def test_synth_generator_matches_notebook(pack, corruption, extra):
    from rogii_wellbore.config import CORRUPTION_PRESETS
    sig = CORRUPTION_PRESETS[corruption]
    ns = load_defs(MASTER, [7], master_namespace(EVENT_PACK=pack, SIGMA_DRIFT=sig['sigma_drift'],
                                                  SIGMA_TVT_OFFSET=sig['sigma_tvt_offset'],
                                                  SIGMA_MEAS=sig['sigma_meas'], SIGMA_CALIB=sig['sigma_calib'], **extra))
    cfg = SynthConfig(event_pack=pack, corruption=corruption, pinch_prob=extra.get('PINCH_PROB', 0.0))
    rng = np.random.RandomState(5)
    n_events = 0
    for seed in range(150):
        tw = _typewell(rng) if seed % 25 == 0 else tw
        a = _nb_generate(ns, seed, tw, HINT)
        b = generate_well(tw, HINT, seed, cfg)
        assert _eq(a, b), f'seed {seed} differs'
        if b is not None and b['_pack_info']:
            n_events += len(b['_pack_info']['events'])
    assert pack == 'v3' or n_events > 10        # the pack is live in this test


def test_hint_dist_matches_notebook(fake_data):
    store = WellStore.train(str(fake_data))
    ns = load_defs(MASTER, [7], master_namespace(well_hw_path=store.hw_path))
    wids = store.well_ids()[:12]
    a, _ = ns['build_hint_real_dist'](wids)
    b, _ = build_hint_real_dist(store, wids)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize('geom', [Geometry(), Geometry.from_height(224), Geometry(H=64, W=128, bin_size=4.0)])
def test_image_matches_notebook(fake_data, geom):
    store = WellStore.train(str(fake_data))
    ns = load_defs(MASTER, [13, 16], master_namespace(H=geom.H, W=geom.W, BIN_SIZE=geom.bin_size, K_MAX=geom.k_max))
    for wid in store.well_ids()[:6]:
        hw, tw = store.read_hw(wid), store.read_tw(wid)
        a = ns['build_image_real'](hw, tw)
        b = build_image(**arrays_from_frames(hw, tw), geom=geom)
        for k in ('image_in', 'truth_row', 'hint_row', 'post_ps_mask', 'hw_mask', 'dz_per_col',
                  'tw_gr_per_row_ext', 'hw_gr_per_col'):
            assert np.array_equal(a[k], b[k], equal_nan=True), (wid, k)
        assert a['tvt_lo'] == b['tvt_lo'] and a['depth_center'] == b['depth_center']
        for k_off in (-geom.k_max, -3, 0, 5, geom.k_max):          # anchor jitter
            ia, ta, ma = ns['make_jittered_image'](a, k_off)
            ib, tb, mb = render(b, k_off, geom)
            assert np.array_equal(ia, ib) and np.array_equal(ta, tb) and np.array_equal(ma, mb)


def test_synth_image_matches_notebook():
    geom = Geometry()
    ns = load_defs(MASTER, [13], master_namespace())
    rng = np.random.RandomState(1)
    tw = _typewell(rng)
    for seed in range(5):
        syn = generate_well(tw, HINT, seed, SynthConfig())
        if syn is None:
            continue
        a = ns['build_image_synth'](syn); b = build_image_synth(syn, geom)
        assert np.array_equal(a['image_in'], b['image_in']) and np.array_equal(a['truth_row'], b['truth_row'])
