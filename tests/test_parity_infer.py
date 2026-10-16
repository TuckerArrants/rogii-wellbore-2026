"""The Kaggle path: image, refeed and per-well prediction (with and without PhaseJitter TTA),
the port against final-submission-equal-weight-no-tta.ipynb."""
import numpy as np
import pytest
import torch

from conftest import INFER, load_defs, offline_smp
from rogii_wellbore.config import Geometry, ModelConfig
from rogii_wellbore.data import WellStore
from rogii_wellbore.image import arrays_from_frames, build_image
from rogii_wellbore.infer import predict_well
from rogii_wellbore.model import StackedUNet, iter_logits

ENC = 'tu-efficientnet_b0'


def _ns(geom):
    import os, glob, pandas as pd, torch.nn as nn, torch.nn.functional as F
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, F=F, os=os, glob=glob, smp=offline_smp(),
              DEFAULT_H=geom.H, DEFAULT_W=geom.W, DEFAULT_BIN_SIZE=geom.bin_size, DEFAULT_K_MAX=geom.k_max,
              IN_CH=5, STACK_DETACH=True, DECODER_INTERP='nearest', DECODER_CHANNELS=(512, 256, 128, 64, 32),
              DEVICE=torch.device('cpu'), CKPT_DEVICE=torch.device('cpu'), PHASE_CHUNK=8,
              _IMG_CACHE={}, _IMG_CACHE_BYTES=0, IMG_CACHE_GB=8.0, _IMG_STATS={'built': 0, 'hits': 0},
              _row_idx_cache={})
    return load_defs(INFER, [5, 6, 8], ns)


def _test_wells(fake_data):
    store = WellStore.test(str(fake_data))
    return {w: arrays_from_frames(store.read_hw(w), store.read_tw(w), with_truth=False) for w in store.well_ids()}


@pytest.mark.parametrize('geom', [Geometry(), Geometry.from_height(224)])
def test_inference_image_matches_notebook(fake_data, geom):
    ns = _ns(geom)
    for wid, a in _test_wells(fake_data).items():
        x = ns['build_image_from_arrays'](**a, H=geom.H, W=geom.W, BIN_SIZE=geom.bin_size, K_MAX=geom.k_max)
        y = build_image(**a, geom=geom)
        for k in ('image_in', 'post_ps_mask', 'hint_row', 'bin_edges'):
            assert np.array_equal(x[k], y[k]), (wid, k)
        assert x['tvt_lo'] == y['tvt_lo']


@pytest.mark.parametrize('k,tta', [(1, False), (3, False), (3, True)])
def test_member_prediction_matches_notebook(fake_data, k, tta):
    geom = Geometry(H=64, W=128, bin_size=4.0, k_max=4)
    ns = _ns(geom)
    entry = dict(stack_n=2, encoder_name=ENC, handoff_t=1.5, refeed_t=1.0)
    models_nb, models_port = [], []
    for s in range(2):
        torch.manual_seed(s)
        m = ns['build_model'](entry).eval()
        p = StackedUNet(ModelConfig(encoder=ENC, handoff_t=1.5), encoder_weights=None, train_mode=False).eval()
        p.load_state_dict(m.state_dict())
        models_nb.append(m); models_port.append(p)
    x = torch.randn(2, 5, geom.H, geom.W)
    with torch.no_grad():
        a = ns['_iter_logits'](models_nb[0], x, 3); b, _ = iter_logits(models_port[0], x, 3)
        assert all(torch.equal(a[j], b[j]) for j in (1, 2, 3))
    offsets = ns['PHASE_TTA_OFFSETS'] if 'PHASE_TTA_OFFSETS' in ns else [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    for wid, arr in _test_wells(fake_data).items():
        preds, _ = ns['predict_well_streams'](arr, models_nb, k, tta, offsets, geom=dict(
            H=geom.H, W=geom.W, BIN_SIZE=geom.bin_size, K_MAX=geom.k_max))
        q = predict_well(arr, models_port, k, geom, tta)
        np.testing.assert_array_equal(preds[f'k{k}'], q)
