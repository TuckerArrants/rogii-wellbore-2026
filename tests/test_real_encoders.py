"""The four members' real encoders, built without weights on the meta device (no memory, no
download): the package model must have exactly the parameter names and shapes of the models
the training notebook saved and the Kaggle notebook loaded, so the real checkpoints load into it.
The other parity tests use a small encoder; this one closes that gap for the encoders that shipped."""
import os

import pytest
import torch

from conftest import INFER, MASTER, load_defs, master_namespace, offline_smp
from rogii_wellbore import config as C
from rogii_wellbore.model import StackedUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = ['v2l', 'v2xl', 'v2l_small', 'l2']


def _shapes(m):
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


def _training_nb(enc, cfg):
    ns = master_namespace(H=cfg.geometry.H, W=cfg.geometry.W, BIN_SIZE=cfg.geometry.bin_size)
    ns.update(smp=offline_smp(), DECODER_CHANNELS=tuple(cfg.model.decoder_channels), DECODER_INTERP='nearest',
              DROP_PATH_RATE=0.2, DROP_CONNECT_RATE=0.2, POSTERIOR_DILATION=False, MULTISCALE_POST_INJECT=False,
              POST_DIL_EMBED=128, POST_DIL_RATES=(1, 2, 4, 8, 16), HANDOFF_TRAIN_T=1.5, REFEED_TRAIN_T=1.0,
              STACK_N=2, STACK_DETACH=True, ENCODER=enc, IN_CH=5, UNROLL_ENABLE=True, UNROLL_P=0.5,
              UNROLL_K_CHOICES=(2, 3, 4), UNROLL_K_WEIGHTS=(1.0, 1.0, 1.0), UNROLL_DEEP_SUP_ROUNDS=True,
              EMA_DECAY=0.997)
    return load_defs(MASTER, [18], ns)['build_model'](enc, 5)


def _kaggle_nb(enc, cfg):
    import glob, numpy as np, pandas as pd, torch.nn as nn, torch.nn.functional as F
    g = cfg.geometry
    ns = dict(np=np, pd=pd, torch=torch, nn=nn, F=F, os=os, glob=glob, smp=offline_smp(),
              DEFAULT_H=g.H, DEFAULT_W=g.W, DEFAULT_BIN_SIZE=g.bin_size, DEFAULT_K_MAX=g.k_max,
              IN_CH=5, STACK_DETACH=True, DECODER_INTERP='nearest', DECODER_CHANNELS=tuple(cfg.model.decoder_channels),
              DEVICE=torch.device('cpu'), CKPT_DEVICE=torch.device('cpu'), PHASE_CHUNK=8,
              _IMG_CACHE={}, _IMG_CACHE_BYTES=0, IMG_CACHE_GB=8.0, _IMG_STATS={'built': 0, 'hits': 0},
              _row_idx_cache={})
    ns = load_defs(INFER, [5, 6, 8], ns)
    return ns['build_model'](dict(stack_n=2, encoder_name=enc, handoff_t=1.5, refeed_t=1.0))


@pytest.mark.parametrize('name', MEMBERS)
def test_real_encoder_state_dict_matches_both_notebooks(name):
    cfg = C.load(os.path.join(REPO, 'configs/members', f'{name}.yaml'))
    enc = cfg.model.encoder
    with torch.device('meta'):
        port_train = _shapes(StackedUNet(cfg.model, encoder_weights=None, train_mode=True))
        port_infer = _shapes(StackedUNet(cfg.model, encoder_weights=None, train_mode=False))
        nb_train = _shapes(_training_nb(enc, cfg))
        nb_kaggle = _shapes(_kaggle_nb(enc, cfg))
    assert port_train == nb_train, set(port_train) ^ set(nb_train)
    assert port_infer == nb_kaggle, set(port_infer) ^ set(nb_kaggle)
    n = sum(int(torch.Size(s).numel()) for k, s in port_infer.items() if not k.endswith('num_batches_tracked'))
    print(f'{name}: {enc}, {n / 1e6:.0f}M parameters (both stages)')
