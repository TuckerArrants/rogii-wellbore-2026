"""Datasets, model, refeed and a full short training phase: the port against 2SUNET_MASTER.ipynb.

The model tests use a small timm encoder at 64 x 128 so they run on CPU in seconds; the
architecture code is the same for every encoder.
"""
import zlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import MASTER, load_defs, master_namespace, offline_smp
from rogii_wellbore.augment import TrainDataset, ValDataset
from rogii_wellbore.config import AugConfig, Geometry, ModelConfig, RunConfig, TrainConfig
from rogii_wellbore.data import TruthCache, WellStore
from rogii_wellbore.image import arrays_from_frames, build_image
from rogii_wellbore.model import StackedUNet, UnrollSampler, hard_ce_loss, iter_logits
from rogii_wellbore.train import EMAState, cosine_lambda, train_phase

ENC = 'tu-efficientnet_b0'
SMALL = Geometry(H=64, W=128, bin_size=4.0, k_max=4)


def _recs(fake_data, geom):
    store = WellStore.train(str(fake_data))
    return store, {w: build_image(**arrays_from_frames(store.read_hw(w), store.read_tw(w)), geom=geom)
                   for w in store.well_ids()}


def _nb_ns(fake_data, geom, recs, **extra):
    store = WellStore.train(str(fake_data))
    ns = master_namespace(H=geom.H, W=geom.W, BIN_SIZE=geom.bin_size, K_MAX=geom.k_max,
                          well_images_real=recs, well_hw_path=store.hw_path, **extra)
    return load_defs(MASTER, [7, 13, 16], ns)


@pytest.mark.parametrize('geom', [Geometry(), Geometry.from_height(224)])
def test_finetune_dataset_matches_notebook(fake_data, geom):
    store, recs = _recs(fake_data, geom)
    ns = _nb_ns(fake_data, geom, recs)
    aug = AugConfig()
    wids = store.well_ids()[:8]
    seed = 68
    ft_seed = (seed + 1_000_000) & 0xFFFFFFFF
    s, ac, p = aug.md_warp(geom); bs, bac, bp = aug.bedwarp(geom)
    nb = ns['TrainDataset'](wids, synth_recs=None, base_seed=ft_seed, h_flip_p=0.5, hint_drop_p=0.3,
                            md_warp_strength=s, md_warp_autocorr=ac, md_warp_p=p,
                            shear_std_rows=3.0, shear_autocorr=128, shear_p=0.0, shear_max_rows=9,
                            bedwarp_strength=bs, bedwarp_autocorr=bac, bedwarp_p=bp)
    port = TrainDataset.finetune(recs, wids, geom, seed, aug)
    for epoch in (0, 3):
        nb.set_epoch(epoch); port.set_epoch(epoch)
        for i in range(len(wids)):
            a, b = nb[i], port[i]
            for k in ('image', 'truth_row', 'main_mask'):
                assert torch.equal(a[k], b[k]), (epoch, i, k)


def test_pretrain_and_val_datasets_match_notebook(fake_data):
    geom = Geometry()
    store, recs = _recs(fake_data, geom)
    ns = _nb_ns(fake_data, geom, recs)
    synth = dict(list(recs.items())[:5])
    nb = ns['TrainDataset']([], synth, base_seed=34, h_flip_p=0.0, hint_drop_p=0.0)
    port = TrainDataset.pretrain(synth, geom, 34, AugConfig())
    for i in range(len(synth)):
        assert torch.equal(nb[i]['image'], port[i]['image']) and torch.equal(nb[i]['truth_row'], port[i]['truth_row'])
    nbv, pv = ns['RealValDataset'](list(recs)[:4]), ValDataset(recs, list(recs)[:4])
    for i in range(4):
        assert all(torch.equal(nbv[i][k], pv[i][k]) for k in ('image', 'truth_row', 'main_mask'))


def _model_ns(handoff_t=1.5, refeed_t=1.0, unroll=True):
    ns = master_namespace(H=SMALL.H, W=SMALL.W, BIN_SIZE=SMALL.bin_size, K_MAX=SMALL.k_max)
    ns.update(smp=offline_smp(), DECODER_CHANNELS=(512, 256, 128, 64, 32), DECODER_INTERP='nearest',
              DROP_PATH_RATE=0.2, DROP_CONNECT_RATE=0.2, POSTERIOR_DILATION=False, MULTISCALE_POST_INJECT=False,
              POST_DIL_EMBED=128, POST_DIL_RATES=(1, 2, 4, 8, 16), HANDOFF_TRAIN_T=handoff_t, REFEED_TRAIN_T=refeed_t,
              STACK_N=2, STACK_DETACH=True, ENCODER=ENC, IN_CH=5, UNROLL_ENABLE=unroll, UNROLL_P=0.5 if unroll else 0.0,
              UNROLL_K_CHOICES=(2, 3, 4), UNROLL_K_WEIGHTS=(1.0, 1.0, 1.0), UNROLL_DEEP_SUP_ROUNDS=True,
              EMA_DECAY=0.997)
    return load_defs(MASTER, [18], ns)


def _pair(handoff_t=1.5, refeed_t=1.0, unroll=True):
    ns = _model_ns(handoff_t, refeed_t, unroll)
    torch.manual_seed(0)
    nbm = ns['build_model'](ENC, 5)
    port = StackedUNet(ModelConfig(encoder=ENC, handoff_t=handoff_t, refeed_t=refeed_t), encoder_weights=None)
    port.load_state_dict(nbm.state_dict(), strict=True)
    return ns, nbm, port


@pytest.mark.parametrize('temps', [(1.5, 1.0), (1.0, 1.0), (1.5, 1.3)])
def test_model_forward_refeed_match_notebook(temps):
    ns, nbm, port = _pair(*temps)
    nbm.eval(); port.eval()
    x = torch.randn(2, 5, SMALL.H, SMALL.W)
    with torch.no_grad():
        a2, a1 = nbm(x, return_aux=True); b2, b1 = port(x, return_aux=True)
        assert torch.equal(a2, b2) and torch.equal(a1, b1)
        ns2 = load_defs(MASTER, [22], dict(ns), names={'_eval_iter_logits'})
        ka, la = ns2['_eval_iter_logits'](nbm, x, 4)
        kb, lb = iter_logits(port, x, 4)
        assert torch.equal(la, lb) and all(torch.equal(ka[k], kb[k]) for k in range(1, 5))
        for k in (2, 3, 4):                               # training-time unroll, rounds included
            ra = nbm.unroll_forward(x, k, return_aux=True, deep_sup_rounds=True)
            rb = port.unroll_forward(x, k, return_rounds=True)
            assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1])
            assert len(ra[2]) == len(rb[2]) == k - 1 and all(torch.equal(p, q) for p, q in zip(ra[2], rb[2]))


def test_unroll_sampler_and_loss_match_notebook():
    ns = _model_ns()
    tc = TrainConfig()
    s = UnrollSampler(tc, 'f0_s34', 'finetune')
    g = torch.Generator(); g.manual_seed(1234); s.gen.manual_seed(1234)
    draws_nb = [ns['_unroll_sample_k'](g) for _ in range(500)]
    draws_port = [s.sample() for _ in range(500)]
    assert [k if u else 1 for u, k in draws_nb] == draws_port
    logits = torch.randn(3, 64, 128); tgt = torch.randint(0, 64, (3, 128)); msk = torch.rand(3, 128) > 0.3
    assert torch.equal(ns['hard_ce_loss'](logits, tgt, msk), hard_ce_loss(logits, tgt, msk))


def test_training_phase_matches_notebook(fake_data, tmp_path):
    """Two fine-tuning epochs with EMA, unrolling and k=3 selection, run by the notebook's
    ``_train_one_phase`` and by the port from the same weights: identical logs, best state and
    final weights. (The notebook seeded its unroll generator with Python's salted ``hash()``;
    here the notebook gets the port's crc32 so both draw the same k.)"""
    geom = SMALL
    store, recs = _recs(fake_data, geom)
    wids = store.well_ids()
    train_ids, val_ids = wids[:12], wids[12:16]
    cfg = RunConfig(name='t', geometry=geom, model=ModelConfig(encoder=ENC),
                    train=TrainConfig(batch_size=4, num_workers=0, amp=None, finetune_epochs=2, patience=5))
    ns = _model_ns()
    load_defs(MASTER, [7, 13, 16, 18, 22], ns)
    ns.update(well_images_real=recs, well_hw_path=store.hw_path, DEVICE=torch.device('cpu'),
              row_idx_t=torch.arange(geom.H, dtype=torch.float32), STACK_DEEP_SUP_WEIGHT=0.5,
              UNROLL_ROUND_SUP_WEIGHT=0.5, UNROLL_RNG_SEED=20260707, DEPLOY_K_LOG=[2, 3], CHECKPOINT_K=3,
              USE_AMP=False, AMP_DTYPE='bfloat16', _EXP_ARMED=False, exp_loss_weight_at=lambda e: 0.0,
              EXP_LOSS_DELTA=8.0, CKPT_DIR=str(tmp_path), IN_COLAB=False,
              hash=lambda t: zlib.crc32(f'{t[0]}|{t[1]}'.encode()))
    torch.manual_seed(0)
    nbm = ns['build_model'](ENC, 5)
    port = StackedUNet(cfg.model, encoder_weights=None); port.load_state_dict(nbm.state_dict())

    def loaders(ds_cls_nb):
        aug = cfg.aug
        s, ac, p = aug.md_warp(geom); bs, bac, bp = aug.bedwarp(geom)
        tr_nb = ns['TrainDataset'](train_ids, None, base_seed=1_000_034, h_flip_p=0.5, hint_drop_p=0.3,
                                   md_warp_strength=s, md_warp_autocorr=ac, md_warp_p=p,
                                   bedwarp_strength=bs, bedwarp_autocorr=bac, bedwarp_p=bp)
        tr_port = TrainDataset.finetune(recs, train_ids, geom, 34, aug)
        g1 = torch.Generator(); g1.manual_seed(1_000_034); g2 = torch.Generator(); g2.manual_seed(1_000_034)
        return (DataLoader(tr_nb, batch_size=4, shuffle=True, generator=g1),
                DataLoader(ns['RealValDataset'](val_ids), batch_size=4),
                DataLoader(tr_port, batch_size=4, shuffle=True, generator=g2),
                DataLoader(ValDataset(recs, val_ids), batch_size=4))

    dl_nb, vl_nb, dl_p, vl_p = loaders(None)
    logs_nb, logs_p = [], []
    import builtins
    real_print = builtins.print
    ns['print'] = lambda *a, **k: logs_nb.append(' '.join(map(str, a)))

    ema_nb = ns['EMAState'](nbm, decay=0.997)
    opt = torch.optim.AdamW(nbm.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, ns['make_lr_lambda'](0, 2))
    torch.manual_seed(7)
    r_nb = ns['_train_one_phase'](nbm, ema_nb, dl_nb, vl_nb, opt, sch, n_epochs=2, patience=5,
                                  run_label='f0_s34', phase_tag='finetune', do_real_val=True, do_early_stop=True)

    ema_p = EMAState(port, 0.997)
    opt2 = torch.optim.AdamW(port.parameters(), lr=1e-3, weight_decay=1e-3)
    sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, cosine_lambda(2))
    torch.manual_seed(7)
    r_p = train_phase(port, ema_p, dl_p, vl_p, opt2, sch2, n_epochs=2, patience=5, run_label='f0_s34',
                      phase='finetune', cfg=cfg, recs=recs, truths=TruthCache(store), device=torch.device('cpu'),
                      log=logs_p.append)
    assert r_nb[0] == r_p.best_rmse and r_nb[2] == r_p.best_epoch and r_nb[4] == r_p.val_traj
    for k, v in r_nb[1].items():
        assert torch.equal(v, r_p.best_state[k]), k
    for (k, a), b in zip(nbm.state_dict().items(), port.state_dict().values()):
        assert torch.equal(a, b), k
    # the numbers in the epoch lines agree (the line format changed a little)
    import re
    num = lambda s: re.findall(r'k1_TVT=([\d.]+).*?\| s1 TVT=([\d.]+).*?k2:([\d.]+).*?k3:([\d.]+)', s)
    nums_nb = [num(l) for l in logs_nb if 'k1_TVT' in l]
    assert len(nums_nb) == 2 and all(nums_nb) and nums_nb == [num(l) for l in logs_p if 'k1_TVT' in l]
