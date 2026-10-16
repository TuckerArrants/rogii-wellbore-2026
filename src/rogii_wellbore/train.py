"""Training one member: synthetic pretraining, real fine-tuning, per-seed checkpoints.

One job is one (fold, seed). Pretraining (5 epochs, synthetic wells of that fold's pool, no
EMA, AdamW + cosine) is skipped when ``pretrained_dir`` holds ``pretrain_f{fold}_s{seed}_ep5.pt``,
which is how every final member was trained. Fine-tuning (real training wells of the fold,
fresh AdamW + cosine, EMA 0.997) evaluates the EMA weights on the fold's validation wells
after every epoch, at k=1 and at each depth in ``deploy_k_log``, keeps the best state at
``checkpoint_k`` and stops after ``patience`` epochs without improvement. The loss is
CE(stage 2) + 0.5 * CE(stage 1), plus ``unroll_round_sup_weight`` * mean CE of the
intermediate rounds on unrolled steps.
"""
from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import TrainDataset, ValDataset
from .config import RunConfig
from .data import TruthCache
from .metrics import decode_rows, well_residual
from .model import StackedUNet, UnrollSampler, expected_row, hard_ce_loss, iter_logits


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def loader_generator(seed: int):
    g = torch.Generator(); g.manual_seed(int(seed) & 0xFFFFFFFF)
    return g


class EMAState:
    """Exponential moving average of every floating-point state-dict entry."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def swap_into(self, model):
        live = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    live[k] = v.detach().clone(); v.copy_(self.shadow[k])
        return live

    def restore(self, model, live):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in live:
                    v.copy_(live[k])


def cosine_lambda(total_epochs: int):
    def f(epoch):
        progress = epoch / max(1, total_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, progress))))
    return f


def autocast(cfg: RunConfig, device):
    if not cfg.train.amp or device.type != 'cuda':
        return nullcontext()
    dt = torch.bfloat16 if cfg.train.amp == 'bfloat16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dt)


# ---------------------------------------------------------------------------------------------
# evaluation during training
# ---------------------------------------------------------------------------------------------
def evaluate(model, loader, recs, truths: TruthCache, cfg: RunConfig, device, k_list=()):
    """Stage-2 RMSE and CE at k=1, stage-1 RMSE and CE, and {k: (RMSE, CE)} for k in ``k_list``."""
    model.eval()
    geom = cfg.geometry
    ks = sorted({int(k) for k in k_list if int(k) >= 2})
    kmax = max([1] + ks)
    errs = {k: [] for k in [1] + ks}; errs1 = []
    ce = {k: 0.0 for k in [1] + ks}; ce1 = 0.0; nb = 0
    row_idx = torch.arange(geom.H, device=device, dtype=torch.float32)
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device)
            tgt = batch['truth_row'].to(device); msk = batch['main_mask'].to(device)
            kl, logits1 = iter_logits(model, img, kmax)
            nb += 1
            ce1 += hard_ce_loss(logits1, tgt, msk).item()
            er1 = expected_row(logits1, row_idx).cpu().numpy()
            ers = {}
            for k in [1] + ks:
                ce[k] += hard_ce_loss(kl[k], tgt, msk).item()
                ers[k] = expected_row(kl[k], row_idx).cpu().numpy()
            for b, wid in enumerate(batch['well_id']):
                t = truths(wid); rec = recs[wid]
                for k in [1] + ks:
                    _, pf = decode_rows(ers[k][b], rec['tvt_lo'], geom.bin_size, t.n_hw, geom.W)
                    errs[k].append(well_residual(pf, t)[1] ** 2)
                _, pf = decode_rows(er1[b], rec['tvt_lo'], geom.bin_size, t.n_hw, geom.W)
                errs1.append(well_residual(pf, t)[1] ** 2)
    r = lambda e: float(np.sqrt(np.concatenate(e).mean()))
    return {'rmse': r(errs[1]), 'ce': ce[1] / max(nb, 1), 'rmse1': r(errs1), 'ce1': ce1 / max(nb, 1),
            'multik': {k: (r(errs[k]), ce[k] / max(nb, 1)) for k in ks}}


# ---------------------------------------------------------------------------------------------
# one phase
# ---------------------------------------------------------------------------------------------
@dataclass
class PhaseResult:
    best_rmse: float = float('inf')
    best_state: dict | None = None
    best_epoch: int = -1
    final_epoch: int = 0
    val_traj: list = field(default_factory=list)


def train_phase(model, ema, train_loader, val_loader, opt, sched, *, n_epochs, patience, run_label,
                phase, cfg: RunConfig, recs, truths, device, early_stop=True, save_dir=None, log=print):
    tc = cfg.train
    finetune = phase == 'finetune'
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and tc.amp == 'float16'))
    sampler = UnrollSampler(tc, run_label, phase)
    unroll_on = finetune and tc.unroll_p > 0
    round_sup = tc.unroll_round_sup_weight
    deploy_ks = sorted({int(k) for k in tc.deploy_k_log if int(k) >= 2}) if finetune else []
    res = PhaseResult(); since = 0
    for epoch in range(n_epochs):
        res.final_epoch = epoch + 1
        train_loader.dataset.set_epoch(epoch)
        model.train(); s_loss = s_ce2 = s_ce1 = 0.0; nb = 0
        for batch in train_loader:
            img = batch['image'].to(device); tgt = batch['truth_row'].to(device); msk = batch['main_mask'].to(device)
            with autocast(cfg, device):
                k = sampler.sample() if unroll_on else 1
                rounds = []
                if k > 1:
                    out = model.unroll_forward(img, k, return_rounds=bool(round_sup))
                    logits, logits1 = out[0], out[1]
                    rounds = out[2] if round_sup else []
                else:
                    logits, logits1 = model(img, return_aux=True)
                ce2 = hard_ce_loss(logits, tgt, msk); ce1 = hard_ce_loss(logits1, tgt, msk)
                loss = ce2 + tc.deep_sup_weight * ce1
                if rounds:
                    loss = loss + round_sup * torch.stack([hard_ce_loss(r, tgt, msk) for r in rounds]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'{run_label} [{phase}]: diverged at epoch {epoch}')
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            s_loss += loss.item(); s_ce2 += ce2.item(); s_ce1 += ce1.item(); nb += 1
        sched.step()
        lr = opt.param_groups[0]['lr']
        live = ema.swap_into(model) if ema is not None else None
        ev = evaluate(model, val_loader, recs, truths, cfg, device, k_list=deploy_ks)
        if live is not None:
            ema.restore(model, live)
        res.val_traj.append(ev['rmse'])
        sel = ev['multik'][tc.checkpoint_k][0] if (tc.checkpoint_k > 1 and tc.checkpoint_k in ev['multik']) else ev['rmse']
        improved = sel < res.best_rmse
        if improved:
            res.best_rmse = sel
            src = ema.shadow if ema is not None else model.state_dict()
            res.best_state = {k: v.detach().cpu().clone() for k, v in src.items()}
            for k, v in model.state_dict().items():
                res.best_state.setdefault(k, v.detach().cpu().clone())
            res.best_epoch = epoch + 1; since = 0
        else:
            since += 1
        mk = '  | ' + ' '.join(f'k{k}:{ev["multik"][k][0]:.3f}({ev["multik"][k][1]:.3f})' for k in deploy_ks) if deploy_ks else ''
        log(f'  [{run_label}|{phase}] ep {epoch + 1:2d}  lr={lr:.2e}  L={s_loss / nb:.3f} '
            f'(CE2={s_ce2 / nb:.3f} CE1={s_ce1 / nb:.3f})  k1_TVT={ev["rmse"]:.3f}(CE{ev["ce"]:.3f})'
            f'  | s1 TVT={ev["rmse1"]:.3f}{mk}{" *" if improved else ""}')
        if phase == 'pretrain' and save_dir and epoch >= n_epochs - 2:
            fn = f'pretrain_{run_label}_ep{epoch + 1}.pt'
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(save_dir, fn))
            log(f'  [{run_label}|{phase}] saved {fn}')
        if early_stop and since >= patience:
            log(f'  [{run_label}|{phase}] early stop at epoch {epoch + 1} (best {res.best_rmse:.3f} at {res.best_epoch})')
            break
    return res


# ---------------------------------------------------------------------------------------------
# one job
# ---------------------------------------------------------------------------------------------
def train_one_run(cfg: RunConfig, fold: int, seed: int, folds: dict, real_recs: dict, synth_recs: dict | None,
                  truths: TruthCache, device, out_dir: str, pretrained_state: dict | None = None,
                  encoder_weights='imagenet', log=print) -> dict:
    tc, geom = cfg.train, cfg.geometry
    seed_everything(seed)
    label = f'f{fold}_s{seed}'
    val_loader = DataLoader(ValDataset(real_recs, folds[fold]['val']), batch_size=tc.batch_size, shuffle=False,
                            num_workers=tc.num_workers, worker_init_fn=seed_worker, pin_memory=device.type == 'cuda')
    model = StackedUNet(cfg.model, encoder_weights=encoder_weights).to(device)
    log(f'  model {cfg.model.encoder}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')

    if pretrained_state is None:
        assert synth_recs, f'{label}: no pretrained checkpoint and no synthetic pool'
        ds = TrainDataset.pretrain(synth_recs, geom, seed, cfg.aug)
        dl = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers,
                        worker_init_fn=seed_worker, generator=loader_generator(seed), pin_memory=device.type == 'cuda')
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(tc.pretrain_epochs))
        log(f'  pretrain: {len(ds)} synthetic wells, {tc.pretrain_epochs} epochs')
        train_phase(model, None, dl, val_loader, opt, sched, n_epochs=tc.pretrain_epochs, patience=tc.patience,
                    run_label=label, phase='pretrain', cfg=cfg, recs=real_recs, truths=truths, device=device,
                    early_stop=False, save_dir=out_dir, log=log)
        pt_reused = False
    else:
        model.load_state_dict(pretrained_state, strict=True)
        pt_reused = True
        log('  pretrain: reused checkpoint')

    ema = EMAState(model, tc.ema_decay)
    ds = TrainDataset.finetune(real_recs, folds[fold]['train'], geom, seed, cfg.aug)
    dl = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers, worker_init_fn=seed_worker,
                    generator=loader_generator(ds.base_seed), pin_memory=device.type == 'cuda')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc.lr, weight_decay=tc.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(tc.finetune_epochs))
    log(f'  finetune: {len(ds)} real wells, up to {tc.finetune_epochs} epochs')
    res = train_phase(model, ema, dl, val_loader, opt, sched, n_epochs=tc.finetune_epochs, patience=tc.patience,
                      run_label=label, phase='finetune', cfg=cfg, recs=real_recs, truths=truths, device=device,
                      early_stop=True, log=log)
    return {'best_rmse': res.best_rmse, 'best_state': res.best_state, 'best_epoch': res.best_epoch,
            'final_epoch': res.final_epoch, 'val_traj': res.val_traj, 'pt_reused': pt_reused}
