"""The two-stage ("stacked") U-Net.

Stage 1 reads the 5-channel image and gives row logits per column (B, H, W). Its softmax over
rows, softened by ``handoff_t`` and detached, is concatenated to the image as a 6th channel for
stage 2, which gives the refined logits. ``iter_logits`` re-feeds stage 2 its own softened
belief for rounds 2..k (``refeed_t``); ``unroll_forward`` is the training version, which keeps
the graph only for the last round and optionally returns the intermediate rounds for their
own loss. State-dict keys (``stage1.*``, ``stage2.*``) match the notebooks' checkpoints.

Temperatures live on the instance, so members trained at different temperatures can share a
process (the notebooks read them from module globals in training and from the instance in the
Kaggle kernel).
"""
from __future__ import annotations

import zlib

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainConfig


def _unet(encoder: str, in_ch: int, cfg: ModelConfig, encoder_weights, drop_path_rate):
    kw = dict(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_ch, classes=1,
              decoder_channels=tuple(cfg.decoder_channels), decoder_interpolation=cfg.decoder_interp)
    if drop_path_rate:
        kw['drop_path_rate'] = drop_path_rate
    return smp.Unet(**kw)


class StackedUNet(nn.Module):
    def __init__(self, cfg: ModelConfig, encoder_weights: str | None = 'imagenet', train_mode: bool = True):
        """``encoder_weights='imagenet'`` for training (downloads the timm weights once);
        ``None`` for loading checkpoints. ``train_mode=False`` builds without stochastic depth,
        as the Kaggle kernel did (drop-path is inert in eval mode either way)."""
        super().__init__()
        self.cfg = cfg
        self.detach_handoff = cfg.detach_handoff
        self.handoff_t = float(cfg.handoff_t)
        self.refeed_t = float(cfg.refeed_t)
        dpr = cfg.drop_path_rate if train_mode else 0.0
        self.stage1 = _unet(cfg.encoder, cfg.in_ch, cfg, encoder_weights, dpr)
        self.stage2 = _unet(cfg.encoder, cfg.in_ch + 1, cfg, encoder_weights, dpr)
        if dpr:
            _assert_droppath_active(self.stage1, cfg.encoder)

    def stage2_from_post(self, x, post):
        return self.stage2(torch.cat([x, post.unsqueeze(1)], dim=1)).squeeze(1)

    def _handoff(self, logits1):
        h = logits1.detach() if self.detach_handoff else logits1
        return F.softmax(h / self.handoff_t, dim=1)

    def forward(self, x, return_aux: bool = False):
        logits1 = self.stage1(x).squeeze(1)
        logits2 = self.stage2_from_post(x, self._handoff(logits1))
        return (logits2, logits1) if return_aux else logits2

    def unroll_forward(self, x, k: int, return_rounds: bool = False):
        """Training-time refeed: rounds 2..k see stage 2's previous output, detached and softened
        by ``refeed_t``. Returns (final logits, stage-1 logits[, rounds 1..k-1 logits])."""
        assert k >= 1
        logits1 = self.stage1(x).squeeze(1)
        logits2 = self.stage2_from_post(x, self._handoff(logits1))
        rounds = []
        for _ in range(2, k + 1):
            if return_rounds:
                rounds.append(logits2)
            post = F.softmax(logits2.detach() / self.refeed_t, dim=1)
            logits2 = self.stage2_from_post(x, post)
        return (logits2, logits1, rounds) if return_rounds else (logits2, logits1)


def iter_logits(model: StackedUNet, x, kmax: int):
    """Evaluation-time refeed. Returns ({k: stage-2 logits for k=1..kmax}, stage-1 logits);
    k=1 equals ``forward``. Matches the training notebook's ``_eval_iter_logits`` and the
    Kaggle kernel's ``_iter_logits``."""
    logits1 = model.stage1(x).squeeze(1)
    logits2 = model.stage2_from_post(x, model._handoff(logits1))
    out = {1: logits2}
    for k in range(2, kmax + 1):
        logits2 = model.stage2_from_post(x, F.softmax(logits2 / model.refeed_t, dim=1))
        out[k] = logits2
    return out, logits1


def _assert_droppath_active(unet, tag):
    from timm.layers import DropPath
    live = [m.drop_prob for m in unet.modules() if isinstance(m, DropPath) and getattr(m, 'drop_prob', 0.0)]
    assert live, f'drop_path_rate requested but {tag} built no active DropPath layers (silent no-op)'


def hard_ce_loss(logits, truth_row, mask):
    """Cross-entropy over rows, per column, averaged over the masked (post-PS, in-window) columns."""
    B, Hh, Ww = logits.shape
    log_probs = F.log_softmax(logits, dim=1).permute(0, 2, 1).reshape(B * Ww, Hh)
    tgt = truth_row.reshape(B * Ww); msk = mask.reshape(B * Ww)
    nll = -log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
    return (nll * msk.float()).sum() / msk.float().sum().clamp(min=1.0)


def expected_row(logits, row_idx=None):
    """Probability-weighted row per column (B, W): the decode."""
    if row_idx is None:
        row_idx = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)
    return (F.softmax(logits, dim=1) * row_idx[None, :, None]).sum(dim=1)


class UnrollSampler:
    """Per-step (unroll?, k) draws for fine-tuning: with probability ``unroll_p`` pick k from
    ``unroll_k_choices`` by ``unroll_k_weights``, else k=1. One torch.Generator per
    (run, phase), seeded ``unroll_rng_seed + crc32(run_label|phase)``.

    The notebook seeded this generator with Python's ``hash()`` of the label, which is salted
    per process, so its unroll draws were not reproducible between sessions; crc32 is."""

    def __init__(self, cfg: TrainConfig, run_label: str, phase: str):
        self.cfg = cfg
        self.gen = torch.Generator(device='cpu')
        salt = zlib.crc32(f'{run_label}|{phase}'.encode())
        self.gen.manual_seed((cfg.unroll_rng_seed + salt) & 0x7FFFFFFF)

    def sample(self) -> int:
        c = self.cfg
        if c.unroll_p <= 0.0:
            return 1
        if float(torch.rand((), generator=self.gen)) >= c.unroll_p:
            return 1
        w = torch.tensor(c.unroll_k_weights, dtype=torch.float32)
        return int(c.unroll_k_choices[int(torch.multinomial(w, 1, generator=self.gen))])


def load_checkpoint(model: nn.Module, path: str, map_location='cpu', strict_bn_counters: bool = False):
    """Load an EMA checkpoint. The EMA shadow holds only float tensors, so BatchNorm
    ``num_batches_tracked`` counters may be missing; anything else missing or unexpected fails."""
    try:
        state = torch.load(path, map_location=map_location, mmap=True)
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if strict_bn_counters or not k.endswith('num_batches_tracked')]
    if unexpected or bad_missing:
        raise RuntimeError(f'{path}: checkpoint does not match the model '
                           f'(unexpected {list(unexpected)[:4]}, missing {bad_missing[:4]})')
    return model
