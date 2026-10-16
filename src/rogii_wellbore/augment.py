"""Training and validation datasets.

Every random choice for sample ``idx`` in epoch ``e`` comes from
``RandomState((base_seed * 1_000_003 + e * 7919 + idx) mod 2**32)`` (the anchor jitter) and from
one derived stream per augmentation, ``RandomState(rng_seed * HASH mod 2**32)``, so switching one
augmentation on or off never changes another's draws and the result does not depend on
DataLoader worker scheduling. Order of application: anchor jitter -> HFlip -> HintDrop ->
MDwarp -> BedWarp.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import AugConfig, Geometry
from .image import render, val_example
from .synth import make_correlated_noise

# derived-stream hash constants (registered in the ledger's RNG registry)
HASH_HFLIP = 3266489917
HASH_HINTDROP = 2654435761
HASH_MDWARP = 1234567891
HASH_BEDWARP = 3203431781


def sample_seed(base_seed: int, epoch: int, idx: int) -> int:
    return (base_seed * 1_000_003 + epoch * 7919 + idx) & 0xFFFFFFFF


class TrainDataset(Dataset):
    """``recs`` maps an id to a rendered record (``image.build_image``); ``ids`` is the order.

    Pretraining uses only synthetic records with HFlip at ``h_flip_p_pretrain`` (0) and no other
    augmentation; fine-tuning uses only real records with every augmentation in ``aug``."""

    def __init__(self, recs: dict, ids, geom: Geometry, base_seed: int,
                 h_flip_p: float = 0.0, hint_drop_p: float = 0.0,
                 md_warp: tuple = (0.0, 128, 1.0), bedwarp: tuple = (0.0, 64, 1.0)):
        self.recs, self.ids, self.geom = recs, list(ids), geom
        self.base_seed = int(base_seed) & 0xFFFFFFFF
        self.epoch = 0
        self.h_flip_p, self.hint_drop_p = float(h_flip_p), float(hint_drop_p)
        self.md_warp_strength, self.md_warp_autocorr, self.md_warp_p = float(md_warp[0]), int(md_warp[1]), float(md_warp[2])
        self.bedwarp_strength, self.bedwarp_autocorr, self.bedwarp_p = float(bedwarp[0]), int(bedwarp[1]), float(bedwarp[2])

    @classmethod
    def pretrain(cls, recs, geom, seed, aug: AugConfig):
        return cls(recs, list(recs), geom, base_seed=seed, h_flip_p=aug.h_flip_p_pretrain)

    @classmethod
    def finetune(cls, recs, ids, geom, seed, aug: AugConfig):
        return cls(recs, ids, geom, base_seed=(seed + 1_000_000) & 0xFFFFFFFF,
                   h_flip_p=aug.h_flip_p, hint_drop_p=aug.hint_drop_p,
                   md_warp=aug.md_warp(geom), bedwarp=aug.bedwarp(geom))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        H, W, K = self.geom.H, self.geom.W, self.geom.k_max
        wid = self.ids[idx]
        rec = self.recs[wid]
        rng_seed = sample_seed(self.base_seed, self.epoch, idx)
        k = np.random.RandomState(rng_seed).randint(-K, K + 1)
        image, truth, mask = render(rec, k, self.geom)

        if self.h_flip_p > 0.0:              # mirror along MD; dZ/dMD changes sign
            if np.random.RandomState((rng_seed * HASH_HFLIP) & 0xFFFFFFFF).random_sample() < self.h_flip_p:
                image = image[:, :, ::-1].copy()
                image[4] *= -1
                truth = truth[::-1].copy()
                mask = mask[::-1].copy()

        if self.hint_drop_p > 0.0:           # hide a share of the pinned pre-PS columns
            hint_rng = np.random.RandomState((rng_seed * HASH_HINTDROP) & 0xFFFFFFFF)
            has_pin = (image[2] > 0).any(axis=0)
            drop = has_pin & (hint_rng.random_sample(W) < self.hint_drop_p)
            if drop.any():
                image[2, :, drop] = 0.0

        if self.md_warp_strength > 0.0:      # smooth monotone reindex of columns (MD)
            warp_rng = np.random.RandomState((rng_seed * HASH_MDWARP) & 0xFFFFFFFF)
            if warp_rng.random_sample() < self.md_warp_p:
                g = make_correlated_noise(W, autocorr_length=self.md_warp_autocorr,
                                          target_std=self.md_warp_strength, rng=warp_rng)
                cum = np.cumsum(np.exp(g)); cum -= cum[0]
                if cum[-1] > 0:
                    src = np.clip(np.round(cum / cum[-1] * (W - 1)).astype(np.int64), 0, W - 1)
                    image, truth, mask = image[:, :, src], truth[src], mask[src]

        if self.bedwarp_strength > 0.0:      # smooth monotone reindex of rows (bed thickness)
            bw_rng = np.random.RandomState((rng_seed * HASH_BEDWARP) & 0xFFFFFFFF)
            if bw_rng.random_sample() < self.bedwarp_p:
                gh = make_correlated_noise(H, autocorr_length=self.bedwarp_autocorr,
                                           target_std=self.bedwarp_strength, rng=bw_rng)
                cumh = np.cumsum(np.exp(gh)); cumh -= cumh[0]
                if cumh[-1] > 0:
                    src_f = cumh / cumh[-1] * (H - 1)
                    src_r = np.clip(np.round(src_f).astype(np.int64), 0, H - 1)
                    image = image[:, src_r, :]
                    t_new = np.interp(truth.astype(np.float64), src_f, np.arange(H, dtype=np.float64))
                    truth = np.clip(np.round(t_new), 0, H - 1).astype(truth.dtype)

        return {'image': torch.from_numpy(image).clone(), 'truth_row': torch.from_numpy(truth).long(),
                'main_mask': torch.from_numpy(mask).clone(), 'well_id': wid}


class ValDataset(Dataset):
    """Canonical images of real wells, for scoring."""

    def __init__(self, recs: dict, ids):
        self.recs, self.ids = recs, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        wid = self.ids[idx]
        image, truth, mask = val_example(self.recs[wid])
        return {'image': torch.from_numpy(image).clone(), 'truth_row': torch.from_numpy(truth).long(),
                'main_mask': torch.from_numpy(mask).clone(), 'well_id': wid}
