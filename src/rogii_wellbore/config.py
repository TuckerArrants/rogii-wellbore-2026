"""Run configuration: one YAML per ensemble member, everything the notebook held in globals.

A config file may name a parent with ``base: <relative path>``; the child's keys are merged
over the parent's, section by section (nested dicts merge, everything else replaces). The
dataclasses below are the full set of knobs the final members used. Knobs the notebook
carried switched off (shear, PhaseJitter/ToeCrop/GR augmentation in training, fault-zone
events, posterior dilation, multi-scale injection, warm restarts, the integral-regression
loss, knowledge distillation) were not ported; see docs/PORTING.md.
"""
from __future__ import annotations

import copy
import dataclasses as dc
import json
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class Geometry:
    """Image geometry. The vertical extent is H * bin_size feet around the last known TVT."""
    H: int = 256
    W: int = 512
    bin_size: float = 1.0
    k_max: int = 12          # anchor-jitter range in rows; also the TW-log padding

    def __post_init__(self):
        assert self.H % 32 == 0 and self.W % 32 == 0, f'H, W must be multiples of 32: {self}'
        assert self.bin_size > 0 and self.k_max >= 0

    @classmethod
    def from_height(cls, H: int, extent_ft: float = 256.0, k_max: int = 12) -> 'Geometry':
        """The house formula: W = 2H, bin_size = 256 / H (vertical extent fixed at 256 ft)."""
        return cls(H=H, W=2 * H, bin_size=extent_ft / H, k_max=k_max)


@dc.dataclass(frozen=True)
class SynthConfig:
    """Synthetic pretraining pool (generator "Method A v3" + event pack A)."""
    n_wells: int = 25000                     # per fold
    corruption: str = 'clean'                # key into CORRUPTION_PRESETS
    wobble_total_pct: float = 0.15
    wobble_split: tuple = (0.50, 0.30, 0.20)
    wavelength_divisors: tuple = (1.0, 5.0, 20.0)
    drift_autocorr: int = 40
    tvt_offset_autocorr: int = 200
    # faults (FT39j)
    fault_prob: float = 0.33
    fault_magnitude_min: float = 5.0
    fault_magnitude_max: float = 50.0
    # unconformities (FT39k)
    unconf_prob: float = 0.33
    unconf_max_cumulative_shift: float = 30.0
    # variable bed thickness (FT39l)
    var_bed_prob: float = 0.50
    var_bed_amp_std: float = 0.40
    var_bed_autocorr_length: int = 200
    var_bed_amp_min: float = 0.30
    var_bed_amp_max: float = 2.00
    # synthetic Z
    dz_dip_frac: float = 0.10
    dz_dip_autocorr_frac: float = 0.20
    dz_noise_std: float = 0.5
    # event pack A (EVEX-A): 'v3' = no pack, 'v3_expA' = dip breaks + repeated sections
    event_pack: str = 'v3_expA'
    dipbreak_prob: float = 0.35
    dipbreak_second_p: float = 0.30
    dipbreak_fault_couple_p: float = 0.30
    dipbreak_dslope_min: float = 0.008
    dipbreak_dslope_max: float = 0.035
    pinch_prob: float = 0.0                  # machinery kept; rate 0 in every member
    pinch_h_min: float = 10.0
    pinch_h_max: float = 30.0
    pinch_c_min: float = 0.05
    pinch_c_max: float = 0.20
    repeat_prob: float = 0.12
    repeat_shift_min: float = 10.0
    repeat_shift_max: float = 40.0
    repeat_damp_min: float = 0.2
    repeat_damp_max: float = 0.5
    repeat_l_min: int = 150
    repeat_l_max: int = 400
    max_abs_dip: float = 0.050
    exc_cap_ft: float = 320.0
    expa_hash: int = 3714159261

    def sigmas(self) -> dict:
        return dict(CORRUPTION_PRESETS[self.corruption])


CORRUPTION_PRESETS = {
    'baseline':  dict(sigma_drift=8.0, sigma_tvt_offset=3.5, sigma_meas=1.5, sigma_calib=0.06),
    'reduced_B': dict(sigma_drift=3.0, sigma_tvt_offset=1.0, sigma_meas=1.0, sigma_calib=0.06),
    'reduced_C': dict(sigma_drift=1.5, sigma_tvt_offset=0.5, sigma_meas=0.7, sigma_calib=0.06),
    'clean':     dict(sigma_drift=0.5, sigma_tvt_offset=0.2, sigma_meas=0.5, sigma_calib=0.06),
}


@dc.dataclass(frozen=True)
class AugConfig:
    """Fine-tuning augmentation. Strengths are given at the reference geometry (W=512 for
    MDwarp, H=256 for BedWarp) and scaled to the run's geometry, as the notebook did
    (``0.36 * W / 512``); autocorrelation lengths are W//4 and H//4 unless set."""
    h_flip_p: float = 0.5
    h_flip_p_pretrain: float = 0.0
    hint_drop_p: float = 0.3
    md_warp_p: float = 0.5
    md_warp_strength_ref: float = 0.36
    md_warp_autocorr: int | None = None
    bedwarp_p: float = 0.5
    bedwarp_strength_ref: float = 0.36
    bedwarp_autocorr: int | None = None

    def md_warp(self, geom: Geometry) -> tuple[float, int, float]:
        return (self.md_warp_strength_ref * geom.W / 512,
                self.md_warp_autocorr or geom.W // 4, self.md_warp_p)

    def bedwarp(self, geom: Geometry) -> tuple[float, int, float]:
        return (self.bedwarp_strength_ref * geom.H / 256,
                self.bedwarp_autocorr or geom.H // 4, self.bedwarp_p)


@dc.dataclass(frozen=True)
class ModelConfig:
    encoder: str = 'tu-tf_efficientnetv2_l.in21k_ft_in1k'
    in_ch: int = 5
    decoder_channels: tuple = (512, 256, 128, 64, 32)
    decoder_interp: str = 'nearest'
    drop_path_rate: float = 0.2
    detach_handoff: bool = True
    handoff_t: float = 1.5            # stage-1 -> stage-2 softmax temperature (train AND eval)
    refeed_t: float = 1.0             # temperature on re-fed stage-2 beliefs (rounds 2..k)


@dc.dataclass(frozen=True)
class TrainConfig:
    folds: tuple = (0, 1, 2, 3, 4)
    seeds: tuple = (34, 68, 102, 136, 170, 204, 238)
    pretrain_epochs: int = 5
    finetune_epochs: int = 35
    lr: float = 1e-3
    wd: float = 1e-3
    batch_size: int = 8
    patience: int = 15
    ema_decay: float = 0.997
    grad_clip: float = 5.0
    deep_sup_weight: float = 0.5        # CE on stage-1 logits
    amp: str | None = 'bfloat16'         # None | 'bfloat16' | 'float16'
    # unrolled refeed training (fine-tuning only)
    unroll_p: float = 0.5
    unroll_k_choices: tuple = (2, 3, 4)
    unroll_k_weights: tuple = (1.0, 1.0, 1.0)
    unroll_round_sup_weight: float | None = 0.5   # None = supervise the final round only
    unroll_rng_seed: int = 20260707
    # checkpoint selection on the deployment depth
    checkpoint_k: int = 3
    deploy_k_log: tuple = (2, 3)
    num_workers: int = 4


@dc.dataclass(frozen=True)
class CVConfig:
    seed: int = 34
    n_folds: int = 5


@dc.dataclass(frozen=True)
class RunConfig:
    name: str
    geometry: Geometry = Geometry()
    synth: SynthConfig = SynthConfig()
    aug: AugConfig = AugConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    cv: CVConfig = CVConfig()
    notes: str = ''

    # --- derived -------------------------------------------------------------------------
    def experiment_name(self) -> str:
        """The run-folder name the notebook derived, e.g.
        5fold_effnetv2_l_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt."""
        enc_short = ENCODER_SHORT.get(self.model.encoder, self.model.encoder.replace('tu-', ''))
        bits = [f'{len(self.train.folds)}fold', enc_short, 'stack2', 'ema', f'{self.synth.n_wells}pt',
                f'mdwarp{round(self.aug.md_warp_strength_ref * 100)}',
                f'bedwarp{round(self.aug.bedwarp_strength_ref * 100)}']
        if self.train.unroll_p > 0:
            bits += [f'unroll{round(self.train.unroll_p * 100)}']
            if self.train.unroll_round_sup_weight:
                bits += ['deepsup']
        if self.model.handoff_t != 1.0:
            bits += [f"handoffT{str(self.model.handoff_t).replace('.', '')}"]
        if self.synth.event_pack == 'v3_expA':
            bits += ['expa']
        name = '_'.join(bits) + '_pt'
        g = self.geometry
        if (g.H, g.W, g.bin_size) != (256, 512, 1.0):
            name += f'_H{g.H}_W{g.W}_bin{g.bin_size}'
        return name

    def to_dict(self) -> dict:
        return json.loads(json.dumps(dc.asdict(self)))


ENCODER_SHORT = {
    'tu-tf_efficientnetv2_l.in21k_ft_in1k': 'effnetv2_l',
    'tu-tf_efficientnetv2_xl.in21k_ft_in1k': 'effnetv2_xl',
    'tu-tf_efficientnet_l2.ns_jft_in1k_475': 'l2',
}


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
_SECTIONS = {'geometry': Geometry, 'synth': SynthConfig, 'aug': AugConfig, 'model': ModelConfig,
             'train': TrainConfig, 'cv': CVConfig}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop('base', None)
    if base is not None:
        raw = _merge(load_raw(path.parent / base), raw)
    return raw


def _build(cls, d: dict):
    names = {f.name: f for f in dc.fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise KeyError(f'{cls.__name__}: unknown keys {sorted(unknown)}')
    kw = {}
    for k, v in d.items():
        default = names[k].default
        kw[k] = tuple(v) if isinstance(default, tuple) and isinstance(v, list) else v
    return cls(**kw)


def from_dict(raw: dict) -> RunConfig:
    raw = dict(raw)
    unknown = set(raw) - set(_SECTIONS) - {'name', 'notes'}
    if unknown:
        raise KeyError(f'RunConfig: unknown sections {sorted(unknown)}')
    kw = {k: _build(cls, raw.get(k, {})) for k, cls in _SECTIONS.items()}
    if 'name' not in raw:
        raise KeyError('a run config needs a name')
    return RunConfig(name=raw['name'], notes=raw.get('notes', ''), **kw)


def load(path: str | Path) -> RunConfig:
    return from_dict(load_raw(path))
