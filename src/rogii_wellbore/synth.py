"""Synthetic wells for pretraining ("Method A v3" + event pack A).

A synthetic well takes a real typewell (GR as a function of TVT) and invents a trajectory
through it: a smooth landing before the PS point, then a dipping, wobbling post-PS path,
plus optional faults, an unconformity, variable bed thickness and (event pack A) dip breaks
and a repeated section. The well's GR is read from the typewell along that path with
calibration, drift and measurement noise. Z follows -TVT with a smooth dip term, computed
BEFORE faults so the wellbore stays continuous across them.

Every random draw comes from ``RandomState(seed)`` in a fixed order; event pack A draws from
its own stream ``RandomState(seed * expa_hash mod 2**32)``, so ``event_pack='v3'`` reproduces
the pack-free generator exactly. This module is a line-for-line port of the training
notebook's ``generate_synthetic_well_method_a_v3`` with the fault-zone bundle (off in every
member) removed; tests/test_parity_synth.py checks byte equality against the notebook.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import SynthConfig
from .data import WellStore


def make_correlated_noise(n, autocorr_length, target_std, rng):
    """AR(1) noise with unit stationary variance, scaled to ``target_std``."""
    phi = np.exp(-1.0 / autocorr_length)
    white = rng.normal(0, 1, n)
    smoothed = np.zeros(n)
    smoothed[0] = white[0]
    for i in range(1, n):
        smoothed[i] = phi * smoothed[i - 1] + np.sqrt(1 - phi ** 2) * white[i]
    return smoothed * target_std


def generate_well(real_tw_df: pd.DataFrame, hint_real_dist: dict, seed: int, cfg: SynthConfig):
    """One synthetic well, or None when the typewell or the drawn well is too short."""
    sig = cfg.sigmas()
    sigma_drift, sigma_calib = sig['sigma_drift'], sig['sigma_calib']
    sigma_tvt_offset, sigma_meas = sig['sigma_tvt_offset'], sig['sigma_meas']
    wobble_split, wavelength_divisors = cfg.wobble_split, cfg.wavelength_divisors

    rng = np.random.RandomState(seed)
    tw_tvt = real_tw_df['TVT'].values
    tw_gr_arr = real_tw_df['GR'].values
    ok = ~np.isnan(tw_tvt) & ~np.isnan(tw_gr_arr)
    tw_tvt = tw_tvt[ok]; tw_gr_arr = tw_gr_arr[ok]
    if len(tw_tvt) < 100:
        return None
    sort = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[sort]; tw_gr_arr = tw_gr_arr[sort]
    n_md = int(rng.choice(hint_real_dist['n_md']))
    ps_md_pct = float(rng.choice(hint_real_dist['ps_md_pct']))
    tvt_range = float(rng.choice(hint_real_dist['tvt_range']))
    tw_lo, tw_hi = tw_tvt[0], tw_tvt[-1]
    margin = tvt_range * 1.5
    if tw_hi - tw_lo < 2 * margin:
        heel_tvt = (tw_lo + tw_hi) / 2
    else:
        heel_tvt = float(rng.uniform(tw_lo + margin, tw_hi - margin))
    md_arr = np.arange(n_md).astype(np.float64)
    lateral_md = n_md * (1 - ps_md_pct / 100)
    if lateral_md < 100:
        return None
    dip_sign = rng.choice([-1, 1])
    dip_rate = dip_sign * tvt_range / max(lateral_md, 1)
    wobble_total_amp = tvt_range * cfg.wobble_total_pct * rng.uniform(0.7, 1.3)
    amp_lo = wobble_total_amp * wobble_split[0] * rng.uniform(0.5, 1.5)
    amp_mid = wobble_total_amp * wobble_split[1] * rng.uniform(0.5, 1.5)
    amp_hi = wobble_total_amp * wobble_split[2] * rng.uniform(0.5, 1.5)
    omega_lo = 2 * np.pi / (lateral_md * wavelength_divisors[0] * rng.uniform(0.7, 1.3))
    omega_mid = 2 * np.pi / (lateral_md / wavelength_divisors[1] * rng.uniform(0.7, 1.3))
    omega_hi = 2 * np.pi / (lateral_md / wavelength_divisors[2] * rng.uniform(0.7, 1.3))
    phi_lo = rng.uniform(0, 2 * np.pi)
    phi_mid = rng.uniform(0, 2 * np.pi)
    phi_hi = rng.uniform(0, 2 * np.pi)
    ps_md = int(ps_md_pct / 100 * n_md)
    ps_md = max(ps_md, 100); ps_md = min(ps_md, n_md - 100)
    pre_md = md_arr[:ps_md]
    pre_start_tvt = max(tw_lo, heel_tvt - tvt_range * 8)
    pre_t = pre_md / max(ps_md - 1, 1)
    pre_tvt = pre_start_tvt + (heel_tvt - pre_start_tvt) * (0.5 - 0.5 * np.cos(pre_t * np.pi))
    post_md = md_arr[ps_md:] - md_arr[ps_md]

    # --- variable bed thickness: smooth modulation of the wobble amplitudes ---
    var_bed_info = None
    if cfg.var_bed_prob > 0 and rng.uniform() < cfg.var_bed_prob:
        var_bed_mod_full = make_correlated_noise(n_md, autocorr_length=cfg.var_bed_autocorr_length,
                                                 target_std=cfg.var_bed_amp_std, rng=rng) + 1.0
        var_bed_mod_full = np.clip(var_bed_mod_full, cfg.var_bed_amp_min, cfg.var_bed_amp_max).astype(np.float64)
        var_bed_info = {'mod_std_observed': float(np.std(var_bed_mod_full)),
                        'mod_min': float(var_bed_mod_full.min()), 'mod_max': float(var_bed_mod_full.max())}
    else:
        var_bed_mod_full = np.ones(n_md, dtype=np.float64)
    var_bed_mod_post = var_bed_mod_full[ps_md:ps_md + len(post_md)]

    post_tvt = (heel_tvt + dip_rate * post_md
                + var_bed_mod_post * amp_lo * np.sin(omega_lo * post_md + phi_lo)
                + var_bed_mod_post * amp_mid * np.sin(omega_mid * post_md + phi_mid)
                + var_bed_mod_post * amp_hi * np.sin(omega_hi * post_md + phi_hi))
    tvt_full = np.concatenate([pre_tvt, post_tvt])

    # --- faults: TVT jumps in the 20-80% window of the post-PS section; Z uses the pre-fault TVT ---
    tvt_full_for_z = tvt_full.copy()
    fault_info = []
    if cfg.fault_prob > 0 and rng.uniform() < cfg.fault_prob:
        n_faults = 1 if rng.uniform() < 0.7 else 2
        post_md_lo = ps_md + int(0.20 * (n_md - ps_md))
        post_md_hi = ps_md + int(0.80 * (n_md - ps_md))
        if post_md_hi - post_md_lo >= 50:
            n_faults_eff = min(n_faults, post_md_hi - post_md_lo)
            positions = np.sort(rng.choice(np.arange(post_md_lo, post_md_hi), size=n_faults_eff, replace=False))
            magnitudes = rng.uniform(cfg.fault_magnitude_min, cfg.fault_magnitude_max,
                                     size=n_faults_eff) * rng.choice([-1, 1], size=n_faults_eff)
            for pos, mag in zip(positions, magnitudes):
                tvt_full[pos:] += mag
                fault_info.append({'md_idx': int(pos), 'magnitude': float(mag)})

    # --- unconformity: a bounded slope change in the 20-70% window ---
    unconf_info = None
    if cfg.unconf_prob > 0 and rng.uniform() < cfg.unconf_prob:
        unconf_lo = ps_md + int(0.20 * (n_md - ps_md))
        unconf_hi = ps_md + int(0.70 * (n_md - ps_md))
        if unconf_hi - unconf_lo >= 100:
            unconf_pos = int(rng.choice(np.arange(unconf_lo, unconf_hi)))
            lateral_remaining = n_md - unconf_pos
            max_delta_slope = cfg.unconf_max_cumulative_shift / max(lateral_remaining, 1)
            unconf_delta_slope = float(rng.uniform(-max_delta_slope, max_delta_slope))
            md_relative = (np.arange(n_md) - unconf_pos).astype(np.float64)
            md_relative[md_relative < 0] = 0.0
            tvt_full = tvt_full + unconf_delta_slope * md_relative
            unconf_info = {'md_idx': unconf_pos, 'delta_slope': unconf_delta_slope,
                           'cumulative_shift': float(unconf_delta_slope * lateral_remaining)}

    # --- event pack A: dip breaks + repeated section (+ dormant pinch), own RNG stream ---
    pack_info = None
    pinch_map_fn = None
    if cfg.event_pack == 'v3_expA':
        tvt_full, pack_info, pinch_map_fn = _event_pack_a(tvt_full, fault_info, ps_md, n_md, seed, cfg)
    elif cfg.event_pack != 'v3':
        raise ValueError(f'unknown event_pack {cfg.event_pack!r}')

    tvt_offset = make_correlated_noise(n_md, autocorr_length=cfg.tvt_offset_autocorr,
                                       target_std=sigma_tvt_offset, rng=rng)
    lk = tvt_full if pinch_map_fn is None else pinch_map_fn(tvt_full)
    tvt_for_lookup = np.clip(lk + tvt_offset, tw_tvt[0], tw_tvt[-1])
    hw_gr_lookup = np.interp(tvt_for_lookup, tw_tvt, tw_gr_arr)
    calib_factor = 1.0 + rng.normal(0, sigma_calib)
    drift_noise = make_correlated_noise(n_md, autocorr_length=cfg.drift_autocorr,
                                        target_std=sigma_drift, rng=rng)
    meas_noise = rng.normal(0, sigma_meas, n_md)
    hw_gr_synth = hw_gr_lookup * calib_factor + drift_noise + meas_noise
    hw_gr_synth = np.clip(hw_gr_synth, 10, 250).astype(np.float32)
    tvt_input = np.full(n_md, np.nan)
    tvt_input[:ps_md] = tvt_full[:ps_md]
    synth_z_dip_amp = tvt_range * cfg.dz_dip_frac * rng.uniform(0.5, 1.5)
    synth_z_dip_acl = max(200, int(n_md * cfg.dz_dip_autocorr_frac))
    synth_z_dip = make_correlated_noise(n_md, autocorr_length=synth_z_dip_acl,
                                        target_std=synth_z_dip_amp, rng=rng)
    synth_z_drill_noise = rng.normal(0, cfg.dz_noise_std, n_md)
    z_full = (-tvt_full_for_z + synth_z_dip + synth_z_drill_noise).astype(np.float32)
    return {
        'MD': md_arr.astype(np.float32), 'TVT': tvt_full.astype(np.float32),
        'TVT_input': tvt_input.astype(np.float32), 'GR': hw_gr_synth, 'Z': z_full,
        'ps_md_idx': ps_md, '_tw_tvt': tw_tvt.astype(np.float32), '_tw_gr': tw_gr_arr.astype(np.float32),
        '_fault_info': fault_info, '_unconf_info': unconf_info, '_var_bed_info': var_bed_info,
        '_pack_info': pack_info,
    }


def _event_pack_a(tvt_full, fault_info, ps_md, n_md, seed, cfg: SynthConfig):
    """Dip breaks (A1), a repeated section (A3-lite) and the pinch lens (rate 0 in every
    member, but its gate draw is still consumed so the stream matches the notebook).
    Guards act on the pack's added effect: added |slope| <= max_abs_dip and the post-PS
    excursion may not exceed max(base excursion, exc_cap_ft); a breach redraws (6 tries),
    then the pack is dropped for this well."""
    prng = np.random.RandomState((int(seed if seed is not None else 0) * cfg.expa_hash) & 0xFFFFFFFF)
    plo = ps_md + int(0.20 * (n_md - ps_md)); phi = ps_md + int(0.80 * (n_md - ps_md))
    base_tvt = tvt_full.copy()
    b_post = base_tvt[ps_md:]
    b_exc = float(np.nanmax(b_post) - np.nanmin(b_post))
    pack_info = {'events': [], 'dropped': False, 'tries': 0}
    pinch_map_fn = None
    for _try in range(6):
        tvt_full = base_tvt.copy(); ev = []; pinch = None; rep_span = None
        if phi - plo >= 50 and prng.uniform() < cfg.dipbreak_prob:
            nb = 2 if prng.uniform() < cfg.dipbreak_second_p else 1
            for _ in range(nb):
                cpl = bool(fault_info) and prng.uniform() < cfg.dipbreak_fault_couple_p
                if cpl:
                    p = int(prng.choice([f['md_idx'] for f in fault_info]))
                else:
                    p = int(prng.choice(np.arange(plo, phi)))
                ds = float(prng.uniform(cfg.dipbreak_dslope_min, cfg.dipbreak_dslope_max))
                ds *= (1.0 if prng.uniform() < 0.5 else -1.0)
                rel = (np.arange(n_md) - p).astype(np.float64); rel[rel < 0] = 0.0
                tvt_full = tvt_full + ds * rel
                ev.append({'type': 'dipbreak', 'md_idx': p, 'delta_slope': ds, 'coupled': cpl})
        if phi - plo >= 50 and prng.uniform() < cfg.repeat_prob:
            p = int(prng.choice(np.arange(plo, phi)))
            band = tvt_full[ps_md:p]
            blo = float(np.nanmin(band)); bhi = float(np.nanmax(band)); ext = bhi - blo
            if ext >= max(8.0, cfg.repeat_shift_min):
                target = float(prng.uniform(blo + 0.15 * ext, bhi - 0.15 * ext))
                s = float(np.clip(target - tvt_full[p], -cfg.repeat_shift_max, cfg.repeat_shift_max))
                if abs(s) >= 2.0 and (blo <= tvt_full[p] + s <= bhi):
                    L = int(prng.uniform(cfg.repeat_l_min, cfg.repeat_l_max))
                    dp = float(prng.uniform(cfg.repeat_damp_min, cfg.repeat_damp_max))
                    tvt_full[p:] += s
                    e = min(p + L, n_md)
                    if e - p >= 3:
                        seg = tvt_full[p:e].copy()
                        new = seg[0] + np.concatenate([[0.0], np.cumsum(np.diff(seg) * dp)])
                        tail = new[-1] - seg[-1]
                        tvt_full[p:e] = new
                        tvt_full[e:] += tail
                    rep_span = (p, e)
                    ev.append({'type': 'repeat', 'md_idx': p, 'shift': s, 'damp': dp, 'L': int(e - p)})
        if prng.uniform() < cfg.pinch_prob:
            pt = tvt_full[ps_md:]
            t0r = float(np.nanmin(pt)); t1r = float(np.nanmax(pt))
            h = float(prng.uniform(cfg.pinch_h_min, cfg.pinch_h_max))
            if t1r - t0r > h + 2.0:
                t0 = float(prng.uniform(t0r, t1r - h)); c = float(prng.uniform(cfg.pinch_c_min, cfg.pinch_c_max))
                rec = float((1.0 - c) * h * prng.uniform(1.2, 2.0))
                pinch = (t0, h, c, rec)
                ev.append({'type': 'pinch', 't0': t0, 'h': h, 'c': c, 'rec': rec})
        dd = np.abs(np.diff((tvt_full - base_tvt)[ps_md:]))
        if rep_span is not None:
            a = max(rep_span[0] - ps_md - 1, 0)
            b = min(rep_span[1] - ps_md + 1, len(dd))
            if b > a:
                dd[a:b] = 0.0
        pack_slope = float(dd.max()) if len(dd) else 0.0
        post = tvt_full[ps_md:]
        exc = float(np.nanmax(post) - np.nanmin(post))
        if pack_slope <= cfg.max_abs_dip and exc <= max(b_exc, cfg.exc_cap_ft):
            pack_info['events'] = ev; pack_info['tries'] = _try + 1
            if pinch is not None:
                pinch_map_fn = _pinch_map(*pinch)
            break
    else:
        tvt_full = base_tvt
        pack_info['dropped'] = True; pack_info['tries'] = 6
    return tvt_full, pack_info, pinch_map_fn


def _pinch_map(t0, h, c, rec):
    def fn(t):
        u = np.asarray(t, dtype=np.float64).copy()
        d = (1.0 - c) * h
        mid = (u > t0) & (u < t0 + c * h)
        rcv = (u >= t0 + c * h) & (u < t0 + c * h + rec)
        u[mid] = t0 + (u[mid] - t0) / c
        u[rcv] = u[rcv] + d * (1.0 - (u[rcv] - (t0 + c * h)) / rec)
        return u
    return fn


# ---------------------------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------------------------
def build_pool(store: WellStore, train_wids, hint_dist: dict, fold: int, cfg: SynthConfig,
               cv_seed: int = 34, n_wells: int | None = None, log=print) -> list[dict]:
    """The fold's pretraining pool. Typewells are drawn (with replacement) from the fold's
    training wells only, with ``RandomState(cv_seed + 5000 + fold)``; synthetic well i uses
    seed ``cv_seed + 7000 + fold * 100000 + i``."""
    n = cfg.n_wells if n_wells is None else n_wells
    picks = np.random.RandomState(cv_seed + 5000 + fold).choice(train_wids, size=n, replace=True)
    pool, t0 = [], time.time()
    tw_cache: dict = {}
    for i, tw_wid in enumerate(picks):
        tw = tw_cache.get(tw_wid)
        if tw is None:
            tw = tw_cache[tw_wid] = store.read_tw(tw_wid, usecols=['TVT', 'GR'])
        syn = generate_well(tw, hint_dist, seed=cv_seed + 7000 + fold * 100000 + i, cfg=cfg)
        if syn is not None:
            syn['_synth_id'] = f'synth_f{fold}_pretrain_{i:04d}'
            syn['_seed_tw_wid'] = tw_wid
            pool.append(syn)
    log(f'  fold {fold}: {len(pool)} synthetic wells in {time.time() - t0:.0f}s; ' + pool_stats(pool))
    return pool


def pool_stats(pool) -> str:
    n = max(len(pool), 1)
    faulted = sum(1 for w in pool if w.get('_fault_info'))
    unconf = sum(1 for w in pool if w.get('_unconf_info'))
    varbed = sum(1 for w in pool if w.get('_var_bed_info'))
    packs = [w['_pack_info'] for w in pool if w.get('_pack_info')]
    ev = [e for p in packs for e in p['events']]
    return (f'faults {faulted / n:.0%}, unconformities {unconf / n:.0%}, variable beds {varbed / n:.0%}'
            + (f', dip breaks {sum(e["type"] == "dipbreak" for e in ev)}, repeats '
               f'{sum(e["type"] == "repeat" for e in ev)}, guard-dropped {sum(p["dropped"] for p in packs)}'
               if packs else ''))


def leakage_audit(pool, fold_split) -> None:
    """Hard stop if any synthetic well was built from a typewell outside the fold's training set."""
    train = set(fold_split['train']); val = set(fold_split['val'])
    seeds = [s['_seed_tw_wid'] for s in pool]
    leaked = [w for w in seeds if w in val]
    outside = [w for w in seeds if w not in train]
    assert not leaked and not outside, f'synthetic pool leakage: {len(leaked)} val, {len(outside)} non-train typewells'


def match_corr(hw_gr, hw_tvt, tw_tvt, tw_gr) -> float:
    """Correlation between a well's GR and the typewell GR at the true TVT (post-PS)."""
    t = np.interp(hw_tvt, tw_tvt, tw_gr)
    ok = np.isfinite(hw_gr) & np.isfinite(t) & np.isfinite(hw_tvt)
    if ok.sum() < 20:
        return np.nan
    a, b = hw_gr[ok], t[ok]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
