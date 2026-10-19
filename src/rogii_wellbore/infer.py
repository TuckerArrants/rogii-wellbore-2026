"""Kaggle inference: several members' checkpoints -> one blended ``submission.csv``.

    python -m rogii_wellbore.infer --config configs/submit/final_equal_weight.yaml \
        --comp-root /kaggle/input/competitions/rogii-wellbore-geology-prediction --out submission.csv

Per member (a folder holding ``*_kaggle_inference_manifest.json`` and its checkpoints): the
manifest sets the encoder and image geometry; the submit config sets the blend weight, the
refeed depth ``eval_k`` and the temperatures, which are checked against the manifest when it
records them (by field, or in the free-text notes of the older manifests). Each checkpoint
predicts every test well at depth ``eval_k``; a member's prediction is the mean over its
checkpoints (and over PhaseJitter shifts when ``tta`` is on); the blend is the weighted mean
of the members, per native MD row. Rows a member cannot cover are filled forward, and rows
with no prediction at all fall back to the last known TVT.

Port of ``final-submission-equal-weight-no-tta.ipynb``; tests/test_parity_infer.py checks the
image, the refeed and the per-well prediction against that notebook's functions.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import yaml

from .config import Geometry, ModelConfig
from .data import WellStore
from .image import arrays_from_frames, build_image
from .model import StackedUNet, expected_row, iter_logits, load_checkpoint

PHASE_TTA_OFFSETS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


# ---------------------------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------------------------
@dataclass
class Member:
    name: str
    weight: float
    dir: str
    eval_k: int
    model_cfg: ModelConfig
    geom: Geometry
    ckpt_paths: list
    tta: bool = False
    manifest: dict = field(default_factory=dict)


def find_manifest(ckpt_dir: str) -> str:
    hits = sorted(glob.glob(os.path.join(ckpt_dir, '*_kaggle_inference_manifest.json')))
    if not hits:
        hits = sorted(glob.glob(os.path.join(ckpt_dir, '**', '*_kaggle_inference_manifest.json'), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f'{ckpt_dir}: expected exactly one *_kaggle_inference_manifest.json, found {len(hits)}')
    return hits[0]


_T_PATTERNS = {'handoff_T': re.compile(r'\bhandoff[_\- ]?T\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)', re.I),
               'refeed_T': re.compile(r'\brefeed[_\- ]?T\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)', re.I)}


def recorded_temperature(manifest: dict, key: str):
    """(value, source) from the recipe field, else from the manifest's free text; (None, why) if absent."""
    if key in manifest.get('recipe', {}):
        return float(manifest['recipe'][key]), 'recipe field'
    vals = sorted({float(v) for v in _T_PATTERNS[key].findall(json.dumps(manifest))})
    if len(vals) == 1:
        return vals[0], 'free-text note'
    return None, ('ambiguous: %s' % vals) if vals else 'absent'


def _entry_fold(e):
    if 'fold' in e:
        return int(e['fold'])
    return int(re.search(r'_f(\d+)_s', e['filename']).group(1))


def resolve_member(name: str, spec: dict, root: str = '', log=print) -> Member:
    ckpt_dir = os.path.join(root, spec['dir']) if root else spec['dir']
    mpath = find_manifest(ckpt_dir)
    member_dir = os.path.dirname(mpath)
    man = json.load(open(mpath))
    rec = man['recipe']
    ic = man['image_constants']
    geom = Geometry(H=int(ic['H']), W=int(ic['W']), bin_size=float(ic['BIN_SIZE']), k_max=int(ic['K_MAX']))
    assert int(rec['in_ch']) == 5 and int(rec.get('stack_n', 1)) >= 2, f'[{name}] expected a 5-channel stacked artifact'
    for key, local in (('handoff_T', float(spec['handoff_t'])), ('refeed_T', float(spec.get('refeed_t', 1.0)))):
        val, src = recorded_temperature(man, key)
        if val is None:
            log(f'  [{name}] {key} not recorded in the manifest ({src}); using {local}')
        elif val != local:
            raise AssertionError(f'[{name}] {key}: manifest {val} ({src}) != config {local}')
    mcfg = ModelConfig(encoder=rec['encoder_name'], in_ch=5,
                       decoder_channels=tuple(rec.get('decoder_channels', ModelConfig.decoder_channels)),
                       decoder_interp=rec.get('decoder_interp', 'nearest'),
                       handoff_t=float(spec['handoff_t']), refeed_t=float(spec.get('refeed_t', 1.0)))
    folds = spec.get('folds')
    entries = [e for e in man['checkpoints'] if folds is None or _entry_fold(e) in set(folds)]
    paths = []
    for e in entries:
        p = os.path.join(member_dir, e['filename'])
        if not os.path.exists(p):
            hits = sorted(glob.glob(os.path.join(member_dir, e['filename'][:-3] + '_*.pt')))
            if not hits:
                raise FileNotFoundError(f'[{name}] missing checkpoint {e["filename"]} in {member_dir}')
            p = hits[0]
        paths.append(p)
    log(f'[{name}] {len(paths)} checkpoints, {rec["encoder_name"]}, {geom.H}x{geom.W}@{geom.bin_size:.4f}, '
        f'k={spec["eval_k"]}, weight {spec["weight"]}' + (f', manifest CV {man["full_cv_metric"]["ens_rmse"]:.3f}'
                                                            if 'ens_rmse' in man.get('full_cv_metric', {}) else ''))
    return Member(name, float(spec['weight']), member_dir, int(spec['eval_k']), mcfg, geom, paths,
                  tta=bool(spec.get('tta', False)), manifest=man)


def check_weights(members) -> None:
    total = float(sum(m.weight for m in members))
    if any(not (0 < m.weight <= 1) for m in members) or total != 1.0:
        raise ValueError(f'member weights must be in (0, 1] and sum to exactly 1.0, got {[m.weight for m in members]}')


# ---------------------------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------------------------
def phase_records(arrays: dict, geom: Geometry, offsets) -> list:
    """The well rendered once per PhaseJitter offset: drop ``round(off * n/W)`` toe rows and
    re-bin, which moves every column boundary by part of a bin. Offset 0 is the plain render."""
    n_hw = len(arrays['hw_md'])
    bin_w = n_hw / float(geom.W)
    recs = []
    for off in offsets:
        drop = int(round(float(off) * bin_w))
        drop = min(drop, n_hw - geom.W - 1) if n_hw - geom.W - 1 > 0 else 0
        end = n_hw - drop
        a = {k: (v[:end] if isinstance(v, np.ndarray) and v.shape[0] == n_hw else v) for k, v in arrays.items()}
        try:
            recs.append(build_image(**a, geom=geom))
        except Exception:
            continue
    return recs


def predict_well(arrays: dict, models: list, k: int, geom: Geometry, tta: bool, phase_chunk: int = 8) -> np.ndarray:
    """Mean prediction of ``models`` (one member) on every native row of one well, NaN-free."""
    n_hw = len(arrays['hw_md'])
    recs = phase_records(arrays, geom, PHASE_TTA_OFFSETS if tta else (0.0,))
    if not recs:
        return np.full(n_hw, np.nan)
    P = len(recs)
    imgs = torch.from_numpy(np.stack([r['image_in'] for r in recs]))

    def run(group):
        local = np.zeros((P, geom.W), dtype=np.float64)
        with torch.no_grad():
            for m in group:
                dev = next(m.parameters()).device
                ri = torch.arange(geom.H, device=dev, dtype=torch.float32)
                for lo in range(0, P, phase_chunk):
                    x = imgs[lo:lo + phase_chunk].to(dev)
                    kl, _ = iter_logits(m, x, k)
                    local[lo:lo + x.shape[0]] += expected_row(kl[k], ri).cpu().numpy()
        return local

    by_dev: dict = {}
    for m in models:
        by_dev.setdefault(str(next(m.parameters()).device), []).append(m)
    er_sum = np.zeros((P, geom.W))
    if len(by_dev) > 1:
        with ThreadPoolExecutor(max_workers=len(by_dev)) as ex:
            for local in ex.map(run, by_dev.values()):
                er_sum += local
    else:
        for g in by_dev.values():
            er_sum += run(g)

    acc = np.zeros(n_hw); cnt = np.zeros(n_hw)
    for pi, rec in enumerate(recs):
        be = rec['bin_edges']; widths = np.diff(be)
        pred_col = rec['tvt_lo'] + (er_sum[pi] / len(models) + 0.5) * geom.bin_size
        acc[be[0]:be[-1]] += np.repeat(pred_col, widths)
        cnt[be[0]:be[-1]] += np.repeat(np.ones(geom.W), widths)
    p = np.full(n_hw, np.nan)
    cov = cnt > 0
    p[cov] = acc[cov] / cnt[cov]
    if not cov.all():
        last = np.nan
        for r in range(n_hw):
            if cov[r]:
                last = p[r]
            elif not np.isnan(last):
                p[r] = last
        if np.isnan(p[0]):
            p[:np.argmax(cov)] = p[cov][0] if cov.any() else 0.0
    return p


def _devices():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [torch.device(f'cuda:{i}') for i in range(n)] or [torch.device('cpu')]


def member_batch_size(paths, n_dev, cap=8, vram_gb=13.5, reserve_gb=3.5) -> int:
    """How many checkpoints to hold at once: what fits per GPU after an activation reserve."""
    try:
        gib = max(os.path.getsize(p) for p in paths) / 2 ** 30
    except OSError:
        return max(1, min(cap, n_dev))
    per_dev = int((vram_gb - reserve_gb) // max(gib * 1.10, 1e-9))
    return max(1, min(cap, max(per_dev, 1) * n_dev))


def predict_member(member: Member, wells: dict, devices=None, log=print) -> dict:
    devices = devices or _devices()
    bs = member_batch_size(member.ckpt_paths, len(devices))
    sums = {w: np.zeros(len(a['hw_md'])) for w, a in wells.items()}
    n = 0
    for lo in range(0, len(member.ckpt_paths), bs):
        models = []
        for j, p in enumerate(member.ckpt_paths[lo:lo + bs]):
            m = StackedUNet(member.model_cfg, encoder_weights=None, train_mode=False)
            models.append(load_checkpoint(m, p).to(devices[j % len(devices)]).eval())
        for w, a in wells.items():
            sums[w] += predict_well(a, models, member.eval_k, member.geom, member.tta) * len(models)
        n += len(models)
        models.clear(); gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f'  [{member.name}] {n}/{len(member.ckpt_paths)} checkpoints')
    return {w: s / n for w, s in sums.items()}


def read_test_wells(comp_root: str) -> tuple[dict, dict]:
    store = WellStore.test(comp_root)
    wells, last_known = {}, {}
    for wid in store.well_ids():
        hw = store.read_hw(wid); tw = store.read_tw(wid)
        wells[wid] = arrays_from_frames(hw, tw, with_truth=False)
        known = hw['TVT_input'].dropna()
        last_known[wid] = float(known.iloc[-1]) if len(known) else float('nan')
    return wells, last_known


def blend(member_preds: dict, members: list) -> dict:
    check_weights(members)
    wids = next(iter(member_preds.values())).keys()
    return {w: sum(m.weight * member_preds[m.name][w] for m in members) for w in wids}


def make_submission(preds: dict, wells: dict, last_known: dict, sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Fill ``sample_sub`` (id = ``{well}_{md_row}``) from per-row predictions on post-PS rows."""
    recs = []
    for w, p in preds.items():
        post = np.where(np.isnan(wells[w]['hw_tvti']))[0]
        recs += [(w, int(r), float(p[r])) for r in post]
    pred_df = pd.DataFrame(recs, columns=['well_id', 'md_row', 'tvt'])
    sub = sample_sub[['id']].copy()
    parts = sub['id'].str.rsplit('_', n=1, expand=True)
    sub['well_id'] = parts[0]; sub['md_row'] = parts[1].astype(int)
    sub = sub.merge(pred_df, on=['well_id', 'md_row'], how='left')
    miss = sub['tvt'].isna()
    if miss.any():
        sub.loc[miss, 'tvt'] = sub.loc[miss, 'well_id'].map(pd.Series(last_known))
    sub['tvt'] = sub['tvt'].fillna(sub['tvt'].mean())
    if sub['tvt'].abs().max() < 1.0:
        raise RuntimeError('predictions are all ~0: the merge failed')
    return sub[['id', 'tvt']]


def run(config_path: str, comp_root: str, out_path: str = 'submission.csv', root: str = '', log=print) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path))
    root = root or cfg.get('root', '')
    members = [resolve_member(n, s, root, log) for n, s in cfg['members'].items()]
    check_weights(members)
    wells, last_known = read_test_wells(comp_root)
    log(f'{len(wells)} test wells')
    preds = {m.name: predict_member(m, wells, log=log) for m in members}
    sub = make_submission(blend(preds, members), wells, last_known, pd.read_csv(os.path.join(comp_root, 'sample_submission.csv')))
    sub.to_csv(out_path, index=False)
    log(f'wrote {out_path}: {len(sub)} rows, tvt {sub.tvt.min():.2f}..{sub.tvt.max():.2f}')
    return sub


def check_reference(sub: pd.DataFrame, ref: dict, row_tol: float = 1e-3, stat_tol: float = 0.01, log=print) -> bool:
    """Compare a submission with the numbers the notebook of record printed for the same wells
    (``pipeline/03_submit/reference_rows.json``). Tolerances cover the printed rounding plus GPU
    arithmetic order. Returns True when everything is within them."""
    by_id = sub.set_index('id')['tvt']
    checks = []
    if len(sub) != ref['n_rows']:
        log(f'rows: {len(sub)} here, {ref["n_rows"]} in the reference (a different test set?)')
        return False
    for rid, v in ref['rows'].items():
        checks.append((f'row {rid}', float(by_id[rid]), v, row_tol))
    for k, v in ref['stats'].items():
        checks.append((f'{k} of all rows', float(getattr(sub['tvt'], k)()), v, stat_tol))
    wid = sub['id'].str.rsplit('_', n=1).str[0]
    for w, v in ref['well_mean'].items():
        checks.append((f'mean of well {w}', float(sub['tvt'][wid == w].mean()), v, stat_tol))
    worst = max(abs(a - b) for _, a, b, _ in checks)
    bad = [(n, a, b) for n, a, b, t in checks if abs(a - b) > t]
    for n, a, b in bad:
        log(f'  {n}: {a:.6f} here, {b:.6f} in the reference')
    log(f'{len(checks) - len(bad)}/{len(checks)} reference values match (largest difference {worst:.4f} ft)')
    return not bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--comp-root', required=True)
    ap.add_argument('--root', default='', help='folder the member dirs are relative to')
    ap.add_argument('--out', default='submission.csv')
    a = ap.parse_args(argv)
    run(a.config, a.comp_root, a.out, a.root)


if __name__ == '__main__':
    main()
