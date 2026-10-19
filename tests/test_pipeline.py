"""The port end to end on fake data (CPU, tiny encoder, one fold, one epoch per phase), and the
member configs against the run folders of record."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from rogii_wellbore import config as C
from rogii_wellbore.eval import evaluate_blend
from rogii_wellbore.infer import check_reference, run as infer_run
from rogii_wellbore.run import run as train_run
from rogii_wellbore.verify import replay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUN_FOLDERS = {   # Drive / Kaggle folder names of the four final members
    'v2xl': '5fold_effnetv2_xl_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt',
    'v2l': '5fold_effnetv2_l_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt',
    'v2l_small': '5fold_effnetv2_l_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt'
                 '_H224_W448_bin1.1428571428571428',
    'l2': '5fold_l2_stack2_ema_25000pt_mdwarp42_bedwarp42_handoffT15_expa_pt',
}


@pytest.mark.parametrize('name', sorted(RUN_FOLDERS))
def test_member_config_names_its_run_folder(name):
    cfg = C.load(os.path.join(REPO, 'configs/members', f'{name}.yaml'))
    assert cfg.name == name and cfg.experiment_name() == RUN_FOLDERS[name]


def test_member_configs_match_recorded_recipes():
    """Values from each member's ft39l_summary.json (the 'experiment' string and recipe block)."""
    m = {n: C.load(os.path.join(REPO, 'configs/members', f'{n}.yaml')) for n in RUN_FOLDERS}
    assert m['v2l_small'].aug.md_warp(m['v2l_small'].geometry) == (0.315, 112, 0.5)
    assert m['v2l_small'].aug.bedwarp(m['v2l_small'].geometry) == (0.315, 56, 0.5)
    assert m['l2'].aug.md_warp(m['l2'].geometry) == (0.42, 128, 0.5)
    assert [m[n].train.patience for n in ('v2l', 'v2xl', 'v2l_small', 'l2')] == [15, 12, 15, 12]
    assert [m[n].train.finetune_epochs for n in ('v2l', 'v2xl', 'v2l_small', 'l2')] == [35, 35, 35, 30]
    assert m['l2'].train.seeds == (34, 68, 102, 136, 170, 204, 272)
    assert all(m[n].model.handoff_t == 1.5 and m[n].synth.event_pack == 'v3_expA' for n in m)


def test_submit_config_is_the_final_submission():
    cfg = yaml.safe_load(open(os.path.join(REPO, 'configs/submit/final_equal_weight.yaml')))
    ms = cfg['members']
    assert sum(v['weight'] for v in ms.values()) == 1.0
    assert {n: v['eval_k'] for n, v in ms.items()} == {'l2_hT15_k1': 1, 'xl_hT15_k3': 3, 'l_hT15_k3': 3,
                                                        'l_hT15_k3_small': 3}
    assert all(v['dir'].split('/')[-1] in RUN_FOLDERS.values() for v in ms.values())


@pytest.fixture(scope='module')
def tiny_run(fake_data, tmp_path_factory):
    out = tmp_path_factory.mktemp('member')
    cfg = C.from_dict(dict(
        name='tiny', geometry=dict(H=64, W=128, bin_size=4.0, k_max=4),
        synth=dict(n_wells=24), model=dict(encoder='tu-efficientnet_b0'),
        train=dict(folds=[0], seeds=[34, 68], pretrain_epochs=1, finetune_epochs=1, batch_size=4,
                   num_workers=0, amp=None, patience=3)))
    summary = train_run(cfg, str(fake_data), str(out), device=torch.device('cpu'), encoder_weights=None,
                        log=lambda *a: None)
    return cfg, out, summary


def test_run_writes_the_member_folder(tiny_run):
    cfg, out, summary = tiny_run
    names = set(os.listdir(out))
    for f in ('ckpt_f0_s34.pt', 'ckpt_f0_s68.pt', 'result_f0_s34.npz', 'oof_f0_s34.npz', 'oof_f0_s68.npz',
              'pretrain_f0_s34_ep1.pt', 'summary.json', 'fold_splits.json', 'config.yaml',
              'tiny_kaggle_inference_manifest.json'):
        assert f in names, f
    man = json.load(open(out / 'tiny_kaggle_inference_manifest.json'))
    assert man['recipe']['handoff_T'] == 1.5 and man['n_checkpoints'] == 2
    assert np.isfinite(summary['full_cv']['ens_rmse'])


def test_run_resumes(tiny_run, fake_data):
    cfg, out, summary = tiny_run
    logs = []
    s2 = train_run(cfg, str(fake_data), str(out), device=torch.device('cpu'), encoder_weights=None, log=logs.append)
    assert any('resumed' in l for l in logs) and s2['full_cv'] == summary['full_cv']


def test_eval_reproduces_the_training_score(tiny_run, fake_data):
    cfg, out, summary = tiny_run
    exact = evaluate_blend({'tiny': str(out)}, {'tiny': 1.0}, data_root=str(fake_data), log=lambda *a: None)
    assert exact['members']['tiny']['full_cv'] == summary['full_cv']['ens_rmse']
    approx = evaluate_blend({'tiny': str(out)}, {'tiny': 1.0}, log=lambda *a: None)
    assert abs(approx['blend']['full_cv'] - summary['full_cv']['ens_rmse']) < 1e-3


def test_infer_writes_a_submission(tiny_run, fake_data, tmp_path):
    cfg, out, _ = tiny_run
    spec = {'members': {'a': {'dir': str(out), 'weight': 0.5, 'eval_k': 3, 'handoff_t': 1.5, 'refeed_t': 1.0},
                        'b': {'dir': str(out), 'weight': 0.5, 'eval_k': 1, 'handoff_t': 1.5, 'refeed_t': 1.0,
                              'tta': True}}}
    p = tmp_path / 'submit.yaml'; p.write_text(yaml.safe_dump(spec))
    sub = infer_run(str(p), str(fake_data), str(tmp_path / 'submission.csv'), log=lambda *a: None)
    sample = pd.read_csv(fake_data / 'sample_submission.csv')
    assert list(sub['id']) == list(sample['id']) and sub['tvt'].notna().all() and sub['tvt'].abs().min() > 1000
    bad = dict(spec); bad['members'] = {**spec['members'], 'a': {**spec['members']['a'], 'handoff_t': 1.0}}
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(AssertionError, match='handoff_T'):
        infer_run(str(p), str(fake_data), str(tmp_path / 's2.csv'), log=lambda *a: None)


def test_verify_replays_a_checkpoint(tiny_run, fake_data):
    """Checkpoint replay against the run's own OOF file: identical on CPU, and the split matches."""
    cfg, out, _ = tiny_run
    res = replay(cfg, str(fake_data), str(out), fold=0, seeds=[34, 68], device=torch.device('cpu'),
                 log=lambda *a: None)
    assert res['ok'] and res['fold_split_ok'] is True
    assert all(r['max_d_pred'] == 0.0 and r['d_rmse'] == 0.0 for r in res['seeds'].values())


def test_verify_catches_a_wrong_checkpoint(tiny_run, fake_data, tmp_path):
    cfg, out, _ = tiny_run
    import shutil
    for f in os.listdir(out):
        shutil.copy(out / f, tmp_path / f)
    shutil.copy(out / 'ckpt_f0_s68.pt', tmp_path / 'ckpt_f0_s34.pt')      # seed 68's weights under 34's name
    res = replay(cfg, str(fake_data), str(tmp_path), fold=0, seeds=[34], device=torch.device('cpu'),
                 log=lambda *a: None)
    assert not res['ok']


def test_check_reference():
    ref = json.load(open(os.path.join(REPO, 'pipeline/03_submit/reference_rows.json')))
    rows = {f'000d7d20_{i}': 11744.38 for i in range(10)}
    rows.update({f'00bbac68_{i}': 12220.94 for i in range(10)}); rows.update({f'00e12e8b_{i}': 11603.86 for i in range(10)})
    rows.update(ref['rows'])
    sub = pd.DataFrame({'id': list(rows), 'tvt': list(rows.values())})
    ref = dict(ref, n_rows=len(sub), stats={'min': float(sub.tvt.min()), 'mean': float(sub.tvt.mean()),
                                             'max': float(sub.tvt.max())},
               well_mean={w: float(sub.tvt[sub.id.str.startswith(w)].mean()) for w in ref['well_mean']})
    assert check_reference(sub, ref, log=lambda *a: None)
    sub.loc[sub.id == '00e12e8b_6383', 'tvt'] += 0.01
    assert not check_reference(sub, ref, log=lambda *a: None)


def test_smoke_config_is_v2l_made_small():
    s = C.load(os.path.join(REPO, 'configs/members/smoke.yaml'))
    v = C.load(os.path.join(REPO, 'configs/members/v2l.yaml'))
    assert s.model == v.model and s.geometry == v.geometry and s.aug == v.aug
    assert (s.synth.n_wells, list(s.train.folds), list(s.train.seeds), s.train.pretrain_epochs,
            s.train.finetune_epochs) == (200, [0], [34], 1, 2)
    assert s.experiment_name() != v.experiment_name()     # never writes into a member's run folder


def test_verify_fold_check_skips_folds_missing_from_the_record(tiny_run, fake_data, tmp_path):
    """The notebook's ft39l_fold_splits.json can hold only its last session's folds; the check
    compares the folds it holds, and the OOF files still pin the replayed fold."""
    import shutil
    from rogii_wellbore.cv import load_folds, save_folds
    cfg, out, _ = tiny_run
    for f in os.listdir(out):
        if f != 'fold_splits.json':
            shutil.copy(out / f, tmp_path / f)
    folds = load_folds(str(out / 'fold_splits.json'))
    partial = {f: (v if f == 4 else {'val': [], 'train': []}) for f, v in folds.items()}
    save_folds(partial, str(tmp_path / 'ft39l_fold_splits.json'))
    res = replay(cfg, str(fake_data), str(tmp_path), fold=0, seeds=[34], device=torch.device('cpu'),
                 log=lambda *a: None)
    assert res['ok'] and res['fold_split_ok'] and res['oof_folds'] == [0]
    wrong = {f: {'val': folds[(f + 1) % 5]['val'], 'train': []} for f in folds}
    save_folds(wrong, str(tmp_path / 'ft39l_fold_splits.json'))
    assert not replay(cfg, str(fake_data), str(tmp_path), fold=0, seeds=[34], device=torch.device('cpu'),
                      log=lambda *a: None)['ok']
