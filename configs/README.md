# configs

One YAML per run. A file may start with `base: <relative path>`; its keys are merged over the base's, section by
section. Sections and every key: `src/rogii_wellbore/config.py`.

| folder | read by | files |
|---|---|---|
| `members/` | `python -m rogii_wellbore.run` | `base.yaml` (shared), `v2xl`, `v2l`, `v2l_small` (inherits `v2l`), `l2` |
| `eval/` | `python -m rogii_wellbore.eval` | `final_blend.yaml`: the four members' Drive run folders, 0.25 each |
| `submit/` | `python -m rogii_wellbore.infer` | `final_equal_weight.yaml`: the 26th-place submission |

Where the member values come from: the training notebook's settings of record (`pipeline/01_train/2SUNET_MASTER.ipynb`,
cells 2, 3 and 24) and each member's `ft39l_summary.json` (encoder, augmentation strengths, epochs, patience, seeds,
image geometry). `tests/test_pipeline.py` checks that each member config derives its run folder's exact name and the
recorded recipe values. One value is inferred rather than recorded: `checkpoint_k: 1` for `l2`.

A member's run folder name is derived from its config (`RunConfig.experiment_name`), the same way the notebook named
its folders, so the existing Drive and Kaggle folders line up with the configs.
