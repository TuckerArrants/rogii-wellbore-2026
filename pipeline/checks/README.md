# Checks · does it run on the real data?

The tests in `tests/` run on CPU with fake data and a small encoder. They show that every function gives the
notebooks' numbers. They cannot show that the package runs on the real files, with the real encoders and
checkpoints, on a GPU. These checks do, without retraining anything.

## Why not retrain and compare?

Two runs of the same recipe differed by more than most of the changes we tested (seed and GPU noise), so one
retrained seed proves nothing and five cost days of GPU, most of it synthetic pretraining. Inference has no
such noise. So the checks replay the saved checkpoints and compare with what the notebooks saved from the same
weights. Training is checked for running, not for its score.

## What covers what

| | where | GPU time | what it shows |
|---|---|---|---|
| every function equals the notebooks' | `tests/` (CPU, fake data) | none | the port is faithful, bit for bit, including two full fine-tuning epochs |
| the four real encoders | `tests/test_real_encoders.py` | none | the package model has exactly the parameter names and shapes of the saved checkpoints |
| **1. score the ensemble from its OOF files** | [`checks_colab.ipynb`](checks_colab.ipynb) | none | the real OOF files load, and the members and blend score 5.136 / 5.157 / 5.252 / 5.260 / **5.038** |
| **2. replay a checkpoint** | [`checks_colab.ipynb`](checks_colab.ipynb) | 1–2 min per seed | real data, fold split, rendering, a real checkpoint, refeed and decode give the notebook's OOF numbers |
| **3. tiny training run** | [`checks_colab.ipynb`](checks_colab.ipynb) | 10–15 min | the whole training path runs on the real data on a GPU (`configs/members/smoke.yaml`) |
| **4. the submission** | [`../03_submit/submit_kaggle.ipynb`](../03_submit/submit_kaggle.ipynb) | roughly 15–30 min on Kaggle | the Kaggle path (manifests, batched loading over 2 GPUs, blend, submission) gives the saved submission's numbers on the 3 visible test wells (`reference_rows.json`): to ~0.001 ft with all four members, or within 0.5 ft with the three V2 members only (`configs/submit/check_v2_only.yaml`) |

Checks 1, 2 and 4 compare with numbers of record and print PASS or FAIL. Check 3 only has to finish.

Not covered, on purpose: reproducing a member's CV by retraining it.

From the command line, check 2 is:

```bash
python -m rogii_wellbore.verify --config configs/members/v2l.yaml --data-root <data> \
    --run-dir <Drive run folder> [--ckpt-dir <folder with ckpt_f*_s*.pt>] [--fold 0] [--seeds 34]
```

On CPU it takes about 3 s per well for EfficientNetV2-L (153 wells in a fold), so it does not need a GPU.

## Results (18 October 2026; checks 1–3 on a Colab GPU, check 4 on Kaggle 2× T4)

**1. OOF scores: PASS.** Every number of record, to the printed precision:

| | full CV | per fold |
|---|---|---|
| L2 | 5.2601 | 5.657 5.115 5.428 5.263 4.806 |
| V2-XL | 5.1362 | 5.291 4.986 5.109 5.413 4.874 |
| V2-L | 5.1569 | 5.324 5.008 5.197 5.301 4.948 |
| V2-L small | 5.2525 | 5.631 5.068 5.237 5.548 4.745 |
| **blend** | **5.0383** | 5.319 4.842 5.079 5.250 4.683 |

Blend offset/drift 3.54 / 3.58 ft; member error correlations 0.89–0.96.

**2. Checkpoint replay: PASS.** V2-L, fold 0, seed 34, k = 3, 153 wells: RMSE 5.7179 in the notebook's OOF file,
5.7180 replayed; largest single-row difference 0.032 ft (p99 0.008 ft). The first run printed FAIL on the fold
split: the run folder's `ft39l_fold_splits.json` holds fold 4 only (it matched), and the check counted the four
missing folds as differences. The check now compares against the OOF files, which name every fold's wells, and
against the recorded file only for the folds it holds.

**3. Tiny training run: PASS** (it only had to finish). Fold 0: 620 train / 153 validation wells (constant
15–16 ft), 773 wells rendered in 15 s, a 200-well pool, ImageNet weights downloaded, 1 pretraining and 2
fine-tuning epochs with unrolling and EMA, then the OOF file, summary and manifest. Best 15.238 at epoch 1
(constant 16.124), as expected for a run this short.

**4. Submission on Kaggle: PASS** (three V2 members; the L2 model is no longer on Kaggle). Internet on, code
cloned from GitHub, `segmentation_models_pytorch` and `timm` from PyPI. All 105 V2 checkpoints resolved from
their manifests and ran over both GPUs; 14,151 rows written, every one from the models, range 11,592.77–12,239.13
ft (saved submission: 11,592.99–12,239.20). All 16 reference values (10 rows, min/mean/max, 3 per-well means)
within the 0.49 ft bound; the largest difference was 0.219 ft. The format-1 manifests do not record `refeed_T`,
so the config's 1.0 is used, as in the notebook of record.

All four checks pass. The package runs on the real data and reproduces the notebooks' results without retraining.
