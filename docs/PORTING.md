# From notebooks to the package

The competition was run from three notebooks. After it ended, their code was moved into `src/rogii_wellbore/`,
with one YAML per member replacing the notebook's global settings. The notebooks are kept unchanged in `pipeline/`
(outputs stripped), and the tests load their functions straight from those files to check the port.

## Where the code went

| notebook (cell) | what | package |
|---|---|---|
| `2SUNET_MASTER` (2, 3, 24) | settings | `config.py`, `configs/members/*.yaml` |
| `2SUNET_MASTER` (5) | 5-fold split | `cv.make_folds` |
| `2SUNET_MASTER` (7, 9, 10, 11) | synthetic generator, per-fold pools, leakage audit, match_corr | `synth.py`, `data.build_hint_real_dist` |
| `2SUNET_MASTER` (13), Kaggle (6) | image rendering (two copies) | `image.py` (one copy) |
| `2SUNET_MASTER` (16) | anchor jitter, augmentation, datasets | `augment.py`, `image.render` |
| `2SUNET_MASTER` (18), Kaggle (5) | two-stage U-Net, refeed, unroll, loss | `model.py` |
| `2SUNET_MASTER` (18, 22) | EMA, schedule, training phases, evaluation | `train.py` |
| `2SUNET_MASTER` (24, 27, 29) | job loop with resume, seed ensemble, per-seed OOF | `run.py`, `oof.py` |
| `2SUNET_MASTER` (30-37) | offset/drift, tail, position, modes, summary, manifest | `metrics.py`, `run.summarize`, `run.manifest` |
| `ENSEMBLE_OOF_postproc` (12) | member correlation and blend | `eval.py` |
| Kaggle (3, 4, 7-10) | member registry and checks, batched loading, blend, submission | `infer.py`, `configs/submit/*.yaml` |

## What the tests check

All on CPU, without the competition data (`tests/conftest.py` writes a small fake data folder from synthetic wells).
"Equal" means `np.array_equal` / `torch.equal`, not a tolerance.

| test | port vs notebook |
|---|---|
| `test_parity_data` | fold split equal; synthetic wells equal for 150 seeds in three settings (the members' pool, event pack off, noisy corruption with the dormant pinch switched on); generator inputs equal; rendered images equal at 256 × 512, 224 × 448 and 64 × 128, at five anchor-jitter offsets |
| `test_parity_train` | fine-tuning samples equal (all augmentations, two geometries, two epochs); pretraining and validation samples equal; model outputs equal for forward, evaluation refeed to k = 4 and training unroll with its intermediate rounds, at three temperature settings; unroll draws and loss equal; **two full fine-tuning epochs** (EMA, unrolling, round supervision, k = 3 selection) give equal epoch scores, best state and final weights |
| `test_parity_infer` | Kaggle image equal at two geometries; refeed equal; per-well member prediction equal at k = 1 and k = 3 |
| `test_real_encoders` | the four members' real encoders (EfficientNetV2-L, V2-XL, L2), built without weights: the package model's parameter names and shapes equal both notebooks' models, so the saved checkpoints load |
| `test_pipeline` | member configs name their Drive/Kaggle run folders and match the recorded recipes; a full member run (pool, pretraining, fine-tuning, OOF, summary, manifest) on fake data; resume; `eval` reproduces the training score exactly with the data and within 0.001 without it; `infer` writes a complete submission and refuses a temperature that disagrees with the manifest; checkpoint replay (`verify`) reproduces a run's OOF file exactly and catches a swapped checkpoint; the submission reference check |

## Deliberate differences

1. **Unroll draws are reproducible.** The notebook seeded its per-step unroll generator with Python's `hash()` of the
   run label, which is salted per process, so which steps were unrolled (and to what depth) changed between sessions.
   The package uses `crc32` of the same label. The parity test gives the notebook the same seed to compare.
2. **One image function.** The Kaggle notebook's copy filled a constant cost volume with `cost - mean` where training
   used zeros. The two only differ for a well whose cost channel has no variance at all; the package uses the training
   version everywhere.
3. **Truth is read once.** The notebook re-read every validation CSV every epoch; the package caches it. Same numbers.
4. **Manifests carry the temperatures.** `run.py` writes `<name>_kaggle_inference_manifest.json` (format 2) with
   `handoff_T`, `refeed_T`, `checkpoint_k` and the decoder settings as fields. `infer.py` still reads the format-1
   manifests of the four final members, where the temperatures sit in free text.
5. **`torch.amp`** replaces the deprecated `torch.cuda.amp` calls. Same behaviour.

## Not ported

Switched off in every final member, so left in the notebook: vertical shear, ToeCrop / GR augmentation
during training, the fault-zone event bundle, posterior MD dilation, multi-scale posterior injection, warm restarts,
cycle-aware EMA, the integral-regression loss, learning-rate warmup, the single-U-Net path, the per-fold pretraining
donor (`REUSE_PT`), and the planted-scenario smoke cells. Knowledge distillation had already been removed from the
notebook. Each is recorded in the ledger with its result.

The neighbour-prior model (`pipeline/03_submit/stacked-unet-inference.ipynb`) is kept as its Kaggle notebook only.
Its training notebook is not in the repo, and neither is ported.

## Checks on the real data

The tests above use fake data and a small encoder. What they cannot show (the real files, encoders,
checkpoints and a GPU) is covered by four checks that replay saved results instead of retraining:
[pipeline/checks/](../pipeline/checks/). Retraining is not one of them: seed and GPU noise make a single
retrained seed uninformative, and enough seeds to compare would cost days of GPU.
