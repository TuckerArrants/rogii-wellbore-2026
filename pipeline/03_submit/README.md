# 03 · Kaggle submission

`final-submission-equal-weight-no-tta.ipynb` is the submission that scored **6.600 on the private leaderboard (26th
place)**. It runs offline on Kaggle (T4 × 2), reads four members' checkpoints, and writes `submission.csv`.
**As run, outputs stripped.**

## Inputs

| input | what |
|---|---|
| competition data | `rogii-wellbore-geology-prediction` |
| `tuckerarrants/smp-offline` (dataset) | wheels for `segmentation_models_pytorch` 0.5.0 and `timm` 1.0.27, installed with internet off |
| `tuckerarrants/rogii-v2s` (dataset) | the V2-XL, V2-L and V2-L small members: 35 checkpoints each plus their inference manifests |
| `tuckerarrants/rogii-l2` (model, `pytorch/default/1`) | the L2 member: 35 checkpoints plus its manifest |

## How it works

1. **The member registry (`MODEL_LIST`).** Each member names its folder, blend weight, refeed depth `eval_k`, and
   temperatures. Everything else (encoder, stack depth, image size) is read from the member's own manifest. Any
   setting given in the registry that disagrees with the manifest stops the notebook.
2. **Guards.** The weights must sum to exactly 1.0. Handoff temperatures are checked against each manifest. The refeed
   temperature is not recorded in these manifests, so the notebook prints a warning and uses 1.0, the training value.
3. **Per-member geometry.** Each member renders the test wells at its own image size (the small V2-L at 224 × 448);
   predictions are compared only after they are decoded back to one TVT per row, so members of different sizes
   average directly.
4. **Batched loading.** Checkpoints are loaded 4–8 at a time (the L2's are 3.6 GB each) and spread over both GPUs;
   the result is the same as loading all at once.
5. **Blend.** Each member's 35 checkpoints are averaged; the four member predictions are then averaged with equal
   weights:

| member | refeed | weight |
|---|---|---|
| `l2_hT15_k1` | k = 1 | 0.25 |
| `xl_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3_small` | k = 3 | 0.25 |

`DO_PHASE_JITTER_TTA = False`: one pass per checkpoint, no test-time augmentation.

6. **Output.** `submission.csv` in `sample_submission.csv` order (`id`, `tvt`). A last cell plots each well's
   prediction from the blend and from each member, as a visual check.

## The saved run

The run stored with the notebook is the interactive one, on the 3 wells (14,151 rows) Kaggle shows outside scoring:
all 140 checkpoints resolved, every row filled by the models (none by the fallback), predictions between 11,593 and
12,239 ft.
