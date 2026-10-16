# 03 · Kaggle submission

**Code:** `rogii_wellbore.infer`. **Config:** `configs/submit/final_equal_weight.yaml`. **Driver:**
`submit_kaggle.ipynb`. **Notebook of record:** `final-submission-equal-weight-no-tta.ipynb`, the submission that scored
**6.600 on the private leaderboard (26th place)**, as run, outputs stripped.

```bash
python -m rogii_wellbore.infer --config configs/submit/final_equal_weight.yaml --comp-root <competition data> --out submission.csv
```

## On Kaggle (internet off)

Attach: the competition data; `tuckerarrants/smp-offline` (wheels for `segmentation_models_pytorch` 0.5.0 and `timm`
1.0.27); `tuckerarrants/rogii-v2s` and the `tuckerarrants/rogii-l2` model (checkpoints); and this repo's code as a
dataset, built with `python scripts/make_kaggle_src.py` and uploaded with the Kaggle CLI. The driver finds the package
and the config anywhere under `/kaggle/input`.

## What it does

1. **Members.** For each entry in the config: find the folder's `*_kaggle_inference_manifest.json`; take the encoder,
   decoder and image geometry from it; take the weight, refeed depth `eval_k` and temperatures from the config, and stop
   if the manifest records a different temperature (as a field, or in the free text of the older manifests).
2. **Weights** must each be in (0, 1] and sum to exactly 1.0.
3. **Prediction.** Each member renders the test wells at its own geometry. Checkpoints are loaded a few at a time (as
   many as fit a T4 after an activation reserve; the L2's are 3.6 GB each) and spread over the GPUs. Each predicts
   every well at depth `eval_k`; a member's prediction is the mean over its checkpoints, per native MD row. With
   `tta: true`, also the mean over 8 PhaseJitter shifts (drop part of a bin of toe rows, re-bin).
4. **Blend** of the members with their weights; post-PS rows go into `sample_submission.csv` order. Rows without a
   prediction fall back to the well's last known TVT.

| member | refeed | weight |
|---|---|---|
| `l2_hT15_k1` | k = 1 | 0.25 |
| `xl_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3_small` | k = 3 | 0.25 |

No TTA in the final submission.

## The saved run

The run stored with the notebook of record is the interactive one, on the 3 wells (14,151 rows) Kaggle shows outside
scoring: all 140 checkpoints resolved, every row filled by the models (none by the fallback), predictions between
11,593 and 12,239 ft. The package's per-well prediction equals the notebook's, with and without TTA
(`tests/test_parity_infer.py`).
