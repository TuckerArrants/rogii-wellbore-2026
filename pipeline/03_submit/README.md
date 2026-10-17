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
   every well at depth `eval_k`; a member's prediction is the mean over its checkpoints, per native MD row.
4. **Blend** of the members with their weights; post-PS rows go into `sample_submission.csv` order. Rows without a
   prediction fall back to the well's last known TVT.

| member | refeed | weight |
|---|---|---|
| `l2_hT15_k1` | k = 1 | 0.25 |
| `xl_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3` | k = 3 | 0.25 |
| `l_hT15_k3_small` | k = 3 | 0.25 |


## The saved run

The run stored with the notebook of record is the interactive one, on the 3 wells (14,151 rows) Kaggle shows outside
scoring: all 140 checkpoints resolved, every row filled by the models (none by the fallback), predictions between
11,593 and 12,239 ft. The package's per-well prediction equals the notebook's
(`tests/test_parity_infer.py`).

## The submission we did not pick

`stacked-unet-inference.ipynb` (outputs stripped) is the Kaggle notebook of the best single model: EfficientNetV2-XL,
5 folds × 5 seeds, CV **5.027**, with the neighbour prior. It would have placed 7th on the private board; it was not
selected. Why, and what we learned: [the top-level README](../../README.md#the-best-model-we-did-not-submit).

How it differs from the final submission:

- **One more input channel.** For each test well, the prior is built from all 773 training wells: the layer's tilt
  near each piece of the well (a weighted fit over the closest pieces of neighbouring wells, corrected for the
  direction the well runs), added up from the well's last known TVT on top of the well's own change in Z. It is drawn
  as a soft band around one row per column. Stage 1 reads the usual 5 channels; stage 2 reads 7: those 5, the
  prior, and stage 1's guess.
- **The prior enters at round 2.** Stage 2's first pass sees the prior channel blank; rounds 2 and 3 (k = 3) see it.
- **A learned gate** at the output decides how much of the prior to use.

It is not ported to the package: it runs as its own notebook, with checkpoints from the
`tuckerarrants/rogii-unet` dataset.
