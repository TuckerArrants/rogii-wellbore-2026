# 02 · Ensemble analysis

`ENSEMBLE_OOF_postproc.ipynb` loads the final members' checkpoints, produces their out-of-fold predictions on the
real wells, and studies how they combine. It reuses the training notebook's data, image and model code unchanged, so
its OOF predictions match the ones each member's training run saved. **As run, outputs stripped.**

**Runs on:** Colab with a GPU. **Cache first:** for each member it loads the per-seed OOF files from that member's
Drive run folder when they exist and only runs the checkpoints for folds or seeds that are missing.

## Sections

| # | section | what it does |
|---|---|---|
| 1–4 | setup and folds | Drive, data, imports, the same 5 folds as training |
| 5–6 | real-image cache | renders the 773 real wells once (5 channels, including dZ) |
| 7–9 | model and scorer | the two-stage U-Net and decode, and `_eval_model_pool`: the seed-averaged scorer used for every CV number |
| 10 | member registry | V2-L, V2-L small (224 × 448), V2-XL and L2, each with its Drive folder, encoder, refeed depth and temperature |
| 11 | OOF driver | OOF predictions for every member at its own settings |
| 12 | correlation and blend | member-to-member error correlation (all rows and worst decile), the best linear blend, and the blend's offset/drift split and error by position along the well |
| 13 | CV-inflation check | does a well's OOF error depend on its distance (or typewell similarity) to the training wells of its fold? |
| 16–17 | PhaseJitter TTA and fp16 | each of 8 grid shifts scored alone, every combination of them scored with fold rotation, and the cost of running in half precision |

## Recorded result: PhaseJitter TTA on the V2-L member

One pass, scored on OOF. Each shift alone is 0.05–0.13 ft worse than no shift; their errors are highly correlated
(0.96–0.998).

| combination | CV | cost |
|---|---|---|
| no shift (single pass) | 5.2606 | — |
| all 8 shifts averaged | 5.3011 | +0.041 |
| best subset, fold-rotated | 5.2859 | +0.025 |
| free weights, fold-rotated | 5.2817 | +0.021 |

No cheaper subset passed the bar set in the notebook (at least 4 shifts and at least 0.02 ft saved). That left one
choice: no TTA (the CV bet) or all 8 shifts (the test-set bet, which had gained about 0.26 ft on the public board on
earlier lines). The final submission went without TTA.

The member OOF tables and the blend weights from section 12 were not saved with the notebook.
