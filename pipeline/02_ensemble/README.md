# 02 · Scoring members and the blend

**Code:** `rogii_wellbore.eval`. **Config:** `configs/eval/final_blend.yaml`. **Driver:** `eval_colab.ipynb`.
**Notebook of record:** `ENSEMBLE_OOF_postproc.ipynb`, as run, outputs stripped.

```bash
python -m rogii_wellbore.eval --config configs/eval/final_blend.yaml [--root <runs dir>] [--data-root <data>] [--out blend.json]
```

Reads each member's `oof_f*_s*.npz` (no GPU, no checkpoints) and prints:

- each member's seed-ensemble CV, per fold and pooled;
- the blend: the weighted per-row average of the members' predictions, scored like a member, with offset/drift, tail
  and heel-to-toe read-outs per fold;
- the error correlation between each pair of members, over all rows and over the worst 10% of wells.

With `--data-root`, truth comes from the competition files and member scores are exactly the training-time numbers
(the test suite checks this). Without it, truth is rebuilt from the OOF files and scores can move in the 4th decimal.

**Expected on the final members:** 5.136 (V2-XL), 5.157 (V2-L), 5.252 (V2-L small), 5.260 (L2); blend **5.038**. Not
yet run from the package; see [docs/PORTING.md](../../docs/PORTING.md).

## The notebook of record

`ENSEMBLE_OOF_postproc.ipynb` did more than the blend: a check of whether a well's error depends on its distance to
the fold's training wells, and the cost of fp16 inference. Only the blend and the correlations were ported.
