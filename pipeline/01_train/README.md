# 01 · Training a member

**Code:** `rogii_wellbore.run` (with `cv`, `synth`, `image`, `augment`, `model`, `train`, `oof`, `metrics`).
**Configs:** `configs/members/{v2xl,v2l,v2l_small,l2}.yaml`. **Driver:** `train_colab.ipynb`.
**Notebook of record:** `2SUNET_MASTER.ipynb`, as run, outputs stripped. Every function the package uses was
checked against it ([docs/PORTING.md](../../docs/PORTING.md)).

```bash
python -m rogii_wellbore.run --config configs/members/v2l.yaml --data-root <unzipped data> --out <runs dir> \
    [--pretrained-dir <folder with pretrain_f*_s*_ep5.pt>] [--folds 0 1] [--seeds 34 68]
```

## What a run does

1. **Folds.** `cv.make_folds(seed=34)`: 5 folds stratified by whole-well TVT range (153/153/153/153/161 validation
   wells). Written to `fold_splits.json` with each fold's constant baseline.
2. **Images.** Every real well is rendered once at the config's geometry (`image.build_image`).
3. **Per fold, per seed (one job):**
   - **Pretraining**, unless `--pretrained-dir` holds `pretrain_f{fold}_s{seed}_ep5.pt`. The fold's pool (25,000
     synthetic wells from its training typewells; `synth.build_pool`) is built once per fold, audited for leakage,
     and rendered; 5 epochs, AdamW 1e-3 / wd 1e-3, cosine, no EMA, HFlip off. The last two epochs are saved.
   - **Fine-tuning** on the fold's real training wells: fresh AdamW and cosine, EMA 0.997, batch 8, gradient clip 5,
     bf16 autocast; anchor jitter ±12 rows, HFlip 0.5, HintDrop 0.3, MDwarp and BedWarp at p 0.5. Unrolled members
     run stage 2 for k ∈ {2, 3, 4} rounds on half the steps, with a 0.5-weighted loss on the intermediate rounds.
     After each epoch the EMA weights are scored on the validation wells at k = 1 and at the deployment depths; the
     best epoch at `checkpoint_k` is kept, and training stops after `patience` epochs without improvement.
   - Writes `ckpt_f{fold}_s{seed}.pt` and `result_f{fold}_s{seed}.npz`. A re-run skips finished jobs.
4. **OOF and summary.** Each checkpoint predicts its fold's validation wells at `checkpoint_k`
   (`oof_f{fold}_s{seed}.npz`); the seeds are averaged per fold; `summary.json` holds per-fold and full-CV scores with
   the offset/drift, tail and position diagnostics; `<name>_kaggle_inference_manifest.json` is what stage 03 reads.

## The four final members

All: 5 folds × 7 seeds (35 checkpoints), pretrained on 25,000-well pools with event pack A, handoff temperature 1.5,
deep supervision 0.5.

| config | encoder | image | MDwarp / BedWarp | unroll, depth | fine-tune epochs / patience | full-CV of record |
|---|---|---|---|---|---|---|
| `v2xl` | `tu-tf_efficientnetv2_xl.in21k_ft_in1k` | 256 × 512, 1 ft rows | 0.36 / 0.36 | yes, k = 3 | 35 / 12 | 5.136 |
| `v2l` | `tu-tf_efficientnetv2_l.in21k_ft_in1k` | 256 × 512, 1 ft rows | 0.36 / 0.36 | yes, k = 3 | 35 / 15 | 5.157 |
| `v2l_small` | `tu-tf_efficientnetv2_l.in21k_ft_in1k` | 224 × 448, 1.14 ft rows | 0.315 / 0.315 | yes, k = 3 | 35 / 15 | 5.252 |
| `l2` | `tu-tf_efficientnet_l2.ns_jft_in1k_475` | 256 × 512, 1 ft rows | 0.42 / 0.42 | no, k = 1 | 30 / 12 | 5.260 |

Full-CV is over 773 wells and 3,783,989 rows; the constant baseline on the same rows is 15.910. Offset and drift are
each about 3.6–3.7 ft on every member; the worst 10% of wells carry 58–60% of the squared error.

**Recorded quirks.** The four `ft39l_summary.json` files list `synth_pretrain_pool: 15`, a debug value of
`N_SYNTH_PRETRAIN` that was in the notebook when they were written; the run folders are named for the 25,000-well
pool, the saved notebook sets 25,000, and so does the config. The L2 run used seed 272 in place of 238. Its
`checkpoint_k: 1` is inferred from how it was deployed; the summary does not record it.

## The notebook of record

The saved `2SUNET_MASTER.ipynb` was last run as a probe (EfficientNetV2-L at 288 × 576, fold 0 only, CV 5.492,
null), so its settings cell shows that probe. It still holds the switched-off experiments the package dropped
(listed in [docs/PORTING.md](../../docs/PORTING.md)).
