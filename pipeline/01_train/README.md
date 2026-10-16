# 01 · Training

`2SUNET_MASTER.ipynb` trains one ensemble member end to end: data, folds, synthetic pool, pretraining, fine-tuning,
out-of-fold evaluation, and the files the Kaggle notebook needs. The four final members were all trained from this
notebook, changing only the settings listed below. **As run, outputs stripped.**

**Runs on:** Colab with a GPU, Google Drive mounted. Reads the competition zip from
`MyDrive/Kaggle/ROGII-Wellbore/`, caches Hugging Face encoder weights there (`hf_home/`), and writes each run to a
folder named after its settings (for example `5fold_effnetv2_l_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt`).

**State of the saved copy:** the notebook was last saved after a probe (EfficientNetV2-L at 288 × 576, fold 0 only,
CV 5.492, null). Its settings cell shows that probe, not a final member. Each member's exact settings are in the
`ft39l_summary.json` in its run folder ([ARTIFACTS.md](../../ARTIFACTS.md)).

## What the notebook does, section by section

| section | cells | what happens |
|---|---|---|
| setup | 0–3 | mount Drive; fetch encoder weights once into a Drive cache; unzip data; the settings cell; Hugging Face token from the Colab secret `HF_TOKEN` |
| 1 · data and folds | 5 | list the 773 wells; 5 folds stratified by each well's TVT range |
| 2 · synthetic pool | 7–11 | the synthetic well generator (drift, noise, faults, unconformities, dip breaks, repeated sections); planted-scenario checks run before any pool is built; one pool per fold from that fold's training typewells only; a hard leakage assert on every pool; a check that the pool's GR-matching difficulty landed on target |
| 3 · images | 13–14 | turn every real and synthetic well into a 5-channel 256 × 512 image (see [GLOSSARY.md](../../docs/GLOSSARY.md)) |
| 4 · dataset | 16 | training frames: anchor jitter, MDwarp, BedWarp, HFlip, HintDrop, each on its own random stream |
| 5 · model | 18–20 | the two-stage U-Net (`smp.Unet` with a `timm` encoder, decoder channels 512→32); stage 2 reads the image plus stage 1's tempered probability map |
| 6 · train and eval | 22–24 | the decode (probability-weighted row → TVT); pretrain 5 epochs on synthetic wells, fine-tune 30–35 epochs on real wells with EMA weights, unrolled refeed rounds and early stopping |
| §C · results | 26–34 | the seed ensemble per fold and full-CV; per-seed OOF files; offset/drift split; tail counts; error by position along the well; per-fold heel/toe; residual modes |
| §D · artifacts | 36–37 | `ft39l_summary.json` (settings and all metrics), `ft39l_kaggle_inference_manifest.json` (what the Kaggle notebook checks), fold splits, checkpoints; mirrored to Drive |

## The four final members

All: 5 folds × 7 seeds (35 checkpoints), 5 input channels, synthetic pretraining on 25,000-well pools with event
pack A, two-stage U-Net with deep supervision 0.5, handoff temperature 1.5, learning rate 1e-3, weight decay 1e-3,
batch 8, EMA 0.997, HFlip 0.5, HintDrop 0.3.

| member | encoder | image | MDwarp / BedWarp strength | unroll | fine-tune epochs | full-CV |
|---|---|---|---|---|---|---|
| V2-XL | `tu-tf_efficientnetv2_xl.in21k_ft_in1k` | 256 × 512, 1 ft rows | 0.36 / 0.36 | p 0.5, k 2–4 | 35 | 5.136 |
| V2-L | `tu-tf_efficientnetv2_l.in21k_ft_in1k` | 256 × 512, 1 ft rows | 0.36 / 0.36 | p 0.5, k 2–4 | 35 | 5.157 |
| V2-L small | `tu-tf_efficientnetv2_l.in21k_ft_in1k` | 224 × 448, 1.14 ft rows | 0.315 / 0.315 | p 0.5, k 2–4 | 35 | 5.252 |
| L2 | `tu-tf_efficientnet_l2.ns_jft_in1k_475` | 256 × 512, 1 ft rows | 0.42 / 0.42 | none | 30 | 5.260 |

Full-CV is over 773 wells and 3,783,989 rows; the constant baseline on the same rows is 15.910. Offset and drift
are each about 3.6–3.7 ft on every member; the worst 10% of wells carry 58–60% of the squared error.

**A recorded discrepancy:** the four summaries list `synth_pretrain_pool: 15`, a debug value of `N_SYNTH_PRETRAIN`
that was in the notebook when they were written. The run folders are named for the 25,000-well pool, and the saved
notebook now sets `N_SYNTH_PRETRAIN = 25000`. The field is kept as written.
