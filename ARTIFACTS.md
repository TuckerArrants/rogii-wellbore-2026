# External artifacts

Everything the notebooks read or wrote that is not in git.

## Kaggle

| artifact | link | contents | used by |
|---|---|---|---|
| competition data | [rogii-wellbore-geology-prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data) | `train/` (773 wells, two CSVs each), `test/`, `sample_submission.csv` | all stages; under the competition's rules, not redistributed |
| offline wheels | [tuckerarrants/smp-offline](https://www.kaggle.com/datasets/tuckerarrants/smp-offline) | `segmentation_models_pytorch` 0.5.0, `timm` 1.0.27 | 03 |
| V2 members | [tuckerarrants/rogii-v2s](https://www.kaggle.com/datasets/tuckerarrants/rogii-v2s) | V2-XL, V2-L and V2-L small: 35 checkpoints each + `ft39l_kaggle_inference_manifest.json` | 03 |
| L2 member | [tuckerarrants/rogii-l2](https://www.kaggle.com/models/tuckerarrants/rogii-l2) (PyTorch, `default`, version 1) | L2: 35 checkpoints + manifest | 03 |
| repo code | `tuckerarrants/rogii-wellbore-src` (not yet uploaded) | `src/rogii_wellbore` + `configs/`, built by `scripts/make_kaggle_src.py` | 03 driver |

## Google Drive (`MyDrive/Kaggle/ROGII-Wellbore/`)

One folder per final member, written by the training notebook (the package writes the same layout, with
`summary.json` in place of `ft39l_summary.json`). Each holds all 35 per-seed OOF files (`oof_f{fold}_s{seed}.npz`,
5 folds × 7 seeds; format in `src/rogii_wellbore/oof.py`), per-seed metrics (`result_f{fold}_s{seed}.npz`),
`ft39l_summary.json` (settings and all CV metrics), `ft39l_kaggle_inference_manifest.json`, and, depending on the
folder, the fold splits and some or all of the checkpoints and pretraining checkpoints. `configs/eval/final_blend.yaml`
points at these folders.

| member | folder |
|---|---|
| V2-XL | [`5fold_effnetv2_xl_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt`](https://drive.google.com/drive/folders/1_EgeRD9085RUNv2bW2TWoeZUhLyaycb3) |
| V2-L | [`5fold_effnetv2_l_stack2_ema_25000pt_mdwarp36_bedwarp36_unroll50_deepsup_handoffT15_expa_pt`](https://drive.google.com/drive/folders/1ILEUVOP49cN4lzIMRNsSy3qoMyKS2In9) |
| V2-L small | [`…_handoffT15_expa_pt_H224_W448_bin1.1428571428571428`](https://drive.google.com/drive/folders/1_cpsV15GKfBHTEI4L2PeuH7g96Aowmw2) |
| L2 | [`5fold_l2_stack2_ema_25000pt_mdwarp42_bedwarp42_handoffT15_expa_pt`](https://drive.google.com/drive/folders/12rnzFgTiK1o2Q9HDvTkMb916gogBFFPa) |

The XL folder also holds `board_noise_summary.json`: the public/private board-noise simulation from that member's
errors (40,000 draws of 50 and 200 wells).

The same Drive folder holds the competition zip the training notebook reads
(`rogii-wellbore-geology-prediction.zip`) and the Hugging Face weight cache (`hf_home/`).
