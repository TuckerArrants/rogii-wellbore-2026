# ROGII 2026 Wellbore Geology Prediction

Notebooks and records for the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
competition: predict where a horizontal well sits in the rock layers (its TVT, true vertical thickness) along the part
of the well that has no interpretation yet, from its gamma-ray log, a reference log and its trajectory. Scored by RMSE
in feet. The problem and the data: [docs/TASK.md](docs/TASK.md). Words used throughout:
[docs/GLOSSARY.md](docs/GLOSSARY.md).

**Result: 26th place, private leaderboard RMSE 6.600 ft.**

The work was done in Colab and Kaggle notebooks between May and August 2026. This repo documents it; it is not set up
to be rerun. The notebooks are kept as run, with their outputs stripped; the numbers worth keeping are in the stage
READMEs and in `docs/`.

## How it works

1. **The well becomes an image.** Each well is drawn as a 256 × 512 picture: columns are steps along the well, rows
   are 1 ft steps in depth around the last known position. Five channels: the reference log down the rows, the well's
   own log across the columns, the known positions before the interpretation stops, a matching cost (how badly the
   two logs disagree at every row and column), and the well's dip. Predicting TVT becomes picking one row per column.
2. **Synthetic pretraining.** Only 773 real wells are labelled. Each model first trains on 25,000 synthetic wells
   built from the real reference logs, with drift, faults, dip breaks and repeated sections added more often than
   in real wells. Then it is fine-tuned on the real wells.
3. **A two-stage U-Net.** An EfficientNet U-Net gives a probability for every row in every column. A second U-Net
   reads the image plus that probability map and refines it. Three of the four members are also trained to refine
   their own output again, and are run for three rounds at prediction time. The answer is the probability-weighted
   row.
4. **Ensemble.** Four model families (EfficientNetV2-L at two image sizes, EfficientNetV2-XL, EfficientNet-L2),
   each 5 folds × 7 seeds, averaged with equal weights. No test-time augmentation.

The hard part, measured many ways: about half of the remaining error is one constant offset per well that the inputs
do not determine. What was tried against it is in [docs/WHAT_DIDNT_WORK.md](docs/WHAT_DIDNT_WORK.md).

## Results

| | CV (773 wells, 5 folds) | public LB | private LB |
|---|---|---|---|
| final 4-member ensemble, equal weights, no TTA | members 5.14–5.26 (below) | <!-- owner: public score --> | **6.600** (26th) |
| best earlier single line on the public board | 5.592 | 5.444 | — |

Every recorded score, and how CV and the board came apart: [docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## Pipeline

| # | stage | notebook | runs on | produces |
|---|---|---|---|---|
| 01 | [training](pipeline/01_train/) | `2SUNET_MASTER.ipynb` | Colab (GPU) | folds, synthetic pool, checkpoints and per-seed OOF predictions for one member |
| 02 | [ensemble analysis](pipeline/02_ensemble/) | `ENSEMBLE_OOF_postproc.ipynb` | Colab | member OOF correlations, blend weights, error breakdown, TTA check |
| 03 | [submission](pipeline/03_submit/) | `final-submission-equal-weight-no-tta.ipynb` | Kaggle (T4 × 2, offline) | `submission.csv` from the four members |

## Layout

```
pipeline/NN_*/   one folder per stage: the notebook as run (outputs stripped) and a README
                 with inputs, outputs and the numbers from the run of record
docs/            the task, a glossary, validation rules, what didn't work, the leaderboard record
scripts/         strip_notebooks.py: removes outputs before a notebook is committed
ARTIFACTS.md     everything outside git: Kaggle datasets and models, Drive folders with OOFs
```

**Not in git:** competition data, checkpoints, OOF predictions, fold lists. They are on Kaggle and Google Drive and
listed in [ARTIFACTS.md](ARTIFACTS.md).

## Credits

- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and
  [timm](https://github.com/huggingface/pytorch-image-models) for the U-Net and the EfficientNet encoders.
- ROGII and Kaggle for the competition and data.
