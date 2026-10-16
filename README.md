# ROGII 2026 Wellbore Geology Prediction

Code for the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
competition: predict where a horizontal well sits in the rock layers (its TVT, true vertical thickness) along the part
of the well that has no interpretation yet, from its gamma-ray log, a reference log and its trajectory. Scored by RMSE
in feet. The problem and the data: [docs/TASK.md](docs/TASK.md). Words used throughout:
[docs/GLOSSARY.md](docs/GLOSSARY.md).

**Result: 26th place, private leaderboard RMSE 6.600; cross-validation 5.038 for the submitted ensemble.**

The work was done in Colab and Kaggle notebooks between May and August 2026 and moved into this repo afterwards as the
`rogii_wellbore` package. The notebooks of record are kept, as run, in `pipeline/`; the tests check the package against
them function by function ([docs/PORTING.md](docs/PORTING.md)).

## How it works

1. **Folds** (`rogii_wellbore.cv`). 5 folds over the 773 training wells, stratified by each well's TVT range. Every run
   uses the same split.
2. **Synthetic pretraining pool** (`synth`). Per fold, 25,000 synthetic wells built from that fold's training
   typewells: a drifting, dipping path through the reference log, with faults, unconformities, variable bed
   thickness, dip breaks and repeated sections shown more often than in real wells.
3. **The well as an image** (`image`). 256 × 512 by default: columns are steps along the well, rows are 1 ft of TVT
   around the last known position. Five channels: the reference log, the well's log, the known pre-PS positions, a
   matching cost, and the well's dip. Predicting TVT becomes picking one row per column.
4. **Two-stage U-Net** (`model`, `train`). An EfficientNet U-Net gives row probabilities per column; a second U-Net
   reads the image plus those probabilities (softened, temperature 1.5) and refines them. Three of the four members
   are also trained to refine their own output again ("unrolled") and are run for three rounds. Pretrain 5 epochs on
   synthetic wells, fine-tune up to 35 on real wells with EMA weights, keep the best epoch at the deployment depth.
5. **Ensemble and submission** (`eval`, `infer`). Four members (EfficientNetV2-L at two image sizes,
   EfficientNetV2-XL, EfficientNet-L2), each 5 folds × 7 seeds, averaged with equal weights, no test-time
   augmentation.

About half of the remaining error is one constant offset per well that the inputs do not determine. What was tried
against it, and the mistakes caught along the way: [docs/WHAT_DIDNT_WORK.md](docs/WHAT_DIDNT_WORK.md).

## Results

| | CV (773 wells) | private LB |
|---|---|---|
| **final ensemble: 4 members, equal weights, no TTA** | **5.038** | **6.600 (26th)** |
| member: EfficientNetV2-XL | 5.136 | |
| member: EfficientNetV2-L | 5.157 | |
| member: EfficientNetV2-L, 224 × 448 | 5.252 | |
| member: EfficientNet-L2 | 5.260 | |

How the CV line came down from about 12 ft, every recorded leaderboard score, and why the private score sits above CV:
[docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## Pipeline

| # | stage | code | config | driver | notebook of record |
|---|---|---|---|---|---|
| 01 | [train a member](pipeline/01_train/) | `rogii_wellbore.run` (cv, synth, image, augment, model, train, oof) | `configs/members/<member>.yaml` | `train_colab.ipynb` | `2SUNET_MASTER.ipynb` |
| 02 | [score members and the blend](pipeline/02_ensemble/) | `rogii_wellbore.eval` | `configs/eval/final_blend.yaml` | `eval_colab.ipynb` | `ENSEMBLE_OOF_postproc.ipynb` |
| 03 | [Kaggle submission](pipeline/03_submit/) | `rogii_wellbore.infer` | `configs/submit/final_equal_weight.yaml` | `submit_kaggle.ipynb` | `final-submission-equal-weight-no-tta.ipynb` |

## Layout

```
src/rogii_wellbore/   the package (pip install -e .)
  config.py             one dataclass per section; YAML loader with `base:` inheritance
  data.py  cv.py        reading wells; the 5-fold split
  synth.py              synthetic wells and per-fold pools
  image.py  augment.py  rendering; training/validation datasets and augmentation
  model.py  train.py    two-stage U-Net, refeed, loss; one (fold, seed) job
  run.py                a whole member: jobs, OOF files, summary, Kaggle manifest
  metrics.py  oof.py    decode, offset/drift, tail, position; per-seed OOF files and ensembles
  eval.py               member and blend CV from OOF files
  infer.py              Kaggle inference and submission.csv
configs/              members/, eval/, submit/ (one YAML each)
pipeline/NN_*/        per stage: the notebook of record (outputs stripped), a driver notebook, a README
tests/                parity with the notebooks of record, and an end-to-end run on fake data
scripts/              strip_notebooks.py; make_kaggle_src.py (code as a Kaggle dataset)
docs/                 task, glossary, validation, what didn't work, scores, porting notes
ARTIFACTS.md          Kaggle datasets and Drive run folders
```

## Running it

```bash
pip install -e .[test]
python -m pytest -q                                    # about 1.5 minutes on CPU, no data needed

# train a member (GPU); folds/seeds default to the config's
python -m rogii_wellbore.run --config configs/members/v2l.yaml --data-root <unzipped data> --out <runs dir> \
    [--pretrained-dir <folder with pretrain_f*_s*_ep5.pt>] [--folds 0] [--seeds 34]

# score the four members and the blend from their OOF files
python -m rogii_wellbore.eval --config configs/eval/final_blend.yaml [--root <runs dir>] [--data-root <data>]

# the submission (on Kaggle: pipeline/03_submit/submit_kaggle.ipynb)
python -m rogii_wellbore.infer --config configs/submit/final_equal_weight.yaml --comp-root <competition data>
```

**Not in git:** competition data, checkpoints, OOF predictions, fold lists (they name competition wells). They are on
Kaggle and Google Drive and listed in [ARTIFACTS.md](ARTIFACTS.md).

## Credits

- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and
  [timm](https://github.com/huggingface/pytorch-image-models) for the U-Net and the EfficientNet encoders.
- ROGII and Kaggle for the competition and data.
