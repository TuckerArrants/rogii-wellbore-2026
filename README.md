# ROGII 2026 Wellbore Geology Prediction

Code for the [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
competition: predict where a horizontal well sits in the rock layers (its TVT, true vertical thickness) along the part
of the well that has no interpretation yet, from its gamma-ray log, a reference log and its trajectory. Scored by RMSE
in feet. The problem and the data: [docs/TASK.md](docs/TASK.md). Words used throughout:
[docs/GLOSSARY.md](docs/GLOSSARY.md).

**Result: 26th place, private leaderboard RMSE 6.600; cross-validation 5.038 for the submitted ensemble.**
Write-up on Kaggle: [26th place solution: stacked U-Net, synthetic wells](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/26th-place-solution-stacked-unet-synthetic-wel).

In plain words: an oil well is drilled down, then turned to run sideways through a thin layer of rock for a mile or
more. To keep it in the layer, a geologist compares the gamma-ray readings taken along the well with a reference log
from a nearby vertical well, and works out where the well sits in the layers. Each well's position is given up to a
point (the PS point); the task is to predict the rest.

The work was done in Colab and Kaggle notebooks between May and August 2026 and moved into this repo afterwards as the
`rogii_wellbore` package. The notebooks of record are kept, as run, in `pipeline/`; the tests check the package against
them function by function ([docs/PORTING.md](docs/PORTING.md)).

## The idea: draw the well as a picture

A geologist solves this by lining up two wiggly lines. We drew each well as an image instead:

![Left: a cross-section with a vertical typewell and a horizontal well running through bent, dipping layers, known up to the PS point. Right: the same well as an image, where the layers become flat bands and the well becomes a line across them. Below: the five channels the network reads.](docs/img/well_as_image.svg)

- Each **column** is a step along the well (512 columns). Each **row** is one foot of depth in the rock (256 rows,
  centred on the last known position).
- The reference log runs down the rows, so the rock layers become flat bands. The well's own log runs across the
  columns.
- Three more channels: the known positions, a match cost that marks where the two logs agree, and the well's dip.

The task is now to **pick one row in every column**: draw the line that carries on from the known part. That is an
image segmentation problem, and images come with tools that wiggly lines do not:

- **Pretrained U-Nets.** A U-Net is the standard network for marking things in an image pixel by pixel. Its encoder
  can start from weights learned on ImageNet, millions of everyday photos. Photos are not rocks, but a network that
  already finds edges, bands and smooth lines is a good place to start. Larger encoders tended to help, so the final
  models use big EfficientNets.
- **As much practice data as we want.** The image is built from a reference log and a path through it. So we can make
  up a path, read the reference log along it, add noise and geology (faults, dip breaks, repeated layers), and get a
  training image with a known answer. Every model trains on 25,000 of these before it sees a real well. There are only
  773 real ones.

Most top placing solutions formulated the problem this way, despite the task reading as a standard "tabular" one, where GBDT's would beat neural networks. Once the approach is reformulated as "well as an image", the UNet approach is far superior. 

## How it works

1. **Folds** (`rogii_wellbore.cv`). 5 folds over the 773 training wells, stratified by each well's TVT range. Every run
   uses the same split.
2. **Synthetic wells** (`synth`). Per fold, 25,000 made-up wells built from that fold's training typewells: a
   drifting, dipping path through the reference log, with faults, unconformities, variable bed thickness, dip breaks
   and repeated sections shown more often than in real wells. The hard cases are the ambiguous ones, so they are
   over-represented on purpose.
3. **The well as an image** (`image`). 256 × 512 by default, five channels, as above.
4. **Two-stage U-Net** (`model`, `train`). The first U-Net makes a guess: for every column, a chance for every row. The
   second U-Net sees the image plus that guess and sharpens it, and its output can be fed back into it again:

   ![The five-channel image goes into the stage 1 U-Net, which gives a first guess of row probabilities. Stage 2 reads the image plus that guess and gives a sharper one, which is fed back in for three rounds. The average row in each column, in feet, is the prediction. Below: training on 25,000 synthetic wells, then about 620 real wells, then four models averaged.](docs/img/two_stage_unet.svg)

   The prediction in each column is the probability-weighted average row, turned back into feet. Each model
   pretrains for 5 epochs on synthetic wells, then fine-tunes for up to 35 on real wells with EMA weights; the best
   epoch at the deployment depth is kept. Three of the four members run three rounds; the L2 runs one.
5. **Ensemble and submission** (`eval`, `infer`). Four members (EfficientNetV2-L at two image sizes,
   EfficientNetV2-XL, EfficientNet-L2), each 5 folds × 7 seeds, averaged with equal weights.

About half of the remaining error is one constant offset per well that the inputs do not pin down: the logs can match
almost as well several feet higher or lower. What was tried against it:
[docs/WHAT_DIDNT_WORK.md](docs/WHAT_DIDNT_WORK.md).

## Results

| | CV (773 wells) | private LB |
|---|---|---|
| **final ensemble: 4 members, equal weights** | **5.038** | **6.600 (26th)** |
| member: EfficientNetV2-XL | 5.136 | |
| member: EfficientNetV2-L | 5.157 | |
| member: EfficientNetV2-L, 224 × 448 | 5.252 | |
| member: EfficientNet-L2 | 5.260 | |
| *not submitted:* EfficientNetV2-XL with the neighbour prior, one model | 5.027 | 6.11 (would have placed **7th**) |

How the CV line came down from about 12 ft, every recorded leaderboard score, and why the private score sits above CV:
[docs/LEADERBOARD.md](docs/LEADERBOARD.md).

## The best model I did not submit

The final ensemble left out one input: a **neighbour prior**. Nearby training wells show how the rock layer tilts in
each place. The prior reads those tilts along the new well's path (from the closest pieces of neighbouring wells,
corrected for the direction the well runs) and adds them up from the well's last known position. On its own it is a
rough guess: 13.3 ft in CV, against 15.9 for a constant. Drawn as a soft band in one more channel (the 7th input to
stage 2, after the five base channels and stage 1's guess), it gave the U-Net a second opinion on where the line
should go. A learned gate at the output decided how much to trust it.

It took a single EfficientNetV2-XL (5 folds × 5 seeds) from 5.180 to **5.027**, the best CV of the project. Then we
dropped it. Three things pointed the same way:

- **The public board.** On its roughly 50 wells, the model scored 6.09 with the prior and 5.7 with the prior channel
  blanked. The prior alone scored 24.7, against 13.3 in CV.
- **A plausible story.** Random folds put each validation well's neighbours in the training set. If the test wells
  sit further from the training wells, a neighbour feature looks better in CV than it is. That seemed to explain the
  board.
- **A harder CV that agreed.** Folds that held out whole regions scored the prior alone at 51 ft. But those folds were
  weighted to reproduce the 24.7 public score, so they could not check it; they repeated it.

It looked too good to be true, so we checked for leakage and found none: the prior's inputs were the same numbers at
training and at test. We dropped it anyway, on 18 July, and built the final ensemble without it.

The private board, the other three quarters of the test wells, went the other way:

| | CV | public LB | private LB |
|---|---|---|---|
| the prior alone | 13.3 | 24.7 | **13.2** |
| EfficientNetV2-XL with the prior (5.027 model) | 5.027 | 6.09 | **6.11** |
| final ensemble, no prior | 5.038 | | 6.600 (26th) |

The prior alone scored 13.2 on the private board, the same as its 13.3 in CV. The CV was right; the public board's 50
wells were the odd ones out. Every submission that used the prior, as the 7th channel or as a simple blend with the
model's output, scored much better than the final ensemble. The 5.027 model's Kaggle notebook is kept as
[`pipeline/03_submit/stacked-unet-inference.ipynb`](pipeline/03_submit/stacked-unet-inference.ipynb); it is not
ported to the package.

What we take from it:

- **About 50 wells should not overrule CV on 773.** Our own simulation put one public score at ±0.85 ft, and we
  acted on a 0.4 ft gap. The 24.7 looked too large to be noise, but the private board put the prior back at 13.2.
- **"Too good to be true" is a reason to look for a leak, not a verdict.** Once the leak check came back clean, the CV
  result should have stood.
- **A check tuned to agree with a number cannot confirm it.**

This was the most costly decision of the project.

## Pipeline

| # | stage | code | config | driver | notebook of record |
|---|---|---|---|---|---|
| 01 | [train a member](pipeline/01_train/) | `rogii_wellbore.run` (cv, synth, image, augment, model, train, oof) | `configs/members/<member>.yaml` | `train_colab.ipynb` | `2SUNET_MASTER.ipynb` |
| 02 | [score members and the blend](pipeline/02_ensemble/) | `rogii_wellbore.eval` | `configs/eval/final_blend.yaml` | `eval_colab.ipynb` | `ENSEMBLE_OOF_postproc.ipynb` |
| 03 | [Kaggle submission](pipeline/03_submit/) | `rogii_wellbore.infer` | `configs/submit/final_equal_weight.yaml` | `submit_kaggle.ipynb` | `final-submission-equal-weight-no-tta.ipynb`; `stacked-unet-inference.ipynb` (not submitted) |

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
  verify.py             checkpoint replay against the notebooks' OOF files
configs/              members/ (plus smoke.yaml, a tiny check run), eval/, submit/ (one YAML each)
pipeline/NN_*/        per stage: the notebook of record (outputs stripped), a driver notebook, a README
pipeline/checks/      checks on the real data: OOF scores, checkpoint replay, a tiny training run
tests/                parity with the notebooks of record, and an end-to-end run on fake data
scripts/              strip_notebooks.py; make_kaggle_src.py (code as a Kaggle dataset); make_diagrams.py
docs/                 task, glossary, validation, what didn't work, scores, porting notes; img/ diagrams
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

# replay a saved checkpoint and compare with the notebook's OOF file (CPU is fine)
python -m rogii_wellbore.verify --config configs/members/v2l.yaml --data-root <data> --run-dir <run folder>
```

The checks that the package runs on the real data, without retraining anything: [pipeline/checks/](pipeline/checks/).

**Not in git:** competition data, checkpoints, OOF predictions, fold lists (they name competition wells). They are on
Kaggle and Google Drive and listed in [ARTIFACTS.md](ARTIFACTS.md).

## Credits

- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) and
  [timm](https://github.com/huggingface/pytorch-image-models) for the U-Net and the EfficientNet encoders.
- ROGII and Kaggle for the competition and data.
