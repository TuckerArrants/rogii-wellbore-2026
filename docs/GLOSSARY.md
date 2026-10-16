# Glossary

Words the READMEs and notebooks use without explaining. The task itself: [TASK.md](TASK.md).

## Data

| term | meaning |
|---|---|
| **HW** | the horizontal well: its own GR log, trajectory (Z, X, Y) and known TVT up to the PS point |
| **TW, typewell** | the reference log for a well: GR as a function of TVT, from a vertical well nearby |
| **MD** | measured depth: distance along the well. One row per foot |
| **TVT** | true vertical thickness: the well's position within the rock layers, the quantity predicted |
| **PS point** | where the given TVT stops and prediction starts. "Pre-PS" and "post-PS" are the rows before and after |
| **heel, toe** | the start and the far end of the horizontal part of the well. Errors grow from heel to toe |
| **ANCC** | TVT + Z: the elevation of the target formation top. A smooth surface, which is why TVT = −Z + ANCC |

## The image

| term | meaning |
|---|---|
| **image, frame** | one well drawn as H × W = 256 × 512: W columns along MD, H rows of 1 ft in TVT, centred on the last known TVT |
| **channels** | five per image: `TW_GR` (reference log down the rows), `HW_GR` (the well's log across the columns), `hint` (the known pre-PS rows marked), `cost volume` (squared GR difference between the two at every row and column), `dZ` (the well's dip per column) |
| **cost volume** | the matching channel: low where the well's GR equals the reference log's GR at that depth. The single largest early gain |
| **expected-row decode** | the prediction: in each column, the probability-weighted average row, converted back to TVT |
| **supervision window** | only ±128 ft around the PS depth is inside the image; truth outside it is not trained on |

## Models

| term | meaning |
|---|---|
| **2SUNet, stack-2** | two U-Nets in a row. Stage 1 reads the image; stage 2 reads the image plus stage 1's probability map ("the handoff") and refines it. Stage 1's gradient is cut at the handoff ("detached") |
| **handoff temperature (handoffT)** | stage 1's probabilities are softened (divided by T = 1.5) before stage 2 sees them |
| **k, refeed, self-iteration** | running stage 2 again on its own output. k = 1 is one pass; k = 3 is the final members' setting |
| **unroll** | training stage 2 to do that: in half the batches, stage 2 is run 2–4 rounds and every round is supervised. Lets k = 3 help at prediction time instead of hurt |
| **deep supervision** | stage 1's output also gets a loss (weight 0.5), not only the final stage |
| **EMA** | the weights used are an exponential moving average over training steps (decay 0.997) |
| **member** | one model family in the final ensemble (encoder, image size, recipe), trained as 5 folds × 7 seeds = 35 checkpoints |

## Synthetic data and augmentation

| term | meaning |
|---|---|
| **synth pool, pretrain** | 25,000 synthetic wells per fold, built from that fold's training typewells: a trajectory and a drifting formation surface are generated, and the well's GR is read from the reference log along it with noise added. Each model trains on these for 5 epochs before the real wells |
| **events** | rare structures put into synthetic wells: faults, unconformities, dip breaks, repeated sections. Shown more often than in real wells on purpose: they are the ambiguous cases |
| **EVEX-A, event pack A** | the final pool's event mix: dip breaks at probability 0.35 and repeated sections at 0.12, on top of the earlier faults |
| **MDwarp** | augmentation: stretch and squeeze the well along MD (the GR and the truth move together) |
| **BedWarp** | augmentation: stretch and squeeze the layers in TVT (the reference log and the truth move together) |
| **HFlip** | augmentation: mirror the well along MD (probability 0.5, fine-tuning only) |
| **HintDrop** | augmentation: hide part of the known pre-PS rows (probability 0.3) |
| **PhaseJitter** | shifting the TVT grid by part of a row. Tried as augmentation and as test-time averaging (TTA) |

## Evaluation

| term | meaning |
|---|---|
| **OOF** | out-of-fold: each well predicted by the models that did not train on it |
| **full-CV** | RMSE over all 773 wells' OOF predictions, pooled over rows: the headline number |
| **offset, drift** (older notes: **datum**) | split of a well's error: the offset is its mean, the drift is what is left. RMS of each is reported |
| **tail** | the worst wells. Reported as the count of wells above 8 / 10 / 12 / 15 ft and the share of squared error in the worst 10% |
| **const** | the baseline that predicts the PS-point TVT for every row: 15.9 ft |
| **DNR** | "do not repeat": an idea measured and closed, numbered in the working ledger |
