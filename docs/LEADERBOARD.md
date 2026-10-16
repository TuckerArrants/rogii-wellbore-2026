# Scores over time

The cross-validation line, oldest first, with the public leaderboard score where one was recorded. The detailed records
behind these are internal working notes, not published.

**How to read the board.** The public leaderboard scored about 50 wells. Simulated from our own errors, a single public
score is uncertain by about ±0.85 ft, and two related submissions differ by chance by about 0.16 ft
([VALIDATION.md](VALIDATION.md)). A public score that moves by less than that is not evidence.

## The CV line

| when | change | CV | public LB |
|---|---|---|---|
| May | 4-channel U-Net, no matching cost, no dip | ~12 | |
| May | + the matching-cost channel | ~10.3 | |
| May | 5-fold protocol fixed | 9.8 | |
| late May | + the dip (dZ) channel | 8.546 | 6.94 |
| early Jun | + HFlip and HintDrop augmentation | 7.900 | 6.663 |
| Jun | synthetic wells with GR matching as hard as real wells | 7.202 | |
| Jun | EfficientNet-B7 encoder | 6.789 | |
| mid Jun | + MDwarp and BedWarp augmentation: the "lock" baseline | 6.541 | |
| mid Jun | two-stage U-Net (stage 1 detached) | 6.262 | |
| Jun | decoder 2× wider | ~6.1 | |
| late Jun | synthetic pool 7,500 → 15,000 wells | 5.80 | |
| late Jun | EfficientNetV2-L, stronger augmentation | 5.691 | |
| 2 Jul | EfficientNetV2-XL, 25,000-well pool | 5.592 | 5.444 (with TTA) |
| 11 Jul | augmentation strengths re-tuned on the new line | 5.474 | |
| 11 Jul | rebuilt pool + dip breaks and repeated sections (event pack A) | 5.307 | |
| 12 Jul | unrolled training + supervision of every round, k = 3 | 5.180 | |
| 16–17 Jul | + spatial prior from neighbour wells | 5.027 | 6.2 (prior on), 5.7 (prior off) |

The spatial prior was then dropped ([VALIDATION.md](VALIDATION.md)); the final ensemble does not use it.

## The final ensemble

Each member: 5 folds × 7 seeds, 25,000-well synthetic pretraining with event pack A, two-stage U-Net, handoff
temperature 1.5, trained late July to early August.

| member | encoder | image | refeed | CV |
|---|---|---|---|---|
| `xl_hT15_k3` | EfficientNetV2-XL | 256 × 512 | k = 3, unrolled | 5.136 |
| `l_hT15_k3` | EfficientNetV2-L | 256 × 512 | k = 3, unrolled | 5.157 |
| `l_hT15_k3_small` | EfficientNetV2-L | 224 × 448 | k = 3, unrolled | 5.252 |
| `l2_hT15_k1` | EfficientNet-L2 | 256 × 512 | k = 1 | 5.260 |

Submitted as an equal-weight average (0.25 each), no TTA:

| submission | CV | private LB |
|---|---|---|
| `final-submission-equal-weight-no-tta` | 5.038 | **6.600 — 26th place** |

## Other recorded public scores

These are from the July inference work on the 5.592 / 5.477 lines (the 5.477 line was later retracted as leakage; see
[WHAT_DIDNT_WORK.md](WHAT_DIDNT_WORK.md)).

| submission | public LB | note |
|---|---|---|
| distillation line, one pass | 5.821 | |
| same, two refeed rounds pooled across seeds | 5.864 | the CV gain (−0.11) did not show on the board; within noise |
| same, PhaseJitter TTA (8 grid shifts averaged) | **5.566** | best public score of these. TTA costs CV but helps the off-grid test wells; the same gap appeared on a second model line |
| refeed + TTA together | 5.603 | |
| TTA on high-uncertainty wells only | 5.738 | the TTA gain is spread over many wells |

Probe submissions (built to measure a property of the test set, not to score well) are not listed.

## Why the private score is above CV

The ensemble scores 5.038 on CV (members 5.14–5.26); the private leaderboard gave 6.600. The public board had run above CV for the
later lines too (5.7 for a 5.03 CV line). Two differences between the test set and training were measured: about 16%
of test wells are off the MD grid that every training well uses, and the test wells sit further from the training
wells than random folds assume ([VALIDATION.md](VALIDATION.md)). How much of the gap each one explains, if any, was
not measured.
