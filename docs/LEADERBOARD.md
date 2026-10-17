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
| 2 Jul | EfficientNetV2-XL, 25,000-well pool | 5.592 | 5.444 |
| 11 Jul | augmentation strengths re-tuned on the new line | 5.474 | |
| 11 Jul | rebuilt pool + dip breaks and repeated sections (event pack A) | 5.307 | |
| 12 Jul | unrolled training + supervision of every round, k = 3 | 5.180 | |
| 16–17 Jul | + neighbour prior as an extra channel, with a learned gate | 5.027 | 6.091 (prior on), 5.7 (prior blanked) |

The neighbour prior was then dropped on the public board's reading, and the final ensemble does not use it. That was a
mistake: on the private board this model scored 6.11 and would have placed 7th
([the top-level README](../README.md#the-best-model-we-did-not-submit)).

## The final ensemble

Each member: 5 folds × 7 seeds, 25,000-well synthetic pretraining with event pack A, two-stage U-Net, handoff
temperature 1.5, trained late July to early August.

| member | encoder | image | refeed | CV |
|---|---|---|---|---|
| `xl_hT15_k3` | EfficientNetV2-XL | 256 × 512 | k = 3, unrolled | 5.136 |
| `l_hT15_k3` | EfficientNetV2-L | 256 × 512 | k = 3, unrolled | 5.157 |
| `l_hT15_k3_small` | EfficientNetV2-L | 224 × 448 | k = 3, unrolled | 5.252 |
| `l2_hT15_k1` | EfficientNet-L2 | 256 × 512 | k = 1 | 5.260 |

Submitted as an equal-weight average (0.25 each):

| submission | CV | private LB |
|---|---|---|
| `final-submission-equal-weight-no-tta` | 5.038 | **6.600 — 26th place** |

## The submission we did not pick

| submission | CV | public LB | private LB |
|---|---|---|---|
| `stacked-unet-inference`: EfficientNetV2-XL with the neighbour prior, 5 folds × 5 seeds | **5.027** | 6.091 | **6.11** (would have placed 7th) |
| the neighbour prior alone (no model) | 13.3 | 24.7 | 13.2 |

The prior alone scored the same on the private board as in CV; the public board was the outlier. Every submission that used the neighbour prior, as an input channel or as a blend with the model's output, scored
much better on the private board than the final ensemble.

## Why the private score is above CV

The ensemble scores 5.038 on CV (members 5.14–5.26); the private leaderboard gave 6.600. Two things are known. About
16% of test wells are off the MD grid that every training well uses ([VALIDATION.md](VALIDATION.md)). And the
neighbour-prior model, with almost the same CV (5.027), scored 6.11: on the private wells, the prior was worth about
0.5 ft more than the final ensemble's extra members. How much of the gap each explains was not measured.
