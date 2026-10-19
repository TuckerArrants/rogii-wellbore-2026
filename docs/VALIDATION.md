# Validation

How a change was judged, and how the public leaderboard was read.

## Cross-validation

- **Folds:** 5 folds over the 773 training wells (153/153/153/153/161 for validation), stratified by each well's TVT
  range so every fold has the same mix of quiet and strongly drifting wells. The split is fixed by seed; every run
  in the project uses the same folds.
- **Seeds:** 5 per fold during development, 7 for the final members. A fold's prediction is the mean of its seeds;
  the headline is RMSE over all 773 wells' out-of-fold rows ("full-CV").
- **Leakage:** the synthetic pretraining pool for a fold is built only from that fold's training typewells, and the
  matching statistics it copies come only from that fold's training wells. A hard assert in
  [stage 01](../pipeline/01_train/) checks both pools before training.

## How a result was read

- **Set the bar before the run.** Each experiment named its baseline and its pass mark before training: the change had
  to beat the baseline by a set margin (0.20 ft early on, 0.10 ft late) with no fold worse by more than 0.5 ft. A
  result inside the margin was recorded as null, not as a small win.
- **Read every fold, and the single-seed numbers.** A gain on one fold, or a gain that appears only after averaging
  seeds, was not enough. An early three-fold pass (column masking) reversed on the five-fold run, and from then on
  nothing was adopted on three folds.
- **The tail counts as much as the mean.** Every run reports, beside full-CV, the offset/drift split, the number of
  wells above 8 / 10 / 12 / 15 ft, the worst-10% share of squared error, and heel-to-toe error. A change that lowered
  the mean by making the worst wells worse was not adopted.
- **Inference settings are measured too.** Test-time augmentation, refeed depth and temperatures were each measured on
  OOF predictions of the checkpoints they would ship with. Several "free" ideas cost CV when measured.

## The public leaderboard

- **Its noise was measured, not guessed.** Drawing 50-well subsets from our own OOF errors: one public score is
  uncertain by about ±0.85 ft (one standard deviation); two submissions of related models differ by chance by about
  0.16 ft. At that size, a real 0.1 ft improvement is a one-sigma read. CV decided model changes; the board was used
  for gross checks. The one time the board overruled CV, it was wrong (below).

## When the public board misled us

The neighbour prior, a guess built from nearby training wells and given to the model as an extra channel, took the
best single model from 5.18 to 5.03 in CV. On the public board it scored worse with the prior (6.09) than with it
blanked (5.7), and the prior alone scored 24.7 against 13.3 in CV. We concluded that random folds overstate anything
built from neighbours, built region-held-out folds that agreed (they had been weighted to reproduce the public
score), and dropped the prior. The private board showed the opposite: the prior alone scored 13.2, matching CV, and
the model scored 6.11, which would have placed 7th. The full account:
[the top-level README](../README.md#the-best-model-we-did-not-submit).
