# Validation

How a change was judged, and the two places where cross-validation turned out not to describe the test set.

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
  for gross checks.
- **It was also used as an instrument.** The kernel cannot return test data, but the score can encode one number per
  submission. Adding a large constant to the predictions of wells that meet some condition, and reading the score,
  measures what share of the public rows meet it. This is how the off-grid test wells were found: about 16% of test
  wells do not sit on the 1.0 ft MD grid that every training well uses.

## Where CV and the test set came apart

1. **The MD grid.** Training wells are all on one grid, so no CV can reward robustness to grid shift. PhaseJitter TTA
   cost about 0.04 ft on CV and improved the public score by about 0.26 ft, twice, on two different model lines.
   The final submission did not use it (see [LEADERBOARD.md](LEADERBOARD.md)).
2. **Neighbour wells.** A feature built from nearby training wells (a spatial prior on the formation surface)
   improved CV from 5.18 to 5.03, then scored worse on the leaderboard with it switched on (6.2) than off (5.7).
   Random folds put a validation well's neighbours in the training set; the test wells are further from the training
   wells than that. The rule that came out of it: anything built from other wells is priced on folds that hold out
   whole regions, and checked against the board before it is trusted. The spatial feature was dropped, and the final
   ensemble does not use it.
