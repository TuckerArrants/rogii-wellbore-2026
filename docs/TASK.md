# The task

[ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction): a
horizontal well is drilled sideways through a thin rock layer. A geologist follows it by matching the gamma-ray (GR)
log measured in the well against a reference log from a nearby vertical well. That matching gives the well's
**TVT** (true vertical thickness): its depth position within the layered rock. For each well, the TVT is given up to
a point; the task is to predict it for the rest of the well.

**Metric:** RMSE in feet, pooled over every predicted row of every test well. Long wells count more, and the far end
of each well (the toe), where errors are largest, counts most.

**Format:** a code competition. The submitted Kaggle notebook runs offline on hidden test wells; only the score comes
back. The public leaderboard scored roughly a quarter of the test wells (about 50); the private leaderboard scored the
rest.

## The data

Per well, two CSV files:

| file | columns used | what it is |
|---|---|---|
| `<well>__horizontal_well.csv` | `MD`, `GR`, `Z`, `X`, `Y`, `TVT_input`, `TVT` | the well itself, one row per foot of measured depth (MD). `TVT_input` is filled up to the **PS point** and empty after it; `TVT` is the answer (training only) |
| `<well>__typewell.csv` | `TVT`, `GR` | the reference log: GR as a function of TVT |

- **Training:** 773 wells, 3.78 million rows to predict (the rows after the PS point).
- **Coordinates:** `X`, `Y` are Texas state-plane feet; the wells are in the Eagle Ford / Austin Chalk / Buda section of
  South Texas.
- **Grid:** in training, every well is on an exact 1.0 ft MD grid. About 16% of test wells are not (measured through
  the leaderboard; see [LEADERBOARD.md](LEADERBOARD.md)).

## What the labels are

The TVT labels are a geosteering interpretation made in StarSteer. They follow **TVT = −Z + C**, where C is the
elevation of the target formation top along the well: a piecewise-linear surface with about 35 control points per
well. Locally the rock is flat, so a change in Z shows up one-for-one in TVT. Over a whole lateral, though, the
surface drifts by about 157 ft (median; 90th percentile 242 ft). Predicting TVT after the PS point means predicting
that drift.

Predicting a constant from the PS point scores **15.9 ft**. The final models score **5.1–5.3 ft** in cross-validation.

## What makes it hard

1. **The GR log does not say where the well is, only what it looks like.** The well runs nearly parallel to the
   layers and stays inside a band of about 27 ft of TVT after the PS point (median). Layered rock repeats itself, so a
   stretch of GR often matches the reference log equally well at two or more depths. In a test with the true dip
   given, 92–93% of columns still had a rival depth that matched as well as the right one.
2. **About half of the remaining error is one offset per well.** Split each well's error into its mean (the offset)
   and the rest (the drift): at the final models, offset ≈ 3.6 ft and drift ≈ 3.7 ft. Every attempt to predict the
   offset from the inputs failed (summary in [WHAT_DIDNT_WORK.md](WHAT_DIDNT_WORK.md)). In real drilling, the offset
   comes from information this dataset does not include, such as formation tops picked in nearby wells.
3. **The error sits in a few wells.** The worst 10% of wells carry about 60% of the squared error. A small change in
   how a model handles those wells moves the score more than anything that happens on typical wells.
4. **Few wells, and a test set that is not quite like training.** 773 wells is small for a deep network, so most of
   the gains came from synthetic data and augmentation. The test set differs from training in ways cross-validation
   cannot show: the off-grid wells above, and the test wells' positions relative to the training wells (stage notes
   in [VALIDATION.md](VALIDATION.md)).
