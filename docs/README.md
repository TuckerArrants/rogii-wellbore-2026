# docs

Start here if you are new to the project:

- [The write-up on Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/26th-place-solution-stacked-unet-synthetic-wel): the solution in one post.

- [TASK.md](TASK.md): the competition, the data, what the labels are, and what makes it hard.
- [GLOSSARY.md](GLOSSARY.md): the words the rest of the repo uses without explaining.
- [WHAT_DIDNT_WORK.md](WHAT_DIDNT_WORK.md): ideas we measured and dropped.

Records:

- [LEADERBOARD.md](LEADERBOARD.md): the CV line from ~12 ft to the final ensemble (5.038), every recorded public score,
  the final private result, and the better submission we did not pick.
- [VALIDATION.md](VALIDATION.md): how folds were built, how a result was read, how the leaderboard's noise was priced,
  and the two places where CV and the public board disagreed.
- [PORTING.md](PORTING.md): how the notebooks became the `rogii_wellbore` package, what the tests check against the
  notebooks, the deliberate differences, and what is still owed.

Diagrams (`img/`, drawn by `scripts/make_diagrams.py`): [the well as an image](img/well_as_image.svg) and
[the two-stage U-Net and its training](img/two_stage_unet.svg). Both are explained in the
[top-level README](../README.md#the-idea-draw-the-well-as-a-picture).
