# What didn't work

Ideas we measured and dropped, and mistakes we caught. Numbers are RMSE in feet; lower is better. "CV" is the
5-fold out-of-fold score. Results are quoted against the baseline of their time, which fell from about 12 ft to about
5.1 ft over the project, so compare numbers within a row, not across rows. Terms: [GLOSSARY.md](GLOSSARY.md).

## Recovering the per-well offset

About half of the final error is one offset per well (see [TASK.md](TASK.md)). It was attacked from every side we
could think of; none of these moved it.

| idea | what happened |
|---|---|
| **Predict the offset from features of the well** | Hand-built features: R² −0.25. Sequence features: R² −0.175. The U-Net's own internal representation: R² +0.002. The offset is not in the inputs |
| **A per-well latent plus a loss on each well's mean error** | The model learned to remove the bias on training wells and was unchanged on validation wells: it memorised each well's GR, nothing that transferred |
| **Map the formation surface across wells** (kriging, a physics-informed surface) | The surface between wells cannot be pinned better than about 29 ft, by any method we tried. The wells are far apart: half the rows are more than 400 ft from any other well |
| **Use the label formula** (TVT = −Z + C) directly | Assuming the layer stays flat: 101 ft. A shared surface: 122 ft. A local plane fitted on each well: 76 ft. The model at the time: about 5.6 |
| **Shift, shrink or re-anchor each well's prediction** | Capped at about 0.1 ft. The error is not a constant offset: it grows from heel to toe |
| **Test-time training on each test well's known rows** | Made things worse (+2.15 ft on typical wells). How well a model fits a well's known rows says nothing about the rows after them |
| **Neighbour wells as an input** (spatial prior) | Better CV (5.18 → 5.03), worse leaderboard (6.2 against 5.7 without it). Random folds overstate anything built from nearby wells; details in [VALIDATION.md](VALIDATION.md) |

## Matching the logs

| idea | what happened |
|---|---|
| **Match against the well's own pre-PS GR** instead of the typewell | A sharper matcher on its own (7.33 ft against 14.09), but beside the trained model it added about 0.03 ft |
| **Correct a level difference between typewell and well GR** | The per-well difference is small (75th percentile 2.9 API units) and is not a constant: the mismatch is in the shape, which a level shift cannot fix |
| **Other matching costs** (rank and census transforms, multi-scale windows, learned cost volumes) | Census replacing the squared difference: worse. The absolute GR value is what places a depth; shape-only costs lose it. With the true dip given, a better cost still found the same rival depths |
| **Multi-answer heads** (K candidate depths per column) | Either collapsed to one answer, or spread so far that even the best of K was worse on the hard wells (17.1 → 19.7 ft) |
| **Combine models for the worst wells** | On the worst 10% of wells, every model is wrong in the same direction (error correlation 0.93–0.95 across architectures). Averaging cannot cancel it |

## Training

| idea | what happened |
|---|---|
| **Extra loss terms** (auxiliary heads, consistency between views, per-well loss weights) | 0 for 9. Per-well weighting looked good on fold 4 (−0.17) and failed on fold 0 (+0.30), the fold it was designed for |
| **A wider decoder** | 2× the default width was the best measured. 3× was worse by 1.04 ft per seed |
| **Other backbones** (ConvNeXtV2 and similar patch-stem networks, ResNet-50, MiT, U-Net++, DeepLabV3+, FPN) | None beat EfficientNet on this image format. The patch-stem networks were given their own learning-rate recipe first and still lost |
| **SAM optimizer** | Slightly worse on the worst wells |
| **Freeze the encoder during fine-tuning** | 12.76 against 8.00 |
| **Self-supervised pretraining** (masked GR) | About 12.8 against 8.85 for synthetic supervised pretraining: the network learned a shortcut instead of the task |
| **More realistic synthetic wells** | Closer to the real data was not better. Moving the pool's statistics toward the real wells hurt; a pool rebuilt for fidelity tied the old one. What helped was more of the hard cases (events), not more realism |
| **Synthetic wells mixed into fine-tuning** | Worse. Synthetic data helps as a starting point, not as extra training data |

## Inference

| idea | what happened |
|---|---|
| **Horizontal-flip TTA** | The flipped prediction differs from the plain one (0.69 ft on average) and averaging them hurt |
| **Shift-by-part-of-a-row TTA** (PhaseJitter) | −0.04 ft on CV, +0.26 ft on the public board, twice: a test-set effect from the off-grid wells. Not in the final submission |
| **GR gain/offset TTA** | Every setting hurt (+0.41 for the average of nine) |
| **Refeed deeper than k = 2** without training for it | Better on average, but worse on single folds and on the worst wells. Unrolled training is what made k = 3 safe |

## Mistakes we caught

- **A distillation gain that was leakage.** Training each model to also match the averaged predictions of three
  earlier models scored 5.477 against 5.592, the best result at the time. But those earlier models had been trained on
  the validation wells of the model being scored. Three versions that removed that exposure (nested folds, shuffled
  targets, teachers that saw neither fold) each gave back the whole gain. The result was retracted.
- **A three-fold pass that reversed.** Column masking passed on three folds and failed all four checks on five.
  Since then, nothing has been adopted without five folds.
- **A "3-sigma" leaderboard surprise that was 1.7 sigma.** The board noise had been guessed, two to three times too
  small. Measured from our own errors, the surprise was ordinary. Board noise is now always simulated
  ([VALIDATION.md](VALIDATION.md)).
- **A grid assumption nobody had checked.** That training wells sit on an exact 1 ft MD grid was taken from a data
  summary. Checked well by well, it held for all 773 (3.78 million steps), which is what made the off-grid test wells
  a clean finding.
- **Settings that silently did nothing.** Loading saved pretraining checkpoints skips pretraining, so a changed
  synthetic-pool setting has no effect; and a debug value (`N_SYNTH_PRETRAIN = 15`) stayed in the notebook under
  the line meant to remove it. Both were caught before a run was spent on them. The fix is an assert, not care: any
  pool setting fails loudly when pretraining is being reused.
