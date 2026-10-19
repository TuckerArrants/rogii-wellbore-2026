# Checks · does it run on the real data?

The tests in `tests/` run on CPU with fake data and a small encoder. They show that every function gives the
notebooks' numbers. They cannot show that the package runs on the real files, with the real encoders and
checkpoints, on a GPU. These checks do, without retraining anything.

## Why not retrain and compare?

Two runs of the same recipe differed by more than most of the changes we tested (seed and GPU noise), so one
retrained seed proves nothing and five cost days of GPU, most of it synthetic pretraining. Inference has no
such noise. So the checks replay the saved checkpoints and compare with what the notebooks saved from the same
weights. Training is checked for running, not for its score.

## What covers what

| | where | GPU time | what it shows |
|---|---|---|---|
| every function equals the notebooks' | `tests/` (CPU, fake data) | none | the port is faithful, bit for bit, including two full fine-tuning epochs |
| the four real encoders | `tests/test_real_encoders.py` | none | the package model has exactly the parameter names and shapes of the saved checkpoints |
| **1. score the ensemble from its OOF files** | [`checks_colab.ipynb`](checks_colab.ipynb) | none | the real OOF files load, and the members and blend score 5.136 / 5.157 / 5.252 / 5.260 / **5.038** |
| **2. replay a checkpoint** | [`checks_colab.ipynb`](checks_colab.ipynb) | 1–2 min per seed | real data, fold split, rendering, a real checkpoint, refeed and decode give the notebook's OOF numbers |
| **3. tiny training run** | [`checks_colab.ipynb`](checks_colab.ipynb) | 10–15 min | the whole training path runs on the real data on a GPU (`configs/members/smoke.yaml`) |
| **4. the full submission** | [`../03_submit/submit_kaggle.ipynb`](../03_submit/submit_kaggle.ipynb) | roughly 15–30 min on Kaggle | all 140 checkpoints give the saved submission's numbers on the 3 visible test wells (`reference_rows.json`) |

Checks 1, 2 and 4 compare with numbers of record and print PASS or FAIL. Check 3 only has to finish.

Not covered, on purpose: reproducing a member's CV by retraining it.

From the command line, check 2 is:

```bash
python -m rogii_wellbore.verify --config configs/members/v2l.yaml --data-root <data> \
    --run-dir <Drive run folder> [--ckpt-dir <folder with ckpt_f*_s*.pt>] [--fold 0] [--seeds 34]
```

On CPU it takes about 3 s per well for EfficientNetV2-L (153 wells in a fold), so it does not need a GPU.

## Results

Not run yet. When they are, record them here: date, runtime, and each check's printout.
