"""ROGII 2026 wellbore geology prediction: the two-stage U-Net pipeline, ported from the
competition notebooks. Modules: config, data, cv, synth, image, augment, model, train, run,
metrics, oof, eval, infer. Command lines: ``python -m rogii_wellbore.run`` (train a member),
``python -m rogii_wellbore.eval`` (score members and blends from OOF files),
``python -m rogii_wellbore.infer`` (Kaggle submission)."""
__version__ = '1.0.0'
