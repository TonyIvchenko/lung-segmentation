# Hyperparameter Guide

Key knobs exposed by `scripts/train.py`:

- `--learning-rate`
- `--weight-decay`
- `--batch-size`
- `--epochs`
- `--lr-step-size`, `--lr-gamma`
- `--grad-clip`
- `--patience`

## Practical defaults
- Start with `lr=5e-4`, `batch_size=4`.
- Use `patience` to stop unstable long runs.
- Add small `weight_decay` for regularization when overfitting appears.
