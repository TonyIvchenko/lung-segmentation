# Reproducibility Notes

## Randomness controls
- `scripts/train.py` uses `set_seed(seed)` for Python, NumPy, and Torch.
- `scripts/make_splits.py` uses explicit `--seed` for dataset partitioning.

## Artifacts to version
- `splits.pk`
- training config (`args` saved in checkpoints)
- metric history (`--history-output`, `--history-csv`)

## Recommendations
- Keep immutable checkpoints for key experiments.
- Record git commit hashes alongside benchmark metrics.
- Prefer CLI scripts over ad-hoc notebook state for official runs.
