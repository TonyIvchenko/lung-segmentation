# Evaluate CLI Guide

Use `scripts/evaluate.py` to score a saved checkpoint on `train`, `val`, or `test` data.

## Example
```bash
python3 scripts/evaluate.py \
  --checkpoint models/unet-cli.pt \
  --split test \
  --model auto
```

## Outputs
- Printed summary: loss, Jaccard, Dice, precision, recall, F1
- Optional JSON summary with `--output-json`
- Optional per-sample CSV with `--output-samples-csv`

## Notes
- `--model auto` infers architecture from checkpoint metadata when available.
- Use `--cpu` for CPU-only environments.
