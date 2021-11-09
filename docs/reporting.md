# Reporting Guide

Use `scripts/report.py` to summarize a training history JSON into best-epoch metrics.

## Example
```bash
python3 scripts/report.py \
  --history outputs/run-01/history.json \
  --output outputs/run-01/summary.json
```

This helps standardize experiment result snapshots before sharing.
