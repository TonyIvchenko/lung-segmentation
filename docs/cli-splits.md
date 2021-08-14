# Split Generation Guide

Use `scripts/make_splits.py` to regenerate train/val/test splits reproducibly.

## Example
```bash
python3 scripts/make_splits.py \
  --data-folder input/dataset \
  --output splits.pk \
  --test-size 0.2 \
  --val-size 0.1 \
  --seed 42
```

## Notes
- Only masks with matching origin images are kept.
- The script logs dropped pairs if source images are missing.
- Store the generated `splits.pk` in version control for repeatability.
