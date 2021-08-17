# Dataset Preview Guide

Use `scripts/preview_dataset.py` to export an image grid of random samples.

## Example
```bash
python3 scripts/preview_dataset.py \
  --split train \
  --count 9 \
  --image-size 512 \
  --output images/dataset-preview.png
```

## Why use it
- Quickly validate image/mask alignment.
- Spot split issues before training.
- Share visual QA artifacts in experiment reports.
