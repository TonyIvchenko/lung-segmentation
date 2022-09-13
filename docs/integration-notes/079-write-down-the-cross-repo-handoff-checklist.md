# Write down the cross-repo handoff checklist

## Why this matters
A repeatable handoff from images and masks to splits keeps training and inference reproducible.

## Quick command
```bash
cd ~/git/lung-segmentation
python3 scripts/build_pair_index.py \
  --images-folder workspace/images \
  --masks-folder workspace/masks \
  --mask-suffix "" \
  --output workspace/index/pairs.json

python3 scripts/make_splits.py \
  --data-folder workspace \
  --mask-suffix "" \
  --output workspace/splits/splits.pk
```

## What to check
- Pair index has no unexpected missing masks.
- Split counts look sensible before training starts.
