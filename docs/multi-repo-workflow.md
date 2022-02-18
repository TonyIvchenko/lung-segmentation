# Multi-Repo Workflow (Converter + lungmask + Training)

This guide describes a consistent pipeline across:
- `nifti-image-converter`
- `lungmask`
- `lung-segmentation`

## 1) Convert input volume(s) to PNG slices

Use identical naming settings for every case:

```bash
python3 python/nii2png.py \
  -i data/case001.nii.gz \
  -o workspace/images \
  --axis z \
  --prefix case001 \
  --index-width 3 \
  --manifest-json workspace/images/case001-manifest.json \
  --yes
```

## 2) Generate mask slices with matching names

```bash
lungmask data/case001.nii.gz workspace/case001-mask.nii.gz \
  --export-png-dir workspace/masks \
  --png-prefix case001 \
  --axis z \
  --index-width 3 \
  --manifest-json workspace/masks/case001-manifest.json
```

For PNG-only outputs, add `--skip-volume-output`.

## 3) Build training pair index

```bash
python3 scripts/build_pair_index.py \
  --images-folder workspace/images \
  --masks-folder workspace/masks \
  --mask-suffix "" \
  --output workspace/pairs.json
```

## 4) Build train/val/test splits

```bash
python3 scripts/make_splits.py \
  --data-folder workspace \
  --mask-suffix "" \
  --output workspace/splits.pk
```

## 5) Train and run batch inference

Train:

```bash
python3 scripts/train.py \
  --data-folder workspace \
  --splits workspace/splits.pk \
  --output workspace/models/unet.pt
```

Batch predict:

```bash
python3 scripts/predict.py \
  --checkpoint workspace/models/unet.pt \
  --image-dir workspace/images \
  --output-mask-dir workspace/predictions \
  --glob "*.png"
```
