# Train CLI Guide

Use `scripts/train.py` to run supervised segmentation training without notebooks.

## Required inputs
- Dataset folder: `input/dataset/images` and `input/dataset/masks`
- Split file: `splits.pk`

## Example
```bash
python3 scripts/train.py \
  --data-folder input/dataset \
  --splits splits.pk \
  --model pretrained-unet \
  --batch-norm \
  --epochs 100 \
  --output models/unet-cli.pt
```

## Useful flags
- `--resume`: continue from a checkpoint
- `--save-every`: periodic snapshots
- `--patience`: early stopping by validation Jaccard
- `--history-output` / `--history-csv`: export epoch metrics
- `--no-pretrained-encoder`: train pretrained UNet architecture without ImageNet weights
