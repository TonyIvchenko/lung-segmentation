# FAQ

## Why are tests skipped locally?
Some tests intentionally skip when optional dependencies (`torch`, `torchvision`) are unavailable.

## Why do predicted masks look blurry?
Use nearest-neighbor interpolation for masks and confirm postprocessing thresholding.

## How do I train from scratch without pretrained encoder?
Run `scripts/train.py --model pretrained-unet --no-pretrained-encoder` (or use `--model unet`).

## How do I run everything on CPU?
Use `--cpu` in train/evaluate/predict scripts.
