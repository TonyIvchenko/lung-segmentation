# Predict CLI Guide

Use `scripts/predict.py` for single-image inference.

## Example
```bash
python3 scripts/predict.py \
  --checkpoint models/unet-cli.pt \
  --image input/dataset/images/CHNCXR_0042_0.png \
  --output-mask outputs/CHNCXR_0042_0-mask.png \
  --output-overlay outputs/CHNCXR_0042_0-overlay.png
```

## Behavior
- Input image is converted to grayscale and resized to `--image-size`.
- Predicted class map is saved as `--output-mask`.
- If `--output-overlay` is provided, an RGB blend is saved for quick QA.
- If `--output-probability` is provided, class-1 softmax probabilities are saved.
