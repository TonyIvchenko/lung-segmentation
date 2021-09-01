# Troubleshooting

## `FileNotFoundError` in `LungDataset`
- Verify `input/dataset/images` and `input/dataset/masks` exist.
- Ensure split pair stems match actual `.png` file names.

## Torchvision pretrained download issues
- For offline environments, use checkpoints trained earlier.
- Use `pretrained=False` for model initialization in tests.

## CUDA not available
- Run scripts with `--cpu`.
- Confirm torch installation includes CUDA build if GPU is expected.

## Metrics unexpectedly low
- Check mask interpolation mode (must be nearest for masks).
- Validate split quality with `scripts/preview_dataset.py`.
