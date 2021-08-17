# Data Format Reference

Expected dataset structure:

```text
input/dataset/
  images/
    <image_id>.png
  masks/
    <image_id>_mask.png
```

## Constraints
- Origins must be grayscale-compatible PNG files.
- Masks are thresholded to binary labels (`0`, `1`) in `LungDataset`.
- Each pair in `splits.pk` stores `(origin_stem, mask_stem)` without `.png`.

## Validation behavior
`LungDataset` raises `FileNotFoundError` for missing source files to fail fast.
