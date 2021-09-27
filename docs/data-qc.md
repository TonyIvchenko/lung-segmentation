# Data Quality Checklist

Before training:

- Verify image/mask counts are close (allowing unlabeled images).
- Check random overlays with `scripts/preview_dataset.py`.
- Confirm masks are not empty for most samples.
- Ensure split file uses current dataset version.
- Rebuild splits if major dataset changes were made.
