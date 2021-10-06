# Batch Inference Strategy

Current `scripts/predict.py` is single-image oriented.

For bulk inference:

1. Iterate image paths in a shell or Python wrapper.
2. Call `scripts/predict.py` with stable output naming.
3. Store masks and overlays in separate folders.
4. Run post-processing/evaluation from saved outputs.

A future enhancement can add native directory-level batch inference.
