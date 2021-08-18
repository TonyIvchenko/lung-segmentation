# Metrics Guide

Implemented segmentation metrics:

- `jaccard` (IoU)
- `dice` (F1-style overlap)
- logits wrappers: `jaccard_from_logits`, `dice_from_logits`

Additional analysis helpers:
- confusion totals (`tp`, `fp`, `fn`, `tn`)
- `precision_recall_f1`

## Practical notes
- Metrics flatten per-image masks before aggregation.
- Metric helpers cast inputs to float for robust dtype handling.
- Evaluation CLI reports overlap and confusion-derived quality scores.
