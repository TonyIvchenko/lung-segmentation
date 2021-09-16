# Migration Guide

## From notebook to CLI
1. Generate or reuse `splits.pk`.
2. Train with `scripts/train.py`.
3. Evaluate with `scripts/evaluate.py`.
4. Infer masks with `scripts/predict.py`.

## Benefits
- Versionable command history
- Easier automation
- Reproducible artifact layout

## Compatibility
Model files from notebook runs can still be evaluated with the CLI if state dict keys match.
