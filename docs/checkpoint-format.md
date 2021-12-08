# Checkpoint Format

Saved checkpoints are standard PyTorch dictionaries.

## Keys
- `model`: state dict (required)
- `args`: serialized training arguments (optional)
- `metrics`: summary metrics like `best_val_jaccard` (optional)
- `history`: epoch-by-epoch metrics (optional)

## Produced by
- `scripts/train.py`
- `src.checkpoints.save_checkpoint`

## Consumed by
- `scripts/evaluate.py`
- `scripts/predict.py`
- `src.checkpoints.load_checkpoint`

## Compatibility
The loader can infer model type from checkpoint `args` when `--model auto` is used.
Batch norm and upsampling mode can also be inferred from checkpoint args metadata.
