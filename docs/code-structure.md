# Code Structure

## `src/`
- `data.py`: dataset and paired transforms
- `models.py`: UNet variants
- `metrics.py`: Dice/IoU and logits helpers
- `analysis.py`: confusion-derived metrics
- `checkpoints.py`: model save/load helpers
- `utils.py`: seed/device/parameter utilities
- `config.py`: typed training defaults

## `scripts/`
- executable workflow entry points

## `tests/`
- unit tests for core modules
