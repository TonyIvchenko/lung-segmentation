# Performance Tips

## Data loading
- Increase `--num-workers` for faster host-side preprocessing.
- Keep dataset on fast local storage.

## Training
- Use `--batch-size` as high as memory allows.
- Use scheduler (`--lr-step-size`, `--lr-gamma`) for stable long runs.
- Enable gradient clipping (`--grad-clip`) when training becomes unstable.

## Evaluation
- Batch inference with `--batch-size` to improve throughput.
