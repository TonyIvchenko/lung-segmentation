# Hardware Notes

## GPU
- Default scripts prefer CUDA when available.
- Use `--cpu` to force CPU inference or evaluation.

## Memory
- `--batch-size` is the primary VRAM lever.
- Lower image size (`--image-size`) if memory is constrained.

## Throughput
- Increase `--num-workers` for faster data loading on multi-core systems.
- Use `--progress` to monitor long epochs.
