# Seed Policy

Default seed is `42` across scripts unless overridden.

## What is seeded
- Python `random`
- NumPy random generator
- Torch CPU generator
- Torch CUDA generators (if available)

## Why this matters
Consistent seeds reduce variance when comparing architecture or hyperparameter changes.
