# Model Selection Guidance

## Start with `pretrained-unet` when:
- Training data is limited
- Faster convergence is needed
- You can access pretrained encoder weights

## Start with `unet` when:
- You need fully offline reproducibility
- You want architecture simplicity
- You are benchmarking augmentation/loss effects first

Compare models using the same split, seed, and evaluation script outputs.
