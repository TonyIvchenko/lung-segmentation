# Model Variants

Two model families are supported in code and CLI:

## `unet`
- Encoder-decoder UNet from `src.models.UNet`
- Supports optional batch norm
- Configurable upsampling mode

## `pretrained-unet`
- VGG11 encoder backbone via `src.models.PretrainedUNet`
- Supports pretrained or randomly initialized encoder
- Compatible with both old and new torchvision weight APIs

## Selection
Use `--model unet` or `--model pretrained-unet` in training/evaluation/prediction scripts.
