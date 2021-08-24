# Augmentation Guide

Paired transforms keep image and mask operations synchronized:

- `Pad(max_padding)`
- `Crop(max_shift)`
- `RandomHorizontalFlipPair(p)`
- `RandomVerticalFlipPair(p)`
- `Resize(output_size)`
- `ComposePair([...])`

## Safety guarantees
- `Pad` and `Crop` support zero values without errors.
- `Resize` uses bilinear interpolation for images and nearest for masks.
- Masks remain binary after thresholding in `LungDataset`.
