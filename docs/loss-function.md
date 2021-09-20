# Loss Function Notes

Training and evaluation currently use categorical cross-entropy (`torch.nn.functional.cross_entropy`).

## Rationale
- Stable baseline for 2-class pixel segmentation
- Directly compatible with integer class masks
- Avoids explicit softmax in model forward pass

## Extensions
Potential future additions:
- Dice loss
- Focal loss
- Composite CE + Dice objective
