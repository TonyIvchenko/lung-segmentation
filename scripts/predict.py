#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image

from src.checkpoints import load_checkpoint
from src.data import blend
from src.utils import resolve_device


def preprocess_image(image_path, image_size):
    origin = Image.open(image_path).convert("L")
    resized = torchvision.transforms.functional.resize(
        origin,
        (image_size, image_size),
        interpolation=Image.BILINEAR,
    )
    tensor = torchvision.transforms.functional.to_tensor(resized) - 0.5
    return tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Run lung segmentation inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path)
    parser.add_argument("--model", choices=["unet", "pretrained-unet"], default="pretrained-unet")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(prefer_cuda=not args.cpu)

    model, _ = load_checkpoint(
        path=args.checkpoint,
        model_name=args.model,
        device=device,
        batch_norm=args.model == "pretrained-unet",
        upscale_mode="bilinear",
    )

    origin = preprocess_image(args.image, args.image_size)
    input_tensor = origin.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1)[0].cpu()

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask_image = torchvision.transforms.functional.to_pil_image(prediction.float())
    mask_image.save(args.output_mask)

    if args.output_overlay is not None:
        args.output_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = blend(origin, mask2=prediction)
        overlay.save(args.output_overlay)


if __name__ == "__main__":
    main()
