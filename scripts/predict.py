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


def validate_prediction_args(args, parser):
    if args.image is not None:
        if args.output_mask is None:
            parser.error("--output-mask is required when --image is provided")
        if args.output_mask_dir is not None:
            parser.error("--output-mask-dir cannot be used with --image")
        if args.output_overlay_dir is not None or args.output_probability_dir is not None:
            parser.error("directory outputs require --image-dir")
        return

    if args.output_mask_dir is None:
        parser.error("--output-mask-dir is required when --image-dir is provided")
    if args.output_mask is not None:
        parser.error("--output-mask cannot be used with --image-dir")
    if args.output_overlay is not None or args.output_probability is not None:
        parser.error("single-image outputs cannot be used with --image-dir")


def infer_mask_and_probability(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1)[0].cpu()
        probability = torch.softmax(logits, dim=1)[0, 1].cpu()
    return prediction, probability


def save_single_prediction(origin, prediction, probability, output_mask, output_overlay=None, output_probability=None):
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask_image = torchvision.transforms.functional.to_pil_image(prediction.float())
    mask_image.save(output_mask)

    if output_overlay is not None:
        output_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = blend(origin, mask2=prediction)
        overlay.save(output_overlay)

    if output_probability is not None:
        output_probability.parent.mkdir(parents=True, exist_ok=True)
        prob_image = torchvision.transforms.functional.to_pil_image(probability)
        prob_image.save(output_probability)


def parse_args():
    parser = argparse.ArgumentParser(description="Run lung segmentation inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=Path)
    source_group.add_argument("--image-dir", type=Path)
    parser.add_argument("--output-mask", type=Path)
    parser.add_argument("--output-mask-dir", type=Path)
    parser.add_argument("--output-overlay", type=Path)
    parser.add_argument("--output-overlay-dir", type=Path)
    parser.add_argument("--output-probability", type=Path)
    parser.add_argument("--output-probability-dir", type=Path)
    parser.add_argument("--glob", default="*.png")
    parser.add_argument("--model", choices=["auto", "unet", "pretrained-unet"], default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    validate_prediction_args(args, parser)
    return args


def main():
    args = parse_args()
    device = resolve_device(prefer_cuda=not args.cpu)
    model_name = None if args.model == "auto" else args.model

    model, _ = load_checkpoint(
        path=args.checkpoint,
        model_name=model_name,
        device=device,
        batch_norm=False,
    )

    if args.image is not None:
        origin = preprocess_image(args.image, args.image_size)
        input_tensor = origin.unsqueeze(0).to(device)
        prediction, probability = infer_mask_and_probability(model, input_tensor)
        save_single_prediction(
            origin=origin,
            prediction=prediction,
            probability=probability,
            output_mask=args.output_mask,
            output_overlay=args.output_overlay,
            output_probability=args.output_probability,
        )
        return

    args.output_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.output_overlay_dir is not None:
        args.output_overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.output_probability_dir is not None:
        args.output_probability_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.image_dir.glob(args.glob))
    if not image_paths:
        raise FileNotFoundError(
            f"No files matching '{args.glob}' found in directory: {args.image_dir}"
        )

    for image_path in image_paths:
        origin = preprocess_image(image_path, args.image_size)
        input_tensor = origin.unsqueeze(0).to(device)
        prediction, probability = infer_mask_and_probability(model, input_tensor)
        output_overlay = None
        if args.output_overlay_dir is not None:
            output_overlay = args.output_overlay_dir / image_path.name
        output_probability = None
        if args.output_probability_dir is not None:
            output_probability = args.output_probability_dir / image_path.name
        save_single_prediction(
            origin=origin,
            prediction=prediction,
            probability=probability,
            output_mask=args.output_mask_dir / image_path.name,
            output_overlay=output_overlay,
            output_probability=output_probability,
        )
    print(f"saved predictions for {len(image_paths)} images to {args.output_mask_dir}")


if __name__ == "__main__":
    main()
