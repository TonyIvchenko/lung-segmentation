#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.checkpoints import load_checkpoint
from src.data import ComposePair, LungDataset, Resize
from src.metrics import dice_from_logits, jaccard_from_logits
from src.utils import resolve_device


def evaluate(model, dataloader, device):
    total = 0
    total_loss = 0.0
    total_jaccard = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for origins, masks in dataloader:
            origins = origins.to(device)
            masks = masks.to(device)

            logits = model(origins)
            num = origins.size(0)

            total += num
            total_loss += torch.nn.functional.cross_entropy(logits, masks).item() * num
            total_jaccard += jaccard_from_logits(masks.float(), logits).item() * num
            total_dice += dice_from_logits(masks.float(), logits).item() * num

    return {
        "loss": total_loss / total,
        "jaccard": total_jaccard / total,
        "dice": total_dice / total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lung segmentation checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-folder", type=Path, default=Path("input/dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits.pk"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--model", choices=["auto", "unet", "pretrained-unet"], default="auto")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


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

    with open(args.splits, "rb") as split_file:
        splits = pickle.load(split_file)

    transforms = ComposePair([Resize((args.image_size, args.image_size))])
    dataset = LungDataset(
        splits[args.split],
        args.data_folder / "images",
        args.data_folder / "masks",
        transforms=transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate(model, dataloader, device)
    print(
        "{split} metrics: loss={loss:.6f}, jaccard={jaccard:.6f}, dice={dice:.6f}".format(
            split=args.split,
            loss=metrics["loss"],
            jaccard=metrics["jaccard"],
            dice=metrics["dice"],
        )
    )


if __name__ == "__main__":
    main()
