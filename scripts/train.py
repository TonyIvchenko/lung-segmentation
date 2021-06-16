#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.checkpoints import build_model, save_checkpoint
from src.data import (
    ComposePair,
    Crop,
    LungDataset,
    Pad,
    RandomHorizontalFlipPair,
    RandomVerticalFlipPair,
    Resize,
)
from src.metrics import dice_from_logits, jaccard_from_logits
from src.utils import resolve_device, set_seed


def build_transforms(image_size):
    train_transforms = ComposePair(
        [
            Pad(max_padding=24),
            Crop(max_shift=24),
            RandomHorizontalFlipPair(p=0.5),
            RandomVerticalFlipPair(p=0.1),
            Resize((image_size, image_size)),
        ]
    )
    eval_transforms = ComposePair([Resize((image_size, image_size))])
    return train_transforms, eval_transforms


def evaluate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    val_jaccard = 0.0
    val_dice = 0.0
    total = 0

    with torch.no_grad():
        for origins, masks in dataloader:
            origins = origins.to(device)
            masks = masks.to(device)

            logits = model(origins)
            loss = torch.nn.functional.cross_entropy(logits, masks)

            num = origins.size(0)
            total += num
            val_loss += loss.item() * num
            val_jaccard += jaccard_from_logits(masks.float(), logits).item() * num
            val_dice += dice_from_logits(masks.float(), logits).item() * num

    return {
        "loss": val_loss / total,
        "jaccard": val_jaccard / total,
        "dice": val_dice / total,
    }


def train(args):
    set_seed(args.seed)

    data_folder = Path(args.data_folder)
    origins_folder = data_folder / "images"
    masks_folder = data_folder / "masks"

    with open(args.splits, "rb") as split_file:
        splits = pickle.load(split_file)

    train_transforms, eval_transforms = build_transforms(args.image_size)

    datasets = {
        "train": LungDataset(
            splits["train"],
            origins_folder,
            masks_folder,
            transforms=train_transforms,
        ),
        "val": LungDataset(
            splits["val"],
            origins_folder,
            masks_folder,
            transforms=eval_transforms,
        ),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }

    device = resolve_device(prefer_cuda=not args.cpu)

    model = build_model(
        args.model,
        batch_norm=args.batch_norm,
        upscale_mode=args.upscale_mode,
        pretrained=args.pretrained_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_jaccard = float("-inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0

        for origins, masks in dataloaders["train"]:
            origins = origins.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(origins)
            loss = torch.nn.functional.cross_entropy(logits, masks)
            loss.backward()
            optimizer.step()

            num = origins.size(0)
            seen += num
            train_loss += loss.item() * num

        val_metrics = evaluate(model, dataloaders["val"], device)
        log_line = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "val_loss": val_metrics["loss"],
            "val_jaccard": val_metrics["jaccard"],
            "val_dice": val_metrics["dice"],
        }
        history.append(log_line)

        print(
            "epoch: {epoch}/{total}, train loss: {train_loss:.6f}, "
            "val loss: {val_loss:.6f}, val jaccard: {val_jaccard:.6f}, "
            "val dice: {val_dice:.6f}".format(
                epoch=epoch,
                total=args.epochs,
                train_loss=log_line["train_loss"],
                val_loss=log_line["val_loss"],
                val_jaccard=log_line["val_jaccard"],
                val_dice=log_line["val_dice"],
            )
        )

        if log_line["val_jaccard"] >= best_val_jaccard:
            best_val_jaccard = log_line["val_jaccard"]
            save_checkpoint(
                path=args.output,
                model=model,
                args=vars(args),
                metrics={"best_val_jaccard": best_val_jaccard},
                history=history,
            )
            print(f"saved checkpoint to {args.output}")

    if args.history_output is not None:
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_output, "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train lung segmentation models")
    parser.add_argument("--data-folder", type=Path, default=Path("input/dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits.pk"))
    parser.add_argument("--output", type=Path, default=Path("models/unet-cli.pt"))
    parser.add_argument("--history-output", type=Path)
    parser.add_argument("--model", choices=["unet", "pretrained-unet"], default="pretrained-unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upscale-mode", default="bilinear")
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--pretrained-encoder", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.model == "pretrained-unet" and not args.pretrained_encoder:
        # Keep backward-compatible default behavior for pretrained UNet.
        args.pretrained_encoder = True

    return args


if __name__ == "__main__":
    train(parse_args())
