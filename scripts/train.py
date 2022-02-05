#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.checkpoints import build_model, load_checkpoint, save_checkpoint
from src.config import TrainConfig
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


def iterate_batches(dataloader, progress, desc):
    if not progress:
        return dataloader

    from tqdm.auto import tqdm

    return tqdm(dataloader, desc=desc, leave=False)


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


def evaluate(model, dataloader, device, progress=False):
    model.eval()
    val_loss = 0.0
    val_jaccard = 0.0
    val_dice = 0.0
    total = 0

    with torch.no_grad():
        for origins, masks in iterate_batches(dataloader, progress=progress, desc="val"):
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
    if args.max_samples is not None:
        datasets["train"] = Subset(
            datasets["train"],
            range(min(args.max_samples, len(datasets["train"]))),
        )
        datasets["val"] = Subset(
            datasets["val"],
            range(min(args.max_samples, len(datasets["val"]))),
        )

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    best_val_jaccard = float("-inf")
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    if args.resume is not None:
        model, checkpoint = load_checkpoint(
            path=args.resume,
            model_name=args.model,
            device=device,
            batch_norm=args.batch_norm,
            upscale_mode=args.upscale_mode,
        )
        history = checkpoint.get("history", [])
        best_val_jaccard = checkpoint.get("metrics", {}).get(
            "best_val_jaccard",
            best_val_jaccard,
        )
        start_epoch = len(history) + 1
        print(f"resumed from {args.resume}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0

        for origins, masks in iterate_batches(dataloaders["train"], progress=args.progress, desc="train"):
            origins = origins.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(origins)
            loss = torch.nn.functional.cross_entropy(logits, masks)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            num = origins.size(0)
            seen += num
            train_loss += loss.item() * num

        val_metrics = evaluate(model, dataloaders["val"], device, progress=args.progress)
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
        scheduler.step()

        if log_line["val_jaccard"] >= best_val_jaccard:
            best_val_jaccard = log_line["val_jaccard"]
            epochs_without_improvement = 0
            save_checkpoint(
                path=args.output,
                model=model,
                args=vars(args),
                metrics={"best_val_jaccard": best_val_jaccard},
                history=history,
            )
            print(f"saved checkpoint to {args.output}")
        else:
            epochs_without_improvement += 1

        if args.save_every > 0 and epoch % args.save_every == 0:
            snapshot_path = args.output.with_name(
                f"{args.output.stem}-epoch-{epoch}{args.output.suffix}"
            )
            save_checkpoint(
                path=snapshot_path,
                model=model,
                args=vars(args),
                metrics={"epoch": epoch, "val_jaccard": log_line["val_jaccard"]},
                history=history,
            )
            print(f"saved snapshot checkpoint to {snapshot_path}")

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                "early stopping triggered after {count} epochs without validation "
                "Jaccard improvement".format(count=epochs_without_improvement)
            )
            break

    if args.history_output is not None:
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_output, "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=2)

    if args.history_csv is not None:
        args.history_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.history_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["epoch", "train_loss", "val_loss", "val_jaccard", "val_dice"],
            )
            writer.writeheader()
            writer.writerows(history)


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train lung segmentation models")
    parser.add_argument("--data-folder", type=Path, default=defaults.data_folder)
    parser.add_argument("--splits", type=Path, default=defaults.splits)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--history-output", type=Path)
    parser.add_argument("--history-csv", type=Path)
    parser.add_argument("--model", choices=["unet", "pretrained-unet"], default=defaults.model)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--upscale-mode", default=defaults.upscale_mode)
    parser.add_argument("--batch-norm", action="store_true")
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained-encoder",
        dest="pretrained_encoder",
        action="store_true",
    )
    pretrained_group.add_argument(
        "--no-pretrained-encoder",
        dest="pretrained_encoder",
        action="store_false",
    )
    parser.set_defaults(pretrained_encoder=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr-step-size", type=int, default=25)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if args.model == "pretrained-unet" and args.pretrained_encoder is None:
        # Keep backward-compatible default behavior for pretrained UNet.
        args.pretrained_encoder = True
    elif args.model == "unet":
        args.pretrained_encoder = False

    return args


if __name__ == "__main__":
    train(parse_args())
