#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data import ComposePair, LungDataset, Resize, blend


def parse_args():
    parser = argparse.ArgumentParser(description="Export a preview image of random dataset samples")
    parser.add_argument("--data-folder", type=Path, default=Path("input/dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits.pk"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("images/dataset-preview.png"))
    return parser.parse_args()


def load_pairs(path):
    import pickle

    with open(path, "rb") as split_file:
        splits = pickle.load(split_file)
    return splits


def main():
    args = parse_args()
    random.seed(args.seed)

    splits = load_pairs(args.splits)
    transforms = ComposePair([Resize((args.image_size, args.image_size))])
    dataset = LungDataset(
        splits[args.split],
        args.data_folder / "images",
        args.data_folder / "masks",
        transforms=transforms,
    )

    indices = random.sample(range(len(dataset)), k=min(args.count, len(dataset)))
    samples = [dataset[index] for index in indices]

    columns = 3
    rows = (len(samples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    flat_axes = np.array(axes).reshape(-1)
    for axis in flat_axes:
        axis.axis("off")

    for axis, (origin, mask) in zip(flat_axes, samples):
        axis.imshow(blend(origin, mask2=mask))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"saved preview to {args.output}")


if __name__ == "__main__":
    main()
