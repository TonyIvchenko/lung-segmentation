#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for lung segmentation")
    parser.add_argument("--data-folder", type=Path, default=Path("input/dataset"))
    parser.add_argument("--output", type=Path, default=Path("splits.pk"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    origins_folder = args.data_folder / "images"
    masks_folder = args.data_folder / "masks"

    origins = {path.stem for path in origins_folder.glob("*.png")}
    masks = sorted(path.stem for path in masks_folder.glob("*.png"))
    origin_mask_pairs = [(mask.replace("_mask", ""), mask) for mask in masks]

    filtered_pairs = [pair for pair in origin_mask_pairs if pair[0] in origins]
    if len(filtered_pairs) != len(origin_mask_pairs):
        missing = len(origin_mask_pairs) - len(filtered_pairs)
        print(f"warning: dropped {missing} masks with missing origin image")

    train_pairs, test_pairs = train_test_split(
        filtered_pairs,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_pairs, val_pairs = train_test_split(
        train_pairs,
        test_size=args.val_size,
        random_state=args.seed,
    )

    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as split_file:
        pickle.dump(splits, split_file)

    print(
        "saved splits to {path}: train={train}, val={val}, test={test}".format(
            path=args.output,
            train=len(train_pairs),
            val=len(val_pairs),
            test=len(test_pairs),
        )
    )


if __name__ == "__main__":
    main()
