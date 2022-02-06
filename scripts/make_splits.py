#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test splits for lung segmentation")
    parser.add_argument("--data-folder", type=Path, default=Path("input/dataset"))
    parser.add_argument("--output", type=Path, default=Path("splits.pk"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--mask-ext", default=".png")
    parser.add_argument(
        "--strict-pairs",
        action="store_true",
        help="Fail if any mask does not have a matching origin image.",
    )
    return parser.parse_args()


def origin_name_from_mask(mask_stem, mask_suffix):
    if not mask_suffix:
        return mask_stem
    if not mask_stem.endswith(mask_suffix):
        raise ValueError(
            f"mask stem '{mask_stem}' does not end with configured suffix '{mask_suffix}'"
        )
    return mask_stem[: -len(mask_suffix)]


def main():
    args = parse_args()
    from sklearn.model_selection import train_test_split

    origins_folder = args.data_folder / "images"
    masks_folder = args.data_folder / "masks"

    origins = {path.stem for path in origins_folder.glob(f"*{args.image_ext}")}
    masks = sorted(path.stem for path in masks_folder.glob(f"*{args.mask_ext}"))
    origin_mask_pairs = [
        (origin_name_from_mask(mask_stem, args.mask_suffix), mask_stem)
        for mask_stem in masks
    ]

    filtered_pairs = [pair for pair in origin_mask_pairs if pair[0] in origins]
    if len(filtered_pairs) != len(origin_mask_pairs):
        missing = len(origin_mask_pairs) - len(filtered_pairs)
        if args.strict_pairs:
            raise ValueError(f"found {missing} masks without matching origin image")
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
