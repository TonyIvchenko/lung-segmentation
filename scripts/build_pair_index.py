#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from scripts.index_helpers import index_summary
from scripts.make_splits import origin_name_from_mask


def build_pairs(
    images_folder,
    masks_folder,
    image_ext=".png",
    mask_ext=".png",
    mask_suffix="",
    strict=False,
):
    images_folder = Path(images_folder)
    masks_folder = Path(masks_folder)
    image_names = {path.stem for path in images_folder.glob(f"*{image_ext}")}
    mask_names = sorted(path.stem for path in masks_folder.glob(f"*{mask_ext}"))

    pairs = []
    missing = []
    for mask_name in mask_names:
        origin_name = origin_name_from_mask(mask_name, mask_suffix)
        if origin_name in image_names:
            pairs.append([origin_name, mask_name])
        else:
            missing.append(mask_name)

    if strict and missing:
        raise ValueError(f"found {len(missing)} masks without matching image")

    return pairs, missing


def parse_args():
    parser = argparse.ArgumentParser(description="Build image/mask pair index for training")
    parser.add_argument("--images-folder", type=Path, required=True)
    parser.add_argument("--masks-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ext", default=".png")
    parser.add_argument("--mask-ext", default=".png")
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs, missing = build_pairs(
        images_folder=args.images_folder,
        masks_folder=args.masks_folder,
        image_ext=args.image_ext,
        mask_ext=args.mask_ext,
        mask_suffix=args.mask_suffix,
        strict=args.strict,
    )

    payload = {
        "images_folder": str(args.images_folder),
        "masks_folder": str(args.masks_folder),
        "pairs": pairs,
        "missing_masks": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = index_summary(payload)
    print(
        "saved pair index to {path}: pairs={pairs}, missing_masks={missing}, missing_ratio={ratio:.3f}".format(
            path=args.output,
            pairs=summary["pairs"],
            missing=summary["missing_masks"],
            ratio=summary["missing_ratio"],
        )
    )


if __name__ == "__main__":
    main()
