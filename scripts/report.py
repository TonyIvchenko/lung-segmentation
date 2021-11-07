#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize training history JSON")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(history):
    best = max(history, key=lambda row: row["val_jaccard"])
    return {
        "epochs": len(history),
        "best_epoch": best["epoch"],
        "best_val_jaccard": best["val_jaccard"],
        "best_val_dice": best["val_dice"],
        "best_val_loss": best["val_loss"],
    }


def main():
    args = parse_args()

    with open(args.history, "r", encoding="utf-8") as history_file:
        history = json.load(history_file)

    summary = summarize(history)
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)


if __name__ == "__main__":
    main()
