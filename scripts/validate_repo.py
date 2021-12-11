#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED_PATHS = [
    Path("src/data.py"),
    Path("src/models.py"),
    Path("src/metrics.py"),
    Path("scripts/train.py"),
    Path("scripts/evaluate.py"),
    Path("scripts/predict.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate repository structure and Python syntax")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        print("missing required paths:")
        for path in missing:
            print(f"- {path}")
        if args.strict:
            return 1

    result = subprocess.run(
        [sys.executable, "-m", "compileall", "src", "scripts", "tests"],
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    print("repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
