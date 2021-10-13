import pytest


pytest.importorskip("torch")

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "train_cli",
    Path(__file__).resolve().parents[1] / "scripts" / "train.py",
)
train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train)


def test_pretrained_unet_defaults_to_pretrained_encoder(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--model", "pretrained-unet"],
    )
    args = train.parse_args()
    assert args.pretrained_encoder is True


def test_no_pretrained_encoder_flag_overrides_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--model", "pretrained-unet", "--no-pretrained-encoder"],
    )
    args = train.parse_args()
    assert args.pretrained_encoder is False
