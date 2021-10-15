import pytest


pytest.importorskip("torch")

from scripts import train


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
