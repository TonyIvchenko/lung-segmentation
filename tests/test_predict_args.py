import pytest


pytest.importorskip("torch")

from scripts import predict


def test_predict_defaults_to_auto_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict.py",
            "--checkpoint",
            "model.pt",
            "--image",
            "image.png",
            "--output-mask",
            "mask.png",
        ],
    )
    args = predict.parse_args()
    assert args.model == "auto"


def test_predict_accepts_probability_output(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict.py",
            "--checkpoint",
            "model.pt",
            "--image",
            "image.png",
            "--output-mask",
            "mask.png",
            "--output-probability",
            "prob.png",
        ],
    )
    args = predict.parse_args()
    assert str(args.output_probability).endswith("prob.png")
