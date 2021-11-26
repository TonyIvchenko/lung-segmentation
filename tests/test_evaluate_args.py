import pytest


pytest.importorskip("torch")

from scripts import evaluate


def test_evaluate_defaults_to_auto_model(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--checkpoint", "model.pt"])
    args = evaluate.parse_args()
    assert args.model == "auto"


def test_evaluate_parses_max_samples(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--checkpoint", "model.pt", "--max-samples", "12"],
    )
    args = evaluate.parse_args()
    assert args.max_samples == 12
