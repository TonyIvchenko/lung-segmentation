import pytest


torch = pytest.importorskip("torch")

from src.analysis import confusion_counts, precision_recall_f1


def test_confusion_counts_binary_masks():
    y_true = torch.tensor([[1, 0, 1, 0]])
    y_pred = torch.tensor([[1, 1, 0, 0]])

    counts = confusion_counts(y_true, y_pred)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_precision_recall_f1_values():
    y_true = torch.tensor([[1, 0, 1, 0]])
    y_pred = torch.tensor([[1, 1, 0, 0]])

    metrics = precision_recall_f1(y_true, y_pred)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
