import pytest


torch = pytest.importorskip("torch")

from src.metrics import dice, dice_from_logits, jaccard, jaccard_from_logits


def test_jaccard_perfect_prediction():
    y_true = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    y_pred = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    assert jaccard(y_true, y_pred).item() == pytest.approx(1.0)


def test_dice_perfect_prediction():
    y_true = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    y_pred = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
    assert dice(y_true, y_pred).item() == pytest.approx(1.0)


def test_logits_wrappers_match_binary_scores():
    y_true = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)
    logits = torch.tensor(
        [[
            [[5.0, 1.0], [1.0, 5.0]],
            [[1.0, 5.0], [5.0, 1.0]],
        ]]
    )

    assert jaccard_from_logits(y_true, logits).item() == pytest.approx(1.0)
    assert dice_from_logits(y_true, logits).item() == pytest.approx(1.0)
