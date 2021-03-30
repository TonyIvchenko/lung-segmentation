import torch


def _flatten_batch(y_true, y_pred):
    if y_true.size(0) != y_pred.size(0):
        raise ValueError("y_true and y_pred must have the same batch size")

    num = y_true.size(0)
    y_true_flat = y_true.float().reshape(num, -1)
    y_pred_flat = y_pred.float().reshape(num, -1)
    return num, y_true_flat, y_pred_flat


def jaccard(y_true, y_pred, eps=1e-7):
    """Jaccard a.k.a IoU score for a batch of masks."""
    _, y_true_flat, y_pred_flat = _flatten_batch(y_true, y_pred)
    intersection = (y_true_flat * y_pred_flat).sum(dim=1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(dim=1)
    score = intersection / (union + eps)
    return score.mean()


def dice(y_true, y_pred, eps=1e-7):
    """Dice a.k.a F1 score for a batch of masks."""
    _, y_true_flat, y_pred_flat = _flatten_batch(y_true, y_pred)
    intersection = (y_true_flat * y_pred_flat).sum(dim=1)
    denominator = y_true_flat.sum(dim=1) + y_pred_flat.sum(dim=1) + eps
    score = (2 * intersection) / denominator
    return score.mean()
