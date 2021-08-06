import torch


def confusion_counts(y_true, y_pred, positive_class=1):
    y_true = (y_true == positive_class).long()
    y_pred = (y_pred == positive_class).long()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true, y_pred, eps=1e-7):
    counts = confusion_counts(y_true, y_pred)
    return precision_recall_f1_from_counts(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        eps=eps,
    )


def precision_recall_f1_from_counts(tp, fp, fn, eps=1e-7):
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
