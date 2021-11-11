from scripts.report import summarize


def test_report_summarize_picks_best_val_jaccard():
    history = [
        {"epoch": 1, "val_jaccard": 0.8, "val_dice": 0.9, "val_loss": 0.2},
        {"epoch": 2, "val_jaccard": 0.85, "val_dice": 0.92, "val_loss": 0.18},
    ]

    summary = summarize(history)

    assert summary["epochs"] == 2
    assert summary["best_epoch"] == 2
    assert summary["best_val_jaccard"] == 0.85
