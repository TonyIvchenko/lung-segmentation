from scripts import report


def test_report_requires_history(monkeypatch):
    monkeypatch.setattr("sys.argv", ["report.py", "--history", "history.json"])
    args = report.parse_args()
    assert str(args.history).endswith("history.json")


def test_report_optional_output(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["report.py", "--history", "history.json", "--output", "summary.json"],
    )
    args = report.parse_args()
    assert str(args.output).endswith("summary.json")
