# Experiment Tracking

Recommended artifacts per run:

- Training checkpoint (`--output`)
- JSON history (`--history-output`)
- CSV history (`--history-csv`)
- Evaluation JSON (`--output-json`)
- Per-sample evaluation CSV (`--output-samples-csv`)

Store each run under a dedicated folder like `outputs/<run-id>/`.
Use run IDs that include model type and timestamp.
