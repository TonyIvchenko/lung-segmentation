# Testing Guide

Run tests locally:

```bash
pytest
```

Or use Make:

```bash
make test
make smoke
```

## Current test coverage
- Metrics behavior
- Dataset loading and file validation
- Transform edge cases
- Model output shape contracts
- Utility helpers and analysis functions

Tests use `pytest.importorskip` for optional heavy dependencies.
