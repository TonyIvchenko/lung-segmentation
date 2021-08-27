# CI Guide

GitHub Actions workflow: `.github/workflows/tests.yml`

## Trigger conditions
- Pushes to `master`
- Pull requests

## CI steps
1. Checkout repository
2. Setup Python
3. Install test dependency (`pytest`)
4. Run unit tests

## Extending CI
- Add linting stage
- Add matrix for multiple Python versions
- Add optional torch-enabled job for full runtime tests
