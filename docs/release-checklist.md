# Release Checklist

Before tagging a release:

- [ ] `make smoke` passes
- [ ] `make test` passes
- [ ] README and docs reflect current CLI flags
- [ ] Example commands run against a sample dataset
- [ ] New checkpoints include metadata (`args`, `metrics`, `history`)
- [ ] CI workflow status is green
