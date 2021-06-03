PYTHON ?= python3

.PHONY: train evaluate predict make-splits preview test smoke

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

predict:
	$(PYTHON) scripts/predict.py

make-splits:
	$(PYTHON) scripts/make_splits.py

preview:
	$(PYTHON) scripts/preview_dataset.py

test:
	pytest

smoke:
	$(PYTHON) -m compileall src scripts tests
