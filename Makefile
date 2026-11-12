# Run inside WSL/Linux. LightGBM and implicit do not install on native ARM64 Windows.
PYTHON ?= python
DATA_DIR ?= data

.PHONY: data test lint baselines

data:
	$(PYTHON) scripts/download_data.py --data-dir $(DATA_DIR)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --data-dir $(DATA_DIR) --output artifacts/baselines.md
