# Run inside WSL/Linux. LightGBM and implicit do not install on native ARM64 Windows.
PYTHON ?= python
DATA_DIR ?= data

.PHONY: data test lint baselines retrieval ranking experiment bundle serve benchmark

data:
	$(PYTHON) scripts/download_data.py --data-dir $(DATA_DIR)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

baselines:
	PYTHONPATH=src $(PYTHON) scripts/run_baselines.py --data-dir $(DATA_DIR) --output artifacts/baselines.md

retrieval:
	PYTHONPATH=src $(PYTHON) scripts/run_retrieval.py --data-dir $(DATA_DIR) --output artifacts/retrieval.md

ranking:
	PYTHONPATH=src $(PYTHON) scripts/run_ranking.py --data-dir $(DATA_DIR) --output artifacts/ranking.md

experiment:
	PYTHONPATH=src $(PYTHON) scripts/run_experiment.py --data-dir $(DATA_DIR) --output artifacts/experiment.md

bundle:
	PYTHONPATH=src $(PYTHON) scripts/build_serving_bundle.py --data-dir $(DATA_DIR) --output artifacts/serving_bundle.joblib

serve:
	PYTHONPATH=src BUNDLE_PATH=artifacts/serving_bundle.joblib $(PYTHON) -m uvicorn recsys.serving.api:create_app_from_env --factory --port 8000

benchmark:
	PYTHONPATH=src $(PYTHON) scripts/benchmark_serving.py --output artifacts/serving.md
