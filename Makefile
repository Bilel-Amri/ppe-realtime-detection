# --------------------------------------------------------------------------- #
# Convenience targets for the PPE detection project.
# --------------------------------------------------------------------------- #
.PHONY: help install check data train evaluate app mlflow samples report docker clean

PYTHON ?= python3
VENV   ?= .venv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install the dependencies
	bash scripts/bootstrap.sh

check:  ## Run every offline check (no GPU, no network)
	bash scripts/run_checks.sh

data:  ## Download and prepare the dataset
	$(PYTHON) scripts/download_dataset.py --kaggle
	$(PYTHON) scripts/prepare_dataset.py

train:  ## Fine-tune YOLO11 on the dataset
	$(PYTHON) -m src.models.train

evaluate:  ## Test-split evaluation + ONNX export + MLflow logging
	$(PYTHON) -m src.models.evaluate

app:  ## Run the API locally
	uvicorn backend.app.main:app --reload --port 8000

mlflow:  ## Start the MLflow tracking server + Model Registry
	bash scripts/launch_mlflow.sh

samples:  ## Extract the sample gallery from the dataset
	$(PYTHON) scripts/extract_samples.py --count 8

report:  ## Build the technical report (PDF)
	bash scripts/build_report.sh

docker:  ## Start the whole stack
	docker compose up --build

clean:  ## Remove the caches and the build artefacts
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache reports/*.aux reports/*.log reports/*.out
