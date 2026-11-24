#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# One-command installation of the PPE detection project.
#
#   bash scripts/bootstrap.sh
#
# It creates the virtual environment, installs the dependencies, initialises
# DVC and prints the next steps.
# --------------------------------------------------------------------------- #
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=============================================================="
echo " PPE detection - bootstrap"
echo "=============================================================="

# --- 1. Python environment ------------------------------------------------ #
if [ ! -d ".venv" ]; then
    echo ">> Creating the virtual environment (.venv)"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> Upgrading pip"
python -m pip install --upgrade pip

echo ">> Installing the runtime dependencies"
pip install -r backend/requirements.txt

echo ">> Installing the development dependencies"
pip install -r requirements-dev.txt || echo "   (dev dependencies are optional)"

# --- 2. Directories ------------------------------------------------------- #
echo ">> Creating the working directories"
mkdir -p data/raw data/processed artifacts reports

# --- 3. DVC --------------------------------------------------------------- #
if command -v dvc >/dev/null 2>&1; then
    if [ ! -d ".dvc" ]; then
        echo ">> Initialising DVC"
        dvc init
    fi
    if ! dvc remote list 2>/dev/null | grep -q "local"; then
        echo ">> Adding the local DVC remote (../dvc_store)"
        dvc remote add -d local ../dvc_store
    fi
else
    echo "!! DVC is not installed - skipping the DVC initialisation"
fi

# --- 4. Sample gallery ---------------------------------------------------- #
if [ -d "data/processed/css-ppe/train/images" ]; then
    echo ">> Extracting the sample gallery"
    python scripts/extract_samples.py --count 8 || true
fi

# --- 5. Verification ------------------------------------------------------- #
echo
echo ">> Verifying the installation"
python scripts/verify_project.py || true

cat <<'MESSAGE'

==============================================================
 Next steps
==============================================================

 1. Get the dataset
      python scripts/download_dataset.py --kaggle

 2. Prepare it (re-split + rebalance)
      python scripts/prepare_dataset.py

 3. Train (needs a GPU for the full run)
      python -m src.models.train

 4. Evaluate on the test split, export ONNX, log to MLflow
      python -m src.models.evaluate

 5. Launch the application
      docker compose up --build
      -> frontend  http://localhost:8080
      -> API docs  http://localhost:8000/docs
      -> MLflow    http://localhost:5000
MESSAGE
