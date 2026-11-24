#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Start a local MLflow tracking server with a Model Registry.
#
#   bash scripts/launch_mlflow.sh
#
# The Model Registry requires a SQL backend store: this script starts a
# PostgreSQL container and points MLflow at it. Without the registry the
# experiments are still tracked, but `mlflow.register_model` fails.
# --------------------------------------------------------------------------- #
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker is required to start the local tracking server."
    echo "   Alternative: pip install mlflow && mlflow server --host 0.0.0.0 --port 5000"
    exit 1
fi

echo ">> Starting PostgreSQL (MLflow backend store)"
docker compose up -d postgres

echo ">> Waiting for PostgreSQL"
until docker compose exec -T postgres pg_isready -U mlflow -d mlflow >/dev/null 2>&1; do
    sleep 2
done

echo ">> Starting the MLflow server"
docker compose up -d mlflow

echo
echo "✅ MLflow is starting on http://localhost:5000"
echo "   Export the tracking URI before training:"
echo "     export PPE_MLFLOW_TRACKING_URI=http://localhost:5000"
