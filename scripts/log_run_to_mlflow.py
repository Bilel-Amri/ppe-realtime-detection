"""Log the Kaggle training run into the local MLflow server (Docker stack).

Usage (from the project root, with `docker compose up` running):

    pip install mlflow==2.16.2 pandas
    python scripts/log_run_to_mlflow.py

Reads:  artifacts/metrics.json, artifacts/train_results.csv,
        artifacts/runs/detect/train/weights/{best.pt,best.onnx},
        artifacts/figures/*.png
Result: one MLflow run (params, metrics, artifacts) + model registered as
        "PPE-YOLO11" and moved to the Staging stage.
"""
import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
WEIGHTS = ART / "runs" / "detect" / "train" / "weights"
URI = os.environ.get("PPE_MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "PPE-YOLO11"

metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))

params = {
    "base_model": metrics.get("base_model", "yolo11n.pt"),
    "imgsz": metrics.get("imgsz", 640),
    "opset": metrics.get("opset", 12),
    "dataset": metrics.get("dataset", "CSS"),
    "trained_on": "Kaggle Tesla T4",
}

csv_path = ART / "train_results.csv"
if csv_path.exists():
    import pandas as pd

    row = pd.read_csv(csv_path).iloc[-1].to_dict()
    for key in ("epochs", "batch", "rect", "fraction", "seed"):
        if key in row:
            params[key] = row[key]
    if "train_time" in row:
        params["train_time_s"] = row["train_time"]

mlflow.set_tracking_uri(URI)
mlflow.set_experiment("ppe-detection")

parser = argparse.ArgumentParser()
parser.add_argument("--register-only", metavar="RUN_ID",
                    help="Skip logging; only register the model of an existing run")
args = parser.parse_args()

client = MlflowClient()
print("MLflow client version:", mlflow.__version__)

if args.register_only:
    run_id = args.register_only
else:
    with mlflow.start_run(run_name="yolo11n_kaggle_run") as run:
        run_id = run.info.run_id
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(
            {
                "test_mAP50": metrics["test_mAP50"],
                "test_mAP50_95": metrics["test_mAP50_95"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
            }
        )
        for name, ap in (metrics.get("per_class_ap50_95") or {}).items():
            if ap is not None:
                mlflow.log_metric("ap50_95_" + name.replace(" ", "_").replace("-", "_"), ap)

        for f in (ART / "metrics.json", csv_path, WEIGHTS / "best.pt", WEIGHTS / "best.onnx"):
            if f.exists():
                mlflow.log_artifact(str(f))
        figures = ART / "figures"
        if figures.exists():
            for img in figures.glob("*.png"):
                mlflow.log_artifact(str(img), artifact_path="figures")
    print("Run logged:", run_id)

# --- Model Registry -------------------------------------------------------- #
# We register with the run's artifact URI directly. The `runs:/` shortcut of a
# recent MLflow client calls an endpoint (logged-models) that the 2.16.2
# server in Docker does not have, which gives a 404.
run_info = client.get_run(run_id).info
source = f"{run_info.artifact_uri.rstrip('/')}/best.pt"

try:
    client.create_registered_model(MODEL_NAME)
except Exception:
    pass  # already exists

version = client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id)
print(f"Model {MODEL_NAME} version {version.version} created")

try:
    client.transition_model_version_stage(MODEL_NAME, version.version, stage="Staging")
    print("Stage: Staging")
except Exception as exc:
    print("Stage transition not available in this client:", exc)
    client.set_registered_model_alias(MODEL_NAME, "staging", version.version)
    print("Alias 'staging' set instead")

print(f"Open {URI}")
