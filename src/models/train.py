"""Training entry point: YOLO11 fine-tuning on the Construction Site Safety set.

Usage (Kaggle / Colab with a GPU, or locally with ``DEVICE=cpu``)::

    python -m src.models.train --data data/processed/css-ppe/dataset.yaml

What this script does
---------------------
1. Resolves the configuration from :mod:`src.config` (overridable with env vars).
2. Launches the Ultralytics fine-tuning with the exact hyper-parameters of the
   reference run.
3. Logs the run to **MLflow** (params + metrics + the best checkpoint as an
   artefact).
4. Appends the run to ``artifacts/train_results.csv`` — the historical
   registry that replaces the notebook's ad-hoc DataFrame.
5. Copies ``best.pt`` to ``artifacts/runs/detect/train/weights/`` so that
   :mod:`src.models.evaluate` and the API find it at a predictable location.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

from src.config import ARTIFACT_DIR, BEST_PT, CFG, DATASET_YAML, ensure_dirs
from src.data.dataset import write_results_csv


def build_train_params() -> Dict[str, object]:
    """Hyper-parameters of the reference run (single source of truth)."""
    return {
        "epochs": CFG.EPOCHS,
        "batch": CFG.BATCH_SIZE,
        "imgsz": CFG.IMGSZ,
        "rect": CFG.RECT,
        "fraction": CFG.FRACTION,
        "seed": CFG.SEED,
        "verbose": CFG.VERBOSE,
        "show_labels": CFG.SHOW_LABELS,
        "show_conf": CFG.SHOW_CONF,
        "show_boxes": CFG.SHOW_BOXES,
        "device": CFG.DEVICE,
        "patience": CFG.PATIENT,
        "workers": CFG.WORKERS,
    }


def resolve_dataset_yaml(explicit: Optional[str]) -> Path:
    """Return the dataset YAML, with a helpful error when it is missing."""
    if explicit:
        path = Path(explicit)
    else:
        path = DATASET_YAML
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at '{path}'.\n"
            "Prepare the dataset first:\n"
            "    python scripts/prepare_dataset.py --source <path-to-css-data>"
        )
    return path


def publish_best_checkpoint(source: Path) -> Path:
    """Copy the best checkpoint to the canonical artefact location."""
    BEST_PT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, BEST_PT)
    return BEST_PT


def log_to_mlflow(
    params: Dict[str, object],
    metrics: Dict[str, float],
    checkpoint: Optional[Path],
    run_name: str,
) -> None:
    """Push the run to the MLflow tracking server (best effort)."""
    try:
        import mlflow
    except ImportError:
        print("⚠️  MLflow is not installed - the run was not tracked.")
        print("    Install it with: pip install mlflow")
        return

    mlflow.set_tracking_uri(CFG.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CFG.MLFLOW_EXPERIMENT)
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in params.items()})
            mlflow.log_metrics(metrics)
            if checkpoint is not None and checkpoint.exists():
                mlflow.log_artifact(str(checkpoint))
            print(f"✅ Run logged to MLflow: {run_name}")
    except Exception as exc:  # pragma: no cover - depends on the server
        print(f"⚠️  MLflow logging failed ({exc}). The local artefacts are intact.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 on the PPE dataset")
    parser.add_argument("--data", default=None, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default="train", help="Name of the Ultralytics run")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None:
        CFG.EPOCHS = args.epochs
    if args.batch is not None:
        CFG.BATCH_SIZE = args.batch
    if args.device is not None:
        CFG.DEVICE = args.device

    ensure_dirs()
    dataset_yaml = resolve_dataset_yaml(args.data)

    from ultralytics import YOLO

    params = build_train_params()
    print("▶️  Training parameters:")
    for key, value in params.items():
        print(f"    {key:>12}: {value}")

    model = YOLO(CFG.BASE_MODEL)
    start = time.time()
    results = model.train(data=str(dataset_yaml), name=args.name, **params)
    duration = time.time() - start

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "train_time_s": round(duration, 2),
    }

    # Ultralytics writes the run inside ./runs/detect/<name>; move the best
    # checkpoint to the canonical artefact folder.
    produced = Path(getattr(results, "save_dir", "./runs/detect/train")) / "weights" / "best.pt"
    checkpoint = publish_best_checkpoint(produced) if produced.exists() else None

    run_id = f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    row = {
        "run_id": run_id,
        "model": CFG.BASE_MODEL,
        "epochs": CFG.EPOCHS,
        "batch": CFG.BATCH_SIZE,
        "imgsz": CFG.IMGSZ,
        "device": CFG.DEVICE,
        "seed": CFG.SEED,
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "train_time_s": metrics["train_time_s"],
    }
    write_results_csv(ARTIFACT_DIR / "train_results.csv", row)

    if not args.no_mlflow:
        log_to_mlflow(
            params,
            metrics,
            checkpoint,
            run_name=os.environ.get("PPE_MLFLOW_RUN_NAME", f"{CFG.BASE_MODEL}_{CFG.EPOCHS}ep"),
        )

    print("\n📊 Final training metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    if checkpoint:
        print(f"\n✅ Best checkpoint published at: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_train_params", "main", "log_to_mlflow", "publish_best_checkpoint"]
