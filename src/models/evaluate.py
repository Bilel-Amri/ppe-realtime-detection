"""Final evaluation, ONNX export and MLflow logging.

This script is the productionised version of the three blocks that were
appended to the training notebook:

1. **Final test-split evaluation** — the model is evaluated on the held-out
   ``test`` split (never on ``val``), and the per-class AP50-95 is printed.
2. **ONNX export** — FP32 and FP16 graphs, both with ``opset=12``,
   ``simplify=True`` and a static shape (a dynamic shape would break the
   fixed-size ONNX Runtime session used by the API).
3. **MLflow logging** — parameters, metrics and artefacts are pushed to the
   tracking server, and the model is registered in the *Model Registry*.

Two deliberate differences with the notebook snippet
----------------------------------------------------
* **The FP16 export can never overwrite the canonical FP32 model.** Ultralytics
  always writes ``best.onnx`` next to the checkpoint, so exporting twice would
  leave a single file holding whichever graph was exported last (FP16). Here
  each graph is renamed right after the export (``best_fp32.onnx`` /
  ``best_fp16.onnx``) and ``best.onnx`` — the file FastAPI loads — is restored
  to the FP32 copy at the end of ``export_onnx()``.
* **The metrics are also written to ``artifacts/metrics.json``.** The API
  (``/model/metadata``) and the LaTeX report read that file, so the numbers
  displayed in the application are exactly the numbers that were measured.

Usage::

    python -m src.models.evaluate --data data/processed/css-ppe/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config import (
    BENCHMARK_JSON,
    BEST_ONNX,
    BEST_ONNX_FP16,
    BEST_PT,
    CFG,
    DATASET_YAML,
    METRICS_JSON,
    ensure_dirs,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def resolve_dataset_yaml(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        path = DATASET_YAML
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at '{path}'. "
            "Run scripts/prepare_dataset.py first."
        )
    return path


def resolve_checkpoint(explicit: Optional[str]) -> Path:
    path = Path(explicit) if explicit else BEST_PT
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. Run `python -m src.models.train` first."
        )
    return path


def per_class_ap(metrics) -> Dict[str, float]:
    """Extract the per-class AP50-95 from an Ultralytics validation result."""
    maps = getattr(metrics.box, "maps", None)
    names = CFG.class_names()
    table: Dict[str, float] = {}
    if maps is None:
        return table
    for index, value in enumerate(maps):
        key = names[index] if index < len(names) else f"class_{index}"
        table[key] = float(value)
    return table


# --------------------------------------------------------------------------- #
# 1) Final test-split evaluation
# --------------------------------------------------------------------------- #


def evaluate_on_test_split(checkpoint: Path, dataset_yaml: Path, imgsz: int):
    """Evaluate the trained model on the **test** split.

    Returns the Ultralytics metrics object (``m``), whose ``box`` attribute
    carries ``map50``, ``map``, ``mp`` (precision), ``mr`` (recall) and
    ``maps`` (per-class AP50-95).
    """
    from ultralytics import YOLO

    best = YOLO(str(checkpoint))
    m = best.val(data=str(dataset_yaml), split="test", imgsz=imgsz)
    return best, m


def print_test_report(m) -> Dict[str, float]:
    """Print the test report and return it as a plain dictionary."""
    names = CFG.class_names()
    maps = list(getattr(m.box, "maps", []) or [])

    print("\n" + "=" * 68)
    print("TEST-SPLIT EVALUATION (held-out split, never used for training)")
    print("=" * 68)
    print(f"TEST mAP50: {m.box.map50:.4f}  mAP50-95: {m.box.map:.4f}  "
          f"P: {m.box.mp:.4f}  R: {m.box.mr:.4f}")

    print(f"\n{'class':<18}{'AP50-95':>10}")
    print("-" * 28)
    for index, name in enumerate(names):
        value = float(maps[index]) if index < len(maps) else float("nan")
        print(f"{name:<18}{value:>10.4f}")

    return {
        "test_mAP50": float(m.box.map50),
        "test_mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class_ap50_95": {
            name: (float(maps[i]) if i < len(maps) else None)
            for i, name in enumerate(names)
        },
    }


# --------------------------------------------------------------------------- #
# 2) ONNX export (FP32 + FP16)
# --------------------------------------------------------------------------- #


def export_onnx(best, imgsz: int = CFG.EXPORT_IMGSZ, opset: int = CFG.EXPORT_OPSET):
    """Export the model to ONNX, in FP32 and in FP16.

    Ultralytics always writes the file under the name ``best.onnx``; the two
    exports are therefore renamed immediately after being produced, and
    ``best.onnx`` (the canonical file loaded by FastAPI) is restored to the
    FP32 copy afterwards, so the FP16 graph can never overwrite it.
    """
    exported: Dict[str, Optional[Path]] = {"fp32": None, "fp16": None}

    # --- FP32 ------------------------------------------------------------- #
    fp32_path = best.export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False
    )
    if fp32_path:
        source = Path(fp32_path)
        BEST_ONNX_FP16.parent.mkdir(parents=True, exist_ok=True)
        fp32_target = BEST_ONNX.parent / "best_fp32.onnx"
        shutil.copyfile(source, fp32_target)
        exported["fp32"] = fp32_target
        print(f"✅ ONNX (FP32) exported: {fp32_target}")

    # --- FP16 ------------------------------------------------------------- #
    fp16_path = best.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
        half=True,
        device=0,
    )
    if fp16_path:
        source = Path(fp16_path)
        target = BEST_ONNX.parent / "best_fp16.onnx"
        shutil.copyfile(source, target)
        exported["fp16"] = target
        print(f"✅ ONNX (FP16) exported: {target}")

    # --- Canonical file ---------------------------------------------------- #
    # The FP16 export above rewrites ``best.onnx`` in place (Ultralytics always
    # exports to that name). Restore it to the FP32 graph: it is the file the
    # FastAPI backend loads (see ``src/config.BEST_ONNX``) and it must never
    # silently become FP16.
    if exported["fp32"] is not None:
        shutil.copyfile(exported["fp32"], BEST_ONNX)
        print(f"✅ Canonical ONNX (FP32) restored: {BEST_ONNX}")

    return exported


def benchmark_onnx(
    onnx_path: Path,
    checkpoint: Path,
    imgsz: int = CFG.EXPORT_IMGSZ,
    runs: int = 50,
) -> Dict[str, float]:
    """Compare the PyTorch and the ONNX Runtime latencies on a dummy image.

    The numbers are *measured*, never estimated: the same synthetic input is
    pushed through both backends and the median latency is reported.
    """
    import numpy as np

    results: Dict[str, float] = {}

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)

    try:
        import onnxruntime as ort

        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        session = ort.InferenceSession(str(onnx_path), providers=providers or None)
        input_name = session.get_inputs()[0].name
        # Warm-up (CUDA context creation, kernel selection, ...)
        for _ in range(5):
            session.run(None, {input_name: dummy})
        timings: List[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            timings.append((time.perf_counter() - start) * 1000.0)
        results["onnx_median_ms"] = float(np.median(timings))
        results["onnx_p90_ms"] = float(np.percentile(timings, 90))
        results["onnx_provider"] = session.get_providers()[0]
    except Exception as exc:
        print(f"⚠️  ONNX Runtime benchmark skipped: {exc}")

    try:
        import torch
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
        tensor = torch.zeros(1, 3, imgsz, imgsz)
        for _ in range(5):
            model.predict(tensor, verbose=False)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model.predict(tensor, verbose=False)
            timings.append((time.perf_counter() - start) * 1000.0)
        results["pytorch_median_ms"] = float(np.median(timings))
        results["pytorch_p90_ms"] = float(np.percentile(timings, 90))
    except Exception as exc:
        print(f"⚠️  PyTorch benchmark skipped: {exc}")

    if "onnx_median_ms" in results and "pytorch_median_ms" in results:
        speedup = results["pytorch_median_ms"] / max(results["onnx_median_ms"], 1e-6)
        results["speedup_x"] = float(speedup)
        print(
            f"⚡ Latency: PyTorch {results['pytorch_median_ms']:.1f} ms → "
            f"ONNX {results['onnx_median_ms']:.1f} ms ({speedup:.2f}x)"
        )
    return results


# --------------------------------------------------------------------------- #
# 3) MLflow logging + Model Registry
# --------------------------------------------------------------------------- #


def log_to_mlflow(
    train_params: Dict[str, object],
    metrics: Dict[str, float],
    checkpoint: Path,
    exported: Dict[str, Optional[Path]],
    run_name: str,
    register: bool = True,
) -> None:
    """Push the evaluation run to MLflow and register the model."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("⚠️  MLflow is not installed - nothing was tracked.")
        print("    Install it with: pip install mlflow")
        return

    mlflow.set_tracking_uri(CFG.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(CFG.MLFLOW_EXPERIMENT)

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**train_params, "model": CFG.BASE_MODEL})
            mlflow.log_metrics(
                {
                    "test_mAP50": metrics["test_mAP50"],
                    "test_mAP50_95": metrics["test_mAP50_95"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )
            for class_name, ap in (metrics.get("per_class_ap50_95") or {}).items():
                if ap is not None:
                    mlflow.log_metric(f"ap50_95/{class_name}", ap)

            mlflow.log_artifact(str(checkpoint))
            if exported.get("fp32"):
                mlflow.log_artifact(str(exported["fp32"]))
            if exported.get("fp16"):
                mlflow.log_artifact(str(exported["fp16"]))

            # --- Model Registry -------------------------------------------- #
            if register:
                try:
                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{checkpoint.name}"
                    mlflow.register_model(model_uri, "PPE-YOLO11")
                    client = MlflowClient()
                    versions = client.search_model_versions(
                        "name='PPE-YOLO11'", order_by=["version_number DESC"], max_results=1
                    )
                    if versions:
                        client.transition_model_version_stage(
                            name="PPE-YOLO11",
                            version=versions[0].version,
                            stage="Staging",
                        )
                        version = versions[0].version
                        print(f"\u2705 Model registered (version {version}).")
                except Exception as exc:
                    print(f"⚠️  Model Registry step skipped: {exc}")

        print(f"✅ Evaluation logged to MLflow: {run_name}")
    except Exception as exc:  # pragma: no cover - depends on the server
        print(f"⚠️  MLflow logging failed ({exc}). Local artefacts are intact.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate on the test split, export ONNX and log to MLflow"
    )
    parser.add_argument("--data", default=None, help="Path to dataset.yaml")
    parser.add_argument("--model", default=None, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=CFG.EXPORT_IMGSZ)
    parser.add_argument("--opset", type=int, default=CFG.EXPORT_OPSET)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    dataset_yaml = resolve_dataset_yaml(args.data)
    checkpoint = resolve_checkpoint(args.model)

    from src.models.train import build_train_params

    train_params = build_train_params()

    # --- 1) test-split evaluation ---------------------------------------- #
    best, m = evaluate_on_test_split(checkpoint, dataset_yaml, args.imgsz)
    metrics = print_test_report(m)

    # --- 2) ONNX export --------------------------------------------------- #
    exported = export_onnx(best, imgsz=args.imgsz, opset=args.opset)

    # --- latency benchmark ------------------------------------------------ #
    benchmark: Dict[str, float] = {}
    if exported.get("fp32"):
        benchmark = benchmark_onnx(
            exported["fp32"], checkpoint, imgsz=args.imgsz, runs=args.benchmark_runs
        )

    # --- 3) MLflow -------------------------------------------------------- #
    if not args.no_mlflow:
        log_to_mlflow(
            train_params,
            metrics,
            checkpoint,
            exported,
            run_name=CFG.MLFLOW_RUN_NAME,
            register=not args.no_register,
        )

    # --- persist the metrics --------------------------------------------- #
    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_yaml": str(dataset_yaml),
        "checkpoint": str(checkpoint),
        "base_model": CFG.BASE_MODEL,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "train_params": train_params,
        **metrics,
        "benchmark": benchmark,
        "onnx_exports": {
            "fp32": str(exported["fp32"]) if exported.get("fp32") else None,
            "fp16": str(exported["fp16"]) if exported.get("fp16") else None,
        },
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    BENCHMARK_JSON.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(f"\n✅ Metrics written to: {METRICS_JSON}")
    print(f"✅ Benchmark written to: {BENCHMARK_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_on_test_split",
    "print_test_report",
    "export_onnx",
    "benchmark_onnx",
    "log_to_mlflow",
    "main",
]
