"""Central configuration for the PPE real-time detection project (Axe 3).

Design goals
------------
1. **Environment agnostic**: the same code must run on Kaggle/Colab (GPU
   notebooks), inside the FastAPI Docker image, and on a plain laptop.
   Every path is resolved from environment variables with safe defaults.
2. **Single source of truth**: the class map, the training hyper-parameters and
   the artefact locations live here and nowhere else. The API, the training
   scripts and the report generator all import from this module so that they
   can never drift apart.
3. **No hidden side effects**: importing this module never touches the disk.

Environment variables
---------------------
``PPE_PROJECT_ROOT``      Root of the repository (auto-detected).
``PPE_DATA_ROOT``         Where the CSS dataset lives (raw + processed).
``PPE_ARTIFACT_DIR``      Where trained weights / ONNX files are written.
``PPE_MLFLOW_TRACKING_URI`` MLflow tracking server URI.
``PPE_DEVICE``            ``cuda`` / ``cpu`` / ``0`` ...
``PPE_BACKEND``           ``auto`` | ``onnx`` | ``ultralytics`` (serving engine).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    """Return the repository root (the folder that contains ``src/``)."""
    env_root = os.environ.get("PPE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    # src/config.py -> src/ -> repo root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _repo_root()

DATA_ROOT: Path = Path(
    os.environ.get("PPE_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser()

RAW_DIR: Path = DATA_ROOT / "raw"
PROCESSED_DIR: Path = DATA_ROOT / "processed"

ARTIFACT_DIR: Path = Path(
    os.environ.get("PPE_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
).expanduser()
RUNS_DIR: Path = ARTIFACT_DIR / "runs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: Directory that holds the YOLO-formatted dataset (train/valid/test).
DATASET_DIR: Path = Path(
    os.environ.get("PPE_DATASET_DIR", PROCESSED_DIR / "css-ppe")
).expanduser()

#: Dataset description consumed by Ultralytics.
DATASET_YAML: Path = Path(
    os.environ.get("PPE_DATASET_YAML", DATASET_DIR / "dataset.yaml")
).expanduser()

#: Folder names used by the CSS dataset after re-splitting.
SPLIT_DIRS: Dict[str, str] = {"train": "train", "valid": "valid", "test": "test"}


# --------------------------------------------------------------------------- #
# Class map (Construction Site Safety dataset, 10 classes)
# --------------------------------------------------------------------------- #


class CFG:
    """Experiment configuration.

    The values mirror the notebook that produced the reference run
    (``notebooks/training_yolo11_ppe.ipynb``) so that re-running the
    pipeline reproduces the published numbers.

    ``DEBUG = True`` shrinks the dataset and the schedule so the whole
    pipeline can be smoke-tested in a couple of minutes on CPU.
    """

    DEBUG: bool = os.environ.get("PPE_DEBUG", "0") == "1"

    SEED: int = 42

    #: Train / validation / test proportions after the stratified re-split.
    TARGET_RATIOS: Dict[str, float] = {"train": 0.80, "valid": 0.15, "test": 0.05}

    #: class_id -> human readable class name (order matters: it defines the
    #: label vector order used by the model, the API and the colour map).
    PPE_CLASSES: Dict[int, str] = {
        0: "Hardhat",
        1: "Mask",
        2: "NO-Hardhat",
        3: "NO-Mask",
        4: "NO-Safety Vest",
        5: "Person",
        6: "Safety Cone",
        7: "Safety Vest",
        8: "Machinery",
        9: "Vehicle",
    }

    NUM_CLASSES: int = len(PPE_CLASSES)

    @classmethod
    def class_names(cls) -> List[str]:
        """Class names ordered by class id."""
        return [cls.PPE_CLASSES[i] for i in range(cls.NUM_CLASSES)]

    @classmethod
    def class_name_to_id(cls) -> Dict[str, int]:
        """Reverse mapping ``name -> class_id``."""
        return {v: k for k, v in cls.PPE_CLASSES.items()}

    # -- training ---------------------------------------------------------- #
    EPOCHS: int = 3 if DEBUG else int(os.environ.get("PPE_EPOCHS", "30"))
    BATCH_SIZE: int = 4 if DEBUG else int(os.environ.get("PPE_BATCH", "12"))
    IMGSZ: int = 640
    RECT: bool = False
    FRACTION: float = 0.05 if DEBUG else 1.0

    VERBOSE: bool = False
    SHOW_LABELS: bool = False
    SHOW_CONF: bool = False
    SHOW_BOXES: bool = False

    BASE_MODEL: str = os.environ.get("PPE_BASE_MODEL", "yolo11n.pt")
    DEVICE: str = os.environ.get("PPE_DEVICE", "cuda")

    PATIENT: int = 50
    WORKERS: int = int(os.environ.get("PPE_WORKERS", "8"))

    # -- augmentation ------------------------------------------------------ #
    #: Rare classes that get extra copies.
    AUGMENT_TARGET_CLASSES: List[str] = ["Safety Cone", "Vehicle"]
    #: Dominant classes excluded from the augmentation candidates so that the
    #: rebalancing does not amplify the imbalance it tries to fix.
    AUGMENT_EXCLUDED_CLASSES: List[str] = ["Person", "Machinery"]
    #: Maximum number of dominant-class instances tolerated in an image.
    AUGMENT_MAX_EXCLUDED: int = 1

    # -- inference --------------------------------------------------------- #
    CONF_THRESHOLD: float = float(os.environ.get("PPE_CONF", "0.25"))
    IOU_THRESHOLD: float = float(os.environ.get("PPE_IOU", "0.45"))
    MAX_DETECTIONS: int = 300
    #: Fixed inference resolution: required for a static ONNX graph.
    EXPORT_IMGSZ: int = 640
    EXPORT_OPSET: int = 12

    # -- MLflow ------------------------------------------------------------ #
    MLFLOW_TRACKING_URI: str = os.environ.get(
        "PPE_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
    )
    MLFLOW_EXPERIMENT: str = os.environ.get("PPE_MLFLOW_EXPERIMENT", "ppe-detection")
    MLFLOW_RUN_NAME: str = os.environ.get("PPE_MLFLOW_RUN_NAME", "yolo11n_30ep")


# --------------------------------------------------------------------------- #
# Artefact locations
# --------------------------------------------------------------------------- #

#: Best checkpoint produced by ``src/models/train.py``.
BEST_PT: Path = RUNS_DIR / "detect" / "train" / "weights" / "best.pt"
#: Canonical ONNX model (FP32) loaded by the FastAPI backend. The FP16 export
#: must never be written here: ``src/models/evaluate.py::export_onnx``
#: restores this file to FP32 after the FP16 export.
BEST_ONNX: Path = RUNS_DIR / "detect" / "train" / "weights" / "best.onnx"
#: Secondary ONNX export (FP16) of the same checkpoint, kept for benchmarking
#: only — never loaded by the backend.
BEST_ONNX_FP16: Path = RUNS_DIR / "detect" / "train" / "weights" / "best_fp16.onnx"
#: Aggregated metrics written by ``src/models/evaluate.py``.
METRICS_JSON: Path = ARTIFACT_DIR / "metrics.json"
#: Latency benchmark written by ``src/models/evaluate.py``.
BENCHMARK_JSON: Path = ARTIFACT_DIR / "benchmark.json"


def ensure_dirs() -> None:
    """Create the writable folders of the project (idempotent)."""
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        DATASET_DIR,
        ARTIFACT_DIR,
        RUNS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CFG",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "DATASET_DIR",
    "DATASET_YAML",
    "ARTIFACT_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "BEST_PT",
    "BEST_ONNX",
    "BEST_ONNX_FP16",
    "METRICS_JSON",
    "BENCHMARK_JSON",
    "ensure_dirs",
]
