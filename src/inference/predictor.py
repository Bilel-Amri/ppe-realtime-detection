"""Unified inference facade used by the FastAPI backend.

The service must start **even when the fine-tuned checkpoint has not been
trained yet** (during the first weeks of the project, and on a fresh clone of
the repository). The loader therefore follows a deterministic resolution
order and reports which model was actually loaded:

``onnx``
    ``artifacts/runs/detect/train/weights/best.onnx`` — ONNX Runtime, the
    production path.
``ultralytics``
    ``artifacts/runs/detect/train/weights/best.pt`` — PyTorch checkpoint.
``coco-fallback``
    ``yolo11n.pt`` pre-trained on COCO (80 classes). This model does **not**
    know the PPE classes; it is only a safety net so that ``docker-compose
    up`` always shows a working application. ``/model/metadata`` advertises
    ``model_kind = "coco-fallback"`` so nobody mistakes it for the trained
    detector.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import BEST_ONNX, BEST_PT, CFG, METRICS_JSON

#: COCO class names of the Ultralytics fallback model (first 80 entries).
COCO_CLASS_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class ModelNotReadyError(RuntimeError):
    """Raised when no model could be instantiated (maps to HTTP 503)."""


class Detector:
    """Thread-safe wrapper around the selected inference backend."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        self.backend_request = (backend or os.environ.get("PPE_BACKEND", "auto")).lower()
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else CFG.CONF_THRESHOLD
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else CFG.IOU_THRESHOLD
        )
        self.explicit_model_path = Path(model_path) if model_path else None

        self.model_kind: str = "unknown"
        self.model_path: Optional[Path] = None
        self.class_names: List[str] = CFG.class_names()
        self.num_classes: int = CFG.NUM_CLASSES
        self.metrics: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._model = None
        self._onnx = None
        self._last_error: Optional[Exception] = None

    # -- loading ----------------------------------------------------------- #
    def load(self) -> None:
        """Instantiate the model according to the resolution order."""
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        explicit = self.explicit_model_path
        candidates: List[Tuple[str, Path]] = []

        if explicit is not None:
            suffix = explicit.suffix.lower()
            if suffix == ".onnx":
                candidates.append(("onnx", explicit))
            else:
                candidates.append(("ultralytics", explicit))
        else:
            if self.backend_request in ("auto", "onnx"):
                candidates.append(("onnx", BEST_ONNX))
            if self.backend_request in ("auto", "ultralytics"):
                candidates.append(("ultralytics", BEST_PT))

        for backend, path in candidates:
            if not path.exists():
                continue
            try:
                if backend == "onnx":
                    from src.inference.onnx_detector import OnnxYoloDetector

                    self._onnx = OnnxYoloDetector(
                        path,
                        num_classes=CFG.NUM_CLASSES,
                        conf_threshold=self.conf_threshold,
                        iou_threshold=self.iou_threshold,
                    )
                else:
                    from ultralytics import YOLO

                    self._model = YOLO(str(path))
                    self._sync_ultralytics_classes()
                self.model_path = path
                self.backend_request = backend
                self.model_kind = "ppe-finetuned"
                return
            except Exception as exc:  # pragma: no cover - depends on runtime
                self._last_error = exc
                continue

        # Nothing found: fall back to the COCO pre-trained model.
        self._load_coco_fallback()

    def _sync_ultralytics_classes(self) -> None:
        """Use the class names declared by the checkpoint (if any)."""
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            self.class_names = [str(n) for n in names]
            self.num_classes = len(self.class_names)

    def _load_coco_fallback(self) -> None:  # pragma: no cover - needs network
        try:
            from ultralytics import YOLO

            self._model = YOLO("yolo11n.pt")
            self.class_names = COCO_CLASS_NAMES
            self.num_classes = len(COCO_CLASS_NAMES)
            self.model_path = Path("yolo11n.pt")
            self.model_kind = "coco-fallback"
            self.backend_request = "ultralytics"
        except Exception as exc:
            raise ModelNotReadyError(
                "No inference backend available. Train the model "
                "(python -m src.models.train) or install ultralytics/onnxruntime."
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._model is not None or self._onnx is not None

    # -- inference --------------------------------------------------------- #
    def predict(
        self,
        image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict], float]:
        """Detect objects in ``image`` (PIL). Returns ``(detections, ms)``.

        ``conf_threshold`` / ``iou_threshold`` override the instance defaults
        **for this call only** - they are never written back to the instance,
        so concurrent requests cannot interfere with each other.
        """
        if self._onnx is not None:
            return self._onnx.timed_predict(
                image,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        if self._model is None:
            self.load()
            if self._model is None:
                raise ModelNotReadyError("Model is not loaded.")

        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        start = time.perf_counter()
        with self._lock:
            results = self._model.predict(
                source=image,
                conf=conf,
                iou=iou,
                verbose=False,
            )[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: List[Dict] = []
        boxes = getattr(results, "boxes", None)
        if boxes is not None:
            for xyxy, conf, cid in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                cid = int(cid)
                detections.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "confidence": float(conf),
                        "class_id": cid,
                        "label": (
                            self.class_names[cid]
                            if 0 <= cid < len(self.class_names)
                            else str(cid)
                        ),
                    }
                )
        return detections, elapsed_ms

    # -- metadata ---------------------------------------------------------- #
    def metadata(self) -> Dict:
        """Everything the frontend and the report need about the model."""
        import json

        metrics = dict(self.metrics)
        if not metrics and METRICS_JSON.exists():
            try:
                metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}

        return {
            "model_kind": self.model_kind,
            "backend": self.backend_request,
            "model_path": str(self.model_path) if self.model_path else None,
            "architecture": "YOLO11 (Ultralytics)",
            "base_model": CFG.BASE_MODEL,
            "num_classes": self.num_classes,
            "classes": [
                {"id": i, "name": name}
                for i, name in enumerate(self.class_names)
            ],
            "imgsz": CFG.IMGSZ,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "metrics": metrics,
        }


#: Process-wide singleton used by the FastAPI application.
_DETECTOR: Optional[Detector] = None


def get_detector() -> Detector:
    """Return the process-wide detector, loading it on first use."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
        _DETECTOR.load()
    return _DETECTOR


def set_detector(detector: Detector) -> None:
    """Override the singleton (used by the tests)."""
    global _DETECTOR
    _DETECTOR = detector


__all__ = [
    "COCO_CLASS_NAMES",
    "Detector",
    "ModelNotReadyError",
    "get_detector",
    "set_detector",
]
