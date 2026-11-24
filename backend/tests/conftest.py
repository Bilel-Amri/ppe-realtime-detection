"""Pytest fixtures shared by the API tests.

The tests never load a real model: a deterministic fake detector is injected
into the service, so the suite runs in a couple of seconds on any machine and
without a GPU.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PPE_CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle",
]


class FakeDetector:
    """A detector that returns two fixed boxes instead of running a model."""

    conf_threshold = 0.25
    iou_threshold = 0.45
    model_kind = "ppe-finetuned"
    is_ready = True

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def predict(
        self,
        image,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        self.calls.append(
            {
                "size": image.size,
                "conf": conf_threshold,
                "iou": iou_threshold,
            }
        )
        detections = [
            {
                "bbox": [10.0, 12.0, 60.0, 80.0],
                "confidence": 0.91,
                "class_id": 0,
                "label": "Hardhat",
            },
            {
                "bbox": [80.0, 30.0, 140.0, 200.0],
                "confidence": 0.74,
                "class_id": 5,
                "label": "Person",
            },
        ]
        return detections, 12.34

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_kind": self.model_kind,
            "backend": "onnx",
            "model_path": "artifacts/runs/detect/train/weights/best.pt",
            "architecture": "YOLO11 (Ultralytics)",
            "base_model": "yolo11n.pt",
            "num_classes": len(PPE_CLASS_NAMES),
            "classes": [
                {"id": index, "name": name}
                for index, name in enumerate(PPE_CLASS_NAMES)
            ],
            "imgsz": 640,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "metrics": {"test_mAP50": 0.707, "test_mAP50_95": 0.467},
        }


class NotReadyDetector(FakeDetector):
    """A detector whose model is missing (triggers the 503 contract)."""

    is_ready = False

    def predict(self, image, conf_threshold=None, iou_threshold=None):
        raise RuntimeError("The detection model is not loaded yet.")


@pytest.fixture()
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture()
def client(monkeypatch, fake_detector):
    """A ``TestClient`` wired to the fake detector."""
    from fastapi.testclient import TestClient

    from backend.app.inference import SERVICE
    from backend.app.main import app

    SERVICE.replace(fake_detector)
    with TestClient(app) as test_client:
        yield test_client
    SERVICE._detector = None  # type: ignore[attr-defined]


@pytest.fixture()
def not_ready_client(monkeypatch):
    """A ``TestClient`` whose model failed to load."""
    from fastapi.testclient import TestClient

    from backend.app.inference import SERVICE
    from backend.app.main import app

    SERVICE.replace(NotReadyDetector())
    with TestClient(app) as test_client:
        yield test_client
    SERVICE._detector = None  # type: ignore[attr-defined]


@pytest.fixture()
def image_bytes() -> bytes:
    """A small, valid JPEG image."""
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 80, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture()
def broken_bytes() -> bytes:
    """A payload that is not an image."""
    return b"this is definitely not a JPEG file"
