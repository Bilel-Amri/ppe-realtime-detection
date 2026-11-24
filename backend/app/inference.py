"""Inference service used by the FastAPI routes.

The heavy lifting (loading the model, running the ONNX Runtime session,
decoding and NMS) lives in :mod:`src.inference`; this module is the thin
service layer that the API talks to, and the single place where the
``src`` package is made importable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The repository root is the parent of ``backend/``: it must be importable so
# that ``from src...`` works both in the Docker image and in a local venv.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference.predictor import (  # noqa: E402  (path bootstrap above)
    Detector,
    ModelNotReadyError,
    get_detector,
    set_detector,
)

APP_VERSION = "1.0.0"


class InvalidImageError(ValueError):
    """Raised when the uploaded payload is not a decodable image."""


# --------------------------------------------------------------------------- #
# Image decoding
# --------------------------------------------------------------------------- #


def decode_image(payload: bytes):
    """Decode raw bytes into a PIL RGB image.

    Raises:
        InvalidImageError: when the payload is empty or not a valid image.
    """
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to decode the uploaded image") from exc

    import io

    try:
        with Image.open(io.BytesIO(payload)) as handle:
            image = handle.convert("RGB")
    except Exception as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid image (expected JPEG, PNG or WEBP)."
        ) from exc

    if image.width < 2 or image.height < 2:
        raise InvalidImageError("The uploaded image is empty or too small.")
    return image


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class InferenceService:
    """Facade around :class:`src.inference.predictor.Detector`."""

    def __init__(self) -> None:
        self._detector: Optional[Detector] = None

    # -- lifecycle --------------------------------------------------------- #
    @property
    def detector(self) -> Detector:
        """Return the detector, loading it on first access."""
        if self._detector is None:
            self._detector = get_detector()
        return self._detector

    def load(self) -> None:
        """Eagerly instantiate the model (called during the app startup)."""
        try:
            self.detector
        except ModelNotReadyError:
            # The application must start anyway: /health reports "degraded"
            # and /predict answers 503 until a model becomes available.
            pass

    def replace(self, detector: Detector) -> None:
        """Inject a detector (used by the unit tests)."""
        self._detector = detector
        set_detector(detector)

    # -- inference --------------------------------------------------------- #
    def predict(
        self,
        image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run the detector on a PIL image."""
        detector = self.detector
        if not detector.is_ready:
            detector.load()
        if not detector.is_ready:
            raise ModelNotReadyError(
                "The detection model is not loaded yet. Train it or mount the "
                "artefacts before calling the API."
            )
        return detector.predict(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def metadata(self) -> Dict[str, Any]:
        return self.detector.metadata()

    def is_ready(self) -> bool:
        try:
            return self.detector.is_ready
        except ModelNotReadyError:
            return False


#: Process-wide service instance.
SERVICE = InferenceService()


__all__ = [
    "APP_VERSION",
    "InvalidImageError",
    "InferenceService",
    "SERVICE",
    "decode_image",
]
