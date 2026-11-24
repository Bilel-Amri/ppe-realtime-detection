"""Pydantic schemas of the PPE detection API.

Every request and every response is validated by these models, which is also
what feeds the OpenAPI documentation served at ``/docs``.

The schemas are written with the plain ``BaseModel`` API so that they work
with both Pydantic v1 and v2 (the API is used by students running different
environments).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ImageSize(BaseModel):
    """Dimensions of the processed image."""

    width: int = Field(..., ge=1, description="Width in pixels")
    height: int = Field(..., ge=1, description="Height in pixels")


class BoundingBox(BaseModel):
    """An axis-aligned box, in pixels of the original image."""

    x1: float = Field(..., description="Left edge")
    y1: float = Field(..., description="Top edge")
    x2: float = Field(..., description="Right edge")
    y2: float = Field(..., description="Bottom edge")
    width: float = Field(..., ge=0, description="Box width in pixels")
    height: float = Field(..., ge=0, description="Box height in pixels")


class Detection(BaseModel):
    """A single detected object."""

    class_id: int = Field(..., ge=0, description="Index of the class in the label map")
    label: str = Field(..., description="Human readable class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection score")
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    """Structured prediction returned by ``POST /predict``."""

    detections: List[Detection] = Field(default_factory=list)
    count: int = Field(..., ge=0, description="Number of detections")
    inference_time_ms: float = Field(
        ..., ge=0.0, description="Server-side inference latency in milliseconds"
    )
    image_size: ImageSize
    model_kind: str = Field(
        ...,
        description=(
            "ppe-finetuned for the PPE detector; coco-fallback for the COCO "
            "pre-trained safety net (it does not know the PPE classes)"
        ),
    )
    conf_threshold: float = Field(..., ge=0.0, le=1.0)
    iou_threshold: float = Field(..., ge=0.0, le=1.0)


class ClassInfo(BaseModel):
    """One entry of the label map."""

    id: int
    name: str


class ModelMetadataResponse(BaseModel):
    """Payload of ``GET /model/metadata``."""

    model_kind: str
    backend: str = Field(..., description="Inference engine: onnx or ultralytics")
    model_path: Optional[str] = None
    architecture: str
    base_model: str
    num_classes: int
    classes: List[ClassInfo]
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    metrics: Dict[str, object] = Field(
        default_factory=dict,
        description="Measured test-split metrics (empty when the model has not been evaluated yet)",
    )


class HealthResponse(BaseModel):
    """Payload of ``GET /health``."""

    status: str = Field(..., description="ok or degraded")
    model_loaded: bool
    model_kind: str
    model_version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    backend: str
    version: str
    uptime_s: float


class SampleImage(BaseModel):
    """One entry of the pre-loaded sample gallery."""

    name: str
    url: str
    width: int
    height: int


class SampleListResponse(BaseModel):
    """Payload of ``GET /samples``."""

    count: int
    samples: List[SampleImage]


class ErrorResponse(BaseModel):
    """Uniform error payload (RFC 7807 style, simplified)."""

    detail: str
    status_code: int
    error: str = Field(..., description="Short machine-readable error code")


__all__ = [
    "ImageSize",
    "BoundingBox",
    "Detection",
    "PredictionResponse",
    "ClassInfo",
    "ModelMetadataResponse",
    "HealthResponse",
    "SampleImage",
    "SampleListResponse",
    "ErrorResponse",
]
