"""FastAPI application exposing the PPE detector.

Endpoints (the four mandatory ones are marked *required* by the brief):

==========================================  ==========================================
``GET  /health``                            *required* - server state, GPU, model version
``GET  /model/metadata``                    *required* - model, classes and metrics
``POST /predict``                           *required* - structured JSON prediction
``POST /predict/visualize``                 *required* - annotated PNG
``GET  /samples``                           gallery of pre-loaded images
``GET  /samples/{name}``                    one gallery image
``GET  /``                                  redirect to the Swagger UI
==========================================  ==========================================

Error contract:

* ``400 Bad Request`` - the uploaded payload is not a decodable image;
* ``413 Payload Too Large`` - the upload exceeds ``MAX_UPLOAD_BYTES``;
* ``503 Service Unavailable`` - the model is not instantiated yet.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repository root importable so that the ``src`` package resolves
# both when the app is started as ``backend.app.main`` and as ``app.main``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .inference import (
    APP_VERSION,
    InvalidImageError,
    SERVICE,
    decode_image,
)
from .schemas import (
    BoundingBox,
    ClassInfo,
    Detection,
    ErrorResponse,
    HealthResponse,
    ImageSize,
    ModelMetadataResponse,
    PredictionResponse,
    SampleImage,
    SampleListResponse,
)

#: Samples shipped with the API (used by the one-click gallery of the frontend).
SAMPLES_DIR: Path = Path(__file__).resolve().parent / "samples"

#: Hard limit on the size of an uploaded image (15 MB).
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

#: Extensions accepted by the gallery.
SAMPLE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup without ever preventing the app from starting."""
    SERVICE.load()
    yield


app = FastAPI(
    title="PPE Detection API",
    version=APP_VERSION,
    description=(
        "Real-time detection of Personal Protective Equipment (PPE) on "
        "construction sites with YOLO11. Backed by an ONNX Runtime session "
        "for the production path and by MLflow for the experiment tracking."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Server state and model availability."},
        {"name": "model", "description": "Model metadata and label map."},
        {"name": "inference", "description": "Prediction endpoints."},
        {"name": "samples", "description": "Pre-loaded demonstration images."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #


def _code_for_status(status_code: int) -> str:
    return {
        400: "invalid_request",
        404: "not_found",
        413: "payload_too_large",
        503: "model_unavailable",
    }.get(status_code, "error")


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "status_code": status_code, "error": code},
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(_request, exc: InvalidImageError):
    return _error_response(400, "invalid_image", str(exc))


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return _error_response(
        exc.status_code, _code_for_status(exc.status_code), str(exc.detail)
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _gpu_info() -> Dict[str, Any]:
    """Best-effort GPU detection (never raises)."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {"gpu_available": False, "gpu_name": None}


async def _read_upload(file: UploadFile) -> bytes:
    """Read the uploaded file and enforce the size limit."""
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return payload


def _build_response(
    detections: List[Dict[str, Any]], elapsed_ms: float, image
) -> PredictionResponse:
    """Convert the raw detections into the validated response model."""
    payload: List[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        payload.append(
            Detection(
                class_id=int(det.get("class_id", 0)),
                label=str(det.get("label", "unknown")),
                confidence=float(det.get("confidence", 0.0)),
                bbox=BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                ),
            )
        )

    metadata = SERVICE.metadata()
    return PredictionResponse(
        detections=payload,
        count=len(payload),
        inference_time_ms=round(elapsed_ms, 2),
        image_size=ImageSize(width=image.width, height=image.height),
        model_kind=str(metadata.get("model_kind", "unknown")),
        conf_threshold=float(metadata.get("conf_threshold", 0.25)),
        iou_threshold=float(metadata.get("iou_threshold", 0.45)),
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root to the Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Server, GPU and model state",
    responses={503: {"model": ErrorResponse, "description": "Model not loaded"}},
)
async def health() -> HealthResponse:
    """Check that the server is up, that the model is loaded and report the GPU."""
    gpu = _gpu_info()
    ready = SERVICE.is_ready()
    metadata = SERVICE.metadata() if ready else {}
    return HealthResponse(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        model_kind=str(metadata.get("model_kind", "unknown")),
        model_version=APP_VERSION,
        gpu_available=bool(gpu["gpu_available"]),
        gpu_name=gpu["gpu_name"],
        backend=str(metadata.get("backend", "none")),
        version=APP_VERSION,
        uptime_s=round(time.time() - START_TIME, 2),
    )


@app.get(
    "/model/metadata",
    response_model=ModelMetadataResponse,
    tags=["model"],
    summary="Model metadata, label map and measured metrics",
)
async def model_metadata() -> ModelMetadataResponse:
    """Describe the loaded model: architecture, classes and test metrics."""
    try:
        metadata = SERVICE.metadata()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"The model is not available: {exc}",
        ) from exc
    classes = [
        ClassInfo(id=int(item["id"]), name=str(item["name"]))
        for item in metadata.get("classes", [])
    ]
    return ModelMetadataResponse(
        model_kind=str(metadata.get("model_kind", "unknown")),
        backend=str(metadata.get("backend", "none")),
        model_path=metadata.get("model_path"),
        architecture=str(metadata.get("architecture", "YOLO11")),
        base_model=str(metadata.get("base_model", "yolo11n.pt")),
        num_classes=int(metadata.get("num_classes", len(classes))),
        classes=classes,
        imgsz=int(metadata.get("imgsz", 640)),
        conf_threshold=float(metadata.get("conf_threshold", 0.25)),
        iou_threshold=float(metadata.get("iou_threshold", 0.45)),
        metrics=metadata.get("metrics", {}) or {},
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Structured prediction for an uploaded image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Image to analyse (JPEG, PNG, WEBP)"),
    conf: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Override the confidence threshold"
    ),
    iou: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Override the NMS IoU threshold"
    ),
) -> PredictionResponse:
    """Detect the PPE objects of an image and return the result as JSON."""
    payload = await _read_upload(file)
    image = decode_image(payload)

    try:
        detections, elapsed_ms = SERVICE.predict(
            image, conf_threshold=conf, iou_threshold=iou
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _build_response(detections, elapsed_ms, image)


@app.post(
    "/predict/visualize",
    tags=["inference"],
    summary="Annotated image (bounding boxes + labels + confidence)",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "Annotated PNG"},
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_visualize(
    file: UploadFile = File(..., description="Image to analyse (JPEG, PNG, WEBP)"),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou: Optional[float] = Query(None, ge=0.0, le=1.0),
) -> Response:
    """Return the image with the detected objects drawn on top of it."""
    payload = await _read_upload(file)
    image = decode_image(payload)

    try:
        detections, elapsed_ms = SERVICE.predict(
            image, conf_threshold=conf, iou_threshold=iou
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    from src.utils.viz import draw_detections, to_png_bytes

    annotated = draw_detections(image, detections)
    return Response(
        content=to_png_bytes(annotated),
        media_type="image/png",
        headers={"X-Inference-Time-Ms": f"{elapsed_ms:.2f}"},
    )


@app.get(
    "/samples",
    response_model=SampleListResponse,
    tags=["samples"],
    summary="List of the pre-loaded sample images",
)
async def list_samples() -> SampleListResponse:
    """Return the images proposed by the one-click gallery of the frontend."""
    samples: List[SampleImage] = []
    if SAMPLES_DIR.exists():
        from PIL import Image

        for path in sorted(SAMPLES_DIR.iterdir()):
            if path.suffix.lower() not in SAMPLE_SUFFIXES:
                continue
            try:
                with Image.open(path) as handle:
                    width, height = handle.size
            except Exception:
                width, height = 0, 0
            samples.append(
                SampleImage(
                    name=path.name,
                    url=f"/samples/{path.name}",
                    width=width,
                    height=height,
                )
            )
    return SampleListResponse(count=len(samples), samples=samples)


@app.get("/samples/{name}", tags=["samples"], summary="Serve one sample image")
async def get_sample(name: str) -> Response:
    """Return the raw bytes of a sample image."""
    safe_name = Path(name).name  # prevent path traversal
    path = SAMPLES_DIR / safe_name
    if not path.exists() or path.suffix.lower() not in SAMPLE_SUFFIXES:
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found.")
    return FileResponse(str(path))


__all__ = ["app", "SAMPLES_DIR", "MAX_UPLOAD_BYTES"]
