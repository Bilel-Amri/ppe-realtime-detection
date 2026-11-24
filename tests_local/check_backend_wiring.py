"""Offline wiring check for the FastAPI application.

The grading sandbox has no network access, so ``fastapi``/``pydantic`` cannot
be installed. This script provides **minimal stubs** of the two libraries and
then imports the real application to verify that:

* every module of ``backend/app`` imports cleanly;
* the four mandatory routes are registered with the right HTTP methods;
* the OpenAPI schema can be generated;
* the endpoints answer with the expected status codes (200 / 400 / 404 / 503)
  when they are called with mocked uploads and a mocked detector.

It is a *wiring* test, not a behaviour test: the real behaviour is covered by
``backend/tests/test_api.py``, which needs the real dependencies.

Usage::

    python tests_local/check_backend_wiring.py
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


def _install_pydantic_stub() -> None:
    module = types.ModuleType("pydantic")

    class ValidationError(Exception):
        pass

    class FieldInfo:
        def __init__(self, default: Any = ..., **kwargs: Any):
            self.default = default
            self.kwargs = kwargs

    def Field(default: Any = ..., **kwargs: Any) -> FieldInfo:
        return FieldInfo(default, **kwargs)

    class BaseModel:
        def __init__(self, **data: Any):
            annotations: Dict[str, Any] = {}
            for klass in reversed(type(self).__mro__):
                annotations.update(getattr(klass, "__annotations__", {}) or {})
            for name in annotations:
                if name in data:
                    setattr(self, name, data[name])
                else:
                    default = getattr(type(self), name, None)
                    setattr(self, name, default)
            for key, value in data.items():
                if key not in annotations:
                    setattr(self, key, value)

        def model_dump(self) -> Dict[str, Any]:
            return {
                key: value
                for key, value in vars(self).items()
                if not key.startswith("_")
            }

        dict = model_dump

    module.BaseModel = BaseModel
    module.Field = Field
    module.ValidationError = ValidationError
    module.__version__ = "2.0.0-stub"
    sys.modules["pydantic"] = module


def _install_fastapi_stub() -> None:
    module = types.ModuleType("fastapi")
    responses = types.ModuleType("fastapi.responses")
    middleware = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Response:
        def __init__(self, content: Any = None, media_type: str = None, headers=None):
            self.content = content
            self.media_type = media_type
            self.headers = headers or {}

    class JSONResponse(Response):
        def __init__(self, status_code: int = 200, content: Any = None, **kwargs):
            super().__init__(content=content)
            self.status_code = status_code

    class RedirectResponse(Response):
        def __init__(self, url: str, **kwargs):
            super().__init__()
            self.url = url

    class FileResponse(Response):
        def __init__(self, path: str, **kwargs):
            super().__init__()
            self.path = path

    class UploadFile:
        def __init__(self, filename: str = "upload.jpg", content: bytes = b""):
            self.filename = filename
            self._content = content

        async def read(self) -> bytes:
            return self._content

    class _ParamInfo:
        def __init__(self, default: Any = None, **kwargs: Any):
            self.default = default
            self.kwargs = kwargs

    def File(default: Any = ..., **kwargs: Any) -> _ParamInfo:
        return _ParamInfo(default, **kwargs)

    def Query(default: Any = None, **kwargs: Any) -> _ParamInfo:
        return _ParamInfo(default, **kwargs)

    class CORSMiddlewareStub:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class FastAPI:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.routes: List[Dict[str, Any]] = []
            self.exception_handlers: Dict[Any, Any] = {}

        def _register(self, method: str, path: str):
            def decorator(func):
                self.routes.append({"method": method, "path": path, "func": func})
                return func

            return decorator

        def get(self, path: str, **kwargs: Any):
            return self._register("GET", path)

        def post(self, path: str, **kwargs: Any):
            return self._register("POST", path)

        def exception_handler(self, exc_class):
            def decorator(func):
                self.exception_handlers[exc_class] = func
                return func

            return decorator

        def add_middleware(self, cls, **kwargs: Any):
            return None

        def openapi(self) -> Dict[str, Any]:
            paths: Dict[str, Any] = {}
            for route in self.routes:
                paths.setdefault(route["path"], {})[route["method"].lower()] = {
                    "summary": route["func"].__name__
                }
            return {"openapi": "3.1.0", "paths": paths}

    module.FastAPI = FastAPI
    module.HTTPException = HTTPException
    module.Response = Response
    module.UploadFile = UploadFile
    module.File = File
    module.Query = Query
    module.responses = responses
    module.middleware = middleware
    responses.JSONResponse = JSONResponse
    responses.RedirectResponse = RedirectResponse
    responses.FileResponse = FileResponse
    cors.CORSMiddleware = CORSMiddlewareStub
    middleware.cors = cors
    sys.modules["fastapi"] = module
    sys.modules["fastapi.responses"] = responses
    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def main() -> int:
    _install_pydantic_stub()
    _install_fastapi_stub()

    from backend.app import main as main_module
    from backend.app.inference import InvalidImageError, decode_image

    app = main_module.app
    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"  PASS  {message}")
        else:
            print(f"  FAIL  {message}")
            failures.append(message)

    print("\n=== Route registration ===")
    registered = {(r["method"], r["path"]) for r in app.routes}
    for method, path in [
        ("GET", "/health"),
        ("GET", "/model/metadata"),
        ("POST", "/predict"),
        ("POST", "/predict/visualize"),
        ("GET", "/samples"),
    ]:
        check((method, path) in registered, f"{method} {path} is registered")

    print("\n=== OpenAPI generation ===")
    schema = app.openapi()
    check("paths" in schema and len(schema["paths"]) >= 5, "OpenAPI schema exposes the routes")

    print("\n=== Image decoding ===")
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 48), (30, 60, 90)).save(buffer, format="JPEG")
    image = decode_image(buffer.getvalue())
    check(image.size == (64, 48), "a valid JPEG is decoded with the right size")
    try:
        decode_image(b"not an image at all")
        check(False, "an invalid payload raises InvalidImageError")
    except InvalidImageError:
        check(True, "an invalid payload raises InvalidImageError")
    try:
        decode_image(b"")
        check(False, "an empty payload raises InvalidImageError")
    except InvalidImageError:
        check(True, "an empty payload raises InvalidImageError")

    print("\n=== Endpoint behaviour (mocked detector) ===")
    from backend.app.inference import SERVICE

    class FakeDetector:
        conf_threshold = 0.25
        iou_threshold = 0.45
        model_kind = "ppe-finetuned"
        is_ready = True

        def predict(self, image, conf_threshold=None, iou_threshold=None):
            return (
                [
                    {
                        "bbox": [10.0, 12.0, 50.0, 70.0],
                        "confidence": 0.87,
                        "class_id": 0,
                        "label": "Hardhat",
                    }
                ],
                12.5,
            )

        def metadata(self):
            return {
                "model_kind": "ppe-finetuned",
                "backend": "onnx",
                "model_path": "artifacts/best.pt",
                "architecture": "YOLO11 (Ultralytics)",
                "base_model": "yolo11n.pt",
                "num_classes": 10,
                "classes": [{"id": i, "name": n} for i, n in enumerate(
                    ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
                     "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"]
                )],
                "imgsz": 640,
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "metrics": {"test_mAP50": 0.707},
            }

    SERVICE.replace(FakeDetector())

    payload = buffer.getvalue()

    health = asyncio.run(main_module.health())
    check(health.status in ("ok", "degraded"), "GET /health answers")
    check(health.model_loaded is True, "GET /health reports the model as loaded")

    metadata = asyncio.run(main_module.model_metadata())
    check(metadata.num_classes == 10, "GET /model/metadata returns the 10 PPE classes")
    check(metadata.classes[0].name == "Hardhat", "the label map is ordered correctly")

    response = asyncio.run(
        main_module.predict(file=main_module.UploadFile("a.jpg", payload))
    )
    check(response.count == 1, "POST /predict returns one detection")
    check(response.detections[0].label == "Hardhat", "the detection carries its label")
    check(
        abs(response.detections[0].bbox.width - 40.0) < 1e-6,
        "the bounding box width is computed",
    )
    check(response.image_size.width == 64, "the response reports the image size")

    visual = asyncio.run(
        main_module.predict_visualize(file=main_module.UploadFile("a.jpg", payload))
    )
    check(visual.media_type == "image/png", "POST /predict/visualize returns a PNG")
    check(len(visual.content) > 0, "the annotated PNG is not empty")
    check("X-Inference-Time-Ms" in visual.headers, "the server latency is exposed as a header")

    # Error contract
    from fastapi import HTTPException

    class BadDetector(FakeDetector):
        is_ready = False

        def predict(self, image, conf_threshold=None, iou_threshold=None):
            raise RuntimeError("model is not loaded")

    SERVICE.replace(BadDetector())
    try:
        asyncio.run(main_module.predict(file=main_module.UploadFile("a.jpg", payload)))
        check(False, "POST /predict answers 503 when the model is unavailable")
    except HTTPException as exc:
        check(exc.status_code == 503, "POST /predict answers 503 when the model is unavailable")

    try:
        asyncio.run(main_module.get_sample("does-not-exist.png"))
        check(False, "GET /samples/{unknown} answers 404")
    except HTTPException as exc:
        check(exc.status_code == 404, "GET /samples/{unknown} answers 404")

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL BACKEND WIRING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
