"""Guards for the canonical ONNX artefact loaded by the FastAPI backend.

Run it with::

    python -m pytest tests_local -q

What it protects against
------------------------
``src/models/evaluate.py::export_onnx()`` exports the model twice (FP32 then
FP16). Ultralytics always writes ``best.onnx``, so a naive export sequence
leaves the **FP16** graph in the file the backend loads — a silent precision
change that no error would reveal. These tests assert that:

* the canonical file exists and loads with ONNX Runtime;
* its input is **float32** (i.e. it is the FP32 graph, not the FP16 one);
* ``Detector(backend="onnx")`` resolves to exactly that file.

They skip cleanly on a fresh clone where the model has not been exported yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import BEST_ONNX

pytestmark = pytest.mark.skipif(
    not BEST_ONNX.exists(),
    reason=f"canonical ONNX not exported yet: {BEST_ONNX}",
)


def test_canonical_onnx_loads_as_float32() -> None:
    """The file FastAPI loads must open and take float32 input."""
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(str(BEST_ONNX), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()

    assert inputs, "the ONNX graph declares no input"
    assert inputs[0].name == "images"
    assert inputs[0].shape == [1, 3, 640, 640]
    # tensor(float) == FP32. tensor(float16) would mean the FP16 export
    # silently overwrote the canonical model.
    assert inputs[0].type == "tensor(float)", (
        f"canonical ONNX is not FP32 (got {inputs[0].type}): "
        f"the FP16 export likely overwrote {BEST_ONNX}"
    )


def test_canonical_onnx_file_is_distinct_from_fp16() -> None:
    """If the FP16 sibling exists, it must not be the same bytes as best.onnx."""
    fp16 = BEST_ONNX.with_name("best_fp16.onnx")
    if not fp16.exists():
        pytest.skip("no FP16 export on disk")

    assert BEST_ONNX.read_bytes() != fp16.read_bytes(), (
        "best.onnx is byte-identical to best_fp16.onnx: "
        "the FP16 export overwrote the canonical FP32 model"
    )


def test_detector_resolves_to_canonical_onnx() -> None:
    """Detector(backend='onnx') must load exactly BEST_ONNX."""
    from src.inference.predictor import Detector

    detector = Detector(backend="onnx")
    detector.load()

    assert detector.model_path == Path(BEST_ONNX)
    assert detector.backend_request == "onnx"
    assert detector.model_kind == "ppe-finetuned"
