"""Unit tests of the FastAPI endpoints (pytest + fastapi TestClient).

The brief requires automated tests on the API, and requires them to run in
the CI pipeline. The suite covers:

* the four mandatory endpoints and their response contracts;
* the semantic HTTP error codes (400 / 404 / 413 / 422 / 503);
* the Pydantic validation of the query parameters;
* the fact that the thresholds given per request do not leak into the next one.
"""

from __future__ import annotations

import io

from PIL import Image


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in ("ok", "degraded")
    assert "gpu_available" in payload
    assert payload["version"]


def test_health_reports_the_loaded_model(client, fake_detector):
    payload = client.get("/health").json()
    assert payload["model_loaded"] is True
    assert payload["model_kind"] == "ppe-finetuned"
    assert payload["backend"] == "onnx"


def test_health_is_degraded_when_the_model_is_missing(not_ready_client):
    payload = not_ready_client.get("/health").json()
    assert payload["status"] == "degraded"
    assert payload["model_loaded"] is False


# --------------------------------------------------------------------------- #
# /model/metadata
# --------------------------------------------------------------------------- #


def test_model_metadata_lists_the_ten_ppe_classes(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_classes"] == 10
    assert [c["name"] for c in payload["classes"]][:3] == [
        "Hardhat",
        "Mask",
        "NO-Hardhat",
    ]
    assert payload["classes"][0]["id"] == 0


def test_model_metadata_exposes_the_measured_metrics(client):
    payload = client.get("/model/metadata").json()
    assert "metrics" in payload
    assert payload["metrics"]["test_mAP50"] == 0.707


def test_model_metadata_documents_the_thresholds(client):
    payload = client.get("/model/metadata").json()
    assert 0.0 <= payload["conf_threshold"] <= 1.0
    assert 0.0 <= payload["iou_threshold"] <= 1.0


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


def test_predict_returns_the_detections(client, image_bytes):
    response = client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["detections"][0]["label"] == "Hardhat"
    assert payload["detections"][0]["confidence"] > 0.9


def test_predict_returns_a_valid_bounding_box(client, image_bytes):
    payload = client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    ).json()
    box = payload["detections"][0]["bbox"]
    assert box["x2"] > box["x1"]
    assert box["y2"] > box["y1"]
    assert box["width"] == box["x2"] - box["x1"]
    assert box["height"] == box["y2"] - box["y1"]


def test_predict_reports_the_image_size(client, image_bytes):
    payload = client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    ).json()
    assert payload["image_size"] == {"width": 320, "height": 240}


def test_predict_reports_the_server_latency(client, image_bytes):
    payload = client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    ).json()
    assert payload["inference_time_ms"] > 0


def test_predict_rejects_a_non_image_payload(client, broken_bytes):
    response = client.post(
        "/predict", files={"file": ("note.txt", broken_bytes, "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_image"


def test_predict_rejects_an_empty_payload(client):
    response = client.post(
        "/predict", files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400


def test_predict_requires_a_file(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_validates_the_confidence_threshold(client, image_bytes):
    response = client.post(
        "/predict?conf=1.5", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 422


def test_predict_forwards_the_threshold_to_the_detector(client, fake_detector, image_bytes):
    client.post(
        "/predict?conf=0.7&iou=0.3",
        files={"file": ("site.jpg", image_bytes, "image/jpeg")},
    )
    assert fake_detector.calls[-1]["conf"] == 0.7
    assert fake_detector.calls[-1]["iou"] == 0.3


def test_predict_does_not_leak_the_threshold_into_the_next_request(
    client, fake_detector, image_bytes
):
    client.post(
        "/predict?conf=0.7", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert fake_detector.calls[-1]["conf"] is None


def test_predict_answers_503_when_the_model_is_not_loaded(not_ready_client, image_bytes):
    response = not_ready_client.post(
        "/predict", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "model_unavailable"


# --------------------------------------------------------------------------- #
# /predict/visualize
# --------------------------------------------------------------------------- #


def test_predict_visualize_returns_a_png(client, image_bytes):
    response = client.post(
        "/predict/visualize", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_visualize_returns_an_image_of_the_same_size(client, image_bytes):
    response = client.post(
        "/predict/visualize", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    annotated = Image.open(io.BytesIO(response.content))
    assert annotated.size == (320, 240)


def test_predict_visualize_exposes_the_latency_header(client, image_bytes):
    response = client.post(
        "/predict/visualize", files={"file": ("site.jpg", image_bytes, "image/jpeg")}
    )
    assert "X-Inference-Time-Ms" in response.headers
    assert float(response.headers["X-Inference-Time-Ms"]) > 0


def test_predict_visualize_rejects_a_non_image(client, broken_bytes):
    response = client.post(
        "/predict/visualize", files={"file": ("note.txt", broken_bytes, "text/plain")}
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# /samples
# --------------------------------------------------------------------------- #


def test_samples_returns_a_list(client):
    response = client.get("/samples")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(payload["samples"])
    for sample in payload["samples"]:
        assert sample["url"].startswith("/samples/")


def test_unknown_sample_returns_404(client):
    response = client.get("/samples/does-not-exist.jpg")
    assert response.status_code == 404


def test_samples_rejects_path_traversal(client):
    response = client.get("/samples/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code in (404, 400)


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #


def test_openapi_schema_is_generated(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/model/metadata" in schema["paths"]
    assert "/predict/visualize" in schema["paths"]


def test_docs_are_served(client):
    response = client.get("/docs")
    assert response.status_code == 200
