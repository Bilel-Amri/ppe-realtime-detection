# PPE Detection — Real-time Object Detection with YOLO11 (Axe 3)

> **Module project — Deep Learning Avancé / MLOps**
> Industrialisation of a computer-vision solution: YOLO11 fine-tuned for
> Personal Protective Equipment (PPE) on construction sites, served by a
> FastAPI backend and a React frontend, with a complete MLOps cycle
> (DVC → MLflow → ONNX → Docker Compose → GitHub Actions).

| | |
|---|---|
| **Dataset** | Construction Site Safety (Roboflow / Kaggle) — 10 classes |
| **Model** | YOLO11n (Ultralytics), fine-tuned, 640×640 |
| **Backend** | FastAPI + Pydantic + ONNX Runtime (async, OpenAPI at `/docs`) |
| **Frontend** | React 18 + Vite + TailwindCSS (drag & drop, gallery, webcam) |
| **MLOps** | DVC, MLflow (+ Model Registry), ONNX FP32/FP16, Docker Compose, GitHub Actions |

---

## 1. Quick start

### 1.1 Run the whole stack (one command)

```bash
git clone <your-repo> ppe-realtime-detection
cd ppe-realtime-detection
cp .env.example .env
docker compose up --build
```

| Service | URL | What it is |
|---|---|---|
| Frontend | <http://localhost:8080> | React application (upload / gallery / webcam) |
| API | <http://localhost:8000> | FastAPI |
| Swagger | <http://localhost:8000/docs> | Interactive OpenAPI documentation |
| MLflow | <http://localhost:5000> | Tracking UI + Model Registry |

> **No model yet?** The API starts anyway: it falls back to the COCO
> pre-trained `yolo11n.pt` and advertises it honestly through
> `model_kind = "coco-fallback"` in `/model/metadata`. The application is
> therefore demonstrable from the very first `docker compose up`.

### 1.2 Run it locally (without Docker)

```bash
bash scripts/bootstrap.sh          # venv + dependencies + DVC init
source .venv/bin/activate

python -m src.models.train          # fine-tuning (GPU recommended)
python -m src.models.evaluate       # test split + ONNX export + MLflow

uvicorn backend.app.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```

### 1.3 Check the installation

```bash
python scripts/verify_project.py
```

The script reports, item by item, what is installed, what is missing and how
to fix it.

---

## 2. The pipeline, end to end

```
Kaggle / Roboflow
        │  scripts/download_dataset.py
        ▼
   data/raw/css-data                       (raw, versioned by DVC)
        │  scripts/prepare_dataset.py
        │    • stratified re-split (class-proportional)
        │    • rebalancing of the rare classes (flip + brightness)
        ▼
   data/processed/css-ppe + dataset.yaml
        │  python -m src.models.train          (YOLO11n, MLflow tracking)
        ▼
   artifacts/runs/.../best.pt
        │  python -m src.models/evaluate
        │    • test-split evaluation  → artifacts/metrics.json
        │    • ONNX export FP32 + FP16
        │    • MLflow logging + Model Registry
        ▼
   best.onnx  ──►  FastAPI (ONNX Runtime)  ──►  React (canvas + bars)
```

Every arrow is one command, and every command is idempotent.

---

## 3. Results

### 3.1 What has been measured

These are the **validation-split** metrics of the reference training run
(yolo11n, 30 epochs, batch 12, 640×640, one T4), taken from the run registry
of `notebooks/training_yolo11_ppe.ipynb`:

| Metric | Value |
|---|---|
| mAP@0.5 | **0.707** |
| mAP@0.5:0.95 | **0.467** |
| Precision | **0.836** |
| Recall | **0.629** |
| Training time | 1211.9 s |

### 3.2 What has **not** been measured yet

The **test-split** metrics (the ones that matter for the final evaluation)
are produced by `python -m src.models/evaluate.py` and written to
`artifacts/metrics.json`. They are deliberately **not** quoted here: no
number appears in this repository, in the API or in the report unless it was
actually computed on the held-out test split.

The same rule applies to the latency figures: they are measured by
`benchmark_onnx()` (median and p90, ONNX Runtime vs PyTorch, with a
warm-up) and written to `artifacts/benchmark.json`.

### 3.3 How to read the metrics from the running application

`GET /model/metadata` returns the content of `artifacts/metrics.json`; the
frontend displays it in the "Session" panel. If the file is absent, the
metrics are simply empty — the application never invents a number.

---

## 4. API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server state, GPU, loaded model and version |
| `GET` | `/model/metadata` | Architecture, label map, measured metrics |
| `POST` | `/predict` | Structured JSON prediction (`multipart/form-data`) |
| `POST` | `/predict/visualize` | Annotated PNG (boxes, labels, confidence) |
| `GET` | `/samples` | List of the gallery images |
| `GET` | `/samples/{name}` | One gallery image |

**Error contract**

| Code | When |
|---|---|
| `400` | The uploaded payload is not a decodable image |
| `413` | The upload exceeds 15 MB |
| `422` | The `conf` / `iou` query parameters are out of range |
| `503` | The model is not instantiated yet |

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@site.jpg" \
     -F "conf=0.3" | jq
```

```json
{
  "detections": [
    {
      "class_id": 0,
      "label": "Hardhat",
      "confidence": 0.91,
      "bbox": { "x1": 10.0, "y1": 12.0, "x2": 60.0, "y2": 80.0,
                "width": 50.0, "height": 68.0 }
    }
  ],
  "count": 1,
  "inference_time_ms": 12.34,
  "image_size": { "width": 320, "height": 240 },
  "model_kind": "ppe-finetuned",
  "conf_threshold": 0.25,
  "iou_threshold": 0.45
}
```

---

## 5. Frontend

* **Drag & drop** (or click, or paste with `Ctrl+V`) to analyse an image;
* **Sample gallery**: one click on a pre-loaded image;
* **Webcam**: single capture or continuous inference (one frame every 600 ms);
* **Bounding boxes** drawn on a canvas, one colour per class (the colour
  palette is shared with the backend, so the PNG and the canvas agree);
* **Confidence bars** animated per class;
* **Latency** measured twice: server-side (`inference_time_ms`) and
  client-side (round trip).

---

## 6. MLOps

| Tool | Usage |
|---|---|
| **DVC** | `dvc.yaml` declares the three stages (prepare → train → evaluate). `data/` and the weights never enter Git. |
| **MLflow** | Parameters, metrics, per-class AP and artefacts are logged; the model is registered in the **Model Registry** (stage `Staging`). |
| **ONNX** | FP32 and FP16 exports, `opset=12`, `simplify=True`, static shape. The API serves the ONNX graph through ONNX Runtime. |
| **Docker Compose** | 4 services: `frontend`, `backend`, `mlflow`, `postgres` (a SQL store is **required** by the Model Registry). |
| **GitHub Actions** | `ci-tests.yml` (flake8, black, pytest, ESLint, vite build, image build) and `docker-build.yml` (publish on tag). |

```bash
dvc init && dvc remote add -d local ../dvc_store
dvc repro                          # runs the whole pipeline
bash scripts/launch_mlflow.sh      # tracking server + registry
```

---

## 7. Project structure

```
ppe-realtime-detection/
├── .github/workflows/       ci-tests.yml, docker-build.yml
├── backend/
│   ├── app/                 main.py, schemas.py, inference.py, samples/
│   ├── tests/               test_api.py, conftest.py
│   └── Dockerfile           multi-stage build
├── frontend/
│   ├── src/components/      Dropzone, BoundingBoxCanvas, ConfidenceBars,
│   │                        SampleGallery, WebcamPanel, LatencyBadge, StatsPanel
│   ├── src/services/        api.js, palette.js
│   └── Dockerfile           build + nginx
├── src/
│   ├── config.py            single source of truth (classes, hyper-parameters)
│   ├── data/dataset.py      stratification, rebalancing, dataset.yaml
│   ├── models/              train.py, evaluate.py
│   ├── inference/           predictor.py, onnx_detector.py (decode + NMS)
│   └── utils/viz.py         letterbox, drawing, colour map
├── scripts/                 download, prepare, extract samples, verify, bootstrap
├── notebooks/               the training notebook (corrected)
├── tests_local/             offline test suite (no GPU needed)
├── data/                    raw/ + processed/ (versioned by DVC)
├── artifacts/               best.pt, best.onnx, metrics.json
├── docker-compose.yml
├── dvc.yaml
└── README.md
```

---

## 8. Tests

```bash
pytest backend/tests -v          # 26 API tests (needs the dependencies)
python tests_local/_minipytest.py        # 17 core tests, no GPU needed
python tests_local/check_backend_wiring.py  # 21 API wiring checks
node tests_local/check_frontend.js         # frontend static checks
```

`tests_local/` was written so that the logic can be verified **without a
GPU and without a network**: the ONNX decoding and the NMS are pure NumPy
functions, and the API wiring is checked against minimal stubs.

---

## 9. Two corrections applied to the original notebook

Both are documented in the report and defended during the defence:

1. **The re-split is now really stratified.** The original notebook filled
   `train` first and then `valid` and `test`; the rare classes ("Safety
   Cone", "Vehicle") ended up concentrated in `train` and almost absent from
   the test split, which makes a test metric meaningless. The new algorithm
   distributes the classes from the rarest to the most common, proportionally
   to the split sizes, and guarantees at least one image of a rare class per
   split.

2. **The 90° rotation was removed from the augmentation.** A 90° rotation
   invalidates every bounding box, and the label file was copied unchanged:
   the boxes no longer matched the pixels. Only geometry-preserving
   transforms are used now (horizontal flip, with `cx → 1 - cx` applied to
   the labels, and a photometric brightness change).

---

## 10. Reference

```bibtex
@misc{construction-site-safety_dataset,
  title  = {Construction Site Safety Dataset},
  type   = {Open Source Dataset},
  author = {Roboflow Universe Projects},
  howpublished = {\url{https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety}},
  year   = {2023}
}
```

Dataset: [Kaggle — construction-site-safety-image-dataset-roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) ·
Framework: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
