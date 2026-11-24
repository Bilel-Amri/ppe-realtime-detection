"""Build the cleaned training notebook from the original one.

The original notebook (``training-yolo-v11-to-detect-ppe-construction-site.ipynb``)
is kept as the historical reference. This script produces
``notebooks/training_yolo11_ppe.ipynb``: the same walkthrough, with

* the **stratification** replaced by the class-proportional algorithm (the
  "fill train first" approach starved the rare classes in valid/test);
* the **90 degree rotation** removed from the augmentation (it invalidates the
  bounding boxes, which were copied unchanged);
* the **final evaluation, ONNX export and MLflow logging** section appended -
  the part that closes the loop with the API.

Usage::

    python scripts/build_notebook.py \
        --source ../attachments/<id>/training-yolo-v11-to-detect-ppe-construction-site.ipynb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def markdown(source: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# --------------------------------------------------------------------------- #
# Replacement cells
# --------------------------------------------------------------------------- #

STRATIFICATION_CELL = '''# ---------------------------------------------
# Stratified re-split (class-proportional)
#
# The original notebook filled `train` until it was full, then `valid`, then
# `test`. That is NOT stratification: the rare classes ("Safety Cone",
# "Vehicle") ended up concentrated in `train` and were almost absent from the
# test split, which makes the test metrics meaningless.
#
# The algorithm below processes the classes from the rarest to the most common
# and distributes their images proportionally to the split sizes, with a
# "coverage" rule that guarantees at least one image of a rare class per
# split. This is the same implementation as `src/data/dataset.py` in the
# repository.
# ---------------------------------------------------------------------------
def largest_remainder(total, ratios, names):
    raw = {n: total * ratios[n] for n in names}
    counts = {n: int(raw[n]) for n in names}
    leftover = total - sum(counts.values())
    order = sorted(names, key=lambda n: (-(raw[n] - counts[n]), names.index(n)))
    for i in range(leftover):
        counts[order[i % len(order)]] += 1
    return counts


def stratified_split(df_total, ratios, seed=CFG.SEED):
    names = list(ratios.keys())
    total = len(df_total)
    target_sizes = largest_remainder(total, ratios, names)
    capacity = dict(target_sizes)

    # Global presence count per class (how rare each class is).
    presence_counts = {cls: int(df_total[cls].sum()) for cls in CFG.PPE_CLASSES}
    class_order = sorted(CFG.PPE_CLASSES, key=lambda c: (presence_counts[c], c))

    assigned = {n: [] for n in names}
    pool = df_total.copy()

    for class_id in class_order:
        candidates = pool[pool[class_id] == 1]
        n = len(candidates)
        if n == 0:
            continue

        quotas = largest_remainder(n, ratios, names)

        # Coverage: at least one image per split when the class allows it.
        if n >= len(names):
            bumped = {name: max(1, quotas[name]) for name in names}
            while sum(bumped.values()) > n:
                reducible = [name for name in names if bumped[name] > 1]
                if not reducible:
                    break
                biggest = max(reducible, key=lambda name: (bumped[name], -names.index(name)))
                bumped[biggest] -= 1
            quotas = bumped

        # Capacity: never exceed the remaining quota of a split.
        quotas = {name: min(quotas[name], capacity[name]) for name in names}
        overflow = n - sum(quotas.values())
        while overflow > 0:
            receivers = [name for name in names if capacity[name] > quotas[name]]
            if not receivers:
                break
            share, extra = divmod(overflow, len(receivers))
            for index, name in enumerate(receivers):
                add = min(share + (1 if index < extra else 0), capacity[name] - quotas[name])
                quotas[name] += add
                overflow -= add
            if all(quotas[name] == capacity[name] for name in receivers):
                break

        candidates = candidates.sample(frac=1, random_state=seed).sort_values("filename")
        cursor = 0
        for name in names:
            take = quotas[name]
            if take <= 0:
                continue
            chosen = candidates.iloc[cursor:cursor + take]
            cursor += take
            assigned[name].append(chosen)
            capacity[name] -= take
        pool = candidates.iloc[cursor:]

    # Fill the remaining capacity with the images that are left.
    leftovers = pool.sort_values("filename")
    cursor = 0
    while cursor < len(leftovers):
        name = max(names, key=lambda n: (capacity[n], -names.index(n)))
        if capacity[name] <= 0:
            break
        assigned[name].append(leftovers.iloc[cursor:cursor + 1])
        capacity[name] -= 1
        cursor += 1

    frames = []
    for name in names:
        part = pd.concat(assigned[name]) if assigned[name] else df_total.iloc[0:0]
        part = part.copy()
        part["origin"] = name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


df_split = stratified_split(df_total, CFG.TARGET_RATIOS)
df_train = df_split[df_split["origin"] == "train"].reset_index(drop=True)
df_valid = df_split[df_split["origin"] == "valid"].reset_index(drop=True)
df_test = df_split[df_split["origin"] == "test"].reset_index(drop=True)

print(f"\\U0001F4CA Train: {len(df_train)} images")
print(f"\\U0001F4CA Validation: {len(df_valid)} images")
print(f"\\U0001F4CA Test: {len(df_test)} images")

# The rare classes must be present in every split, otherwise the test metrics
# cannot be trusted.
for name, frame in (("train", df_train), ("valid", df_valid), ("test", df_test)):
    missing = [
        CFG.PPE_CLASSES[c]
        for c in CFG.PPE_CLASSES
        if frame[c].sum() == 0
    ]
    print(f"   {name:>6}: classes absent -> {missing if missing else 'none'}")
'''

AUGMENTATION_MARKDOWN = """### ⚠️ Corrected augmentation (geometry preserving)

The original notebook produced **three** copies of each selected image:
horizontal flip, **90° rotation** and brightness.

The rotation is a **label corruption bug**: the label file was copied
unchanged, so after a 90° rotation every bounding box pointed at the wrong
pixels. Rotating an image by 90° swaps the axes and mirrors the coordinates -
the boxes must be recomputed.

The augmentation below keeps only the two **geometry preserving** transforms:

| Transform | Geometry | Labels |
|---|---|---|
| Horizontal flip | mirrored on X | `cx -> 1 - cx`, recomputed |
| Brightness (×1.25) | unchanged | unchanged |

The same fix is applied in `src/data/dataset.py` (`augment_train_split`).
"""

AUGMENTATION_CELL = '''# ---------------------------------------------------------------------------
# Rebalancing the rare classes with GEOMETRY PRESERVING augmentations only.
#   - horizontal flip : the image is mirrored, the label x-centers follow
#   - brightness      : photometric only, the geometry does not change
# The 90 degree rotation of the original notebook was removed: it invalidates
# the bounding boxes.
# ---------------------------------------------------------------------------
PPE_CLASSES_INV = {v: k for k, v in CFG.PPE_CLASSES.items()}

target_classes = ["Safety Cone", "Vehicle"]
excluded_classes = ["Person", "Machinery"]

img_dir = os.path.join(CFG.WORKING_PATH, "train", "images")
label_dir = os.path.join(CFG.WORKING_PATH, "train", "labels")

cloned_count = 0

for _, row in df_train.iterrows():
    filename = row["filename"]

    contains_target = any(row[PPE_CLASSES_INV[cls]] == 1 for cls in target_classes)
    excluded_count = sum(row[PPE_CLASSES_INV[cls]] for cls in excluded_classes)

    if not (contains_target and excluded_count <= 1):
        continue

    img_path = os.path.join(img_dir, filename + ".jpg")
    label_path = os.path.join(label_dir, filename + ".txt")

    if not (os.path.exists(img_path) and os.path.exists(label_path)):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    with open(label_path, "r") as f:
        label_lines = [line.strip() for line in f.readlines() if line.strip()]

    # --- horizontal flip: the labels are mirrored with the image ---------- #
    flipped = cv2.flip(img, 1)
    flipped_lines = []
    for line in label_lines:
        parts = line.split()
        cls, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), parts[3], parts[4]
        flipped_lines.append(f"{cls} {1.0 - cx:.6f} {cy:.6f} {w} {h}")

    cv2.imwrite(os.path.join(img_dir, f"{filename}_flip.jpg"), flipped)
    with open(os.path.join(label_dir, f"{filename}_flip.txt"), "w") as f:
        f.write("\\n".join(flipped_lines) + "\\n")

    # --- brightness: photometric only, the labels stay identical ----------- #
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    cv2.imwrite(os.path.join(img_dir, f"{filename}_bright.jpg"), bright)
    with open(os.path.join(label_dir, f"{filename}_bright.txt"), "w") as f:
        f.write("\\n".join(label_lines) + "\\n")

    cloned_count += 1

print(f"\\u2705 {cloned_count} augmented images created (horizontal flip + brightness).")
'''

FINAL_SECTION_MARKDOWN = """---

# 🧪 Final evaluation, ONNX export & MLflow tracking

This is the section that closes the loop with the FastAPI service:

1. the trained model is evaluated on the **test split** (never on `val`);
2. it is exported to **ONNX** in FP32 and FP16 - the API serves the ONNX
   graph through ONNX Runtime;
3. the metrics and the artefacts are pushed to **MLflow**, and the model is
   registered in the Model Registry.

> The numbers printed below are **measured on the test split**. They are
> never estimated and never copied from the validation split.
"""

FINAL_EVAL_CELL = '''# ---------------------------------------------------------------------------
# 1) Final test-split evaluation
# ---------------------------------------------------------------------------
best = YOLO(CFG.MODEL_PATH)
m = best.val(data=dataset_yaml, split="test", imgsz=640)
print("TEST mAP50:", m.box.map50, "mAP50-95:", m.box.map, "P:", m.box.mp, "R:", m.box.mr)
print(pd.DataFrame({"class": list(CFG.PPE_CLASSES.values()), "AP50-95": m.box.maps}))

# The metrics are also written to a JSON file: the API (/model/metadata) and
# the LaTeX report read exactly this file, so the application can never
# display a number that was not measured.
import json

metrics = {
    "measured_at": pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "test_mAP50": float(m.box.map50),
    "test_mAP50_95": float(m.box.map),
    "precision": float(m.box.mp),
    "recall": float(m.box.mr),
    "per_class_ap50_95": {
        name: float(m.box.maps[i]) for i, name in enumerate(CFG.PPE_CLASSES.values())
    },
}
with open(os.path.join(CFG.WORKING_PATH, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("\\u2705 metrics.json written")
'''

ONNX_CELL = '''# ---------------------------------------------------------------------------
# 2) ONNX export (FP32 + FP16)
#
# Ultralytics always writes the file under the name "best.onnx", so exporting
# twice leaves a single file: the FP16 graph would overwrite the FP32 one.
# Each export is therefore renamed right after it is produced.
# ---------------------------------------------------------------------------
best.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
shutil.copyfile(
    os.path.join(CFG.WORKING_PATH, "runs/detect/train/weights/best.onnx"),
    os.path.join(CFG.WORKING_PATH, "runs/detect/train/weights/best_fp32.onnx"),
)

best.export(format="onnx", imgsz=640, opset=12, simplify=True, half=True, device=0)
shutil.copyfile(
    os.path.join(CFG.WORKING_PATH, "runs/detect/train/weights/best.onnx"),
    os.path.join(CFG.WORKING_PATH, "runs/detect/train/weights/best_fp16.onnx"),
)

print("\\u2705 ONNX graphs exported (FP32 + FP16)")
'''

MLFLOW_CELL = '''# ---------------------------------------------------------------------------
# 3) MLflow logging (+ Model Registry)
# ---------------------------------------------------------------------------
import mlflow

mlflow.set_tracking_uri(CFG.MLFLOW_TRACKING_URI)
mlflow.set_experiment("ppe-detection")

with mlflow.start_run(run_name="yolo11n_30ep"):
    mlflow.log_params({**train_data, "model": CFG.BASE_MODEL})
    mlflow.log_metrics({"test_mAP50": m.box.map50, "test_mAP50_95": m.box.map,
                        "precision": m.box.mp, "recall": m.box.mr})
    mlflow.log_artifact(CFG.MODEL_PATH)
    mlflow.log_artifact(CFG.MODEL_PATH.replace(".pt", ".onnx"))

    # Per-class AP50-95
    for i, name in enumerate(CFG.PPE_CLASSES.values()):
        mlflow.log_metric(f"ap50_95/{name}", float(m.box.maps[i]))

    # --- Model Registry --------------------------------------------------- #
    # The registry requires a SQL backend store (PostgreSQL in docker-compose).
    try:
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{os.path.basename(CFG.MODEL_PATH)}"
        mlflow.register_model(model_uri, "PPE-YOLO11")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(
            "name='PPE-YOLO11'", order_by=["version_number DESC"], max_results=1
        )
        if versions:
            client.transition_model_version_stage(
                name="PPE-YOLO11", version=versions[0].version, stage="Staging"
            )
            print(f"\\u2705 Model registered (version {versions[0].version}, stage Staging)")
    except Exception as exc:
        print(f"\\u26A0\\uFE0F Model Registry step skipped: {exc}")

print("\\u2705 Run logged to MLflow")
'''

LATENCY_CELL = '''# ---------------------------------------------------------------------------
# 4) Latency benchmark: PyTorch vs ONNX Runtime
#
# The comparison is measured on a dummy tensor, with a warm-up, and reports
# the median and the 90th percentile.
# ---------------------------------------------------------------------------
import numpy as np

def benchmark(session, input_name, dummy, runs=50):
    for _ in range(5):
        session.run(None, {input_name: dummy})
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings)), float(np.percentile(timings, 90))

try:
    import onnxruntime as ort

    onnx_path = os.path.join(CFG.WORKING_PATH, "runs/detect/train/weights/best_fp32.onnx")
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(onnx_path, providers=providers or None)
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

    onnx_median, onnx_p90 = benchmark(session, input_name, dummy)
    provider = session.get_providers()[0]
    print(f"ONNX Runtime ({provider}): median {onnx_median:.1f} ms, "
          f"p90 {onnx_p90:.1f} ms")

    import torch
    model = YOLO(CFG.MODEL_PATH)
    tensor = torch.zeros(1, 3, 640, 640)
    for _ in range(5):
        model.predict(tensor, verbose=False)
    timings = []
    for _ in range(50):
        start = time.perf_counter()
        model.predict(tensor, verbose=False)
        timings.append((time.perf_counter() - start) * 1000.0)
    torch_median = float(np.median(timings))
    print(f"PyTorch: median {torch_median:.1f} ms")
    print(f"\\u26A1 Speed-up: {torch_median / max(onnx_median, 1e-6):.2f}x")
except Exception as exc:
    print(f"\\u26A0\\uFE0F Benchmark skipped: {exc}")
'''


def build(source_path: Path) -> Dict[str, Any]:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells: List[Dict[str, Any]] = notebook["cells"]

    def source_of(cell: Dict[str, Any]) -> str:
        return "".join(cell.get("source", []))

    # -- 1. replace the naive split by the stratified one ------------------- #
    split_replaced = False
    for index, cell in enumerate(cells):
        text = source_of(cell)
        if "Step 1: Fill TRAIN dataset first" in text:
            cells[index] = code(STRATIFICATION_CELL)
            split_replaced = True
            break
    if not split_replaced:
        raise SystemExit("Could not locate the split cell in the source notebook.")

    # -- 2. replace the augmentation (remove the rotation) ----------------- #
    for index, cell in enumerate(cells):
        text = source_of(cell)
        if "cloned_count" in text and "rotated" in text:
            cells[index] = code(AUGMENTATION_CELL)
            cells.insert(index, markdown(AUGMENTATION_MARKDOWN))
            break

    # -- 3. append the final evaluation section ---------------------------- #
    cells.append(markdown(FINAL_SECTION_MARKDOWN))
    cells.append(code(FINAL_EVAL_CELL))
    cells.append(code(ONNX_CELL))
    cells.append(code(MLFLOW_CELL))
    cells.append(code(LATENCY_CELL))

    # -- 4. refresh the header --------------------------------------------- #
    header = (
        "> **This notebook is the historical reference of the project.**\n"
        "> The production pipeline lives in `src/` (training, evaluation, ONNX\n"
        "> export, MLflow) and the API in `backend/`. The cells below have been\n"
        "> corrected: the stratification is class-proportional and the\n"
        "> augmentation no longer rotates the images (which corrupted the\n"
        "> bounding boxes).\n"
    )
    for cell in cells:
        if cell["cell_type"] == "markdown" and source_of(cell).startswith("# \U0001F4CC"):
            cell["source"] = (header + "".join(cell["source"])).splitlines(keepends=True)
            break

    notebook["cells"] = cells
    notebook["metadata"].setdefault("kernelspec", {})
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.10.12",
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "pygments_lexer": "ipython3",
        "codemirror_mode": {"name": "ipython", "version": 3},
    }
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the cleaned notebook")
    parser.add_argument("--source", required=True, help="Path to the original notebook")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "notebooks" / "training_yolo11_ppe.ipynb"),
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        print(f"❌ Source notebook not found: {source}")
        return 1

    notebook = build(source)
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    print(f"✅ Notebook written to {output}")
    print(f"   cells: {len(notebook['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
