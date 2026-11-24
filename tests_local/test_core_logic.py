"""Local validation suite for the parts of the project that do not need a GPU.

Run it with::

    python -m pytest tests_local -q

The suite covers the pieces that carry real logic and can therefore break
silently:

* the greedy multi-label stratification (``src/data/dataset.py``);
* the horizontal-flip label mirroring;
* the ONNX decode + class-aware NMS (``src/inference/onnx_detector.py``);
* the letterbox / un-letterbox coordinate round-trip (``src/utils/viz.py``).

It does **not** need ultralytics, torch, onnxruntime or a GPU: the decoding
functions are pure NumPy by design.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.config import CFG
from src.data.dataset import (
    class_presence_vector,
    flip_entries,
    parse_label_file,
    stratified_split,
)
from src.inference.onnx_detector import (
    box_iou,
    decode_output,
    nms,
    nms_per_class,
)
from src.utils.viz import letterbox, scale_boxes


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _record(name: str, classes) -> dict:
    return {
        "filename": name,
        "extension": ".jpg",
        "origin": "train",
        "source_path": f"/tmp/{name}.jpg",
        "label_path": f"/tmp/{name}.txt",
        "entries": [(c, 0.5, 0.5, 0.2, 0.2) for c in classes],
        "presence": class_presence_vector([(c,) for c in classes]),
    }


@pytest.fixture()
def corpus():
    """A corpus with a controlled class imbalance.

    40 images with ``Person`` (the dominant class), 10 with ``Vehicle`` and
    6 with ``Safety Cone`` (the rare ones). A naive "fill train first" split
    would starve ``valid`` and ``test`` of the rare classes.
    """
    records = []
    for i in range(40):
        records.append(_record(f"p{i:03d}", [5]))
    for i in range(10):
        records.append(_record(f"v{i:03d}", [5, 9]))
    for i in range(6):
        records.append(_record(f"c{i:03d}", [6]))
    return records


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_parse_label_file_skips_malformed_lines(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.2\n"   # valid
        "broken line\n"          # not enough tokens
        "3 0.1 0.2 0.3\n"        # incomplete coordinates
        "x 0.1 0.2 0.3 0.4\n"    # non numeric class id
        "9 0.1 0.2 0.3 0.4\n"    # valid
    )
    entries = parse_label_file(label)
    assert len(entries) == 2
    assert entries[0][0] == 0
    assert entries[1][0] == 9


def test_class_presence_vector_is_binary():
    vector = class_presence_vector([(5, 0, 0, 0, 0), (5, 0, 0, 0, 0), (9, 0, 0, 0, 0)])
    assert vector[5] == 1
    assert vector[9] == 1
    assert vector[0] == 0
    assert sum(vector) == 2


def test_stratified_split_respects_ratios(corpus):
    splits = stratified_split(corpus)
    total = sum(len(v) for v in splits.values())
    assert total == len(corpus)
    # 80 / 15 / 5 of 56 images
    assert len(splits["train"]) == pytest.approx(44, abs=2)
    assert len(splits["valid"]) == pytest.approx(8, abs=2)
    assert len(splits["test"]) == pytest.approx(3, abs=2)


def test_stratified_split_keeps_rare_classes_in_every_split(corpus):
    """The whole point of the stratification: rare classes reach the test set."""
    splits = stratified_split(corpus)
    for name in ("train", "valid", "test"):
        rare = sum(1 for r in splits[name] if r["presence"][9] or r["presence"][6])
        assert rare > 0, f"no rare-class image landed in '{name}'"


def test_stratified_split_is_deterministic(corpus):
    a = stratified_split(corpus, seed=42)
    b = stratified_split(corpus, seed=42)
    assert {k: [r["filename"] for r in v] for k, v in a.items()} == {
        k: [r["filename"] for r in v] for k, v in b.items()
    }


def test_stratified_split_is_disjoint(corpus):
    splits = stratified_split(corpus)
    names = [r["filename"] for records in splits.values() for r in records]
    assert len(names) == len(set(names))


def test_flip_entries_mirrors_the_x_center():
    entries = [(5, 0.2, 0.3, 0.1, 0.4)]
    flipped = flip_entries(entries)
    assert flipped[0][1] == pytest.approx(0.8)
    assert flipped[0][2] == pytest.approx(0.3)
    assert flipped[0][3] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# ONNX decoding / NMS
# --------------------------------------------------------------------------- #


def test_box_iou_identical_boxes_is_one():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou([0.0, 0.0, 10.0, 10.0], box)
    assert iou[0] == pytest.approx(1.0)


def test_box_iou_disjoint_boxes_is_zero():
    box = np.array([[10.0, 10.0, 20.0, 20.0]])
    iou = box_iou([0.0, 0.0, 10.0, 10.0], box)
    assert iou[0] == pytest.approx(0.0)


def test_nms_removes_duplicates():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 10.5, 10.5],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    assert len(keep) == 2
    assert keep[0] == 0  # highest score first


def test_nms_per_class_keeps_overlapping_different_classes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = np.array([0.9, 0.8])
    classes = np.array([5, 0])  # Person and Hardhat on the same pixels
    keep = nms_per_class(boxes, scores, classes, 0.5)
    assert len(keep) == 2


def test_decode_output_shapes_and_filtering():
    num_classes = CFG.NUM_CLASSES
    num_anchors = 100
    preds = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
    # anchor 0: a confident "Hardhat" at (100, 100) with size 20x20
    preds[0, 0, 0] = 100.0
    preds[0, 1, 0] = 100.0
    preds[0, 2, 0] = 20.0
    preds[0, 3, 0] = 20.0
    preds[0, 4, 0] = 0.95
    # anchor 1: a low score that must be filtered out
    preds[0, 0, 1] = 200.0
    preds[0, 1, 1] = 200.0
    preds[0, 2, 1] = 20.0
    preds[0, 3, 1] = 20.0
    preds[0, 4, 1] = 0.05

    boxes, scores, classes = decode_output(
        preds, num_classes=num_classes, conf_threshold=0.25, iou_threshold=0.45
    )
    assert boxes.shape == (1, 4)
    assert scores.shape == (1,)
    assert classes.shape == (1,)
    assert classes[0] == 0
    assert scores[0] == pytest.approx(0.95)
    assert boxes[0].tolist() == pytest.approx([90.0, 90.0, 110.0, 110.0])


def test_decode_output_returns_empty_when_nothing_passes():
    preds = np.zeros((1, 4 + CFG.NUM_CLASSES, 10), dtype=np.float32)
    boxes, scores, classes = decode_output(preds, CFG.NUM_CLASSES)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)


def test_decode_output_rejects_unexpected_layout():
    with pytest.raises(ValueError):
        decode_output(np.zeros((1, 7, 10), dtype=np.float32), CFG.NUM_CLASSES)


# --------------------------------------------------------------------------- #
# Letterbox round-trip
# --------------------------------------------------------------------------- #


def test_letterbox_round_trip_preserves_coordinates():
    """A box drawn in the source image must survive the letterbox trip."""
    image = Image.new("RGB", (800, 600), (10, 20, 30))
    source_box = [100.0, 50.0, 300.0, 250.0]

    canvas, scale, pad = letterbox(image, (640, 640))
    assert canvas.size == (640, 640)

    # Map the source box into the letterboxed frame.
    pad_x, pad_y = pad
    lb_box = [
        source_box[0] * scale + pad_x,
        source_box[1] * scale + pad_y,
        source_box[2] * scale + pad_x,
        source_box[3] * scale + pad_y,
    ]
    # ...and back.
    restored = scale_boxes([lb_box], scale, pad, image.size)[0]
    assert restored == pytest.approx(source_box, abs=1.0)


def test_letterbox_keeps_aspect_ratio():
    image = Image.new("RGB", (1000, 500), (0, 0, 0))
    canvas, scale, pad = letterbox(image, (640, 640))
    assert scale == pytest.approx(0.64)
    assert canvas.size == (640, 640)
    assert pad[1] > 0  # vertical padding, since the image is wide


def test_scale_boxes_clips_to_image_bounds():
    boxes = scale_boxes([[-50.0, -50.0, 5000.0, 5000.0]], 1.0, (0.0, 0.0), (100, 80))
    assert boxes[0] == [0.0, 0.0, 100.0, 80.0]
