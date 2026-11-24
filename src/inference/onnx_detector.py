"""YOLO11 inference through ONNX Runtime.

Why a hand-written post-processing pipeline?
--------------------------------------------
Ultralytics can load an ``.onnx`` file directly, but it does so through the
very same code path as PyTorch and it hides the graph topology. Writing the
decode + NMS explicitly gives three advantages that matter for this project:

* the **latency** we advertise is really the ONNX Runtime latency, and we can
  measure it separately from the post-processing cost;
* the **NMS threshold is under our control** (the brief explicitly asks for an
  optimised non-maximum suppression);
* the same function is unit-testable with NumPy only, without a GPU, without
  ONNX Runtime and without any checkpoint (see ``tests/test_onnx_decode.py``).

Graph contract
--------------
The model is exported with ``ultralytics.YOLO.export(format="onnx", nms=False)``
which produces a single output ``output0`` of shape ``(1, 4 + nc, 8400)``:

* the first 4 channels are ``cx, cy, w, h`` **in pixels of the letterboxed
  640x640 input**;
* the last ``nc`` channels are the class scores, already sigmoid-activated.

A graph exported *with* NMS (``output0`` of shape ``(1, 300, 6)``) is also
supported: the decoder detects the layout and skips the NMS stage.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.config import CFG
from src.utils.viz import letterbox, scale_boxes

try:  # pragma: no cover - optional dependency, exercised in the Docker image
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Non-maximum suppression
# --------------------------------------------------------------------------- #


def box_iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    """IoU between one ``xyxy`` box and an array of ``xyxy`` boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area_box + area_boxes - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: ``(N, 4)`` array of ``xyxy`` boxes.
        scores: ``(N,)`` array of confidences.
        iou_threshold: boxes overlapping more than this are discarded.

    Returns:
        Indices of the kept boxes, sorted by decreasing confidence.
    """
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """Class-aware NMS.

    Two boxes of *different* classes never suppress each other: a hard hat and
    a person may legitimately overlap. This is the behaviour Ultralytics uses
    (``agnostic=False``) and it is the one we want for PPE scenes.
    """
    if boxes.size == 0:
        return []
    keep: List[int] = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        sub_keep = nms(boxes[idx], scores[idx], iou_threshold)
        keep.extend(int(idx[k]) for k in sub_keep)
    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def decode_output(
    output: np.ndarray,
    num_classes: int,
    conf_threshold: float = CFG.CONF_THRESHOLD,
    iou_threshold: float = CFG.IOU_THRESHOLD,
    max_detections: int = CFG.MAX_DETECTIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLO output tensor into detections.

    Args:
        output: raw tensor of shape ``(1, 4 + nc, N)`` or ``(1, N, 4 + nc)``.
        num_classes: number of detection classes.
        conf_threshold: minimum score to keep a candidate.
        iou_threshold: NMS IoU threshold.
        max_detections: hard cap on the number of returned detections.

    Returns:
        ``(boxes_xyxy, scores, class_ids)`` in the coordinate frame of the
        letterboxed network input.
    """
    preds = np.asarray(output, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape: {np.asarray(output).shape}")

    # Layout detection: (4 + nc, N) is the raw head, (N, 4 + nc) the transposed
    # variant. The first dimension must be the smaller one.
    if preds.shape[0] == 4 + num_classes:
        preds = preds.T
    elif preds.shape[1] == 4 + num_classes:
        pass
    else:
        raise ValueError(
            "ONNX output does not match the expected YOLO layout: "
            f"got {preds.shape}, expected one axis of size {4 + num_classes}."
        )

    if preds.shape[1] < 4 + num_classes:
        raise ValueError(
            f"ONNX output has {preds.shape[1]} channels, "
            f"expected at least {4 + num_classes}."
        )

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4 : 4 + num_classes]

    best_scores = class_scores.max(axis=1)
    best_classes = class_scores.argmax(axis=1)

    keep_mask = best_scores >= conf_threshold
    if not np.any(keep_mask):
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes_cxcywh = boxes_cxcywh[keep_mask]
    scores = best_scores[keep_mask]
    class_ids = best_classes[keep_mask].astype(np.int64)

    cx, cy, w, h = boxes_cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = nms_per_class(boxes, scores, class_ids, iou_threshold)[:max_detections]
    return (
        boxes[keep].astype(np.float32),
        scores[keep].astype(np.float32),
        class_ids[keep],
    )


def decode_nms_output(
    output: np.ndarray, max_detections: int = CFG.MAX_DETECTIONS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a graph exported *with* NMS (``(1, N, 6)`` = xyxy, conf, cls)."""
    preds = np.asarray(output, dtype=np.float32).reshape(-1, 6)[:max_detections]
    return (
        preds[:, :4],
        preds[:, 4],
        preds[:, 5].astype(np.int64),
    )


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


class OnnxYoloDetector:
    """Thin, synchronous wrapper around an ONNX Runtime YOLO session."""

    def __init__(
        self,
        model_path: str | Path,
        num_classes: int = CFG.NUM_CLASSES,
        imgsz: int = CFG.EXPORT_IMGSZ,
        conf_threshold: float = CFG.CONF_THRESHOLD,
        iou_threshold: float = CFG.IOU_THRESHOLD,
        max_detections: int = CFG.MAX_DETECTIONS,
        providers: Sequence[str] | None = None,
    ) -> None:
        if ort is None:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is not installed. Install it with "
                "`pip install onnxruntime-gpu` (or `onnxruntime` on CPU)."
            )
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.num_classes = num_classes
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        available = ort.get_available_providers()
        if providers is None:
            preferred = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ]
            providers = preferred or None
        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    # -- helpers ----------------------------------------------------------- #
    def _preprocess(self, image):
        """Letterbox + normalise to ``(1, 3, H, W)`` float32 in [0, 1]."""
        canvas, scale, pad = letterbox(image, (self.imgsz, self.imgsz))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return array, scale, pad

    def predict(
        self,
        image,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> List[Dict]:
        """Run inference on a PIL image and return normalised detections."""
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        tensor, scale, pad = self._preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        raw = outputs[0]

        if raw.ndim == 3 and raw.shape[-1] == 6:
            boxes, scores, class_ids = decode_nms_output(raw, self.max_detections)
        else:
            boxes, scores, class_ids = decode_output(
                raw,
                num_classes=self.num_classes,
                conf_threshold=conf,
                iou_threshold=iou,
                max_detections=self.max_detections,
            )

        boxes = scale_boxes(boxes.tolist(), scale, pad, image.size)
        names = CFG.class_names()
        detections: List[Dict] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
            cid = int(cid)
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "class_id": cid,
                    "label": names[cid] if 0 <= cid < len(names) else str(cid),
                }
            )
        return detections

    def timed_predict(
        self,
        image,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> Tuple[List[Dict], float]:
        """Same as :meth:`predict` but also returns the inference time in ms."""
        start = time.perf_counter()
        detections = self.predict(
            image, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        return detections, (time.perf_counter() - start) * 1000.0


__all__ = [
    "box_iou",
    "nms",
    "nms_per_class",
    "decode_output",
    "decode_nms_output",
    "OnnxYoloDetector",
]
