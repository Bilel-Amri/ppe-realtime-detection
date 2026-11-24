"""Drawing helpers shared by the API and the notebook.

Everything here is implemented with Pillow only. Pillow is already a hard
dependency of Ultralytics, so the backend container does not need OpenCV just
to draw a bounding box.
"""

from __future__ import annotations

import colorsys
import io
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.config import CFG

#: Stable colour per class id (HSV wheel -> RGB). The mapping only depends on
#: the class id, so the API and the frontend always agree on the colours.
COLOR_CACHE: Dict[int, Tuple[int, int, int]] = {}


def class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a deterministic, well separated RGB colour for ``class_id``."""
    if class_id not in COLOR_CACHE:
        hue = (class_id * 0.618033988749895) % 1.0  # golden ratio offset
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        COLOR_CACHE[class_id] = (int(r * 255), int(g * 255), int(b * 255))
    return COLOR_CACHE[class_id]


def color_map(num_classes: int = CFG.NUM_CLASSES) -> Dict[int, Tuple[int, int, int]]:
    """Full ``class_id -> RGB`` mapping."""
    return {i: class_color(i) for i in range(num_classes)}


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font loading (DejaVu ships with most Linux images)."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def letterbox(
    image: Image.Image,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[float, float]]:
    """Resize ``image`` keeping its aspect ratio and pad the rest.

    Returns the padded image, the scale factor and the ``(pad_x, pad_y)``
    padding applied, in pixels of the *resized* image. Those three values are
    everything needed to map detections back to the original coordinates.
    """
    src_w, src_h = image.size
    dst_h, dst_w = new_shape
    scale = min(dst_h / src_h, dst_w / src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (dst_w, dst_h), color)
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, scale, (pad_x, pad_y)


def scale_boxes(
    boxes: Sequence[Sequence[float]],
    scale: float,
    pad: Tuple[float, float],
    src_size: Tuple[int, int],
) -> List[List[float]]:
    """Map boxes expressed in the letterboxed frame back to the source image.

    Args:
        boxes: ``[[x1, y1, x2, y2], ...]`` in letterboxed pixel coordinates.
        scale: scale factor applied by :func:`letterbox`.
        pad: ``(pad_x, pad_y)`` applied by :func:`letterbox`.
        src_size: ``(width, height)`` of the original image.

    Returns:
        The same boxes clipped to the original image bounds.
    """
    pad_x, pad_y = pad
    src_w, src_h = src_size
    out: List[List[float]] = []
    for x1, y1, x2, y2 in boxes:
        nx1 = (x1 - pad_x) / scale
        ny1 = (y1 - pad_y) / scale
        nx2 = (x2 - pad_x) / scale
        ny2 = (y2 - pad_y) / scale
        nx1 = max(0.0, min(float(src_w), nx1))
        ny1 = max(0.0, min(float(src_h), ny1))
        nx2 = max(0.0, min(float(src_w), nx2))
        ny2 = max(0.0, min(float(src_h), ny2))
        out.append([nx1, ny1, nx2, ny2])
    return out


def draw_detections(
    image: Image.Image,
    detections: Iterable[Dict],
    class_names: Sequence[str] | None = None,
    font_size: int = 18,
) -> Image.Image:
    """Draw bounding boxes, labels and confidence scores on ``image``.

    ``detections`` is an iterable of dicts with the keys ``bbox``
    (``[x1, y1, x2, y2]`` in source pixels), ``class_id``, ``confidence`` and
    optionally ``label``.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(font_size)
    names = list(class_names) if class_names else CFG.class_names()

    for det in detections:
        x1, y1, x2, y2 = (float(v) for v in det["bbox"])
        class_id = int(det.get("class_id", -1))
        conf = float(det.get("confidence", 0.0))
        label = det.get("label") or (
            names[class_id] if 0 <= class_id < len(names) else str(class_id)
        )
        color = class_color(class_id if class_id >= 0 else 0)

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        text = f"{label} {conf:.2f}"
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        except AttributeError:  # very old Pillow
            left, top, right, bottom = draw.textsize(text, font=font)
        text_w, text_h = right - left, bottom - top

        label_top = max(0.0, y1 - text_h - 8)
        draw.rectangle(
            [x1, label_top, x1 + text_w + 10, label_top + text_h + 8], fill=color
        )
        draw.text((x1 + 5, label_top + 2), text, fill=(255, 255, 255), font=font)

    return canvas


def to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes (used by the FastAPI responses)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = [
    "class_color",
    "color_map",
    "letterbox",
    "scale_boxes",
    "draw_detections",
    "to_png_bytes",
]
