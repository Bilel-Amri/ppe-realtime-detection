"""Generate the placeholder images of the sample gallery.

The real gallery must contain **construction site photographs** - run
``python scripts/extract_samples.py`` once the dataset is prepared.

This script only creates synthetic placeholders so that the frontend has
something to display on a fresh clone of the repository (the API and the
gallery work immediately, the detections are meaningless on those images).
The generated images carry a visible "SAMPLE" watermark so that nobody
mistakes them for real dataset photos.

Usage::

    python scripts/generate_placeholder_samples.py --count 6
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "backend" / "app" / "samples"

PALETTE = {
    "sky_top": (126, 160, 195),
    "sky_bottom": (198, 214, 228),
    "ground": (142, 126, 106),
    "ground_dark": (116, 102, 85),
    "helmet": (245, 196, 12),
    "helmet_dark": (196, 150, 8),
    "vest": (243, 106, 26),
    "vest_dark": (190, 80, 16),
    "skin": (205, 160, 120),
    "clothes": (58, 72, 92),
    "cone": (240, 96, 30),
    "cone_light": (255, 255, 255),
    "machine": (108, 116, 124),
    "watermark": (255, 255, 255),
}


def gradient_sky(height: int, width: int) -> Image.Image:
    """Vertical gradient background."""
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    top = PALETTE["sky_top"]
    bottom = PALETTE["sky_bottom"]
    for y in range(height):
        ratio = y / max(1, height - 1)
        colour = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=colour)
    return image


def draw_person(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float, helmet: bool) -> None:
    """Draw a simplified worker."""
    head_r = int(11 * scale)
    body_w = int(26 * scale)
    body_h = int(52 * scale)

    # Legs
    draw.rectangle(
        [x - body_w // 2, y + body_h, x - 2, y + body_h + int(34 * scale)],
        fill=PALETTE["clothes"],
    )
    draw.rectangle(
        [x + 2, y + body_h, x + body_w // 2, y + body_h + int(34 * scale)],
        fill=PALETTE["clothes"],
    )
    # Torso + vest
    draw.rectangle([x - body_w // 2, y, x + body_w // 2, y + body_h], fill=PALETTE["clothes"])
    draw.rectangle(
        [
            x - body_w // 2,
            y + int(6 * scale),
            x + body_w // 2,
            y + body_h - int(6 * scale),
        ],
        fill=PALETTE["vest"],
    )
    draw.rectangle(
        [x - body_w // 2, y + int(20 * scale), x + body_w // 2, y + int(26 * scale)],
        fill=PALETTE["vest_dark"],
    )
    # Head
    draw.ellipse(
        [x - head_r, y - head_r * 2, x + head_r, y],
        fill=PALETTE["skin"],
    )
    # Helmet
    if helmet:
        draw.chord(
            [x - head_r - 2, y - head_r * 2 - 6, x + head_r + 2, y - head_r + 6],
            180,
            360,
            fill=PALETTE["helmet"],
        )
        draw.rectangle(
            [x - head_r - 4, y - head_r - 1, x + head_r + 4, y - head_r + 2],
            fill=PALETTE["helmet_dark"],
        )


def draw_cone(draw: ImageDraw.ImageDraw, x: int, base_y: int, scale: float) -> None:
    """Draw a traffic cone."""
    w = int(34 * scale)
    h = int(44 * scale)
    draw.polygon(
        [(x - w // 2, base_y), (x + w // 2, base_y), (x, base_y - h)],
        fill=PALETTE["cone"],
    )
    draw.rectangle(
        [x - w // 4, base_y - int(h * 0.55), x + w // 4, base_y - int(h * 0.40)],
        fill=PALETTE["cone_light"],
    )
    draw.rectangle(
        [x - w // 2 - 3, base_y - 3, x + w // 2 + 3, base_y + 3],
        fill=PALETTE["helmet_dark"],
    )


def draw_machine(draw: ImageDraw.ImageDraw, x: int, base_y: int, scale: float) -> None:
    """Draw a simplified excavator / vehicle block."""
    w = int(120 * scale)
    h = int(58 * scale)
    draw.rectangle([x - w // 2, base_y - h, x + w // 2, base_y], fill=PALETTE["machine"])
    draw.rectangle(
        [x - w // 4, base_y - h - int(22 * scale), x + w // 4, base_y - h + 4],
        fill=PALETTE["machine"],
    )
    draw.ellipse(
        [x - w // 2 - 8, base_y - 16, x - w // 4 + 4, base_y + 12],
        fill=(40, 40, 44),
    )
    draw.ellipse(
        [x + w // 4 - 4, base_y - 16, x + w // 2 + 8, base_y + 12],
        fill=(40, 40, 44),
    )


def draw_watermark(draw: ImageDraw.ImageDraw, width: int, height: int, index: int) -> None:
    text = f"SAMPLE {index} - replace with dataset images"
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = None
    draw.rectangle([0, height - 42, width, height], fill=(15, 23, 42))
    draw.text((12, height - 34), text, fill=PALETTE["watermark"], font=font)


def make_scene(index: int, width: int = 640, height: int = 480) -> Image.Image:
    random.seed(index)
    image = gradient_sky(height, width)
    draw = ImageDraw.Draw(image)

    # Ground
    horizon = int(height * 0.62)
    draw.rectangle([0, horizon, width, height], fill=PALETTE["ground"])
    draw.rectangle([0, horizon, width, horizon + 8], fill=PALETTE["ground_dark"])

    scale = 1.0 + 0.15 * (index % 3)

    if index % 3 == 0:
        draw_machine(draw, int(width * 0.72), horizon + int(70 * scale), scale * 0.9)
    if index % 2 == 0:
        draw_person(draw, int(width * 0.32), horizon - int(20 * scale), scale, helmet=True)
    else:
        draw_person(draw, int(width * 0.32), horizon - int(20 * scale), scale, helmet=False)
    if index % 4 in (1, 2):
        draw_person(
            draw,
            int(width * 0.55),
            horizon - int(10 * scale),
            scale * 0.9,
            helmet=index % 2 == 1,
        )
    draw_cone(draw, int(width * 0.18), horizon + int(40 * scale), scale)
    draw_cone(draw, int(width * 0.86), horizon + int(60 * scale), scale * 0.8)

    draw_watermark(draw, width, height, index + 1)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the placeholder gallery")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--out", default=str(SAMPLES_DIR))
    args = parser.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    written: List[Tuple[str, Path]] = []
    for index in range(args.count):
        image = make_scene(index)
        path = out / f"sample_{index + 1:02d}.jpg"
        image.save(path, quality=90)
        written.append((path.name, path))

    print(f"✅ {len(written)} placeholder sample(s) written to {out}")
    for name, path in written:
        print(f"   - {name} ({path.stat().st_size // 1024} KB)")
    print()
    print("⚠️  These are synthetic placeholders. Replace them with real dataset")
    print("    images:  python scripts/extract_samples.py --count 8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
