"""Extract a handful of images for the one-click gallery of the frontend.

The gallery lives in ``backend/app/samples/``. When the dataset is available,
this script picks a few representative images (one per detected class when
possible) so the demo works with real construction-site photos.

Usage::

    python scripts/extract_samples.py --count 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CFG, DATASET_DIR  # noqa: E402
from src.data.dataset import class_presence_vector, parse_label_file  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "backend" / "app" / "samples"


def collect_classes(images_dir: Path, labels_dir: Path) -> Dict[str, Path]:
    """Return ``{class_name: image_path}`` - one image per class."""
    picked: Dict[str, Path] = {}
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        entries = parse_label_file(label_path)
        presence = class_presence_vector(entries)
        for class_id, present in enumerate(presence):
            if present and class_id not in picked:
                picked[CFG.PPE_CLASSES[class_id]] = image_path
        if len(picked) == CFG.NUM_CLASSES:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the sample gallery")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_DIR),
        help="Root of the prepared dataset (containing train/images)",
    )
    parser.add_argument("--count", type=int, default=8, help="Number of samples")
    parser.add_argument(
        "--out", default=str(SAMPLES_DIR), help="Destination of the gallery"
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    images_dir = dataset / "train" / "images"
    labels_dir = dataset / "train" / "labels"

    if not images_dir.exists():
        print(f"❌ No images found at '{images_dir}'.")
        print("   Prepare the dataset first: python scripts/prepare_dataset.py")
        return 1

    per_class = collect_classes(images_dir, labels_dir)
    if not per_class:
        print("❌ No labelled image found in the dataset.")
        return 1

    selected: List[Path] = list(per_class.values())[: args.count]
    # Complete the selection with the first images of the split.
    if len(selected) < args.count:
        for image_path in sorted(images_dir.glob("*.jpg")):
            if image_path not in selected:
                selected.append(image_path)
            if len(selected) >= args.count:
                break

    copied = 0
    for image_path in selected[: args.count]:
        target = out / image_path.name
        shutil.copyfile(image_path, target)
        copied += 1

    print(f"✅ {copied} sample(s) copied to {out}")
    for name in sorted(per_class):
        print(f"   - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
