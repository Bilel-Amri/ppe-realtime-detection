"""Dataset preparation: re-split, rebalance and describe the CSS dataset.

This module replaces the ad-hoc notebook cells with a deterministic,
testable pipeline. Two things changed on purpose, and both are worth
defending in the technical report:

1. **Real stratification instead of "fill train first".**
   The notebook filled ``train`` until it was full, then ``valid``, then
   ``test``. That is *not* stratification: the rare classes ended up
   concentrated in ``train`` and almost absent from ``test``, which makes the
   test metrics unusable (the guide explicitly lists this as a critical
   mistake, §9 error #3). Here the split is computed with a greedy
   multi-label stratification that matches the per-class prevalence of every
   split to the global prevalence.

2. **Geometry-preserving augmentation only.**
   The notebook created three copies of the selected images: horizontal
   flip, **90 degree rotation** and brightness. A 90 degree rotation
   invalidates every bounding box: the label file was copied unchanged, so
   the boxes no longer match the pixels. The rotation is dropped here. The
   flip is applied on the image *and* on the labels (``cx -> 1 - cx``), and
   the brightness change is photometric, hence geometry-safe.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from src.config import CFG

#: A record describes one image and its annotations.
Record = Dict[str, object]


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #


def parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO ``.txt`` label file.

    Returns a list of ``(class_id, cx, cy, w, h)`` with normalised coordinates.
    Malformed lines are skipped instead of crashing the pipeline.
    """
    entries: List[Tuple[int, float, float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            entries.append((class_id, cx, cy, w, h))
    return entries


def class_presence_vector(entries: Iterable[Tuple[int, ...]]) -> List[int]:
    """Binary presence vector (1 if the class appears at least once)."""
    vector = [0] * CFG.NUM_CLASSES
    for entry in entries:
        class_id = entry[0]
        if 0 <= class_id < CFG.NUM_CLASSES:
            vector[class_id] = 1
    return vector


def collect_records(
    source_dir: Path, split_names: Sequence[str] = ("train", "valid", "test")
) -> List[Record]:
    """Scan ``source_dir`` and build the record list.

    ``source_dir`` must follow the CSS layout::

        source_dir/train/images/*.jpg
        source_dir/train/labels/*.txt
    """
    records: List[Record] = []
    for split in split_names:
        images_dir = source_dir / split / "images"
        labels_dir = source_dir / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = image_path.stem
            label_path = labels_dir / f"{stem}.txt"
            entries = parse_label_file(label_path) if label_path.exists() else []
            records.append(
                {
                    "filename": stem,
                    "extension": image_path.suffix.lower(),
                    "origin": split,
                    "source_path": str(image_path),
                    "label_path": str(label_path) if label_path.exists() else None,
                    "entries": entries,
                    "presence": class_presence_vector(entries),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Stratification
# --------------------------------------------------------------------------- #


def _global_prevalence(records: Sequence[Record]) -> List[float]:
    """Fraction of images containing each class, over the whole corpus."""
    if not records:
        return [0.0] * CFG.NUM_CLASSES
    counts = [0] * CFG.NUM_CLASSES
    for record in records:
        presence = record["presence"]
        for class_id, present in enumerate(presence):  # type: ignore[arg-type]
            if present:
                counts[class_id] += 1
    total = float(len(records))
    return [c / total for c in counts]


def _largest_remainder(
    total: int, ratios: Dict[str, float], names: Sequence[str]
) -> Dict[str, int]:
    """Split ``total`` items across ``names`` proportionally to ``ratios``.

    Uses the *largest remainder* method so that the returned sizes sum exactly
    to ``total`` (no rounding drift) and the split is deterministic.
    """
    if total <= 0:
        return {name: 0 for name in names}
    raw = {name: total * ratios[name] for name in names}
    counts = {name: int(raw[name]) for name in names}
    leftover = total - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (-(raw[name] - counts[name]), list(names).index(name)),
    )
    for index in range(leftover):
        counts[order[index % len(order)]] += 1
    return counts


def stratified_split(
    records: Sequence[Record],
    ratios: Dict[str, float] | None = None,
    seed: int = CFG.SEED,
) -> Dict[str, List[Record]]:
    """Class-proportional stratification of ``records`` into splits.

    Algorithm (iterative multi-label stratification)
    -------------------------------------------------
    1. The **size** of every split is fixed first: ``ratio * total``, rounded
       with the largest-remainder method so the sizes sum to ``total``.
    2. The classes are then processed **from the rarest to the most common**.
       For each class, its images are distributed across the splits
       proportionally to the split sizes, with two refinements:
       * *coverage* -- when a class has at least one image per split, every
         split receives at least one, so that a rare class is never absent
         from the test set (this is what makes the test metrics meaningful);
       * *capacity* -- a split never receives more images than its remaining
         quota, and the overflow is pushed to the splits that still have room.
    3. The images that contain no still-unassigned class fill the remaining
       capacity.

    The procedure is deterministic: the seed only drives the tie-breaking
    order of images with identical rarity scores.
    """
    ratios = dict(ratios or CFG.TARGET_RATIOS)
    split_names = list(ratios.keys())
    if not records:
        return {name: [] for name in split_names}

    total = len(records)
    target_sizes = _largest_remainder(total, ratios, split_names)
    capacity = dict(target_sizes)

    assigned: Dict[str, List[Record]] = {name: [] for name in split_names}
    pool: List[Record] = list(records)

    rng = random.Random(seed)

    # Global presence counts drive the "rarest class first" ordering.
    presence_counts = [0] * CFG.NUM_CLASSES
    for record in records:
        presence = record["presence"]
        for class_id, present in enumerate(presence):  # type: ignore[arg-type]
            if present:
                presence_counts[class_id] += 1

    def rarity(record: Record) -> float:
        """How hard the image is to place: the rarer its classes, the higher."""
        presence = record["presence"]
        score = 0.0
        for class_id, present in enumerate(presence):  # type: ignore[arg-type]
            if present:
                score += 1.0 / max(presence_counts[class_id], 1)
        return score

    class_order = sorted(
        range(CFG.NUM_CLASSES),
        key=lambda class_id: (presence_counts[class_id], class_id),
    )

    placed: set = set()

    for class_id in class_order:
        candidates = [
            record for record in pool if record["presence"][class_id]  # type: ignore[index]
        ]
        if not candidates:
            continue

        n = len(candidates)
        quotas = _largest_remainder(n, ratios, split_names)

        # Coverage: guarantee at least one image per split when possible.
        if n >= len(split_names):
            bumped = {name: max(1, quotas[name]) for name in split_names}
            while sum(bumped.values()) > n:
                reducible = [name for name in split_names if bumped[name] > 1]
                if not reducible:
                    break
                biggest = max(reducible, key=lambda name: (bumped[name], -split_names.index(name)))
                bumped[biggest] -= 1
            quotas = bumped

        # Capacity: never exceed the remaining quota of a split, push the
        # overflow to the splits that still have room.
        quotas = {name: min(quotas[name], capacity[name]) for name in split_names}
        overflow = n - sum(quotas.values())
        while overflow > 0:
            receivers = [name for name in split_names if capacity[name] > quotas[name]]
            if not receivers:
                break
            share = overflow // len(receivers)
            extra = overflow % len(receivers)
            for index, name in enumerate(receivers):
                add = share + (1 if index < extra else 0)
                add = min(add, capacity[name] - quotas[name])
                quotas[name] += add
                overflow -= add
            if all(quotas[name] == capacity[name] for name in receivers):
                break

        # Deterministic ordering of the images of this class.
        candidates.sort(key=lambda r: (-rarity(r), rng.random(), str(r["filename"])))

        cursor = 0
        for name in split_names:
            take = quotas[name]
            if take <= 0:
                continue
            chosen = candidates[cursor : cursor + take]
            cursor += take
            for record in chosen:
                assigned[name].append(record)
                capacity[name] -= 1
                placed.add(id(record))
        pool = [record for record in pool if id(record) not in placed]

    # Fill the remaining capacity with the images that are left.
    leftovers = sorted(pool, key=lambda r: (-rarity(r), str(r["filename"])))
    cursor = 0
    while cursor < len(leftovers):
        name = max(split_names, key=lambda n: (capacity[n], -split_names.index(n)))
        if capacity[name] <= 0:
            break
        assigned[name].append(leftovers[cursor])
        capacity[name] -= 1
        cursor += 1

    # Anything still unassigned (split sizes already saturated) goes to train.
    if cursor < len(leftovers):
        assigned[split_names[0]].extend(leftovers[cursor:])

    for name in split_names:
        assigned[name].sort(key=lambda r: str(r["filename"]))
    return assigned


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def select_augmentation_candidates(
    train_records: Sequence[Record],
    target_classes: Sequence[str] = tuple(CFG.AUGMENT_TARGET_CLASSES),
    excluded_classes: Sequence[str] = tuple(CFG.AUGMENT_EXCLUDED_CLASSES),
    max_excluded: int = CFG.AUGMENT_MAX_EXCLUDED,
) -> List[Record]:
    """Select the images eligible for a rebalancing copy.

    An image qualifies when it contains at least one *target* class (a rare
    one) and at most ``max_excluded`` instance of the dominant classes, so
    that boosting the rare classes does not amplify the imbalance we are
    trying to fix.
    """
    name_to_id = CFG.class_name_to_id()
    target_ids = {name_to_id[c] for c in target_classes if c in name_to_id}
    excluded_ids = {name_to_id[c] for c in excluded_classes if c in name_to_id}

    candidates: List[Record] = []
    for record in train_records:
        entries = record["entries"]
        class_ids = [int(e[0]) for e in entries]  # type: ignore[arg-type]
        if not any(cid in target_ids for cid in class_ids):
            continue
        excluded_count = sum(1 for cid in class_ids if cid in excluded_ids)
        if excluded_count > max_excluded:
            continue
        candidates.append(record)
    return candidates


def flip_entries(
    entries: Sequence[Tuple[int, float, float, float, float]]
) -> List[Tuple[int, float, float, float, float]]:
    """Mirror the labels of a horizontal flip (``cx -> 1 - cx``)."""
    flipped: List[Tuple[int, float, float, float, float]] = []
    for class_id, cx, cy, w, h in entries:
        flipped.append((class_id, 1.0 - cx, cy, w, h))
    return flipped


def write_label_file(
    path: Path, entries: Sequence[Tuple[int, float, float, float, float]]
) -> None:
    """Write a YOLO label file (6 decimals, the usual convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for class_id, cx, cy, w, h in entries:
            handle.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def augment_train_split(
    train_dir: Path,
    train_records: Sequence[Record],
    apply_flip: bool = True,
    apply_brightness: bool = True,
) -> Dict[str, int]:
    """Create the rebalanced copies inside ``train_dir``.

    The copies are written next to the originals with the suffix
    ``_flip`` / ``_bright``. The original images are left untouched so the
    augmentation is fully reproducible and auditable.

    Returns a small report: ``{"candidates": n, "flipped": n, "bright": n}``.
    """
    try:
        from PIL import Image, ImageEnhance
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for the augmentation step") from exc

    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"
    candidates = select_augmentation_candidates(train_records)

    stats = {"candidates": len(candidates), "flipped": 0, "brightened": 0}
    if not candidates:
        return stats

    for record in candidates:
        source = Path(str(record["source_path"]))
        if not source.exists():
            continue
        try:
            with Image.open(source) as handle:
                image = handle.convert("RGB")
        except OSError:
            continue

        stem = str(record["filename"])
        entries = list(record["entries"])  # type: ignore[arg-type]

        if apply_flip:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            out_image = images_dir / f"{stem}_flip{Path(str(record['extension'])).suffix or '.jpg'}"
            out_label = labels_dir / f"{stem}_flip.txt"
            flipped.save(out_image, quality=95)
            write_label_file(out_label, flip_entries(entries))
            stats["flipped"] += 1

        if apply_brightness:
            bright = ImageEnhance.Brightness(image).enhance(1.25)
            extension = Path(str(record["extension"])).suffix or ".jpg"
            out_image = images_dir / f"{stem}_bright{extension}"
            out_label = labels_dir / f"{stem}_bright.txt"
            bright.save(out_image, quality=95)
            # Photometric change: the geometry is identical, labels unchanged.
            write_label_file(out_label, entries)
            stats["brightened"] += 1

    return stats


# --------------------------------------------------------------------------- #
# Materialisation + description
# --------------------------------------------------------------------------- #


def materialise_split(
    records: Sequence[Record],
    destination: Path,
    split_name: str,
) -> int:
    """Copy the images and write the labels of one split."""
    images_dir = destination / split_name / "images"
    labels_dir = destination / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for record in records:
        source = Path(str(record["source_path"]))
        if not source.exists():
            continue
        extension = str(record.get("extension") or ".jpg")
        shutil.copyfile(source, images_dir / f"{record['filename']}{extension}")
        write_label_file(
            labels_dir / f"{record['filename']}.txt",
            list(record["entries"]),  # type: ignore[arg-type]
        )
        count += 1
    return count


def write_dataset_yaml(path: Path, dataset_dir: Path) -> Path:
    """Write the ``dataset.yaml`` consumed by Ultralytics.

    The file is written by hand (no PyYAML dependency) with a deterministic
    key order, so the produced artefact is byte-stable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = CFG.class_names()
    lines = [
        f"# Auto-generated by src/data/dataset.py - do not edit by hand",
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {CFG.NUM_CLASSES}",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def split_report(
    splits: Dict[str, Sequence[Record]],
    destination: Path,
) -> Dict[str, Dict[str, object]]:
    """Compute and persist the class-distribution report of every split."""
    report: Dict[str, Dict[str, object]] = {}
    for name, records in splits.items():
        counts = {cls: 0 for cls in CFG.PPE_CLASSES}
        instances = {cls: 0 for cls in CFG.PPE_CLASSES}
        for record in records:
            presence = record["presence"]
            for class_id, present in enumerate(presence):  # type: ignore[arg-type]
                if present:
                    counts[class_id] += 1
            for entry in record["entries"]:  # type: ignore[arg-type]
                instances[entry[0]] = instances.get(entry[0], 0) + 1
        report[name] = {
            "n_images": len(records),
            "image_presence": {CFG.PPE_CLASSES[c]: counts[c] for c in counts},
            "instance_count": {CFG.PPE_CLASSES[c]: instances[c] for c in instances},
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_results_csv(path: Path, row: Dict[str, object]) -> None:
    """Append one training run to the historical results registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


__all__ = [
    "parse_label_file",
    "class_presence_vector",
    "collect_records",
    "stratified_split",
    "select_augmentation_candidates",
    "flip_entries",
    "write_label_file",
    "augment_train_split",
    "materialise_split",
    "write_dataset_yaml",
    "split_report",
    "write_results_csv",
]
