"""Prepare the Construction Site Safety dataset.

The raw dataset (downloaded from Kaggle / Roboflow) is already split into
``train/valid/test`` but **badly stratified**: the rare classes are almost
absent from the validation and test splits. This script rebuilds a clean,
class-proportional split and writes it in the layout expected by Ultralytics.

Usage::

    python scripts/prepare_dataset.py --source data/raw/css-data --out data/processed/css-ppe

The source directory must contain::

    css-data/
      train/images/*.jpg   train/labels/*.txt
      valid/images/*.jpg   valid/labels/*.txt
      test/images/*.jpg    test/labels/*.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CFG, ensure_dirs  # noqa: E402
from src.data.dataset import (  # noqa: E402
    augment_train_split,
    collect_records,
    materialise_split,
    split_report,
    stratified_split,
    write_dataset_yaml,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the PPE dataset")
    parser.add_argument(
        "--source",
        default="data/raw/css-data",
        help="Root of the raw dataset (containing train/valid/test)",
    )
    parser.add_argument(
        "--out",
        default="data/processed/css-ppe",
        help="Destination of the re-split dataset",
    )
    parser.add_argument(
        "--ratios",
        default=None,
        help="Comma separated train,valid,test ratios (default: 0.80,0.15,0.05)",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Skip the rebalancing copies of the rare classes",
    )
    parser.add_argument("--seed", type=int, default=CFG.SEED)
    args = parser.parse_args()

    source = Path(args.source).expanduser()
    destination = Path(args.out).expanduser()

    if not source.exists():
        print(f"❌ Raw dataset not found at '{source}'.")
        print("   Download it first:  python scripts/download_dataset.py")
        return 1

    ensure_dirs()

    print(f"▶️  Scanning {source} ...")
    records = collect_records(source)
    if not records:
        print(f"❌ No image found under '{source}'.")
        print("   Expected <source>/train/images/*.jpg and <source>/train/labels/*.txt")
        return 1
    print(f"   {len(records)} images found.")

    ratios = CFG.TARGET_RATIOS
    if args.ratios:
        parts = [float(value) for value in args.ratios.split(",")]
        if len(parts) != 3:
            print("❌ --ratios expects three values: train,valid,test")
            return 1
        ratios = dict(zip(("train", "valid", "test"), parts))

    print(f"▶️  Stratifying with ratios {ratios} (seed={args.seed}) ...")
    splits = stratified_split(records, ratios=ratios, seed=args.seed)
    for name, items in splits.items():
        print(f"   {name:>6}: {len(items):>5} images")

    print(f"▶️  Writing the dataset to {destination} ...")
    for name, items in splits.items():
        count = materialise_split(items, destination, name)
        print(f"   {name:>6}: {count} images copied")

    if not args.no_augmentation:
        print("▶️  Rebalancing the rare classes (horizontal flip + brightness) ...")
        stats = augment_train_split(destination / "train", splits["train"])
        print(
            f"   candidates={stats['candidates']} "
            f"flipped={stats['flipped']} brightened={stats['brightened']}"
        )

    yaml_path = write_dataset_yaml(destination / "dataset.yaml", destination)
    print(f"✅ Dataset description written to {yaml_path}")

    report = split_report(splits, destination / "split_report.json")
    print(f"✅ Split report written to {destination / 'split_report.json'}")
    print(json.dumps(report, indent=2)[:800])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
