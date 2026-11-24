"""Download the Construction Site Safety (CSS) dataset.

The dataset is public and distributed by Roboflow Universe / Kaggle:

* Roboflow Universe - Construction Site Safety
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
* Kaggle - snehilsanyal/construction-site-safety-image-dataset-roboflow
  https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow

Three acquisition paths are supported, in this order of preference:

1. ``--kaggle``   - uses the Kaggle API (needs ``pip install kaggle`` and a
   ``~/.kaggle/kaggle.json`` token). This is the easiest path.
2. ``--roboflow`` - uses the Roboflow API (``pip install roboflow``) with an
   API key, and downloads the dataset in the YOLO format.
3. ``--manual``   - only prints the instructions; the user then places the
   extracted archive under ``data/raw/css-data``.

Usage::

    python scripts/download_dataset.py --kaggle
    python scripts/download_dataset.py --roboflow --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RAW_DIR  # noqa: E402

KAGGLE_DATASET = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
ROBOFLOW_PROJECT = "construction-site-safety/1"
ROBOFLOW_API_KEY = "YOUR_API_KEY"  # placeholder, replaced by the --api-key flag


def _find_dataset_root(base: Path) -> Path:
    """Return the folder that contains the ``train``/``valid``/``test`` splits."""
    candidates = [base, *base.glob("**/")]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any((candidate / split).is_dir() for split in ("train", "valid", "test")):
            return candidate
    return base


def download_with_kaggle(destination: Path) -> bool:
    """Download and extract the dataset with the Kaggle CLI."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌ The Kaggle API is not installed: pip install kaggle")
        return False

    destination.mkdir(parents=True, exist_ok=True)
    print(f"▶️  Downloading {KAGGLE_DATASET} from Kaggle ...")
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(destination),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Kaggle download failed ({exc}). Check ~/.kaggle/kaggle.json")
        return False

    root = _find_dataset_root(destination)
    for split in ("train", "valid", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        if images.exists() and labels.exists():
            print(f"✅ Found the '{split}' split in {root}")
            print(f"   -> {destination / 'css-data'}")
            if root != destination / "css-data":
                if (destination / "css-data").exists():
                    shutil.rmtree(destination / "css-data")
                shutil.copytree(root, destination / "css-data")
            return True
    print("❌ The archive was extracted but no train/valid/test split was found.")
    return False


def download_with_roboflow(destination: Path, api_key: str) -> bool:
    """Download the dataset in YOLO format through the Roboflow API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ The Roboflow SDK is not installed: pip install roboflow")
        return False

    destination.mkdir(parents=True, exist_ok=True)
    print(f"▶️  Downloading {ROBOFLOW_PROJECT} from Roboflow ...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("roboflow-universe-projects").project(ROBOFLOW_PROJECT)
        project.version(1).download(location=str(destination), format="yolov8")
    except Exception as exc:
        print(f"❌ Roboflow download failed ({exc}).")
        return False

    root = _find_dataset_root(destination)
    if root != destination / "css-data":
        if (destination / "css-data").exists():
            shutil.rmtree(destination / "css-data")
        shutil.copytree(root, destination / "css-data")
    print(f"✅ Dataset available in {destination / 'css-data'}")
    return True


def print_manual_instructions(destination: Path) -> None:
    print("\n📥 Manual installation")
    print("=" * 60)
    print("1. Open the Kaggle dataset page:")
    print("   https://www.kaggle.com/datasets/snehilsanyal/")
    print("       construction-site-safety-image-dataset-roboflow")
    print("2. Download the archive (it is a .zip of the YOLO dataset).")
    print("3. Extract it so that you get:")
    print(f"   {destination / 'css-data'}/")
    print("       train/images/*.jpg   train/labels/*.txt")
    print("       valid/images/*.jpg   valid/labels/*.txt")
    print("       test/images/*.jpg    test/labels/*.txt")
    print("4. Then run:")
    print("   python scripts/prepare_dataset.py")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the PPE dataset")
    parser.add_argument("--kaggle", action="store_true", help="Use the Kaggle API")
    parser.add_argument("--roboflow", action="store_true", help="Use the Roboflow API")
    parser.add_argument("--api-key", default=ROBOFLOW_API_KEY, help="Roboflow API key")
    parser.add_argument(
        "--dest",
        default=str(RAW_DIR),
        help="Destination folder (default: data/raw)",
    )
    parser.add_argument(
        "--manual", action="store_true", help="Only print the manual instructions"
    )
    args = parser.parse_args()

    destination = Path(args.dest).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    if args.kaggle and download_with_kaggle(destination):
        return 0
    if args.roboflow and download_with_roboflow(destination, args.api_key):
        return 0

    print_manual_instructions(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
