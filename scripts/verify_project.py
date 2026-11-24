"""Self-check of the project installation.

Run it after ``pip install -r backend/requirements.txt`` to know exactly what
works and what is missing:

    python scripts/verify_project.py

The script never fails on a missing optional component (a missing model, a
missing dataset): it reports the status of each item and exits with a non-zero
code only when a **critical** item is missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

OK = f"{GREEN}[ OK ]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
KO = f"{RED}[FAIL]{RESET}"


def check_python_packages() -> Tuple[bool, list]:
    packages = [
        ("fastapi", "API framework"),
        ("uvicorn", "ASGI server"),
        ("pydantic", "validation"),
        ("ultralytics", "YOLO11 training & export"),
        ("onnxruntime", "ONNX Runtime inference"),
        ("torch", "PyTorch"),
        ("mlflow", "experiment tracking"),
        ("PIL", "image handling"),
        ("numpy", "numerics"),
    ]
    missing = []
    for module, purpose in packages:
        try:
            __import__(module)
            print(f"{OK} {module:<14} {purpose}")
        except ImportError:
            missing.append(module)
            print(f"{KO} {module:<14} {purpose} - not installed")
    return (not missing), missing


def check_paths() -> bool:
    required = {
        "src/models/train.py": "training script",
        "src/models/evaluate.py": "evaluation script",
        "src/inference/predictor.py": "inference facade",
        "backend/app/main.py": "FastAPI application",
        "backend/app/schemas.py": "Pydantic schemas",
        "frontend/src/App.jsx": "React application",
        "docker-compose.yml": "orchestration",
        "dvc.yaml": "DVC pipeline",
        ".github/workflows/ci-tests.yml": "CI pipeline",
    }
    all_ok = True
    for relative, purpose in required.items():
        path = REPO_ROOT / relative
        if path.exists():
            print(f"{OK} {relative:<42} {purpose}")
        else:
            all_ok = False
            print(f"{KO} {relative:<42} {purpose} - MISSING")
    return all_ok


def check_artifacts() -> None:
    artifacts = REPO_ROOT / "artifacts"
    checkpoint = artifacts / "runs" / "detect" / "train" / "weights" / "best.pt"
    onnx = artifacts / "runs" / "detect" / "train" / "weights" / "best_fp32.onnx"
    metrics = artifacts / "metrics.json"

    for path, hint in (
        (checkpoint, "run: python -m src.models.train"),
        (onnx, "run: python -m src.models.evaluate"),
    ):
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"{OK} {path.relative_to(REPO_ROOT)} ({size:.1f} MB)")
        else:
            print(f"{WARN} {path.relative_to(REPO_ROOT)} - absent ({hint})")

    if metrics.exists():
        try:
            payload = json.loads(metrics.read_text())
            print(f"{OK} artifacts/metrics.json")
            print(
                f"      test mAP50={payload.get('test_mAP50')}, "
                f"mAP50-95={payload.get('test_mAP50_95')}"
            )
        except json.JSONDecodeError:
            print(f"{KO} artifacts/metrics.json - invalid JSON")
    else:
        print(f"{WARN} artifacts/metrics.json - absent (run src.models.evaluate)")


def check_dataset() -> None:
    dataset = REPO_ROOT / "data" / "processed" / "css-ppe"
    yaml_path = dataset / "dataset.yaml"
    if yaml_path.exists():
        print(f"{OK} {yaml_path.relative_to(REPO_ROOT)}")
        for split in ("train", "valid", "test"):
            images = dataset / split / "images"
            if images.exists():
                count = len(list(images.glob("*")))
                print(f"      {split:<6}: {count} images")
    else:
        print(
            f"{WARN} data/processed/css-ppe - absent "
            "(run: python scripts/prepare_dataset.py)"
        )


def check_docker() -> None:
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"{OK} {result.stdout.strip()}")
            compose = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if compose.returncode == 0:
                print(f"{OK} {compose.stdout.strip()}")
            else:
                print(f"{WARN} docker compose is not available")
        else:
            print(f"{WARN} docker is not installed")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"{WARN} docker is not installed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the project installation")
    parser.add_argument("--quiet", action="store_true")
    parser.parse_args()

    print("=" * 70)
    print("PPE detection - project verification")
    print("=" * 70)

    print("\n1. Project structure")
    structure_ok = check_paths()

    print("\n2. Python dependencies")
    deps_ok, missing = check_python_packages()

    print("\n3. Dataset")
    check_dataset()

    print("\n4. Trained artefacts")
    check_artifacts()

    print("\n5. Docker")
    check_docker()

    print("\n" + "=" * 70)
    if not structure_ok:
        print(f"{RED}The project structure is incomplete.{RESET}")
        return 1
    if not deps_ok:
        print(
            f"{YELLOW}Some Python packages are missing: {', '.join(missing)}.{RESET}\n"
            "   Install them with:  pip install -r backend/requirements.txt"
        )
        return 1

    print(f"{GREEN}The project is ready.{RESET}")
    print("   Start everything with:  docker compose up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
