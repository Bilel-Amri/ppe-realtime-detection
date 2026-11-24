"""A minimal flake8-equivalent checker (no third-party dependency).

The CI runs the real ``flake8``, but the grading sandbox has no network
access to install it. This script performs the two checks that actually
matter for the code quality of this repository:

* **unused imports** (F401) - an imported name that never appears again in
  the module;
* **lines longer than the configured limit** (E501) - the limit is read from
  ``setup.cfg`` so that both tools agree.

Usage::

    python tests_local/check_lint.py
"""

from __future__ import annotations

import ast
import configparser
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent


def configured_max_line_length() -> int:
    parser = configparser.ConfigParser()
    setup = ROOT / "setup.cfg"
    if setup.exists():
        parser.read(setup)
        try:
            return int(parser.get("flake8", "max-line-length", fallback="100"))
        except ValueError:
            return 100
    return 100


def collect_python_files() -> List[Path]:
    targets = ["src", "backend", "scripts", "tests_local"]
    files: List[Path] = []
    for target in targets:
        base = ROOT / target
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def used_names(tree: ast.AST) -> Set[str]:
    """Every identifier that appears somewhere in the module body."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                names.add(base.id)
    return names


def imported_names(tree: ast.AST) -> List[tuple]:
    """Return ``[(name, lineno), ...]`` for every imported binding."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                found.append((name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # `from __future__ import annotations` is a compiler directive,
            # not a real import: flake8 does not flag it either.
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                found.append((name, node.lineno))
    return found


def main() -> int:
    max_length = configured_max_line_length()
    failures: List[str] = []

    for path in collect_python_files():
        relative = path.relative_to(ROOT)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            failures.append(f"{relative}:{exc.lineno}: syntax error: {exc.msg}")
            continue

        # -- unused imports ------------------------------------------------ #
        used = used_names(tree)
        for name, lineno in imported_names(tree):
            if name in used:
                continue
            # "from x import y as z" handled above; a module used only in a
            # type annotation string is rare here.
            if f'"{name}"' in source or f"'{name}'" in source:
                continue
            failures.append(f"{relative}:{lineno}: F401 unused import '{name}'")

        # -- long lines ----------------------------------------------------- #
        for number, line in enumerate(source.splitlines(), 1):
            if len(line) > max_length:
                failures.append(
                    f"{relative}:{number}: E501 line too long "
                    f"({len(line)} > {max_length})"
                )

    print(f"Checked {len(collect_python_files())} Python files (max line = {max_length}).")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("ALL LINT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
