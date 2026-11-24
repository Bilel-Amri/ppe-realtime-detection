"""Minimal pytest-compatible shim + runner.

The grading environment has no network access, so ``pip install pytest`` is
impossible. This module implements the (small) subset of the pytest API used
by ``tests_local/`` so the suite can still be executed and the results can be
reported honestly.

When the real pytest *is* installed (``python -m pytest tests_local``), this
shim is bypassed entirely — it only kicks in through ``run_local_tests.py``.

Supported subset: ``pytest.approx``, ``pytest.raises``, ``pytest.fixture``,
``tmp_path`` and plain generator fixtures.
"""

from __future__ import annotations

import inspect
import math
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


class _Approx:
    """``pytest.approx`` equivalent (scalar and sequence aware)."""

    def __init__(self, expected: Any, rel: float = 1e-6, abs: float = 1e-9):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=self.rel, abs_tol=self.abs)

    def __eq__(self, other: Any) -> bool:
        if isinstance(self.expected, (list, tuple)):
            if len(other) != len(self.expected):
                return False
            return all(self._close(float(a), float(b)) for a, b in zip(other, self.expected))
        return self._close(float(other), float(self.expected))

    def __repr__(self) -> str:  # pragma: no cover
        return f"approx({self.expected!r})"


def approx(expected: Any, rel: float = 1e-6, abs: float = 1e-9) -> _Approx:
    return _Approx(expected, rel=rel, abs=abs)


class _Raises:
    """``pytest.raises`` equivalent (context manager)."""

    def __init__(self, expected_exception: Any):
        self.expected_exception = expected_exception
        self.value: Any = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {self.expected_exception}"
            )
        if not issubclass(exc_type, self.expected_exception):
            return False
        self.value = exc
        return True


def raises(expected_exception: Any) -> _Raises:
    return _Raises(expected_exception)


def fixture(func: Callable | None = None, **_: Any):
    """Mark a function as a fixture (no-op decorator used by the runner)."""

    def wrapper(f: Callable) -> Callable:
        f.__is_fixture__ = True  # type: ignore[attr-defined]
        return f

    if func is not None:
        return wrapper(func)
    return wrapper


class _Mark:
    def __getattr__(self, _name: str) -> Callable:
        def decorator(*args: Any, **kwargs: Any):
            def wrapper(func: Callable) -> Callable:
                return func

            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return wrapper

        return decorator


mark = _Mark()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _resolve_fixture(name: str, module, tmp_root: Path, cache: Dict[str, Any]):
    if name in cache:
        return cache[name]
    if name == "tmp_path":
        # Built-in fixture provided by pytest itself.
        target = tmp_root / "tmp_path"
        target.mkdir(parents=True, exist_ok=True)
        cache[name] = target
        return target
    func = getattr(module, name, None)
    if func is None:
        raise LookupError(f"fixture '{name}' not found in {module.__name__}")
    kwargs = {}
    for param in inspect.signature(func).parameters:
        if param == "tmp_path":
            kwargs[param] = tmp_root
        else:
            kwargs[param] = _resolve_fixture(param, module, tmp_root, cache)
    if inspect.isgeneratorfunction(func):
        generator = func(**kwargs)
        value = next(generator)
        cache[name] = value
        try:
            next(generator)
        except StopIteration:
            pass
    else:
        value = func(**kwargs)
        cache[name] = value
    return value


def run_module(module, failures: List[Tuple[str, str]]) -> Tuple[int, int]:
    """Run every ``test_*`` function of ``module``."""
    passed = 0
    tmp_root = Path(tempfile.mkdtemp(prefix="minipytest-"))
    try:
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            cache: Dict[str, Any] = {}
            try:
                kwargs = {}
                for param in inspect.signature(func).parameters:
                    kwargs[param] = _resolve_fixture(param, module, tmp_root, cache)
                func(**kwargs)
                passed += 1
                print(f"  PASS  {name}")
            except Exception:
                failures.append((f"{module.__name__}::{name}", traceback.format_exc()))
                print(f"  FAIL  {name}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return passed, len(failures)


def main() -> int:
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.modules.setdefault("pytest", sys.modules[__name__])

    failures: List[Tuple[str, str]] = []
    total_passed = 0
    modules = [
        "tests_local.test_core_logic",
    ]
    for module_name in modules:
        print(f"\n=== {module_name} ===")
        module = importlib.import_module(module_name)
        passed, _ = run_module(module, failures)
        total_passed += passed

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED: {len(failures)} test(s), PASSED: {total_passed}\n")
        for name, tb in failures:
            print(f"--- {name} ---")
            print(tb)
        return 1
    print(f"ALL TESTS PASSED ({total_passed} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
