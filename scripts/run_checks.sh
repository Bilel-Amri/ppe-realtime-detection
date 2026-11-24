#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Run every check that does NOT need a GPU, a network or the heavy
# dependencies. This is the script the CI runs first, and the one to use
# before pushing a commit.
#
#   bash scripts/run_checks.sh
# --------------------------------------------------------------------------- #
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAILED=0

run() {
    local label="$1"; shift
    echo ""
    echo "=============================================================="
    echo " $label"
    echo "=============================================================="
    if "$@"; then
        echo "✅ $label"
    else
        echo "❌ $label"
        FAILED=1
    fi
}

# 1. Every Python file must compile.
run "Python syntax" python3 -m compileall -q src backend scripts tests_local

# 2. Core logic (stratification, NMS, decoding, letterbox).
run "Core logic tests" python3 tests_local/_minipytest.py

# 3. API wiring (routes, error contract, OpenAPI).
run "Backend wiring" python3 tests_local/check_backend_wiring.py

# 4. Frontend static analysis.
run "Frontend checks" node tests_local/check_frontend.js

# 5. The JSON files must be valid.
run "JSON files" python3 - <<'PY'
import json, pathlib, sys
root = pathlib.Path('.')
bad = []
for pattern in ('**/*.json',):
    for path in root.glob(pattern):
        if any(part in ('node_modules', '.git', 'artifacts') for part in path.parts):
            continue
        try:
            json.loads(path.read_text())
        except Exception as exc:
            bad.append(f"{path}: {exc}")
if bad:
    print('\n'.join(bad)); sys.exit(1)
print('All JSON files are valid.')
PY

# 6. Lint-equivalent checks (unused imports, long lines).
run "Lint checks" python3 tests_local/check_lint.py

# 7. The YAML files must not contain tabs (invalid in YAML).
run "YAML sanity" python3 - <<'PY'
import pathlib, sys
root = pathlib.Path('.')
bad = []
for path in root.glob('**/*.y*ml'):
    if any(part in ('node_modules', '.git') for part in path.parts):
        continue
    for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        if '\t' in line:
            bad.append(f'{path}:{number}: tab character')
if bad:
    print('\n'.join(bad)); sys.exit(1)
print('All YAML files are tab-free.')
PY

echo ""
echo "=============================================================="
if [ "$FAILED" -eq 0 ]; then
    echo " ALL CHECKS PASSED"
else
    echo " SOME CHECKS FAILED"
fi
echo "=============================================================="
exit "$FAILED"
