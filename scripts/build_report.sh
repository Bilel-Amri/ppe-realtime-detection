#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# Build the technical report (PDF).
#
#   bash scripts/build_report.sh
#
# The tables of the measured metrics are regenerated from
# artifacts/metrics.json, so the PDF can never display a number that was not
# measured.
# --------------------------------------------------------------------------- #
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "❌ pdflatex is not installed (install texlive-latex-recommended)."
    exit 1
fi

echo ">> Generating the metrics tables"
python scripts/generate_report_tables.py

cd reports

echo ">> Compiling the report (twice, for the table of contents)"
pdflatex -interaction=nonstopmode -halt-on-error rapport_technique.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error rapport_technique.tex >/dev/null

echo "✅ Report built: reports/rapport_technique.pdf"
