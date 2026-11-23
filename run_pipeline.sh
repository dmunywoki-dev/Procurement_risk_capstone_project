#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/raw data/processed artifacts reports/figures reports/tables
mkdir -p /tmp/tgee-capstone-executed

echo "Executing the end-to-end FY2025/2026 notebook..."
jupyter nbconvert --to notebook --execute notebooks/01_TGEE_End_to_End.ipynb \
  --output 01_TGEE_End_to_End.executed.ipynb \
  --output-dir /tmp/tgee-capstone-executed \
  --ExecutePreprocessor.timeout=7200

echo "Building the empirical report and retrospective monitoring diagnostics..."
python scripts/build_report.py
python scripts/monitor_drift.py

echo "Checking notebook syntax..."
python scripts/check_notebooks.py

echo
echo "Pipeline complete."
echo "Inspect artifacts/fy2025_2026/, data/processed/fy2025_2026/ and reports/."
echo "Confirm data_mode in run metadata is official_fy2025_2026 before submission."
