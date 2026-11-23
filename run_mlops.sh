#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Building the empirical report and retrospective drift diagnostics..."
python scripts/build_report.py
python scripts/monitor_drift.py

echo "Tracking the model and registering a candidate in MLflow..."
python scripts/track_mlflow.py

echo "Evaluating promotion gates. A failed gate intentionally blocks champion promotion..."
python scripts/promote_model.py

echo "MLOps run complete. Review reports/fy2025_2026/monitoring/."
