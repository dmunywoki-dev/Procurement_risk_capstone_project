#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw data/processed artifacts reports/figures /tmp/ppip-capstone-executed

for notebook in \
  notebooks/01_data_audit.ipynb \
  notebooks/02_data_preparation.ipynb \
  notebooks/03_model_training.ipynb \
  notebooks/04_evaluation_and_monitoring.ipynb
 do
  echo "Running ${notebook}"
  jupyter nbconvert --to notebook --execute "$notebook" \
    --output "$(basename "$notebook")" \
    --output-dir /tmp/ppip-capstone-executed \
    --ExecutePreprocessor.timeout=7200
 done

echo "Pipeline finished. Review the tables and plots in reports/."
