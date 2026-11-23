#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw data/processed artifacts reports/figures reports/tables /tmp/tgee-capstone-executed
jupyter nbconvert --to notebook --execute notebooks/01_TGEE_End_to_End.ipynb \
  --output 01_TGEE_End_to_End.executed.ipynb \
  --output-dir /tmp/tgee-capstone-executed \
  --ExecutePreprocessor.timeout=7200
printf '\nTGEE pipeline complete. Review reports/, data/processed/ and artifacts/.\n'
