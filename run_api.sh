#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f artifacts/ppip_anomaly_model.joblib ]]; then
  echo "Model not found. Run bash run_pipeline.sh first." >&2
  exit 1
fi
export START_API=1
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
jupyter nbconvert --to notebook --execute notebooks/05_fastapi_service.ipynb \
  --output ppip_api_executed.ipynb --output-dir /tmp \
  --ExecutePreprocessor.timeout=-1
