#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DEFAULT_BUNDLE="$(pwd)/artifacts/fy2025_2026/tgee_fy2025_2026_model_bundle.joblib"
export MODEL_BUNDLE_PATH="$(printenv MODEL_BUNDLE_PATH || printf '%s' "$DEFAULT_BUNDLE")"
export API_HOST="$(printenv API_HOST || printf '%s' '0.0.0.0')"
export API_PORT="$(printenv API_PORT || printf '%s' '8000')"

if [[ ! -f "$MODEL_BUNDLE_PATH" ]]; then
  echo "Model bundle not found: $MODEL_BUNDLE_PATH"
  echo "Run bash run_pipeline.sh first."
  exit 1
fi
exec uvicorn api_service:app --host "$API_HOST" --port "$API_PORT"
