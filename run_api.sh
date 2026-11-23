#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

echo "Exporting FastAPI notebook..."

jupyter nbconvert --to python --output api_service --output-dir . notebooks/02_FastAPI_Model_Service.ipynb

echo "Starting FastAPI..."

export START_API=1
export API_HOST=0.0.0.0
export API_PORT=8000

python api_service.py
