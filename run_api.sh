#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Use the active Python environment. Do not assume a project-local .venv exists
# (the Docker image installs dependencies directly into its Python environment).
echo "Exporting the FY2025/2026 FastAPI notebook..."
jupyter nbconvert --to python --output api_service --output-dir . notebooks/02_FastAPI_Model_Service.ipynb

echo "Starting FastAPI..."
export START_API=1
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

python api_service.py
