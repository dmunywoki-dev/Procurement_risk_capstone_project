FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PPIP_ARTIFACTS_DIR=/app/artifacts \
    START_API=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY notebooks/05_fastapi_service.ipynb notebooks/05_fastapi_service.ipynb
COPY artifacts/ppip_anomaly_model.joblib artifacts/ppip_anomaly_model.joblib
COPY run_api.sh run_api.sh
RUN chmod +x run_api.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1
CMD ["bash", "run_api.sh"]
