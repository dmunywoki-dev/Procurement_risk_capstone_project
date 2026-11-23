FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt ./requirements-api.txt
RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY --chown=10001:10001 . .
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

USER 10001:10001

EXPOSE 8000
CMD ["bash", "run_api.sh"]
