# syntax=docker/dockerfile:1.7
# ─── Build stage: install CPU-only torch + Python deps ───────────────────
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch saves ~250 MB vs the default CUDA build
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ─── Runtime stage: slim image, non-root user ────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=7860 \
    GUNICORN_WORKERS=1 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_GRACEFUL_TIMEOUT=30

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --create-home --home-dir /app app
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=app:app . /app

RUN mkdir -p /app/.data /app/uploads /app/logs && chown -R app:app /app

# Pre-download the embedding model at build time so cold start is fast.
RUN python scripts/download_models.py || true

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/readyz || exit 1

# Entrypoint runs migrations on every container start (idempotent), then exec
# gunicorn so it becomes PID 1 and receives SIGTERM directly for graceful drain.
RUN chmod +x /app/scripts/entrypoint.sh

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
