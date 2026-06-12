FROM python:3.11-slim AS builder

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
RUN python scripts/download_model.py || echo "WARNING: model pre-cache failed (will lazy-load at runtime)"

FROM python:3.11-slim

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models models/

COPY alembic.ini ./
COPY alembic ./alembic
COPY src/ src/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD alembic upgrade head || echo "alembic skipped (in-memory fallback)"; \
    exec uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 7860 \
        --workers 8 \
        --loop asyncio \
        --http httptools \
        --no-access-log
