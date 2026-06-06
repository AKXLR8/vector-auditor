"""JSON logging + Prometheus metrics.

Set LOG_FORMAT=json (default) for one JSON object per line, ready for
Loki/CloudWatch/Datadog. Set LOG_FORMAT=text for human-readable.
"""
import logging
import os
import sys
import time
from typing import Any

try:
    from pythonjsonlogger import jsonlogger

    _HAVE_JSON_LOGGER = True
except ImportError:
    _HAVE_JSON_LOGGER = False

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


def setup_observability(app=None) -> None:
    """Configure root logger. Idempotent."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "json").lower()

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json" and _HAVE_JSON_LOGGER:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "ts", "levelname": "level", "name": "logger"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet noisy libraries
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Prometheus metrics ───────────────────────────────────────────────────────

class Metrics:
    """Single source of truth for all metrics. Lazily initialized."""

    def __init__(self) -> None:
        self.requests_total = Counter(
            "rga_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )
        self.request_duration = Histogram(
            "rga_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )
        self.in_flight = Gauge(
            "rga_in_flight_requests",
            "Current concurrent in-flight requests",
        )

        self.llm_calls_total = Counter(
            "rga_llm_calls_total",
            "LLM calls",
            ["mode", "outcome"],
        )
        self.llm_call_duration = Histogram(
            "rga_llm_call_duration_seconds",
            "LLM call latency",
            ["mode"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
        )

        self.embedding_calls_total = Counter(
            "rga_embedding_calls_total",
            "Embedding calls",
            ["outcome"],
        )

        self.cache_total = Counter(
            "rga_cache_total",
            "Cache operations",
            ["result"],
        )

        self.uploads_total = Counter(
            "rga_uploads_total",
            "Upload pipeline results",
            ["status"],
        )

        self.jobs_total = Counter(
            "rga_jobs_total",
            "Background job lifecycle",
            ["event"],
        )
        self.dlq_size = Gauge(
            "rga_dlq_size",
            "Current dead-letter queue depth",
        )
        self.startup_timestamp = Gauge(
            "rga_startup_timestamp_seconds",
            "Unix timestamp when the app finished booting",
        )

    def observe_request(self, method: str, endpoint: str, status: int, duration_s: float) -> None:
        self.requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration_s)


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
        _metrics.startup_timestamp.set(time.time())
    return _metrics


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
