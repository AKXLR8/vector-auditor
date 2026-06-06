"""Gunicorn config — production worker settings.

Single worker with multiple threads is the right call for this app:
- The embedding model is ~84 MB in RAM. Two workers = 168 MB.
- Async I/O (httpx, asyncpg, redis) is handled by uvicorn's event loop.
- Adding threads lets CPU-bound work (PII detection, JSON serialization)
  run in parallel with the event loop without blocking requests.

Run with:
  gunicorn src.main:app --config gunicorn_conf.py
"""
import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

workers = int(os.getenv("GUNICORN_WORKERS", "1"))
threads = int(os.getenv("GUNICORN_THREADS", str(min(8, (multiprocessing.cpu_count() or 2) * 2))))

worker_class = "uvicorn.workers.UvicornWorker"

timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = 5
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sus'
)

proc_name = "rag-auditor"
preload_app = False


def on_starting(server):
    server.log.info("gunicorn starting: workers=%d threads=%d timeout=%ds", workers, threads, timeout)


def worker_int(worker):
    worker.log.info("worker %s received SIGINT/SIGTERM — draining", worker.pid)


def worker_abort(worker):
    worker.log.warning("worker %s aborted (timeout)", worker.pid)
