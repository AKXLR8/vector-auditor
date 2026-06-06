"""HTTP middleware: request ID, security headers, shutdown gate, metrics."""
import logging
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..observability import get_metrics
from ..shutdown import get_shutdown_manager

logger = logging.getLogger("rga_auditor.middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and start time. Always present."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class ShutdownGateMiddleware(BaseHTTPMiddleware):
    """During shutdown: 503 on new requests, exempt /health, /readyz, /metrics."""

    EXEMPT = {"/health", "/readyz", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        mgr = get_shutdown_manager()
        if mgr.is_shutting_down and request.url.path not in self.EXEMPT:
            return JSONResponse({"detail": "server shutting down"}, status_code=503)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Count requests and latency. Use route template (not full path) for cardinality control."""

    async def dispatch(self, request: Request, call_next):
        # resolve route template after dispatch; before we don't know it
        mgr = get_shutdown_manager()
        if mgr.is_shutting_down and request.url.path not in ShutdownGateMiddleware.EXEMPT:
            # Don't count pre-shutdown-rejected requests
            return await call_next(request)
        try:
            await mgr.begin_request()
        except RuntimeError:
            return JSONResponse({"detail": "server shutting down"}, status_code=503)
        try:
            t0 = time.time()
            response = await call_next(request)
            dt = time.time() - t0
            endpoint = _route_template(request)
            get_metrics().observe_request(request.method, endpoint, response.status_code, dt)
            return response
        finally:
            await mgr.end_request()


def _route_template(request: Request) -> str:
    """Return the route template (e.g. /docs/{doc_id}) to keep cardinality low."""
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path
