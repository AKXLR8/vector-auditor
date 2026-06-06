"""Graceful shutdown manager.

Tracks in-flight requests, drains them on SIGTERM/SIGINT, then triggers
the FastAPI lifespan shutdown. Configurable drain timeout via env.
"""
import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger("rga_auditor.shutdown")


class ShutdownManager:
    def __init__(self, drain_timeout_s: Optional[float] = None) -> None:
        self.drain_timeout_s = drain_timeout_s or float(os.getenv("SHUTDOWN_DRAIN_TIMEOUT", "30"))
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._shutting_down = False
        self._signals_installed = False

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def in_flight_count(self) -> int:
        return self._in_flight

    async def begin_request(self) -> None:
        async with self._lock:
            if self._shutting_down:
                raise RuntimeError("server is shutting down")
            self._in_flight += 1
            self._idle.clear()

    async def end_request(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight == 0:
                self._idle.set()

    async def wait_drain(self) -> None:
        if self._in_flight == 0:
            return
        logger.info("draining %d in-flight request(s) (timeout=%.0fs)", self._in_flight, self.drain_timeout_s)
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=self.drain_timeout_s)
            logger.info("drain complete")
        except asyncio.TimeoutError:
            logger.warning("drain timeout — %d request(s) still in flight", self._in_flight)

    def begin_shutdown(self) -> None:
        if not self._shutting_down:
            logger.info("shutdown initiated")
            self._shutting_down = True
            self._idle.set()  # unblock wait_drain if called

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._signals_installed or os.getenv("DISABLE_SIGNAL_HANDLERS"):
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.begin_shutdown)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for SIGTERM
                pass
        self._signals_installed = True
        logger.info("signal handlers installed (SIGTERM, SIGINT)")


_shutdown: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    global _shutdown
    if _shutdown is None:
        _shutdown = ShutdownManager()
    return _shutdown


@asynccontextmanager
async def tracked_request(manager: Optional[ShutdownManager] = None):
    m = manager or get_shutdown_manager()
    await m.begin_request()
    try:
        yield
    finally:
        await m.end_request()
