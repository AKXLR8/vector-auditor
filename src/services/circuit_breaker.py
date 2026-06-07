"""Lightweight circuit breaker for external calls."""
import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger("rga_auditor.cb")

T = TypeVar("T")


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at > self.recovery_timeout:
            return False  # half-open
        return True

    async def call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        async with self._lock:
            if self._opened_at and time.time() - self._opened_at > self.recovery_timeout:
                logger.info("circuit %s: half-open", self.name)
                self._opened_at = None
                self._failures = 0
            if self._opened_at is not None:
                raise RuntimeError(f"circuit '{self.name}' is open")
        try:
            result = await fn(*args, **kwargs)
        except BaseException:
            async with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._opened_at = time.time()
                    logger.warning("circuit %s: OPENED after %d failures", self.name, self._failures)
            raise
        else:
            async with self._lock:
                if self._failures:
                    logger.info("circuit %s: reset after %d failures", self.name, self._failures)
                self._failures = 0
                self._opened_at = None
            return result
