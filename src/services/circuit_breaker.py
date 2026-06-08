"""Circuit breaker with exponential backoff retry for LLM, Qdrant, and embedding calls."""
import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger("rga_auditor.circuit_breaker")

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_retries: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_retries = half_open_max_retries
        self.state = STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_attempts = 0
        self._lock = asyncio.Lock()

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if self.state == STATE_OPEN:
                if time.monotonic() - self.last_failure_time >= self.recovery_timeout_s:
                    self.state = STATE_HALF_OPEN
                    self.half_open_attempts = 0
                    logger.info("Circuit %s → half_open (recovery timeout elapsed)", self.name)
                else:
                    raise CircuitBreakerOpenError(f"{self.name} circuit is open")

        try:
            result = await fn(*args, **kwargs)
            async with self._lock:
                if self.state == STATE_HALF_OPEN:
                    self.half_open_attempts += 1
                    if self.half_open_attempts >= self.half_open_max_retries:
                        self.state = STATE_CLOSED
                        self.failure_count = 0
                        logger.info("Circuit %s → closed (recovered)", self.name)
                else:
                    self.failure_count = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.monotonic()
                if self.failure_count >= self.failure_threshold and self.state != STATE_OPEN:
                    self.state = STATE_OPEN
                    logger.warning("Circuit %s → open (%d failures)", self.name, self.failure_count)
            raise

    def is_available(self) -> bool:
        if self.state == STATE_OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout_s:
                return True
            return False
        return True


class CircuitBreakerOpenError(Exception):
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 10.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, OSError),
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = min(base_delay_s * (2 ** attempt), max_delay_s)
                        logger.warning("Retry %d/%d for %s after %.2fs: %s", attempt + 1, max_retries, fn.__name__, delay, e)
                        await asyncio.sleep(delay)
                except CircuitBreakerOpenError:
                    raise
                except Exception as e:
                    if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                        last_exc = e
                        if attempt < max_retries:
                            delay = min(base_delay_s * (2 ** attempt), max_delay_s)
                            logger.warning("Retry %d/%d for %s after %.2fs (timeout): %s", attempt + 1, max_retries, fn.__name__, delay, e)
                            await asyncio.sleep(delay)
                    else:
                        raise
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
