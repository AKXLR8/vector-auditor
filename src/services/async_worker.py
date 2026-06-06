"""Run sync functions in the default executor (non-blocking)."""
import asyncio
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
