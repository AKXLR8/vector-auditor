"""Cache backends. Redis when available, otherwise in-process cachetools.TTLCache."""
import hashlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("rga_auditor.cache")


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "")


CACHE_TTL = {
    "llm_response": int(os.getenv("CACHE_TTL_LLM", "3600")),
    "embedding": int(os.getenv("CACHE_TTL_EMBEDDING", "86400")),
    "document": int(os.getenv("CACHE_TTL_DOCUMENT", "300")),
    "query_result": int(os.getenv("CACHE_TTL_QUERY", "600")),
    "user": int(os.getenv("CACHE_TTL_USER", "60")),
}

DEFAULT_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))


class CacheBackend:
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def flush_pattern(self, pattern: str) -> None: ...


class MemoryBackend(CacheBackend):
    def __init__(self, max_size: int = DEFAULT_MAX_ENTRIES) -> None:
        from cachetools import TTLCache
        self._store: "TTLCache[str, Any]" = TTLCache(maxsize=max_size, ttl=DEFAULT_MAX_ENTRIES)

    async def get(self, key: str) -> Optional[Any]:
        try:
            return self._store[key]
        except KeyError:
            return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        try:
            del self._store[key]
        except KeyError:
            pass

    async def flush_pattern(self, pattern: str) -> None:
        for k in [k for k in list(self._store.keys()) if pattern in k]:
            try:
                del self._store[k]
            except KeyError:
                pass


class RedisBackend(CacheBackend):
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis
        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        val = await self._client.get(key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val

    async def set(self, key: str, value: Any, ttl: int) -> None:
        payload = json.dumps(value, default=str)
        await self._client.setex(key, ttl, payload)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def flush_pattern(self, pattern: str) -> None:
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=f"*{pattern}*", count=100)
            if keys:
                await self._client.delete(*keys)
            if cursor == 0:
                break


def _build_backend() -> CacheBackend:
    url = _redis_url()
    if url:
        try:
            backend = RedisBackend(url)
            logger.info("Cache backend: Redis (%s)", url.split("@")[-1] if "@" in url else "local")
            return backend
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-memory cache", e)
    logger.info("Cache backend: in-memory (TTLCache, max=%d)", DEFAULT_MAX_ENTRIES)
    return MemoryBackend()


_cache: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    global _cache
    if _cache is None:
        _cache = _build_backend()
    return _cache


def cache_key(prefix: str, *parts: str) -> str:
    raw = ":".join(parts)
    h = hashlib.sha256(raw.encode()).hexdigest()
    return f"{prefix}:{h}"


async def get_or_compute(key: str, ttl: int, fn):
    cache = get_cache()
    cached = await cache.get(key)
    if cached is not None:
        return cached
    value = await fn() if callable(fn) else fn
    await cache.set(key, value, ttl)
    return value
