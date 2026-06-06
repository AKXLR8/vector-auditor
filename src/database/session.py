"""SQLAlchemy async engine + session factory."""
import logging
import os
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger("rga_auditor.db")

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> Optional[AsyncEngine]:
    return _engine


def get_session_factory() -> Optional[async_sessionmaker[AsyncSession]]:
    return _session_factory


async def init_engine(database_url: Optional[str] = None) -> Optional[AsyncEngine]:
    """Initialize the global async engine. Returns None if no DATABASE_URL is set.

    Designed for Neon Postgres over asyncpg:
    - URL is rewritten to postgresql+asyncpg://
    - libpq-style `sslmode=require` is stripped from the URL and passed
      via connect_args={"ssl": True} instead, because asyncpg.connect()
      rejects `sslmode` as a keyword argument.
    """
    global _engine, _session_factory
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        logger.warning("DATABASE_URL not set — running with file-backed in-memory stores")
        return None
    if _engine is not None:
        return _engine
    # 1) Force the asyncpg driver
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    # 2) Strip sslmode=... and ssl=... from the URL query string.
    #    asyncpg.connect() raises "unexpected keyword argument 'sslmode'" if
    #    these leak through, so we always remove them and re-supply via
    #    connect_args.
    from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    ssl_modes = {k: v[0] for k, v in qs.items() if k in ("sslmode", "ssl")}
    needs_ssl = bool(ssl_modes)
    for k in list(qs.keys()):
        if k in ("sslmode", "ssl"):
            qs.pop(k)
    parsed = parsed._replace(query=urlencode(qs, doseq=True))
    url = urlunparse(parsed)
    # 3) Build connect_args
    connect_args: dict = {}
    if needs_ssl:
        mode = (ssl_modes.get("sslmode") or ssl_modes.get("ssl") or "require").lower()
        if mode in ("disable", "false", "0", "no"):
            connect_args["ssl"] = False
        elif mode in ("require", "verify-ca", "verify-full", "prefer", "allow", "true", "1", "yes"):
            # Neon requires TLS but uses a public CA — default SSLContext is fine.
            connect_args["ssl"] = True
        else:
            connect_args["ssl"] = True
    _engine = create_async_engine(
        url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "5")),
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args or None,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    logger.info(
        "Database engine ready — pool_size=%s, max_overflow=%s",
        os.getenv("DB_POOL_SIZE", "5"),
        os.getenv("DB_MAX_OVERFLOW", "5"),
    )
    return _engine


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_session() -> AsyncIterator[Optional[AsyncSession]]:
    """FastAPI dependency. Yields a session or None if DB is not configured."""
    if _session_factory is None:
        yield None
        return
    async with _session_factory() as session:
        yield session
