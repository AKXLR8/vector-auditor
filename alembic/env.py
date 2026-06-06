"""Alembic environment — async migrations."""
import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Load .env so DATABASE_URL is visible to alembic
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# Make `src` importable when alembic is run as a script
sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override URL from env, with asyncpg + ssl=true conversion
url = os.getenv("DATABASE_URL")
if url:
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    if "sslmode=require" in url or "sslmode=verify-ca" in url or "sslmode=verify-full" in url:
        # Strip sslmode from URL — asyncpg rejects it as a kwarg
        from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=True)
        for k in ("sslmode", "ssl"):
            qs.pop(k, None)
        parsed = parsed._replace(query=urlencode(qs, doseq=True))
        url = urlunparse(parsed)
    config.set_main_option("sqlalchemy.url", url)


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")
    # Pass ssl=True via connect_args for Neon
    connectable = async_engine_from_config(
        cfg, prefix="sqlalchemy.", poolclass=pool.NullPool,
        connect_args={"ssl": True},
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
