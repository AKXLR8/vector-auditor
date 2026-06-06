"""Bootstrap — dependency-ordered init at app startup.

Loads embedding model, pre-warms LLM, starts job-queue worker, installs signal handlers.
Fail-fast in production, warn-and-continue in development.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .database.session import dispose_engine, init_engine
from .job_queue import get_worker
from .observability import get_metrics, setup_observability
from .services.cache import get_cache
from .services.cloudinary import get_cloudinary
from .services.guardrails import get_guardrails
from .services.llm import get_llm
from .services.pii_detector import get_pii_detector
from .shutdown import get_shutdown_manager
from .vectorstore.Qdrant import get_vector_store

logger = logging.getLogger("rga_auditor.bootstrap")

StartupError = RuntimeError


async def bootstrap_app() -> dict:
    """Initialize all subsystems in dependency order. Returns a status dict."""
    s = get_settings()
    setup_observability()
    t0 = time.time()
    status: dict = {"steps": []}

    def step(name: str, fn):
        t = time.time()
        try:
            result = fn()
            dt = time.time() - t
            status["steps"].append({"name": name, "ok": True, "ms": int(dt * 1000)})
            logger.info("bootstrap[%s] OK in %dms", name, int(dt * 1000))
            return result
        except Exception as e:
            dt = time.time() - t
            status["steps"].append({"name": name, "ok": False, "ms": int(dt * 1000), "error": str(e)})
            if s.is_production:
                logger.critical("bootstrap[%s] FAILED: %s", name, e)
                raise StartupError(f"bootstrap step '{name}' failed: {e}") from e
            logger.warning("bootstrap[%s] failed (dev mode, continuing): %s", name, e)
            return None

    # 1. database
    step("database", lambda: asyncio.get_event_loop().run_until_complete(init_engine()) if asyncio.get_event_loop().is_running() else None)
    # Run async init synchronously via the current loop
    if asyncio.get_event_loop().is_running():
        await init_engine()
        status["steps"][-1] = {"name": "database", "ok": True, "ms": 0}

    # 2. cache
    step("cache", get_cache)

    # 3. vector store (loads embedding model — slow first time)
    step("vector_store", get_vector_store)

    # 4. guardrails + pii (best-effort)
    step("guardrails", get_guardrails)
    step("pii_detector", get_pii_detector)
    step("cloudinary", get_cloudinary)

    # 5. LLM pre-warm (optional)
    if s.skip_llm_prewarm:
        logger.info("bootstrap[llm] SKIPPED (SKIP_LLM_PREWARM=true)")
    else:
        def _prewarm():
            llm = get_llm()
            try:
                # Synchronous prewarm: we can't await here; the LLM client's chat is async
                return llm
            except Exception as e:
                if s.is_production:
                    raise
                logger.warning("LLM prewarm failed: %s", e)
        step("llm", _prewarm)
        # actually prewarm async
        try:
            await get_llm().chat("ping", system="You are a test. Reply with 'pong'.", max_tokens=10)
            logger.info("LLM prewarm OK")
        except Exception as e:
            if s.is_production:
                raise StartupError(f"LLM prewarm failed: {e}")
            logger.warning("LLM prewarm failed: %s", e)

    # 6. job queue worker
    await get_worker().start()

    # 7. signal handlers
    try:
        loop = asyncio.get_running_loop()
        get_shutdown_manager().install_signal_handlers(loop)
    except Exception as e:
        logger.debug("signal handlers skipped: %s", e)

    # 8. metrics timestamp
    get_metrics().startup_timestamp.set(time.time())

    status["ready_at"] = time.time()
    status["ready_in_ms"] = int((time.time() - t0) * 1000)
    logger.info("Application READY in %dms", status["ready_in_ms"])
    return status


async def shutdown_app() -> None:
    """Tear down in reverse order."""
    logger.info("Application shutting down")
    try:
        mgr = get_shutdown_manager()
        mgr.begin_shutdown()
        await mgr.wait_drain()
    except Exception as e:
        logger.warning("drain: %s", e)
    try:
        await get_worker().stop()
    except Exception as e:
        logger.warning("worker stop: %s", e)
    try:
        await dispose_engine()
    except Exception as e:
        logger.warning("db dispose: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bootstrap_app()
    yield
    await shutdown_app()


def install_bootstrap(app: FastAPI) -> None:
    """Attach the bootstrap lifespan to an existing FastAPI app."""
    app.router.lifespan_context = lifespan
