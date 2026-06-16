"""Background-job queue (in-memory) with crash recovery.

Stage machine:
  uploading → extracting → chunking → embedding → indexing → completed
                                                              ↘ failed
  duplicate | skipped | stuck are terminal error-like states
"""
import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("rga_auditor.jobs")

PROCESSOR_TIMEOUT_S = float(os.getenv("JOB_PROCESSOR_TIMEOUT", "300"))
WORKER_POLL_INTERVAL_S = float(os.getenv("JOB_POLL_INTERVAL", "1.0"))
MAX_CONCURRENT_JOBS = int(os.getenv("JOB_MAX_CONCURRENT", "5"))

# Allowed stages
STAGE_UPLOADING = "uploading"
STAGE_EXTRACTING = "extracting"
STAGE_CHUNKING = "chunking"
STAGE_EMBEDDING = "embedding"
STAGE_INDEXING = "indexing"
STAGE_COMPLETED = "completed"
STAGE_FAILED = "failed"
STAGE_DUPLICATE = "duplicate"
STAGE_SKIPPED = "skipped"
STAGE_STUCK = "stuck"

ALLOWED_STAGES = {
    STAGE_UPLOADING, STAGE_EXTRACTING, STAGE_CHUNKING, STAGE_EMBEDDING,
    STAGE_INDEXING, STAGE_COMPLETED, STAGE_FAILED, STAGE_DUPLICATE,
    STAGE_SKIPPED, STAGE_STUCK,
}


@dataclass
class JobRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str = ""
    document_id: str = ""
    filename: str = ""
    content_path: str = ""
    privacy: bool = False
    stage: str = STAGE_UPLOADING
    progress: int = 0
    attempts: int = 0
    max_attempts: int = 3
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    next_run_at: datetime = field(default_factory=datetime.utcnow)

    def to_status_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "updated_at": (self.updated_at or self.created_at).isoformat() + "Z" if (self.updated_at or self.created_at) else None,
        }


ProcessorFn = Callable[["JobRecord"], Awaitable[None]]


class InMemoryJobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._wake.set()

    async def enqueue(self, record: JobRecord) -> None:
        async with self._lock:
            self._jobs[record.id] = record
            self._wake.set()

    async def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        async with self._lock:
            r = self._jobs.get(job_id)
            if r is None:
                return None
            for k, v in fields.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            r.updated_at = datetime.utcnow()
            return r

    async def claim(self) -> Optional[JobRecord]:
        async with self._lock:
            now = datetime.utcnow()
            for r in self._jobs.values():
                if r.stage == STAGE_UPLOADING and r.next_run_at <= now:
                    r.stage = STAGE_EXTRACTING
                    r.progress = max(r.progress, 10)
                    r.started_at = now
                    r.updated_at = now
                    r.attempts += 1
                    return r
        return None

    async def complete(self, record: JobRecord) -> None:
        async with self._lock:
            r = self._jobs.get(record.id)
            if r is None:
                return
            r.stage = STAGE_COMPLETED
            r.progress = 100
            r.updated_at = datetime.utcnow()
            self._wake.set()

    async def fail(self, record: JobRecord, error: str) -> None:
        async with self._lock:
            r = self._jobs.get(record.id)
            if r is None:
                return
            r.error = error[:1000]
            r.updated_at = datetime.utcnow()
            if r.attempts >= r.max_attempts:
                r.stage = STAGE_STUCK
            else:
                r.stage = STAGE_UPLOADING
                r.progress = 0
                r.next_run_at = datetime.utcnow() + timedelta(seconds=2 ** r.attempts)
            self._wake.set()

    async def requeue_processing(self) -> int:
        """On startup, any non-terminal jobs are stale — re-queue them."""
        async with self._lock:
            n = 0
            for r in self._jobs.values():
                if r.stage not in (STAGE_COMPLETED, STAGE_FAILED, STAGE_DUPLICATE, STAGE_SKIPPED, STAGE_STUCK):
                    r.stage = STAGE_UPLOADING
                    r.progress = 0
                    r.next_run_at = datetime.utcnow()
                    n += 1
            return n

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def size(self) -> int:
        async with self._lock:
            return sum(1 for r in self._jobs.values() if r.stage in (STAGE_UPLOADING, STAGE_EXTRACTING, STAGE_CHUNKING, STAGE_EMBEDDING, STAGE_INDEXING))


class JobQueueWorker:
    """Background task that polls a queue and runs a processor on each job."""

    def __init__(self, queue: InMemoryJobQueue, processor: Optional[ProcessorFn] = None) -> None:
        self.queue = queue
        self.processor = processor
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._active_tasks: set[asyncio.Task] = set()

    def set_processor(self, fn: ProcessorFn) -> None:
        self.processor = fn

    async def start(self) -> None:
        if self._task is not None:
            return
        await self.queue.requeue_processing()
        self._sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._active_tasks: set[asyncio.Task] = set()
        self._task = asyncio.create_task(self._run(), name="job-queue-worker")
        logger.info("Job queue worker started (max_concurrent=%d)", MAX_CONCURRENT_JOBS)

    async def stop(self) -> None:
        self._stop.set()
        for t in list(self._active_tasks):
            t.cancel()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Job queue worker stopped")

    async def enqueue(self, record: JobRecord) -> None:
        await self.queue.enqueue(record)

    async def _run(self) -> None:
        while not self._stop.is_set():
            self._active_tasks = {t for t in self._active_tasks if not t.done()}
            try:
                await self._sem.acquire()
                record = await self.queue.claim()
                if record is None:
                    self._sem.release()
                    try:
                        await asyncio.wait_for(self.queue._wake.wait(), timeout=WORKER_POLL_INTERVAL_S)
                    except asyncio.TimeoutError:
                        pass
                    self.queue._wake.clear()
                    if await self.queue.size() > 0:
                        self.queue._wake.set()
                    continue
                if self.processor is None:
                    self._sem.release()
                    await self.queue.fail(record, "no processor registered")
                    continue
                task = asyncio.create_task(self._process_one(record), name=f"job-{record.id}")
                self._active_tasks.add(task)
                task.add_done_callback(lambda _t: self._sem.release())
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._sem.release()
                logger.exception("worker loop error")
                await asyncio.sleep(1.0)

    async def _process_one(self, record: JobRecord) -> None:
        try:
            await asyncio.wait_for(self.processor(record), timeout=PROCESSOR_TIMEOUT_S)
            await self.queue.complete(record)
        except asyncio.TimeoutError:
            await self.queue.fail(record, f"timeout after {PROCESSOR_TIMEOUT_S}s")
        except asyncio.CancelledError:
            await self.queue.fail(record, "worker cancelled")
        except Exception as e:
            logger.exception("job %s failed", record.id)
            await self.queue.fail(record, f"{type(e).__name__}: {e}"[:1000])


_queue: Optional[InMemoryJobQueue] = None
_worker: Optional[JobQueueWorker] = None


def get_queue() -> InMemoryJobQueue:
    global _queue
    if _queue is None:
        _queue = InMemoryJobQueue()
    return _queue


def get_worker() -> JobQueueWorker:
    global _worker
    if _worker is None:
        _worker = JobQueueWorker(get_queue())
    return _worker
