"""RabbitMQ worker for background jobs (document ingestion).

    python -m app.worker                      # what the compose `worker` service runs
    docker compose up -d --scale worker=3     # more throughput: one job at a time per worker

Delivery is at-least-once, so every step is idempotent:
  - a job is claimed with a conditional UPDATE; a duplicate message for a finished job is dropped
  - ingestion dedups on (tenant, sha256), so re-running a half-finished job can't create a copy
  - the message is acked only after the job's outcome is committed; a crash means redelivery

Failures: a bad file (PermanentJobError) fails at once and dead-letters; anything else is retried
via jobs.retry until job_max_attempts, then dead-lettered. If Postgres itself is unreachable the
message is requeued untouched. The sweeper re-publishes jobs whose publish never reached the broker.
"""

import asyncio
import json
import logging
import signal
import uuid
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Literal

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage
from redis.asyncio import Redis
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import dispose_engine, get_sessionmaker
from app.models import Job
from app.services.embeddings import get_embedder
from app.services.ingestion import ingest_document
from app.services.jobs import QUEUE, RETRY_QUEUE, declare_topology, job_message
from app.services.parsing import UnsupportedDocumentError
from app.services.resilience import retry_async
from app.services.semantic_cache import bump_kb_version

logger = logging.getLogger("worker")

Outcome = Literal["succeeded", "skipped", "retry", "dead"]


class PermanentJobError(Exception):
    """Retrying can't help (unparseable file, unknown job kind)."""


async def _ingest(session: AsyncSession, job: Job) -> dict:
    try:
        result = await ingest_document(session, job.tenant_id, job.params["filename"], job.payload, job.params.get("title"))
    except UnsupportedDocumentError as exc:
        raise PermanentJobError(str(exc)) from exc
    return {"document_id": str(result.document.id), "created": result.created, "chunk_count": result.document.chunk_count}


HANDLERS: dict[str, Callable[[AsyncSession, Job], Awaitable[dict]]] = {"ingest_document": _ingest}


class Worker:
    def __init__(self, channel: AbstractChannel, redis: Redis) -> None:
        self.channel = channel
        self.redis = redis
        self.settings = get_settings()

    async def on_message(self, message: AbstractIncomingMessage) -> None:
        try:
            job_id = uuid.UUID(json.loads(message.body)["job_id"])
        except (ValueError, KeyError, TypeError):
            logger.error("Malformed message %r; dead-lettering", message.body[:200])
            await message.reject(requeue=False)
            return
        try:
            outcome = await self.run(job_id)
        except Exception:
            # Infrastructure failure (e.g. Postgres down): the job's state is unknown. Back off, then
            # hand the message back to the broker rather than spend an attempt on it.
            logger.exception("Job %s could not be processed; requeueing", job_id)
            await asyncio.sleep(5)
            await message.nack(requeue=True)
            return

        if outcome == "retry":
            await self.channel.default_exchange.publish(job_message(job_id), routing_key=RETRY_QUEUE)
            await message.ack()
        elif outcome == "dead":
            await message.reject(requeue=False)  # dead-letters to jobs.dead
        else:
            await message.ack()

    async def run(self, job_id: uuid.UUID) -> Outcome:
        async with get_sessionmaker()() as session:
            # 'running' is claimable too: that's a redelivery after a worker died mid-job.
            attempts = await session.scalar(
                update(Job)
                .where(Job.id == job_id, Job.status.in_(("queued", "running")))
                .values(status="running", attempts=Job.attempts + 1, updated_at=func.now())
                .returning(Job.attempts)
            )
            await session.commit()
            if attempts is None:
                logger.info("Job %s already finished or unknown; dropping message", job_id)
                return "skipped"

            job = await session.get(Job, job_id)
            handler = HANDLERS.get(job.kind)
            try:
                if handler is None:
                    raise PermanentJobError(f"Unknown job kind '{job.kind}'")
                result = await handler(session, job)
            except Exception as exc:
                await session.rollback()
                permanent = isinstance(exc, PermanentJobError) or attempts >= self.settings.job_max_attempts
                logger.warning(
                    "Job %s attempt %d failed (%s): %s", job_id, attempts, "giving up" if permanent else "will retry", exc,
                    exc_info=not isinstance(exc, PermanentJobError),
                )  # fmt: skip
                await session.execute(
                    update(Job)
                    .where(Job.id == job_id)
                    .values(status="failed" if permanent else "queued", error=f"{type(exc).__name__}: {exc}"[:2000])
                )
                await session.commit()
                return "dead" if permanent else "retry"

            await session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status="succeeded", result=result, error=None, payload=None)  # bytes no longer needed
            )
            await session.commit()

        if job.kind == "ingest_document":
            await bump_kb_version(self.redis, job.tenant_id)
        logger.info("Job %s succeeded on attempt %d: %s", job_id, attempts, result)
        return "succeeded"

    async def sweep(self) -> int:
        """Re-publish jobs stuck in 'queued' (the API's publish failed, or the message was lost)."""
        async with get_sessionmaker()() as session:
            ids = (
                await session.scalars(
                    update(Job)
                    .where(Job.status == "queued", Job.updated_at < func.now() - timedelta(seconds=self.settings.job_stale_seconds))
                    .values(updated_at=func.now())
                    .returning(Job.id)
                )
            ).all()
            await session.commit()
        for job_id in ids:
            await self.channel.default_exchange.publish(job_message(job_id), routing_key=QUEUE)
        if ids:
            logger.warning("Sweeper re-published %d stale job(s)", len(ids))
        return len(ids)

    async def sweep_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.sweep()
            except Exception:
                logger.exception("Sweep failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.settings.job_sweep_interval_seconds)
            except TimeoutError:
                pass


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    await asyncio.to_thread(get_embedder)  # load the model before taking jobs
    redis = Redis.from_url(settings.redis_url, socket_timeout=settings.redis_socket_timeout_seconds)
    connection = await retry_async(
        lambda: aio_pika.connect_robust(settings.rabbitmq_url, timeout=5), attempts=30, base_delay=1, max_delay=5
    )
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=settings.worker_prefetch)
    queue = await declare_topology(channel)
    worker = Worker(channel, redis)
    consumer_tag = await queue.consume(worker.on_message)
    logger.info("Worker consuming %s (prefetch %d)", QUEUE, settings.worker_prefetch)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    sweeper = asyncio.create_task(worker.sweep_forever(stop))
    await stop.wait()

    logger.info("Shutting down")
    await queue.cancel(consumer_tag)  # an unacked in-flight message goes back to the queue
    await sweeper
    await connection.close()
    await redis.aclose()
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
