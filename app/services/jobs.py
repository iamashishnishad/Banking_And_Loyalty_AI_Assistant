"""RabbitMQ topology and the publisher the API uses to hand jobs to workers.

    jobs        work queue. A rejected message dead-letters to jobs.dead.
    jobs.retry  holding queue with a fixed TTL; expired messages dead-letter back into jobs.
                That gives a retry delay without a broker plugin.
    jobs.dead   jobs that failed permanently or ran out of attempts, kept for inspection.

Messages are persistent and carry only {"job_id"}; the jobs table holds everything else.
"""

import asyncio
import json
import logging
import uuid

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractQueue, AbstractRobustConnection

from app.config import get_settings
from app.services.resilience import retry_async

logger = logging.getLogger(__name__)

QUEUE = "jobs"
RETRY_QUEUE = "jobs.retry"
DEAD_QUEUE = "jobs.dead"


async def declare_topology(channel: AbstractChannel) -> AbstractQueue:
    """Idempotent; the API and every worker call it. Returns the work queue."""
    await channel.declare_queue(DEAD_QUEUE, durable=True)
    await channel.declare_queue(
        RETRY_QUEUE,
        durable=True,
        arguments={
            "x-message-ttl": get_settings().job_retry_delay_seconds * 1000,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": QUEUE,
        },
    )
    return await channel.declare_queue(
        QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": "", "x-dead-letter-routing-key": DEAD_QUEUE},
    )


def job_message(job_id: uuid.UUID) -> aio_pika.Message:
    return aio_pika.Message(
        json.dumps({"job_id": str(job_id)}).encode(),
        content_type="application/json",
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        message_id=str(job_id),
    )


class JobPublisher:
    """Connects lazily, so the API starts (and keeps serving chat) while RabbitMQ is down.
    A publish that fails leaves the job 'queued' in Postgres; the worker's sweeper re-publishes it."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._lock = asyncio.Lock()

    async def _get_channel(self) -> AbstractChannel:
        async with self._lock:
            if self._channel is None or self._channel.is_closed:
                if self._connection is None or self._connection.is_closed:
                    self._connection = await aio_pika.connect_robust(self.url, timeout=5)
                # publisher_confirms (the default) makes publish() wait for the broker's ack.
                self._channel = await self._connection.channel()
                await declare_topology(self._channel)
            return self._channel

    async def publish(self, job_id: uuid.UUID) -> None:
        async def attempt() -> None:
            channel = await self._get_channel()
            await channel.default_exchange.publish(job_message(job_id), routing_key=QUEUE, timeout=5)

        await retry_async(attempt, attempts=3)

    async def ping(self) -> bool:
        try:
            await asyncio.wait_for(self._get_channel(), timeout=3)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._connection is not None and not self._connection.is_closed:
            await self._connection.close()
