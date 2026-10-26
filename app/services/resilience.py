"""Circuit breaker and retry-with-backoff, shared by the LLM fallback chain and the job publisher."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """Consecutive-failure breaker.

    closed     calls pass; `failure_threshold` failures in a row open the circuit
    open       calls are rejected without trying, for `reset_seconds`
    half_open  one trial call is let through: success closes the circuit, failure re-opens it

    State is per process, so each API replica learns about an outage on its own. That costs
    each replica `failure_threshold` slow failures, and needs no shared state.
    """

    def __init__(self, name: str, failure_threshold: int, reset_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_started_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "open":
            return False
        now = self._clock()
        # One trial at a time; a trial that never reported back stops blocking after reset_seconds.
        if self._trial_started_at is not None and now - self._trial_started_at < self.reset_seconds:
            return False
        self._trial_started_at = now
        return True

    def record_success(self) -> None:
        if self._opened_at is not None:
            logger.info("Circuit %s closed", self.name)
        self._failures = 0
        self._opened_at = None
        self._trial_started_at = None

    def record_failure(self) -> None:
        self._failures += 1
        self._trial_started_at = None
        if self.state == "half_open" or self._failures >= self.failure_threshold:
            if self.state != "open":
                logger.warning("Circuit %s opened after %d consecutive failures", self.name, self._failures)
            self._opened_at = self._clock()


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential backoff with full jitter, so replicas retrying together don't stampede."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on:
            if attempt == attempts:
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            logger.warning("Attempt %d/%d failed; retrying in %.2fs", attempt, attempts, delay, exc_info=True)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
