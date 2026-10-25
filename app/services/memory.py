"""Per-customer conversation memory and pending-redemption state in Redis.

Only the user's text and the assistant's final answer are stored, not tool calls or thinking
blocks. This keeps replayed history small and cheap. The cost is that the model can't see an
earlier turn's raw tool output, and has to call the tool again if it needs it.
"""

import json
import uuid
from dataclasses import asdict, dataclass

from redis.asyncio import Redis

from app.config import get_settings


@dataclass
class PendingRedemption:
    id: str
    points: int
    redemption_type: str
    created_turn: int


class ConversationMemory:
    def __init__(self, redis: Redis, tenant_id: uuid.UUID, customer_ref: str) -> None:
        self.redis = redis
        prefix = f"chat:{tenant_id}:{customer_ref}"
        self._history_key = f"{prefix}:history"
        self._turn_key = f"{prefix}:turn"
        self._pending_key = f"{prefix}:pending_redemption"
        settings = get_settings()
        self._max_messages = settings.memory_max_messages
        self._ttl = settings.memory_ttl_seconds
        self._pending_ttl = settings.redemption_confirm_ttl_seconds

    async def history(self) -> list[dict]:
        raw = await self.redis.lrange(self._history_key, 0, -1)
        messages = [json.loads(m) for m in raw]
        # LTRIM can cut a user/assistant pair in half; the API needs history to start with a user turn.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages

    async def next_turn(self) -> int:
        """Monotonic per-customer turn number. Redemption confirmation compares turns to prove
        that the customer sent a new message between 'start' and 'confirm'."""
        pipe = self.redis.pipeline()
        pipe.incr(self._turn_key)
        pipe.expire(self._turn_key, self._ttl)
        turn, _ = await pipe.execute()
        return int(turn)

    async def append_exchange(self, user_text: str, assistant_text: str) -> None:
        pipe = self.redis.pipeline()
        pipe.rpush(
            self._history_key,
            json.dumps({"role": "user", "content": user_text}, ensure_ascii=False),
            json.dumps({"role": "assistant", "content": assistant_text}, ensure_ascii=False),
        )
        pipe.ltrim(self._history_key, -self._max_messages, -1)
        pipe.expire(self._history_key, self._ttl)
        await pipe.execute()

    async def clear(self) -> None:
        await self.redis.delete(self._history_key, self._turn_key, self._pending_key)

    async def set_pending_redemption(self, pending: PendingRedemption) -> None:
        await self.redis.set(self._pending_key, json.dumps(asdict(pending)), ex=self._pending_ttl)

    async def get_pending_redemption(self) -> PendingRedemption | None:
        raw = await self.redis.get(self._pending_key)
        return PendingRedemption(**json.loads(raw)) if raw else None

    async def clear_pending_redemption(self) -> None:
        await self.redis.delete(self._pending_key)
