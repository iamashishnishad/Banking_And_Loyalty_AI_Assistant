"""Semantic answer cache: a new question close enough to one already answered reuses the answer.

Entries are Redis hashes indexed by the Redis Query Engine (HNSW, cosine) and found with a
filtered KNN query: same tenant, same reply language, same knowledge-base version. Each entry
expires on its own TTL, and uploading or deleting a document bumps the tenant's KB version, so
answers built on old documents stop matching at once.

A hit also needs the same numbers in both questions. Embeddings barely see numbers: "FD rate
for 1 year" vs "... 2 years" scores 0.970 with e5-small, above most true paraphrases
(scripts/calibrate_cache.py). So the 5 nearest entries are fetched and the first one above the
threshold with matching numbers wins.

Only answers that can't depend on who is asking are stored:
  - the question has no first-person / account words ("my", "mere", "मेरे", "redeem", ...)
  - the agent used only search_knowledge_base, at least once, with no tool errors
  - the answer finished normally and doesn't contain the customer's name
Everything else (balances, redemptions, greetings) always goes to the LLM.

Cache failures never fail a chat turn: every public function logs and carries on.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.config import get_settings

logger = logging.getLogger(__name__)

INDEX = "idx:semcache"
PREFIX = "semcache:"

# Words that make a question about the asker's own account, or a reply to an earlier turn.
_PERSONAL = {
    # English
    "i", "me", "my", "mine", "i'm", "im", "myself", "we", "our", "us",
    # Hinglish
    "mera", "mere", "meri", "mujhe", "maine", "main", "mai", "hum", "hamara", "hamare", "hamari", "humara",
    "apna", "apne", "apni",
    # Hindi
    "मेरा", "मेरे", "मेरी", "मुझे", "मैंने", "मैं", "हम", "हमारा", "हमारे", "अपना", "अपने",
    # Account actions and confirmations
    "balance", "redeem", "redemption", "confirm", "cancel", "yes", "haan", "han", "ok", "okay", "हाँ", "हां",
}
# \w alone splits Devanagari words at their vowel signs (मेरे -> म, र), so add the block explicitly.
_WORD = re.compile(r"[\w\u0900-\u097F']+")

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")  # \d also matches Devanagari digits (२ == 2)
_KNN = 5

_index_available: bool | None = None  # None until the first FT.CREATE attempt


@dataclass
class CacheHit:
    answer: str
    sources: list[dict]
    model: str
    similarity: float


def numbers_in(text: str) -> list[str]:
    """Canonical numbers in a question: '2', '2.0' and '२' are all '2'; '1,000' is '1000'."""
    return sorted({f"{float(n.replace(',', '')):g}" for n in _NUMBER.findall(text)})


def is_cacheable_question(message: str) -> bool:
    return not (set(_WORD.findall(message.lower())) & _PERSONAL)


def is_cacheable_answer(done: dict, customer_name: str | None) -> bool:
    calls = done.get("tool_calls") or []
    if done.get("stop_reason") != "end_turn" or not calls or not done.get("sources"):
        return False
    if any(c["name"] != "search_knowledge_base" or c["is_error"] for c in calls):
        return False
    if customer_name:
        answer = done["answer"].lower()
        if any(len(part) >= 3 and part.lower() in answer for part in customer_name.split()):
            return False
    return True


def _kb_version_key(tenant_id: uuid.UUID) -> str:
    return f"kbv:{tenant_id.hex}"


async def kb_version(redis: Redis, tenant_id: uuid.UUID) -> int:
    return int(await redis.get(_kb_version_key(tenant_id)) or 0)


async def bump_kb_version(redis: Redis, tenant_id: uuid.UUID) -> None:
    """Call after the tenant's documents change: cached answers built on the old set stop matching."""
    try:
        await redis.incr(_kb_version_key(tenant_id))
    except Exception:
        logger.warning("Could not bump KB version for %s; cached answers expire by TTL instead", tenant_id, exc_info=True)


async def _ensure_index(redis: Redis) -> bool:
    global _index_available
    if _index_available is None:
        try:
            await redis.execute_command(
                "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", PREFIX,
                "SCHEMA", "tenant", "TAG", "lang", "TAG", "kbv", "TAG",
                "vec", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(get_settings().embedding_dim),
                "DISTANCE_METRIC", "COSINE",
            )  # fmt: skip
            _index_available = True
        except ResponseError as exc:
            if "already exists" in str(exc).lower():
                _index_available = True
            else:
                # Plain Redis without the Query Engine: run without the cache rather than fail.
                logger.warning("Semantic cache disabled: %s", exc)
                _index_available = False
    return _index_available


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _search_rows(reply) -> list[dict]:
    """FT.SEARCH rows as {field: value}, nearest first. redis-py 8 parses the reply into a map
    ({results: [{id, extra_attributes}]}); older clients return the raw RESP2 array
    [total, key1, [field, value, ...], key2, ...]."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        return [
            {_decode(k): v for k, v in (row.get(b"extra_attributes", row.get("extra_attributes")) or {}).items()}
            for row in results
        ]
    return [{_decode(f[i]): f[i + 1] for i in range(0, len(f), 2)} for f in (reply[2::2] if reply else [])]


class SemanticCache:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        settings = get_settings()
        self.threshold = settings.semantic_cache_threshold
        self.ttl = settings.semantic_cache_ttl_seconds

    async def lookup(
        self, tenant_id: uuid.UUID, language: str, version: int, vector: np.ndarray, question: str
    ) -> CacheHit | None:
        try:
            if not await _ensure_index(self.redis):
                return None
            # Tag values are hex / lowercase words / digits, so they need no escaping.
            query = f"(@tenant:{{{tenant_id.hex}}} @lang:{{{language}}} @kbv:{{{version}}})=>[KNN {_KNN} @vec $vec AS dist]"
            reply = await self.redis.execute_command(
                "FT.SEARCH", INDEX, query, "PARAMS", "2", "vec", vector.astype(np.float32).tobytes(),
                "SORTBY", "dist", "RETURN", "2", "dist", "payload", "LIMIT", "0", str(_KNN), "DIALECT", "2",
            )  # fmt: skip
            wanted = numbers_in(question)
            for doc in _search_rows(reply):
                similarity = 1.0 - float(_decode(doc["dist"]))  # COSINE distance = 1 - cosine similarity
                if similarity < self.threshold:
                    break
                payload = json.loads(doc["payload"])
                if payload["numbers"] == wanted:
                    return CacheHit(
                        answer=payload["answer"], sources=payload["sources"], model=payload["model"], similarity=similarity
                    )
        except Exception:
            logger.warning("Semantic cache lookup failed", exc_info=True)
        return None

    async def store(
        self, tenant_id: uuid.UUID, language: str, version: int, vector: np.ndarray, question: str, done: dict
    ) -> None:
        try:
            if not await _ensure_index(self.redis):
                return
            key = f"{PREFIX}{uuid.uuid4().hex}"
            payload = {
                "question": question,
                "numbers": numbers_in(question),
                "answer": done["answer"],
                "sources": done["sources"],
                "model": done["model"],
            }
            pipe = self.redis.pipeline()
            pipe.hset(
                key,
                mapping={
                    "tenant": tenant_id.hex,
                    "lang": language,
                    "kbv": str(version),
                    "vec": vector.astype(np.float32).tobytes(),
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            )
            pipe.expire(key, self.ttl)
            await pipe.execute()
        except Exception:
            logger.warning("Semantic cache store failed", exc_info=True)
