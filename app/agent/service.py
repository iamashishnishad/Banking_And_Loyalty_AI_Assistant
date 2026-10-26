"""One chat turn end to end: load context and memory, run the agent, save the exchange.

Three ways to answer, cheapest first:
    1. semantic cache   a near-identical policy question was already answered (no LLM call)
    2. the agent        primary LLM, then fallback LLM (see ResilientLLM)
    3. baseline         every LLM is down: keyword router + extractive search answer, flagged degraded
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from sqlalchemy import select

from app.agent.llm import LLMClient
from app.agent.runner import AgentEvent, run_agent
from app.agent.tools import ToolContext
from app.config import get_settings
from app.db import get_sessionmaker
from app.models import Customer, Tenant
from app.services.baseline import baseline_reply
from app.services.embeddings import embed_query
from app.services.language import detect_language
from app.services.memory import ConversationMemory
from app.services.semantic_cache import SemanticCache, is_cacheable_answer, is_cacheable_question, kb_version

logger = logging.getLogger(__name__)


async def chat_turn(
    *,
    llm: LLMClient,
    redis: Redis,
    tenant_id: uuid.UUID,
    customer_ref: str,
    message: str,
    use_cache: bool = True,
) -> AsyncIterator[AgentEvent]:
    """Opens its own DB session. A streaming response outlives FastAPI's request-scoped
    dependencies, so the session has to belong to the generator."""
    started = time.perf_counter()
    language = detect_language(message)
    memory = ConversationMemory(redis, tenant_id, customer_ref)
    history = await memory.history()
    turn = await memory.next_turn()

    cache, vector, version = None, None, 0
    if get_settings().semantic_cache_enabled and is_cacheable_question(message):
        try:
            cache = SemanticCache(redis)
            vector = await embed_query(message, redis)
            version = await kb_version(redis, tenant_id)
        except Exception:
            logger.warning("Semantic cache unavailable for this turn", exc_info=True)
            cache = None
        # use_cache=False skips the lookup only; the fresh answer still refreshes the cache.
        hit = await cache.lookup(tenant_id, language, version, vector, message) if cache and use_cache else None
        if hit is not None:
            yield AgentEvent("meta", {"language": language, "turn": turn, "model": hit.model, "cached": True})
            yield AgentEvent("token", {"text": hit.answer})
            await memory.append_exchange(message, hit.answer)
            yield AgentEvent(
                "done",
                {
                    "answer": hit.answer,
                    "language": language,
                    "model": hit.model,
                    "stop_reason": "end_turn",
                    "tool_calls": [],
                    "sources": hit.sources,
                    "usage": {},
                    "took_ms": round((time.perf_counter() - started) * 1000, 1),
                    "cached": True,
                    "cache_similarity": round(hit.similarity, 4),
                },
            )
            return

    async with get_sessionmaker()() as session:
        bank_name = await session.scalar(select(Tenant.name).where(Tenant.id == tenant_id))
        customer_name = await session.scalar(
            select(Customer.name).where(Customer.tenant_id == tenant_id, Customer.external_ref == customer_ref)
        )
        ctx = ToolContext(
            session=session, redis=redis, tenant_id=tenant_id, customer_ref=customer_ref, memory=memory, turn=turn
        )
        yield AgentEvent("meta", {"language": language, "turn": turn, "model": llm.model, "cached": False})
        streamed_text = False
        async for event in run_agent(
            llm=llm,
            ctx=ctx,
            user_message=message,
            history=history,
            bank_name=bank_name or "the bank",
            customer_name=customer_name,
            language=language,
        ):
            if event.type == "token":
                streamed_text = True
            elif event.type == "done":
                await memory.append_exchange(message, event.data["answer"])
                if cache is not None and is_cacheable_answer(event.data, customer_name):
                    await cache.store(tenant_id, language, version, vector, message, event.data)
            elif event.type == "error" and event.data.get("code") == "llm_unavailable" and not streamed_text:
                # Every provider failed before saying anything: answer without an LLM instead.
                done = await _degraded_answer(session, redis, tenant_id, customer_ref, message, language, memory)
                done.data["took_ms"] = round((time.perf_counter() - started) * 1000, 1)
                yield AgentEvent("token", {"text": done.data["answer"]})
                yield done
                return
            yield event


async def _degraded_answer(session, redis, tenant_id, customer_ref, message, language, memory) -> AgentEvent:
    await session.rollback()  # a failed tool round may have left the transaction aborted
    reply = await baseline_reply(session, tenant_id, customer_ref, message, redis)
    sources = [
        {"n": i, "chunk_id": str(h.chunk_id), "document_title": h.document_title, "content": h.content}
        for i, h in enumerate(reply.hits, start=1)
    ]
    await memory.append_exchange(message, reply.answer)
    logger.warning("All LLM providers unavailable; served baseline answer (%s)", reply.intent)
    return AgentEvent(
        "done",
        {
            "answer": reply.answer,
            "language": language,
            "model": "baseline",
            "stop_reason": "degraded",
            "tool_calls": [],
            "sources": sources,
            "usage": {},
            "took_ms": 0.0,
            "degraded": True,
        },
    )
