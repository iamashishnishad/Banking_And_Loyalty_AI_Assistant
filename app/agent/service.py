"""One chat turn end to end: load context and memory, run the agent, save the exchange."""

import uuid
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from sqlalchemy import select

from app.agent.llm import LLMClient
from app.agent.runner import AgentEvent, run_agent
from app.agent.tools import ToolContext
from app.db import get_sessionmaker
from app.models import Customer, Tenant
from app.services.language import detect_language
from app.services.memory import ConversationMemory


async def chat_turn(
    *,
    llm: LLMClient,
    redis: Redis,
    tenant_id: uuid.UUID,
    customer_ref: str,
    message: str,
) -> AsyncIterator[AgentEvent]:
    """Opens its own DB session. A streaming response outlives FastAPI's request-scoped
    dependencies, so the session has to belong to the generator."""
    language = detect_language(message)
    memory = ConversationMemory(redis, tenant_id, customer_ref)
    history = await memory.history()
    turn = await memory.next_turn()

    async with get_sessionmaker()() as session:
        bank_name = await session.scalar(select(Tenant.name).where(Tenant.id == tenant_id))
        customer_name = await session.scalar(
            select(Customer.name).where(Customer.tenant_id == tenant_id, Customer.external_ref == customer_ref)
        )
        ctx = ToolContext(
            session=session, redis=redis, tenant_id=tenant_id, customer_ref=customer_ref, memory=memory, turn=turn
        )
        yield AgentEvent("meta", {"language": language, "turn": turn, "model": llm.model})
        async for event in run_agent(
            llm=llm,
            ctx=ctx,
            user_message=message,
            history=history,
            bank_name=bank_name or "the bank",
            customer_name=customer_name,
            language=language,
        ):
            if event.type == "done":
                await memory.append_exchange(message, event.data["answer"])
            yield event
