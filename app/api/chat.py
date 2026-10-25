"""LLM agent chat: JSON (/chat) and Server-Sent Events (/chat/stream), plus memory endpoints."""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import select

from app.agent.llm import get_llm
from app.agent.runner import AgentEvent
from app.agent.service import chat_turn
from app.api.deps import RedisDep, SessionDep, TenantDep
from app.models import Customer
from app.schemas import ChatIn, ChatOut, MemoryOut
from app.services.memory import ConversationMemory

router = APIRouter(prefix="/chat", tags=["chat"])


async def ensure_customer(session, tenant_id: uuid.UUID, customer_ref: str) -> None:
    """Checked up front: once a stream has started, a 404 can no longer be sent."""
    exists = await session.scalar(
        select(Customer.id).where(Customer.tenant_id == tenant_id, Customer.external_ref == customer_ref)
    )
    if exists is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")


async def run_chat(redis: Redis, tenant_id: uuid.UUID, customer_ref: str, message: str) -> ChatOut:
    """Run a full turn without streaming and return the final result."""
    llm = get_llm()
    async for event in chat_turn(llm=llm, redis=redis, tenant_id=tenant_id, customer_ref=customer_ref, message=message):
        if event.type == "error":
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, event.data["message"])
        if event.type == "done":
            return ChatOut(**event.data)
    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Agent finished without an answer")


@router.post("", response_model=ChatOut)
async def chat(body: ChatIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> ChatOut:
    await ensure_customer(session, tenant_id, body.customer_ref)
    return await run_chat(redis, tenant_id, body.customer_ref, body.message)


def _sse(event: AgentEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "meta, token, tool_call, tool_result, done | error"}},
)
async def chat_stream(body: ChatIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> StreamingResponse:
    await ensure_customer(session, tenant_id, body.customer_ref)
    llm = get_llm()  # raises before the stream starts, so a missing key is a proper 503

    async def frames() -> AsyncIterator[str]:
        async for event in chat_turn(
            llm=llm, redis=redis, tenant_id=tenant_id, customer_ref=body.customer_ref, message=body.message
        ):
            yield _sse(event)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        # X-Accel-Buffering stops nginx-style proxies from buffering the whole stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/memory/{customer_ref}", response_model=MemoryOut)
async def get_memory(customer_ref: str, tenant_id: TenantDep, redis: RedisDep) -> MemoryOut:
    messages = await ConversationMemory(redis, tenant_id, customer_ref).history()
    return MemoryOut(customer_ref=customer_ref, messages=messages)


@router.delete("/memory/{customer_ref}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_memory(customer_ref: str, tenant_id: TenantDep, redis: RedisDep) -> None:
    await ConversationMemory(redis, tenant_id, customer_ref).clear()
