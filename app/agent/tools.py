"""Tools the agent can call, their JSON schemas, and their implementations.

Security boundary: the tenant and customer come from the request (later, the JWT), never from
the model. No tool takes a customer id, so a prompt-injected "check CUST1002's balance" has
nothing to aim at. Inputs are validated with pydantic before anything runs.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.loyalty import (
    CustomerNotFoundError,
    DuplicateRedemptionError,
    InsufficientPointsError,
    get_balance,
    redeem_points,
)
from app.services.memory import ConversationMemory, PendingRedemption
from app.services.search import SearchHit, hybrid_search

RedemptionType = Literal["statement_credit", "travel", "gift_voucher", "airline_miles"]


@dataclass
class ToolContext:
    session: AsyncSession
    redis: Redis
    tenant_id: uuid.UUID
    customer_ref: str
    memory: ConversationMemory
    turn: int
    # Filled by search_knowledge_base; returned to the client as citations.
    sources: list[SearchHit] = field(default_factory=list)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class NoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StartRedemptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: int = Field(gt=0, le=1_000_000)
    redemption_type: RedemptionType


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


# strict: the API guarantees schema-valid input. eager_input_streaming: inputs stream as generated
# on streamed requests; the server stops validating them then, hence the pydantic check below.
TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the bank's policy documents, FAQs, fee schedules and rate tables. Use this for "
            "any question about products, rules, fees, interest rates, reward earning/expiry/"
            "redemption rules, or procedures. The search understands English, Hindi and Hinglish, "
            "but a short English keyword query usually works best. Returns numbered passages; cite "
            "them as [1], [2]."
        ),
        "input_schema": _schema({"query": {"type": "string", "description": "What to look up"}}, ["query"]),
    },
    {
        "name": "get_points_balance",
        "description": (
            "Get the current customer's reward points balance and how many points expire in the "
            "next 30 days. Takes no input: it always refers to the customer in this chat."
        ),
        "input_schema": _schema({}, []),
    },
    {
        "name": "start_redemption",
        "description": (
            "Step 1 of 2 for redeeming points. Checks the balance and records a pending redemption "
            "that does not spend anything yet. After calling it, tell the customer exactly what will "
            "happen (points and type) and ask them to confirm. Do not call confirm_redemption in "
            "the same reply."
        ),
        "input_schema": _schema(
            {
                "points": {"type": "integer", "description": "Number of points to redeem"},
                "redemption_type": {
                    "type": "string",
                    "enum": ["statement_credit", "travel", "gift_voucher", "airline_miles"],
                },
            },
            ["points", "redemption_type"],
        ),
    },
    {
        "name": "confirm_redemption",
        "description": (
            "Step 2 of 2. Executes the pending redemption. Only call this after the customer has "
            "explicitly agreed (for example 'yes', 'haan', 'confirm', 'हाँ') in a new message after "
            "you asked. Takes no input: it confirms the customer's current pending redemption."
        ),
        "input_schema": _schema({}, []),
    },
]
for _tool in TOOL_DEFINITIONS:
    _tool["strict"] = True
    _tool["eager_input_streaming"] = True


class ToolError(Exception):
    """Expected failure reported back to the model as an is_error tool_result."""


async def _search(ctx: ToolContext, args: SearchInput) -> str:
    result = await hybrid_search(ctx.session, ctx.tenant_id, args.query, top_k=5, redis=ctx.redis)
    if not result.hits:
        return "No matching passages found in the bank's documents."
    passages = []
    for hit in result.hits:
        # Number passages across the whole turn so citations stay unique over repeated searches.
        if hit.chunk_id not in {s.chunk_id for s in ctx.sources}:
            ctx.sources.append(hit)
        n = next(i for i, s in enumerate(ctx.sources, start=1) if s.chunk_id == hit.chunk_id)
        passages.append(f"[{n}] {hit.document_title}\n{hit.content}")
    return "\n\n---\n\n".join(passages)


async def _balance(ctx: ToolContext, _: NoInput) -> str:
    try:
        balance = await get_balance(ctx.session, ctx.tenant_id, ctx.customer_ref)
    except CustomerNotFoundError as exc:
        raise ToolError("Customer not found") from exc
    return json.dumps(
        {
            "customer_name": balance.customer_name,
            "points": balance.points,
            "expiring_within_30_days": balance.expiring_within_30_days,
        },
        ensure_ascii=False,
    )


async def _start_redemption(ctx: ToolContext, args: StartRedemptionInput) -> str:
    try:
        balance = await get_balance(ctx.session, ctx.tenant_id, ctx.customer_ref)
    except CustomerNotFoundError as exc:
        raise ToolError("Customer not found") from exc
    if args.points > balance.points:
        raise ToolError(f"Insufficient points: balance is {balance.points}, requested {args.points}")
    pending = PendingRedemption(
        id=f"rdm-{uuid.uuid4().hex[:12]}",
        points=args.points,
        redemption_type=args.redemption_type,
        created_turn=ctx.turn,
    )
    await ctx.memory.set_pending_redemption(pending)
    return json.dumps(
        {
            "status": "awaiting_customer_confirmation",
            "points": args.points,
            "redemption_type": args.redemption_type,
            "current_balance": balance.points,
            "balance_after": balance.points - args.points,
        }
    )


async def _confirm_redemption(ctx: ToolContext, _: NoInput) -> str:
    pending = await ctx.memory.get_pending_redemption()
    if pending is None:
        raise ToolError("There is no pending redemption (it may have expired). Start a new one.")
    if ctx.turn <= pending.created_turn:
        # Enforced here, not only in the prompt: confirmation must come in a later customer message.
        raise ToolError("The customer has not confirmed yet. Ask them to confirm, then wait for their reply.")
    try:
        new_balance = await redeem_points(
            ctx.session,
            ctx.tenant_id,
            ctx.customer_ref,
            pending.points,
            reason=f"Chat redemption: {pending.redemption_type}",
            reference=pending.id,  # idempotency key: a retried confirm can't double-spend
        )
    except InsufficientPointsError as exc:
        raise ToolError(str(exc)) from exc
    except DuplicateRedemptionError:
        await ctx.memory.clear_pending_redemption()
        raise ToolError("This redemption was already processed.")
    await ctx.memory.clear_pending_redemption()
    return json.dumps(
        {
            "status": "redeemed",
            "points": pending.points,
            "redemption_type": pending.redemption_type,
            "new_balance": new_balance,
            "reference": pending.id,
        }
    )


_HANDLERS = {
    "search_knowledge_base": (SearchInput, _search),
    "get_points_balance": (NoInput, _balance),
    "start_redemption": (StartRedemptionInput, _start_redemption),
    "confirm_redemption": (NoInput, _confirm_redemption),
}


async def run_tool(ctx: ToolContext, name: str, raw_input: object) -> tuple[str, bool]:
    """Validate and run one tool call. Returns (content, is_error); never raises for expected failures."""
    if name not in _HANDLERS:
        return f"Unknown tool '{name}'", True
    model, handler = _HANDLERS[name]
    try:
        args = model.model_validate(raw_input if isinstance(raw_input, dict) else {})
    except ValidationError as exc:
        return json.dumps({"INVALID_INPUT": json.dumps(raw_input, default=str), "errors": exc.errors()}, default=str), True
    try:
        return await handler(ctx, args), False
    except ToolError as exc:
        return str(exc), True
