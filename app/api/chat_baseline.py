"""/chat/baseline: the Phase 1 keyword intent router with extractive answers from hybrid search.

Kept on purpose next to the LLM agent (/chat) so the Phase 4 eval suite can measure how much the
agent improves on it. No LLM call, so it is also /chat's fallback when every provider is down.
"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from app.api.deps import PrincipalDep, RedisDep, SessionDep, TenantDep, authorize_customer
from app.schemas import BaselineChatOut, ChatIn, SearchHitOut
from app.services.baseline import baseline_reply
from app.services.loyalty import CustomerNotFoundError

router = APIRouter(tags=["chat"])


@router.post("/chat/baseline", response_model=BaselineChatOut)
async def chat_baseline(
    body: ChatIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep, principal: PrincipalDep
) -> BaselineChatOut:
    authorize_customer(principal, body.customer_ref)
    try:
        reply = await baseline_reply(session, tenant_id, body.customer_ref, body.message, redis)
    except CustomerNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found") from exc
    return BaselineChatOut(
        intent=reply.intent, answer=reply.answer, sources=[SearchHitOut(**asdict(h)) for h in reply.hits]
    )
