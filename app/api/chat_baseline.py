"""/chat/baseline: the Phase 1 keyword intent router with extractive answers from hybrid search.

Kept on purpose next to the LLM agent (/chat) so the Phase 4 eval suite can measure how much the
agent improves on it. No LLM call, so it is also a zero-cost fallback.
"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from app.api.deps import RedisDep, SessionDep, TenantDep
from app.schemas import BaselineChatOut, ChatIn, SearchHitOut
from app.services.language import detect_language
from app.services.loyalty import CustomerNotFoundError, get_balance
from app.services.search import hybrid_search
from app.services.text import tokenize

router = APIRouter(tags=["chat"])

_BALANCE_TERMS = {"balance", "kitne", "kitna", "कितने", "बैलेंस"}
_POINTS_TERMS = {"points", "rewards"}

_BALANCE_REPLY = {
    "english": "{name}, you have {points} points.{expiring}",
    "hinglish": "{name}, aapke paas {points} points hain.{expiring}",
    "hindi": "{name}, आपके पास {points} पॉइंट्स हैं।{expiring}",
}
_EXPIRING_REPLY = {
    "english": " {n} points expire in the next 30 days.",
    "hinglish": " {n} points agle 30 din mein expire honge.",
    "hindi": " {n} पॉइंट्स अगले 30 दिनों में समाप्त होंगे।",
}
_NO_ANSWER = {
    "english": "Sorry, I couldn't find that in the bank's documents.",
    "hinglish": "Maaf kijiye, yeh jaankari bank ke documents mein nahi mili.",
    "hindi": "क्षमा करें, यह जानकारी बैंक के दस्तावेज़ों में नहीं मिली।",
}


@router.post("/chat/baseline", response_model=BaselineChatOut)
async def chat_baseline(body: ChatIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> BaselineChatOut:
    lang = detect_language(body.message)
    lowered = body.message.lower()
    # Balance terms like "kitne" are stopwords for search, so match them on the raw text.
    asks_balance = any(term in lowered for term in _BALANCE_TERMS)

    if asks_balance and set(tokenize(body.message)) & _POINTS_TERMS:
        try:
            balance = await get_balance(session, tenant_id, body.customer_ref)
        except CustomerNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found") from exc
        expiring = (
            _EXPIRING_REPLY[lang].format(n=balance.expiring_within_30_days) if balance.expiring_within_30_days else ""
        )
        answer = _BALANCE_REPLY[lang].format(name=balance.customer_name, points=balance.points, expiring=expiring)
        return BaselineChatOut(intent="points_balance", answer=answer)

    result = await hybrid_search(session, tenant_id, body.message, top_k=3, redis=redis)
    if not result.hits:
        return BaselineChatOut(intent="faq", answer=_NO_ANSWER[lang])
    return BaselineChatOut(
        intent="faq",
        answer=result.hits[0].content,
        sources=[SearchHitOut(**asdict(h)) for h in result.hits],
    )
