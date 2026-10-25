"""Baseline /chat: keyword intent routing + extractive answers from hybrid search.

This is deliberately the POC behaviour. Phase 2 replaces the routing with LLM tool calling and
the extractive answer with a generated, cited one; keeping the baseline lets the eval suite
measure how much that improves things.
"""

import re
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, status

from app.api.deps import RedisDep, SessionDep, TenantDep
from app.schemas import ChatIn, ChatOut, SearchHitOut
from app.services.loyalty import CustomerNotFoundError, get_balance
from app.services.search import hybrid_search
from app.services.text import tokenize

router = APIRouter(tags=["chat"])

_BALANCE_TERMS = {"balance", "kitne", "kitna", "कितने", "बैलेंस"}
_POINTS_TERMS = {"points", "rewards"}
_DEVANAGARI_RE = re.compile(f"[{chr(0x0900)}-{chr(0x097F)}]")
_HINGLISH_MARKERS = {"kya", "hai", "mere", "mera", "kitne", "kaise", "kab", "nahi", "batao", "kar", "ka", "ki", "ke"}


def detect_language(message: str) -> str:
    if _DEVANAGARI_RE.search(message):
        return "hindi"
    words = set(re.findall(r"[a-z]+", message.lower()))
    return "hinglish" if words & _HINGLISH_MARKERS else "english"


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


@router.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> ChatOut:
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
        return ChatOut(intent="points_balance", answer=answer)

    result = await hybrid_search(session, tenant_id, body.message, top_k=3, redis=redis)
    if not result.hits:
        return ChatOut(intent="faq", answer=_NO_ANSWER[lang])
    return ChatOut(
        intent="faq",
        answer=result.hits[0].content,
        sources=[SearchHitOut(**asdict(h)) for h in result.hits],
    )
