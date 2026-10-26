"""The Phase 1 keyword intent router with extractive answers from hybrid search. No LLM call.

Served at /chat/baseline (the eval baseline), and used by /chat as the last fallback when every
LLM provider is down: a degraded but grounded answer beats an error.
"""

import uuid
from dataclasses import dataclass, field

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.language import detect_language
from app.services.loyalty import get_balance
from app.services.search import SearchHit, hybrid_search
from app.services.text import tokenize

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


@dataclass
class BaselineReply:
    intent: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)


async def baseline_reply(
    session: AsyncSession, tenant_id: uuid.UUID, customer_ref: str, message: str, redis: Redis | None
) -> BaselineReply:
    """Raises CustomerNotFoundError for a balance question about an unknown customer."""
    lang = detect_language(message)
    lowered = message.lower()
    # Balance terms like "kitne" are stopwords for search, so match them on the raw text.
    asks_balance = any(term in lowered for term in _BALANCE_TERMS)

    if asks_balance and set(tokenize(message)) & _POINTS_TERMS:
        balance = await get_balance(session, tenant_id, customer_ref)
        expiring = (
            _EXPIRING_REPLY[lang].format(n=balance.expiring_within_30_days) if balance.expiring_within_30_days else ""
        )
        answer = _BALANCE_REPLY[lang].format(name=balance.customer_name, points=balance.points, expiring=expiring)
        return BaselineReply(intent="points_balance", answer=answer)

    result = await hybrid_search(session, tenant_id, message, top_k=3, redis=redis)
    if not result.hits:
        return BaselineReply(intent="faq", answer=_NO_ANSWER[lang])
    return BaselineReply(intent="faq", answer=result.hits[0].content, hits=result.hits)
