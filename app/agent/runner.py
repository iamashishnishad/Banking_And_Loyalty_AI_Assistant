"""The agent loop: stream a Claude turn, run any tool calls, feed results back, repeat.

run_agent yields AgentEvents that the API layer turns into SSE frames (or collects for the plain
JSON endpoint):

    tool_call   {"name", "input"}
    tool_result {"name", "is_error", "preview"}
    token       {"text"}
    done        {"answer", "sources", "tool_calls", "usage", "stop_reason", "model"}
    error       {"message"}
"""

import logging
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.agent.llm import FinalMessage, LLMClient, LLMError, TextDelta
from app.agent.prompts import FALLBACK_REPLY, SYSTEM_PROMPT, context_note
from app.agent.tools import TOOL_DEFINITIONS, ToolContext, run_tool
from app.config import get_settings
from app.services.language import Language

logger = logging.getLogger(__name__)

_MAX_JSON_RETRIES = 2


@dataclass
class AgentEvent:
    type: str
    data: dict = field(default_factory=dict)


def _usage_counts(message) -> Counter:
    u = message.usage
    return Counter(
        {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None) or 0,
        }
    )


async def run_agent(
    *,
    llm: LLMClient,
    ctx: ToolContext,
    user_message: str,
    history: list[dict],
    bank_name: str,
    customer_name: str | None,
    language: Language,
) -> AsyncIterator[AgentEvent]:
    settings = get_settings()
    started = time.perf_counter()
    messages: list[dict] = [
        *history,
        {"role": "user", "content": user_message},
        # Operator context as a mid-conversation system message: it doesn't disturb the cached
        # prefix, and it carries operator authority rather than looking like user text.
        {"role": "system", "content": context_note(bank_name, customer_name, language)},
    ]
    usage: Counter = Counter()
    tool_calls: list[dict] = []
    answer_parts: list[str] = []
    stop_reason = None

    try:
        for _ in range(settings.agent_max_tool_rounds):
            message = None
            for attempt in range(_MAX_JSON_RETRIES + 1):
                try:
                    round_text: list[str] = []
                    async for event in llm.stream(system=SYSTEM_PROMPT, messages=messages, tools=TOOL_DEFINITIONS):
                        if isinstance(event, TextDelta):
                            round_text.append(event.text)
                            yield AgentEvent("token", {"text": event.text})
                        elif isinstance(event, FinalMessage):
                            message = event.message
                    break
                except ValueError:
                    # The SDK could not parse a streamed tool input at all. There is no tool_use id
                    # to answer, so re-issue the turn. Provider errors are LLMError and propagate.
                    if attempt == _MAX_JSON_RETRIES:
                        raise
                    logger.warning("Unparseable tool input from model; retrying turn")

            usage.update(_usage_counts(message))  # update() keeps zero counts; += drops them
            stop_reason = message.stop_reason
            if round_text:
                answer_parts.append("".join(round_text))
            # Append the assistant turn unchanged (thinking + tool_use blocks must round-trip as-is).
            messages.append({"role": "assistant", "content": message.content})

            if stop_reason == "refusal":
                answer_parts = [FALLBACK_REPLY[language]]
                yield AgentEvent("token", {"text": "\n\n" + FALLBACK_REPLY[language]})
                break

            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                break
            if stop_reason == "max_tokens":
                # A truncated tool input can still parse as a valid partial object: never run it.
                raise RuntimeError("Model output was cut off mid tool call")

            results = []
            for block in tool_uses:
                yield AgentEvent("tool_call", {"name": block.name, "input": block.input})
                content, is_error = await run_tool(ctx, block.name, block.input)
                tool_calls.append({"name": block.name, "input": block.input, "is_error": is_error})
                yield AgentEvent("tool_result", {"name": block.name, "is_error": is_error, "preview": content[:300]})
                result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})
        else:
            logger.warning("Agent hit the tool-round limit (%d)", settings.agent_max_tool_rounds)
            stop_reason = "tool_round_limit"
    except (LLMError, RuntimeError, ValueError) as exc:
        logger.exception("Agent turn failed")
        if isinstance(exc, LLMError) and "not configured" not in str(exc):
            error_message = "The assistant is temporarily unavailable."  # don't leak provider details
        else:
            error_message = str(exc)
        yield AgentEvent("error", {"message": error_message})
        return

    answer = "\n\n".join(p.strip() for p in answer_parts if p.strip()) or FALLBACK_REPLY[language]
    yield AgentEvent(
        "done",
        {
            "answer": answer,
            "language": language,
            "model": llm.model,
            "stop_reason": stop_reason,
            "tool_calls": tool_calls,
            "sources": [
                {"n": i, "chunk_id": str(h.chunk_id), "document_title": h.document_title, "content": h.content}
                for i, h in enumerate(ctx.sources, start=1)
            ],
            "usage": dict(usage),
            "took_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
