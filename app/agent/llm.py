"""LLM providers behind one streaming interface.

The agent loop speaks one message format (Anthropic-style: text / tool_use / tool_result blocks)
and talks to the LLMClient protocol, never to an SDK directly. Each provider adapts to that:

    claude  - Anthropic Messages API (native format)
    gemini  - Google Gemini via google-genai (converted both ways)

That also lets tests use a scripted fake, and Phase 3 put a fallback/circuit breaker in front.
"""

import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import anthropic
import httpx

from app.config import get_settings


@dataclass
class TextDelta:
    text: str


@dataclass
class FinalMessage:
    message: Any  # .content (blocks), .stop_reason, .usage, shaped like anthropic.types.Message


class LLMError(Exception):
    """Provider failure the agent reports to the client instead of crashing the request.
    public=True means the message is safe and useful to show the customer as-is."""

    def __init__(self, message: str, public: bool = False) -> None:
        super().__init__(message)
        self.public = public


_RATE_LIMITED = LLMError("The assistant is busy right now (rate limit reached). Please try again in a minute.", public=True)


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def stream(self, *, system: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | FinalMessage]:
        """Yield TextDelta for each text fragment, then exactly one FinalMessage.
        Raise ValueError for an unparseable tool call (the loop retries), LLMError otherwise."""
        ...


# --- Claude ------------------------------------------------------------------------------------


class ClaudeLLM:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str, effort: str, max_tokens: int) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    async def stream(self, *, system: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | FinalMessage]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                # The system prompt + tools are identical on every request; cache them explicitly, and
                # let top-level cache_control also cache the growing conversation within a tool loop.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                cache_control={"type": "ephemeral"},
                messages=messages,
                tools=tools,
                output_config={"effort": self.effort},
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                yield FinalMessage(await stream.get_final_message())
        except anthropic.AuthenticationError as exc:
            raise LLMError("LLM is not configured: the Anthropic API key was rejected.", public=True) from exc
        except anthropic.RateLimitError as exc:
            raise _RATE_LIMITED from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {type(exc).__name__}") from exc


# --- Gemini ------------------------------------------------------------------------------------

# Finish reasons that mean the provider declined; mapped to the agent's "refusal" handling.
_GEMINI_REFUSALS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class _GeminiContent(list):
    """Assistant content blocks that also carry the raw Gemini parts. Gemini 3 attaches thought
    signatures to function-call parts and rejects a follow-up request that doesn't send them back
    unchanged, so the original parts are replayed rather than rebuilt from the blocks."""

    def __init__(self, blocks: list, parts: list) -> None:
        super().__init__(blocks)
        self.parts = parts


class GeminiLLM:
    def __init__(self, client: Any, model: str, max_tokens: int) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _tools(tools: list[dict]):
        from google.genai import types

        declarations = []
        for t in tools:
            schema = t["input_schema"]
            declarations.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    # Gemini rejects an object schema with no properties; no-arg tools omit it.
                    parameters_json_schema=schema if schema.get("properties") else None,
                )
            )
        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _contents(messages: list[dict]) -> tuple[list, list[str]]:
        """Convert agent messages to Gemini contents. System messages have no place in Gemini's
        contents, so they are returned separately and appended to the system instruction."""
        from google.genai import types

        contents, notes, tool_names = [], [], {}
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                notes.append(content)
            elif isinstance(content, str):
                contents.append(types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=content)]))
            elif isinstance(content, _GeminiContent):
                for block in content:
                    if block.type == "tool_use":
                        tool_names[block.id] = block.name
                contents.append(types.Content(role="model", parts=content.parts))
            else:  # tool results
                parts = []
                for r in content:
                    payload = {"error": r["content"]} if r.get("is_error") else {"result": r["content"]}
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=r["tool_use_id"], name=tool_names.get(r["tool_use_id"], "unknown"), response=payload
                            )
                        )
                    )
                contents.append(types.Content(role="user", parts=parts))
        return contents, notes

    async def stream(self, *, system: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | FinalMessage]:
        from google.genai import errors, types

        contents, notes = self._contents(messages)
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join([system, *(f"Operator note: {n}" for n in notes)]),
            tools=self._tools(tools),
            max_output_tokens=self.max_tokens,
            # The agent loop runs the tools itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        parts, last = [], None
        try:
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                last = chunk
                candidate = chunk.candidates[0] if chunk.candidates else None
                for part in candidate.content.parts if candidate and candidate.content and candidate.content.parts else []:
                    parts.append(part)
                    if part.text and not part.thought:
                        yield TextDelta(part.text)
        except errors.ClientError as exc:
            if exc.code in (401, 403):
                raise LLMError("LLM is not configured: the Gemini API key was rejected.", public=True) from exc
            if exc.code == 429:  # free tier: 5 requests/min on gemini-2.5-flash
                raise _RATE_LIMITED from exc
            raise LLMError(f"Gemini API error {exc.code}") from exc
        except errors.ServerError as exc:
            raise LLMError(f"Gemini is unavailable ({exc.code})") from exc
        except httpx.HTTPError as exc:  # timeouts and connection errors from google-genai's transport
            raise LLMError(f"Gemini request failed: {type(exc).__name__}") from exc

        finish = last.candidates[0].finish_reason.name if last and last.candidates and last.candidates[0].finish_reason else "STOP"
        if finish == "MALFORMED_FUNCTION_CALL":
            raise ValueError("Gemini produced a malformed function call")

        blocks, text = [], "".join(p.text for p in parts if p.text and not p.thought)
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
        for p in parts:
            if p.function_call:
                if not p.function_call.id:
                    # Older models omit call ids; the id only has to be unique within this turn.
                    p.function_call.id = f"call_{uuid.uuid4().hex[:12]}"
                blocks.append(
                    SimpleNamespace(type="tool_use", id=p.function_call.id, name=p.function_call.name, input=dict(p.function_call.args or {}))
                )

        if any(b.type == "tool_use" for b in blocks):
            stop_reason = "max_tokens" if finish == "MAX_TOKENS" else "tool_use"
        elif finish == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif finish in _GEMINI_REFUSALS:
            stop_reason = "refusal"
        else:
            stop_reason = "end_turn"

        u = last.usage_metadata if last else None
        usage = SimpleNamespace(
            input_tokens=(u.prompt_token_count or 0) if u else 0,
            output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
            cache_read_input_tokens=(u.cached_content_token_count or 0) if u else 0,
            cache_creation_input_tokens=0,
        )
        yield FinalMessage(SimpleNamespace(content=_GeminiContent(blocks, parts), stop_reason=stop_reason, usage=usage))


# --- factory -----------------------------------------------------------------------------------

_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """Provider from LLM_PROVIDER. Credentials are checked here, so a missing key becomes a clean
    503 before any response (or stream) has started."""
    global _llm
    s = get_settings()
    if s.llm_provider == "gemini":
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise LLMNotConfiguredError("LLM is not configured: set GEMINI_API_KEY.")
    elif not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise LLMNotConfiguredError("LLM is not configured: set ANTHROPIC_API_KEY.")

    if _llm is None:
        if s.llm_provider == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(
                http_options=types.HttpOptions(
                    timeout=int(s.llm_timeout_seconds * 1000),
                    # Free-tier Gemini returns 503 "high demand" fairly often; retry with backoff.
                    retry_options=types.HttpRetryOptions(attempts=s.llm_max_retries + 1),
                )
            )
            _llm = GeminiLLM(client, s.gemini_model, s.llm_max_tokens)
        else:
            client = anthropic.AsyncAnthropic(timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries)
            _llm = ClaudeLLM(client, s.llm_model, s.llm_effort, s.llm_max_tokens)
    return _llm
