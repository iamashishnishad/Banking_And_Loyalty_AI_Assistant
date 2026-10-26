"""LLM providers behind one streaming interface.

The agent loop speaks one message format (Anthropic-style: text / tool_use / tool_result blocks)
and talks to the LLMClient protocol, never to an SDK directly. Each provider adapts to that:

    claude  - Anthropic Messages API (native format)
    gemini  - Google Gemini via google-genai (converted both ways)
    openai  - any OpenAI-compatible Chat Completions API, e.g. Azure OpenAI (converted both ways)

That also lets tests use a scripted fake. ResilientLLM chains providers (primary, then fallback),
each behind a circuit breaker; get_llm() returns that chain.
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import anthropic
import httpx
import openai

from app.config import get_settings
from app.services.resilience import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class TextDelta:
    text: str


@dataclass
class FinalMessage:
    message: Any  # .content (blocks), .stop_reason, .usage, shaped like anthropic.types.Message
    model: str | None = None  # set by ResilientLLM: the model that actually answered


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


def _claude_messages(messages: list[dict]) -> list[dict]:
    """Assistant turns produced by Gemini or OpenAI (after a mid-turn fallback) are plain
    namespaces; the Anthropic SDK needs dicts. Native Claude blocks pass through untouched."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list) and any(
            isinstance(b, SimpleNamespace) for b in m["content"]
        ):
            blocks = []
            for b in m["content"]:
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            m = {"role": "assistant", "content": blocks}
        out.append(m)
    return out


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
                messages=_claude_messages(messages),
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
            elif role == "assistant":  # Claude blocks, after a mid-turn fallback from Claude
                parts = []
                for block in content:
                    if block.type == "text":
                        parts.append(types.Part(text=block.text))
                    elif block.type == "tool_use":
                        tool_names[block.id] = block.name
                        parts.append(
                            types.Part(function_call=types.FunctionCall(id=block.id, name=block.name, args=block.input))
                        )
                contents.append(types.Content(role="model", parts=parts))
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


# --- OpenAI-compatible (Azure OpenAI, OpenAI, Groq, ...) ----------------------------------------

_OPENAI_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


class OpenAILLM:
    def __init__(self, client: openai.AsyncOpenAI, model: str, max_tokens: int) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools
        ]

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            role, content = m["role"], m["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
            elif role == "assistant":  # blocks from any provider: .type text | tool_use (thinking etc. dropped)
                text = "".join(b.text for b in content if b.type == "text")
                calls = [
                    {"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input, ensure_ascii=False)}}
                    for b in content
                    if b.type == "tool_use"
                ]
                msg: dict = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:  # tool results: one "tool" message each
                for r in content:
                    body = r["content"] if isinstance(r["content"], str) else json.dumps(r["content"], ensure_ascii=False)
                    out.append({"role": "tool", "tool_call_id": r["tool_use_id"], "content": ("ERROR: " + body) if r.get("is_error") else body})
        return out

    async def stream(self, *, system: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | FinalMessage]:
        text: list[str] = []
        calls: dict[int, dict] = {}  # streamed tool calls arrive in fragments, keyed by index
        finish, usage = None, None
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, messages),
                tools=self._tools(tools),
                max_completion_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta and delta.content:
                    text.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in (delta.tool_calls or []) if delta else []:
                    call = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    call["id"] = tc.id or call["id"]
                    if tc.function:
                        call["name"] += tc.function.name or ""
                        call["arguments"] += tc.function.arguments or ""
        except openai.AuthenticationError as exc:
            raise LLMError("LLM is not configured: the OpenAI-compatible API key was rejected.", public=True) from exc
        except openai.RateLimitError as exc:
            raise _RATE_LIMITED from exc
        except openai.BadRequestError as exc:
            if "content_filter" not in str(exc):
                raise LLMError(f"OpenAI-compatible API error {exc.status_code}") from exc
            finish = "content_filter"  # Azure's prompt filter rejects before streaming: a refusal
        except openai.APIError as exc:
            raise LLMError(f"OpenAI-compatible API error: {type(exc).__name__}") from exc

        blocks = []
        if text:
            blocks.append(SimpleNamespace(type="text", text="".join(text)))
        for _, call in sorted(calls.items()):
            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Unparseable arguments for tool {call['name']}") from exc  # the agent loop retries
            blocks.append(
                SimpleNamespace(type="tool_use", id=call["id"] or f"call_{uuid.uuid4().hex[:12]}", name=call["name"], input=args)
            )
        stop_reason = _OPENAI_STOP_REASONS.get(finish, "end_turn")
        if calls and stop_reason == "end_turn":  # some compatible servers report "stop" with tool calls
            stop_reason = "tool_use"
        details = getattr(usage, "prompt_tokens_details", None)
        yield FinalMessage(
            SimpleNamespace(
                content=blocks,
                stop_reason=stop_reason,
                usage=SimpleNamespace(
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cache_read_input_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                    cache_creation_input_tokens=0,
                ),
            )
        )


# --- fallback chain ----------------------------------------------------------------------------


class ResilientLLM:
    """Tries each provider in order, skipping any whose circuit is open.

    A provider is abandoned only if it fails before yielding anything: an LLMError, or no first
    event within `first_event_timeout`. Once text has reached the client, switching providers
    would duplicate it, so a mid-stream failure propagates. ValueError (an unparseable tool call)
    is the model's fault, not an outage: it propagates for the agent loop to retry and doesn't
    count against the circuit.
    """

    def __init__(self, routes: list[tuple[LLMClient, CircuitBreaker]], first_event_timeout: float) -> None:
        self.routes = routes
        self.first_event_timeout = first_event_timeout

    @property
    def model(self) -> str:
        return self.routes[0][0].model

    async def stream(self, *, system: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[TextDelta | FinalMessage]:
        last_error: LLMError | None = None
        for llm, breaker in self.routes:
            if not breaker.allow():
                logger.info("Skipping %s: circuit open", llm.model)
                continue
            events = aiter(llm.stream(system=system, messages=messages, tools=tools))
            try:
                async with asyncio.timeout(self.first_event_timeout):
                    first = await anext(events)
            except ValueError:
                await events.aclose()
                raise
            except (LLMError, TimeoutError, StopAsyncIteration) as exc:
                await events.aclose()
                breaker.record_failure()
                logger.warning("%s failed before its first event (%s); trying next provider", llm.model, type(exc).__name__)
                last_error = exc if isinstance(exc, LLMError) else LLMError(f"{llm.model}: {type(exc).__name__}")
                continue

            try:
                event = first
                while True:
                    if isinstance(event, FinalMessage):
                        event.model = llm.model
                    yield event
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        break
            except LLMError:
                breaker.record_failure()
                raise
            finally:
                await events.aclose()
            breaker.record_success()
            return
        raise last_error or LLMError("All LLM providers are unavailable (circuits open).")


# --- factory -----------------------------------------------------------------------------------

_llm: ResilientLLM | None = None


_KEY_VARS = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}


def _has_key(provider: str) -> bool:
    return any(os.getenv(name) for name in _KEY_VARS[provider])


def _build(provider: str, model: str) -> LLMClient:
    s = get_settings()
    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            http_options=types.HttpOptions(
                timeout=int(s.llm_timeout_seconds * 1000),
                # Free-tier Gemini returns 503 "high demand" fairly often; retry with backoff.
                retry_options=types.HttpRetryOptions(attempts=s.llm_max_retries + 1),
            )
        )
        return GeminiLLM(client, model, s.llm_max_tokens)
    if provider == "openai":
        client = openai.AsyncOpenAI(
            base_url=s.openai_base_url or None, timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries
        )
        return OpenAILLM(client, model, s.llm_max_tokens)
    client = anthropic.AsyncAnthropic(timeout=s.llm_timeout_seconds, max_retries=s.llm_max_retries)
    return ClaudeLLM(client, model, s.llm_effort, s.llm_max_tokens)


def get_llm() -> LLMClient:
    """Primary provider from LLM_PROVIDER, plus LLM_FALLBACK_PROVIDER if set. Credentials are
    checked here, so a missing key becomes a clean 503 before any response (or stream) has started."""
    global _llm
    s = get_settings()
    if not _has_key(s.llm_provider):
        raise LLMNotConfiguredError(f"LLM is not configured: set {_KEY_VARS[s.llm_provider][0]}.")

    if _llm is None:
        primary_model = {"gemini": s.gemini_model, "openai": s.openai_model}.get(s.llm_provider, s.llm_model)
        candidates = [(s.llm_provider, primary_model)]
        if s.llm_fallback_provider:
            if _has_key(s.llm_fallback_provider) and s.llm_fallback_model:
                candidates.append((s.llm_fallback_provider, s.llm_fallback_model))
            else:
                logger.warning("LLM fallback %s ignored: no API key or model set", s.llm_fallback_provider)
        routes = [
            (_build(provider, model), CircuitBreaker(f"llm:{model}", s.circuit_failure_threshold, s.circuit_reset_seconds))
            for provider, model in candidates
        ]
        _llm = ResilientLLM(routes, s.llm_first_event_timeout_seconds)
    return _llm
