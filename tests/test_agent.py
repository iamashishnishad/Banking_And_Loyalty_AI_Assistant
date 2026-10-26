"""Agent loop tests with a scripted fake LLM: no network, no API key."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from app.agent import runner, tools
from app.agent.llm import FinalMessage, LLMError, TextDelta
from app.agent.tools import ToolContext
from app.services.memory import PendingRedemption


def _msg(*blocks, stop_reason="end_turn", input_tokens=100, output_tokens=20):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool_use(name, input_, id_="toolu_1"):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


class FakeLLM:
    model = "fake-model"

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    async def stream(self, *, system, messages, tools):
        self.requests.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for block in turn.content:
            if block.type == "text":
                yield TextDelta(block.text)
        yield FinalMessage(turn)


def _ctx():
    return ToolContext(session=None, redis=None, tenant_id=uuid.uuid4(), customer_ref="C1", memory=None, turn=1)


def _run(llm, monkeypatch, tool_results=None, language="english"):
    calls = []

    async def fake_run_tool(ctx, name, raw_input):
        calls.append((name, raw_input))
        return (tool_results or {}).get(name, ("ok", False))

    monkeypatch.setattr(runner, "run_tool", fake_run_tool)

    async def collect():
        return [
            e
            async for e in runner.run_agent(
                llm=llm, ctx=_ctx(), user_message="hi", history=[], bank_name="Demo Bank",
                customer_name="Ravi", language=language,
            )
        ]

    return asyncio.run(collect()), calls


def test_tool_call_then_answer(monkeypatch):
    llm = FakeLLM(
        [
            _msg(_tool_use("get_points_balance", {}), stop_reason="tool_use"),
            _msg(_text("You have 5500 points."), input_tokens=150, output_tokens=10),
        ]
    )
    events, calls = _run(llm, monkeypatch, {"get_points_balance": ('{"points": 5500}', False)})

    assert calls == [("get_points_balance", {})]
    assert [e.type for e in events] == ["tool_call", "tool_result", "token", "done"]
    done = events[-1].data
    assert done["answer"] == "You have 5500 points."
    assert done["usage"]["input_tokens"] == 250 and done["usage"]["output_tokens"] == 30
    # Second request: assistant tool_use turn, then ONE user message carrying the tool_result.
    second = llm.requests[1]
    assert second[-2]["role"] == "assistant"
    assert second[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"points": 5500}'}],
    }


def test_context_note_is_a_system_message_after_the_user_turn(monkeypatch):
    llm = FakeLLM([_msg(_text("Hello"))])
    _run(llm, monkeypatch, language="hinglish")
    first = llm.requests[0]
    assert first[0] == {"role": "user", "content": "hi"}
    assert first[1]["role"] == "system" and "Hinglish" in first[1]["content"]


def test_tool_error_is_flagged(monkeypatch):
    llm = FakeLLM([_msg(_tool_use("confirm_redemption", {}), stop_reason="tool_use"), _msg(_text("Please confirm first."))])
    _run(llm, monkeypatch, {"confirm_redemption": ("not confirmed", True)})
    assert llm.requests[1][-1]["content"][0]["is_error"] is True


def test_refusal_returns_localised_fallback(monkeypatch):
    events, _ = _run(FakeLLM([_msg(stop_reason="refusal")]), monkeypatch, language="hindi")
    assert "क्षमा करें" in events[-1].data["answer"]


def test_truncated_tool_call_is_not_executed(monkeypatch):
    llm = FakeLLM([_msg(_tool_use("start_redemption", {"points": 5}), stop_reason="max_tokens")])
    events, calls = _run(llm, monkeypatch)
    assert calls == [] and events[-1].type == "error"


def test_tool_round_limit(monkeypatch):
    monkeypatch.setattr(runner.get_settings(), "agent_max_tool_rounds", 2)
    llm = FakeLLM([_msg(_tool_use("get_points_balance", {}), stop_reason="tool_use")] * 2)
    events, calls = _run(llm, monkeypatch)
    assert len(calls) == 2 and events[-1].data["stop_reason"] == "tool_round_limit"


def test_provider_error_becomes_error_event(monkeypatch):
    events, _ = _run(FakeLLM([LLMError("Gemini is unavailable (503)")]), monkeypatch)
    assert events[-1] == runner.AgentEvent(
        "error", {"message": "The assistant is temporarily unavailable.", "code": "llm_unavailable"}
    )


def test_rejected_key_message_is_passed_through(monkeypatch):
    events, _ = _run(FakeLLM([LLMError("LLM is not configured: the Gemini API key was rejected.", public=True)]), monkeypatch)
    assert "Gemini API key" in events[-1].data["message"]


def test_malformed_tool_call_is_retried(monkeypatch):
    llm = FakeLLM([ValueError("malformed"), _msg(_text("Done."))])
    events, _ = _run(llm, monkeypatch)
    assert events[-1].data["answer"] == "Done." and len(llm.requests) == 2


# --- redemption confirmation guard ------------------------------------------------------------


class FakeMemory:
    def __init__(self, pending):
        self.pending = pending

    async def get_pending_redemption(self):
        return self.pending

    async def clear_pending_redemption(self):
        self.pending = None


@pytest.mark.parametrize(("turn", "should_redeem"), [(3, False), (4, True)])
def test_confirm_requires_a_later_customer_turn(monkeypatch, turn, should_redeem):
    redeemed = []

    async def fake_redeem(session, tenant_id, customer_ref, points, reason, reference):
        redeemed.append((points, reference))
        return 1000

    monkeypatch.setattr(tools, "redeem_points", fake_redeem)
    memory = FakeMemory(PendingRedemption(id="rdm-1", points=500, redemption_type="travel", created_turn=3))
    ctx = ToolContext(session=None, redis=None, tenant_id=uuid.uuid4(), customer_ref="C1", memory=memory, turn=turn)

    content, is_error = asyncio.run(tools.run_tool(ctx, "confirm_redemption", {}))

    assert is_error is (not should_redeem)
    assert redeemed == ([(500, "rdm-1")] if should_redeem else [])


def test_invalid_tool_input_rejected():
    content, is_error = asyncio.run(tools.run_tool(_ctx(), "start_redemption", {"points": "lots"}))
    assert is_error and "INVALID_INPUT" in content


class FakePendingMemory(FakeMemory):
    async def set_pending_redemption(self, pending):
        self.pending = pending


def _start(monkeypatch, pending, turn, points=500, redemption_type="travel"):
    async def fake_balance(session, tenant_id, customer_ref):
        return SimpleNamespace(points=5000)

    monkeypatch.setattr(tools, "get_balance", fake_balance)
    memory = FakePendingMemory(pending)
    ctx = ToolContext(session=None, redis=None, tenant_id=uuid.uuid4(), customer_ref="C1", memory=memory, turn=turn)
    content, is_error = asyncio.run(
        tools.run_tool(ctx, "start_redemption", {"points": points, "redemption_type": redemption_type})
    )
    return content, memory.pending


def test_restarting_the_same_redemption_on_yes_keeps_the_original(monkeypatch):
    original = PendingRedemption(id="rdm-1", points=500, redemption_type="travel", created_turn=3)
    content, pending = _start(monkeypatch, original, turn=4)
    assert "already_pending" in content and pending is original  # confirm in turn 4 still allowed


def test_a_different_redemption_replaces_the_pending_one(monkeypatch):
    original = PendingRedemption(id="rdm-1", points=500, redemption_type="travel", created_turn=3)
    content, pending = _start(monkeypatch, original, turn=4, points=800)
    assert "awaiting_customer_confirmation" in content and pending.points == 800 and pending.created_turn == 4
