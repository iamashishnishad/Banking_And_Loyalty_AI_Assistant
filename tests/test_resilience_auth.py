"""Phase 3 unit tests: JWT auth, circuit breaker, LLM fallback chain, semantic-cache gating,
cross-provider message conversion. No network, no Docker."""

import asyncio
import time
from types import SimpleNamespace

import jwt
import pytest

from app.agent import llm as llm_module
from app.agent.llm import FinalMessage, GeminiLLM, LLMError, ResilientLLM, TextDelta, _claude_messages, _GeminiContent
from app.config import get_settings
from app.services import auth
from app.services.resilience import CircuitBreaker, retry_async
from app.services.semantic_cache import _search_rows, is_cacheable_answer, is_cacheable_question, numbers_in

SECRET = "x" * 48


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    monkeypatch.setattr(get_settings(), "jwt_secret", SECRET)


# --- auth ------------------------------------------------------------------------------------


def test_token_round_trip():
    token = auth.issue_token(tenant="demo-bank", role="customer", customer_ref="CUST1001")
    p = auth.decode_token(token)
    assert p == auth.Principal(subject="customer:CUST1001", tenant="demo-bank", role="customer", customer_ref="CUST1001")
    assert p.can_access_customer("CUST1001") and not p.can_access_customer("CUST1002")


def test_staff_can_access_any_customer_in_tenant():
    p = auth.decode_token(auth.issue_token(tenant="demo-bank", role="staff"))
    assert p.can_access_customer("CUST1002")


def test_expired_token_rejected():
    token = auth.issue_token(tenant="demo-bank", role="admin", ttl_seconds=-120)  # beyond the 30s leeway
    with pytest.raises(auth.AuthError, match="expired"):
        auth.decode_token(token)


def test_tampered_token_rejected():
    token = auth.issue_token(tenant="demo-bank", role="customer", customer_ref="CUST1001")
    claims = jwt.decode(token, options={"verify_signature": False})
    forged = jwt.encode({**claims, "tenant": "second-bank"}, "not-the-secret-but-long-enough-xxxxxxxxxxxxxx", algorithm="HS256")
    with pytest.raises(auth.AuthError):
        auth.decode_token(forged)


def test_alg_none_rejected():
    claims = jwt.decode(auth.issue_token(tenant="demo-bank", role="admin"), options={"verify_signature": False})
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    with pytest.raises(auth.AuthError):
        auth.decode_token(unsigned)


def test_wrong_audience_rejected():
    now = int(time.time())
    token = jwt.encode(
        {"sub": "x", "tenant": "demo-bank", "role": "admin", "iss": get_settings().jwt_issuer, "aud": "other-api", "iat": now, "exp": now + 60},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(auth.AuthError):
        auth.decode_token(token)


def test_customer_token_needs_customer_ref():
    with pytest.raises(ValueError):
        auth.issue_token(tenant="demo-bank", role="customer")


def test_weak_secret_fails_startup(monkeypatch):
    monkeypatch.setattr(get_settings(), "jwt_secret", "short")
    with pytest.raises(RuntimeError):
        auth.check_secret()


# --- circuit breaker -------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_circuit_opens_then_half_opens_then_closes():
    clock = Clock()
    cb = CircuitBreaker("t", failure_threshold=2, reset_seconds=10, clock=clock)
    cb.record_failure()
    assert cb.state == "closed" and cb.allow()
    cb.record_failure()
    assert cb.state == "open" and not cb.allow()

    clock.now = 10
    assert cb.state == "half_open"
    assert cb.allow()  # the single trial
    assert not cb.allow()  # no second trial while the first is in flight
    cb.record_success()
    assert cb.state == "closed" and cb.allow()


def test_failed_trial_reopens_circuit():
    clock = Clock()
    cb = CircuitBreaker("t", failure_threshold=1, reset_seconds=5, clock=clock)
    cb.record_failure()
    clock.now = 5
    assert cb.allow()
    cb.record_failure()
    assert cb.state == "open" and not cb.allow()


def test_success_resets_failure_count():
    cb = CircuitBreaker("t", failure_threshold=2, reset_seconds=5)
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.state == "closed"


def test_retry_async_retries_then_succeeds():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "ok"

    assert asyncio.run(retry_async(flaky, attempts=3, base_delay=0)) == "ok"
    assert len(calls) == 3


# --- LLM fallback chain ----------------------------------------------------------------------


def _final(text="hi"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn", usage=None)


class ScriptedLLM:
    def __init__(self, model, behaviour):
        self.model = model
        self.behaviour = behaviour  # "ok" | "fail_first" | "fail_mid" | "hang" | "bad_tool_json"
        self.calls = 0

    async def stream(self, *, system, messages, tools):
        self.calls += 1
        if self.behaviour == "fail_first":
            raise LLMError("503")
        if self.behaviour == "hang":
            await asyncio.sleep(10)
        if self.behaviour == "bad_tool_json":
            raise ValueError("malformed")
        yield TextDelta(f"from {self.model}")
        if self.behaviour == "fail_mid":
            raise LLMError("connection reset")
        yield FinalMessage(_final())


def _chain(*llms, threshold=3, timeout=5.0):
    return ResilientLLM([(m, CircuitBreaker(m.model, threshold, 30)) for m in llms], first_event_timeout=timeout)


def _collect(chain):
    async def go():
        return [e async for e in chain.stream(system="s", messages=[], tools=[])]

    return asyncio.run(go())


def test_fallback_used_when_primary_fails_before_first_token():
    primary, fallback = ScriptedLLM("primary", "fail_first"), ScriptedLLM("fallback", "ok")
    events = _collect(_chain(primary, fallback))
    assert events[0] == TextDelta("from fallback")
    assert events[-1].model == "fallback"


def test_open_circuit_skips_primary_without_calling_it():
    primary, fallback = ScriptedLLM("primary", "fail_first"), ScriptedLLM("fallback", "ok")
    chain = _chain(primary, fallback, threshold=2)
    _collect(chain)
    _collect(chain)
    assert primary.calls == 2
    _collect(chain)
    assert primary.calls == 2 and fallback.calls == 3


def test_mid_stream_failure_is_not_retried_on_fallback():
    primary, fallback = ScriptedLLM("primary", "fail_mid"), ScriptedLLM("fallback", "ok")
    with pytest.raises(LLMError):
        _collect(_chain(primary, fallback))
    assert fallback.calls == 0  # text already reached the client; switching would duplicate it


def test_first_event_timeout_falls_back():
    primary, fallback = ScriptedLLM("primary", "hang"), ScriptedLLM("fallback", "ok")
    events = _collect(_chain(primary, fallback, timeout=0.05))
    assert events[-1].model == "fallback"


def test_malformed_tool_call_propagates_and_does_not_trip_circuit():
    primary, fallback = ScriptedLLM("primary", "bad_tool_json"), ScriptedLLM("fallback", "ok")
    chain = _chain(primary, fallback, threshold=1)
    with pytest.raises(ValueError):
        _collect(chain)
    assert fallback.calls == 0 and chain.routes[0][1].state == "closed"


def test_all_providers_down_raises_llm_error():
    with pytest.raises(LLMError):
        _collect(_chain(ScriptedLLM("a", "fail_first"), ScriptedLLM("b", "fail_first")))


def test_get_llm_without_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "llm_provider", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(llm_module.LLMNotConfiguredError):
        llm_module.get_llm()


# --- cross-provider history ------------------------------------------------------------------


def test_gemini_turns_convert_to_claude_dicts():
    content = _GeminiContent(
        [SimpleNamespace(type="text", text="Checking."), SimpleNamespace(type="tool_use", id="c1", name="get_points_balance", input={})],
        parts=[],
    )
    out = _claude_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": content}])
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "c1", "name": "get_points_balance", "input": {}},
    ]


def test_claude_turns_convert_to_gemini_parts():
    claude_turn = [SimpleNamespace(type="tool_use", id="toolu_1", name="search_knowledge_base", input={"query": "fd rates"})]
    contents, notes = GeminiLLM._contents(
        [
            {"role": "user", "content": "fd rates?"},
            {"role": "assistant", "content": claude_turn},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "[1] 7.1%"}]},
        ]
    )
    call = contents[1].parts[0].function_call
    assert contents[1].role == "model" and call.name == "search_knowledge_base" and call.args == {"query": "fd rates"}
    response = contents[2].parts[0].function_response
    assert response.name == "search_knowledge_base" and response.id == "toolu_1"


# --- semantic cache gating -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "cacheable"),
    [
        ("How long are reward points valid?", True),
        ("senior citizen FD rate 2 years", True),
        ("points kab expire hote hain", True),
        ("When do my points expire?", False),
        ("mere kitne points hai?", False),
        ("मेरे पॉइंट्स कब खत्म होंगे", False),
        ("Redeem 1000 points", False),
        ("haan confirm", False),
    ],
)
def test_personal_questions_bypass_cache(message, cacheable):
    assert is_cacheable_question(message) is cacheable


def test_numbers_are_canonical_across_formats_and_scripts():
    assert numbers_in("FD rate for 2 years") == numbers_in("FD rate 2.0 years") == numbers_in("FD दर २ साल") == ["2"]
    assert numbers_in("fee on 1,000 points") == ["1000"]
    assert numbers_in("FD rate for 1 year") != numbers_in("FD rate for 2 years")
    assert numbers_in("How long are points valid?") == []


def _done(**overrides):
    done = {
        "answer": "Points are valid for 36 months [1].",
        "stop_reason": "end_turn",
        "tool_calls": [{"name": "search_knowledge_base", "input": {"query": "x"}, "is_error": False}],
        "sources": [{"n": 1}],
    }
    return {**done, **overrides}


def test_grounded_policy_answer_is_cacheable():
    assert is_cacheable_answer(_done(), "Ravi Kumar")


@pytest.mark.parametrize(
    "overrides",
    [
        {"tool_calls": [{"name": "get_points_balance", "input": {}, "is_error": False}]},
        {"tool_calls": []},  # ungrounded
        {"sources": []},
        {"stop_reason": "max_tokens"},
        {"tool_calls": [{"name": "search_knowledge_base", "input": {}, "is_error": True}]},
        {"answer": "Ravi ji, points are valid for 36 months [1]."},  # personalised
    ],
)
def test_personal_or_unfinished_answers_are_not_cached(overrides):
    assert not is_cacheable_answer(_done(**overrides), "Ravi Kumar")


def test_search_reply_parsed_in_both_client_formats():
    as_map = {b"total_results": 1, b"results": [{b"id": b"semcache:a", b"extra_attributes": {b"dist": b"0.01", b"payload": b"{}"}}]}
    as_array = [1, b"semcache:a", [b"dist", b"0.01", b"payload", b"{}"]]
    assert _search_rows(as_map) == _search_rows(as_array) == [{"dist": b"0.01", "payload": b"{}"}]
    assert _search_rows({b"total_results": 0, b"results": []}) == _search_rows([0]) == []
