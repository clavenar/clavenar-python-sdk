"""`WardenPending.resolve()` end-to-end: catch the pending, drive the
polling loop, observe the operator's decision flip.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import (
    FAKE_ENDPOINT,
    FakeAnthropicClient,
    FakeAnthropicMessages,
    make_anthropic_message_with_tool_use,
)

from warden_ai.errors import WardenDenied, WardenPending, WardenTransportError
from warden_ai.options import WardenOptions
from warden_ai.wrap import warden_wrap


def _anthropic(response: dict) -> FakeAnthropicClient:
    return FakeAnthropicClient(messages=FakeAnthropicMessages(response=response))


def _pending_body(corr: str = "corr-001") -> dict:
    return {
        "status": "pending",
        "correlation_id": corr,
        "review_reasons": ["needs human"],
    }


def _view(decision: str | None, corr: str = "corr-001", note: str | None = None) -> dict:
    return {
        "correlation_id": corr,
        "agent_id": "agent-a",
        "tool_type": "function",
        "method": "tools/call",
        "review_reasons": ["needs human"],
        "requested_at": "2026-05-12T00:00:00Z",
        "decided_at": None if decision is None else "2026-05-12T00:00:01Z",
        "decision": decision,
        "decider_note": note,
    }


@respx.mock
async def test_resolve_returns_when_operator_approves() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(202, json=_pending_body())
    )
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-001").mock(
        side_effect=[
            httpx.Response(200, json=_view(None)),  # still pending
            httpx.Response(200, json=_view("allow")),  # approved
        ]
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_anthropic(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(WardenPending) as exc:
        await client.messages.create(model="claude-x")
    await exc.value.resolve(poll_interval_s=0.001, timeout_s=2.0)


@respx.mock
async def test_resolve_raises_warden_denied_when_operator_denies() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(202, json=_pending_body())
    )
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-001").mock(
        return_value=httpx.Response(200, json=_view("deny", note="too risky"))
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_anthropic(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(WardenPending) as exc:
        await client.messages.create(model="claude-x")
    with pytest.raises(WardenDenied) as denied:
        await exc.value.resolve(poll_interval_s=0.001, timeout_s=2.0)
    assert denied.value.reasons == ["too risky"]
    assert denied.value.intent_category == "PendingDenied"


@respx.mock
async def test_resolve_swallows_transient_5xx() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(202, json=_pending_body())
    )
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-001").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(503),
            httpx.Response(200, json=_view("allow")),
        ]
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_anthropic(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(WardenPending) as exc:
        await client.messages.create(model="claude-x")
    await exc.value.resolve(poll_interval_s=0.001, timeout_s=2.0)


@respx.mock
async def test_resolve_propagates_terminal_404() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(202, json=_pending_body())
    )
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-001").mock(
        return_value=httpx.Response(404)
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_anthropic(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(WardenPending) as exc:
        await client.messages.create(model="claude-x")
    with pytest.raises(WardenTransportError) as terr:
        await exc.value.resolve(poll_interval_s=0.001, timeout_s=2.0)
    assert terr.value.status == 404


@respx.mock
async def test_resolve_times_out() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(202, json=_pending_body())
    )
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-001").mock(
        return_value=httpx.Response(200, json=_view(None))
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_anthropic(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(WardenPending) as exc:
        await client.messages.create(model="claude-x")
    with pytest.raises(WardenTransportError) as terr:
        await exc.value.resolve(poll_interval_s=0.001, timeout_s=0.05)
    assert "not decided" in str(terr.value)
