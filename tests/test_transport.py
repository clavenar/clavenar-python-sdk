"""HTTP transport behaviour against a respx-mocked warden-lite."""

from __future__ import annotations

import httpx
import pytest
import respx

from warden_ai.errors import WardenTransportError
from warden_ai.options import WardenOptions
from warden_ai.transport import NormalizedToolCall, inspect_tool_use, poll_pending_once

FAKE_ENDPOINT = "http://warden-lite.test"


@respx.mock
async def test_allow_returns_allow_with_correlation_id() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            200, json={"ok": True}, headers={"x-warden-correlation-id": "abc-123"}
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="list", input={}),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "allow"
    assert verdict.correlation_id == "abc-123"


@respx.mock
async def test_deny_403_parses_security_violation_body() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["sql_execute is denied"],
                "review_reasons": [],
                "intent_category": "code_execution",
            },
            headers={"x-warden-correlation-id": "deny-1"},
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="sql_execute", input={}),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "deny"
    assert verdict.reasons == ["sql_execute is denied"]
    assert verdict.intent_category == "code_execution"
    assert verdict.correlation_id == "deny-1"


@respx.mock
async def test_pending_202_parses_review_reasons() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            202,
            json={
                "status": "pending",
                "correlation_id": "corr-7",
                "review_reasons": ["yellow-tier sensitive write"],
            },
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="git_push", input={}),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "pending"
    assert verdict.correlation_id == "corr-7"
    assert verdict.review_reasons == ["yellow-tier sensitive write"]


@respx.mock
async def test_pending_missing_correlation_id_raises() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            202,
            json={"status": "pending", "correlation_id": "", "review_reasons": []},
        )
    )
    with pytest.raises(WardenTransportError, match="missing correlation id"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            WardenOptions(endpoint=FAKE_ENDPOINT),
        )


@respx.mock
async def test_500_raises_transport_error_with_status() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(503, text="upstream unavailable")
    )
    with pytest.raises(WardenTransportError) as exc:
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            WardenOptions(endpoint=FAKE_ENDPOINT),
        )
    assert exc.value.status == 503


@respx.mock
async def test_authorization_header_includes_token() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200)
    )
    await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="op", input={}),
        WardenOptions(endpoint=FAKE_ENDPOINT, token="secret-123"),
    )
    assert route.calls.last.request.headers["authorization"] == "Bearer secret-123"


@respx.mock
async def test_extra_headers_forwarded() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200)
    )
    await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="op", input={}),
        WardenOptions(
            endpoint=FAKE_ENDPOINT,
            extra_headers={"x-warden-demo-prefix": "abcd1234"},
        ),
    )
    assert route.calls.last.request.headers["x-warden-demo-prefix"] == "abcd1234"


@respx.mock
async def test_poll_pending_returns_decision_view() -> None:
    respx.get(f"{FAKE_ENDPOINT}/pending/corr-9").mock(
        return_value=httpx.Response(
            200,
            json={
                "correlation_id": "corr-9",
                "agent_id": "agent-a",
                "tool_type": "function",
                "method": "tools/call",
                "review_reasons": [],
                "requested_at": "2026-05-12T00:00:00Z",
                "decided_at": "2026-05-12T00:01:00Z",
                "decision": "allow",
                "decider_note": None,
            },
        )
    )
    view = await poll_pending_once(
        "corr-9", WardenOptions(endpoint=FAKE_ENDPOINT)
    )
    assert view.decision == "allow"
    assert view.agent_id == "agent-a"


@respx.mock
async def test_poll_pending_terminal_status_raises() -> None:
    respx.get(f"{FAKE_ENDPOINT}/pending/missing").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with pytest.raises(WardenTransportError) as exc:
        await poll_pending_once(
            "missing", WardenOptions(endpoint=FAKE_ENDPOINT)
        )
    assert exc.value.status == 404
