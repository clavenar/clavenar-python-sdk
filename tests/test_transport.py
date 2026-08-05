"""HTTP transport behaviour against a respx-mocked clavenar-lite."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from clavenar_agent_sdk.errors import ClavenarConfigError, ClavenarTransportError
from clavenar_agent_sdk.options import ClavenarOptions
from clavenar_agent_sdk.transport import (
    NormalizedToolCall,
    inspect_tool_use,
    inspect_tool_use_sync,
    inspect_tool_uses,
    poll_pending_once,
)

FAKE_ENDPOINT = "http://clavenar-lite.test"


@respx.mock
async def test_decision_selector_and_atomic_batch() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    await inspect_tool_uses(
        [
            NormalizedToolCall(id="call-a", name="first", input={"n": 1}),
            NormalizedToolCall(id="call-b", name="second", input={"n": 2}),
        ],
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    request = route.calls[0].request
    payload = json.loads(request.read().decode())
    assert request.headers["x-clavenar-decision-contract"] == "clavenar.decision/v1"
    assert request.headers["x-clavenar-idempotency-id"] == payload["id"]
    assert payload["method"] == "clavenar/tools.batch"
    assert [call["id"] for call in payload["params"]["arguments"]["calls"]] == [
        "call-a",
        "call-b",
    ]


@respx.mock
async def test_allow_returns_allow_with_correlation_id() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={"verdict": "allow"},
            headers={"x-clavenar-correlation-id": "abc-123"},
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="list", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "allow"
    assert verdict.correlation_id == "abc-123"


@respx.mock
async def test_exact_decision_allow_envelope() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "contract": "clavenar.decision/v1",
                "decision": "allow",
                "correlation_id": "decision-correlation",
                "executable": False,
            },
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="list", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "allow"
    assert verdict.correlation_id == "decision-correlation"


@respx.mock
async def test_decision_allow_correlation_mismatch_fails_closed() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "contract": "clavenar.decision/v1",
                "decision": "allow",
                "correlation_id": "body-correlation",
                "executable": False,
            },
            headers={"x-clavenar-correlation-id": "header-correlation"},
        )
    )
    with pytest.raises(ClavenarTransportError, match="correlation id header/body mismatch"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="list", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )


@respx.mock
async def test_arbitrary_200_body_fails_closed() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200, json={"verdict": "allow", "unexpected": True})
    )
    with pytest.raises(ClavenarTransportError) as error:
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="list", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )
    assert error.value.status == 200


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
            headers={"x-clavenar-correlation-id": "deny-1"},
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="sql_execute", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "deny"
    assert verdict.reasons == ["sql_execute is denied"]
    assert verdict.intent_category == "code_execution"
    assert verdict.correlation_id == "deny-1"


@respx.mock
async def test_deny_403_parses_full_proxy_envelope() -> None:
    # The full edition uses varied error codes and omits empty
    # review_reasons / absent intent_category; the transport must
    # normalise rather than reject as "unexpected body shape".
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "verdict": "denied",
                "layer": "egress",
                "error": "egress_blocked",
                "reasons": ["Egress blocked — sensitive data detected."],
                "correlation_id": "c-77",
            },
            headers={"x-clavenar-correlation-id": "c-77"},
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="fetch", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "deny"
    assert verdict.layer == "egress"
    assert verdict.review_reasons == []
    assert verdict.intent_category == ""
    assert verdict.correlation_id == "c-77"


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
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
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
    with pytest.raises(ClavenarTransportError, match="missing correlation id"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )


@respx.mock
async def test_pending_header_body_correlation_mismatch_raises() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            202,
            json={"status": "pending", "correlation_id": "body", "review_reasons": []},
            headers={"x-clavenar-correlation-id": "header"},
        )
    )
    with pytest.raises(ClavenarTransportError, match="mismatch"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )


@respx.mock
async def test_429_rate_limited_parses_retry_after() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            429,
            json={
                "verdict": "rate_limited",
                "layer": "proxy",
                "error": "rate_limited",
                "reasons": ["agent request velocity exceeded"],
                "correlation_id": "c-429",
                "retry_after_secs": 17,
            },
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="fetch_user", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "rate_limited"
    assert verdict.code == "rate_limited"
    assert verdict.retry_after_secs == 17
    assert verdict.layer == "proxy"
    assert verdict.correlation_id == "c-429"
    # A 429 is a verdict, not a transient failure — exactly one attempt.
    assert route.call_count == 1


@respx.mock
async def test_429_quota_exceeded_without_retry_after() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            429,
            json={
                "verdict": "quota_exceeded",
                "layer": "proxy",
                "error": "quota_exceeded",
                "reasons": ["tenant monthly spend cap reached"],
            },
        )
    )
    verdict = await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="fetch_user", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "rate_limited"
    assert verdict.code == "quota_exceeded"
    assert verdict.retry_after_secs is None


@respx.mock
async def test_429_malformed_body_raises_transport_error() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(429, json={"wrong": "shape"})
    )
    with pytest.raises(ClavenarTransportError) as exc:
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="fetch_user", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )
    assert exc.value.status == 429


@respx.mock
def test_429_sync_rate_limited_single_attempt() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            429,
            json={
                "verdict": "rate_limited",
                "layer": "proxy",
                "error": "rate_limited",
                "reasons": ["agent request velocity exceeded"],
                "retry_after_secs": 17,
            },
            headers={"x-clavenar-correlation-id": "c-429"},
        )
    )
    verdict = inspect_tool_use_sync(
        NormalizedToolCall(id="toolu_1", name="fetch_user", input={}),
        ClavenarOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "rate_limited"
    assert verdict.code == "rate_limited"
    assert verdict.retry_after_secs == 17
    assert verdict.correlation_id == "c-429"
    assert route.call_count == 1


@respx.mock
async def test_500_raises_transport_error_with_status() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(503, text="upstream unavailable")
    )
    with pytest.raises(ClavenarTransportError) as exc:
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT),
        )
    assert exc.value.status == 503


@respx.mock
async def test_authorization_header_includes_token() -> None:
    endpoint = "https://clavenar-lite.test"
    route = respx.post(f"{endpoint}/mcp").mock(return_value=httpx.Response(200))
    await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="op", input={}),
        ClavenarOptions(endpoint=endpoint, token="secret-123"),
    )
    assert route.calls.last.request.headers["authorization"] == "Bearer secret-123"


@respx.mock
async def test_extra_headers_forwarded() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    await inspect_tool_use(
        NormalizedToolCall(id="toolu_1", name="op", input={}),
        ClavenarOptions(
            endpoint=FAKE_ENDPOINT,
            extra_headers={"x-clavenar-demo-prefix": "abcd1234"},
        ),
    )
    assert route.calls.last.request.headers["x-clavenar-demo-prefix"] == "abcd1234"


async def test_reserved_extra_header_is_rejected_before_network() -> None:
    with pytest.raises(ClavenarConfigError, match="reserved"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            ClavenarOptions(
                endpoint=FAKE_ENDPOINT,
                extra_headers={"Authorization": "attacker-controlled"},
            ),
        )


async def test_credentials_require_https_outside_explicit_loopback() -> None:
    with pytest.raises(ClavenarConfigError, match="require HTTPS"):
        await inspect_tool_use(
            NormalizedToolCall(id="toolu_1", name="op", input={}),
            ClavenarOptions(endpoint=FAKE_ENDPOINT, token="secret"),
        )


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
    view = await poll_pending_once("corr-9", ClavenarOptions(endpoint=FAKE_ENDPOINT))
    assert view.decision == "allow"
    assert view.agent_id == "agent-a"


@respx.mock
async def test_poll_pending_terminal_status_raises() -> None:
    respx.get(f"{FAKE_ENDPOINT}/pending/missing").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with pytest.raises(ClavenarTransportError) as exc:
        await poll_pending_once("missing", ClavenarOptions(endpoint=FAKE_ENDPOINT))
    assert exc.value.status == 404
