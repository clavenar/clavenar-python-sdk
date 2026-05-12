"""Retry semantics: 5xx and network errors retry per
`WardenRetryOptions`; 200/403/202 and 4xx other than 5xx never retry.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import FAKE_ENDPOINT

from warden_ai.errors import WardenTransportError
from warden_ai.options import WardenOptions, WardenRetryOptions
from warden_ai.transport import NormalizedToolCall, inspect_tool_use


def _call() -> NormalizedToolCall:
    return NormalizedToolCall(id="t1", name="list_files", input={})


@respx.mock
async def test_retries_502_then_succeeds() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        side_effect=[
            httpx.Response(502, text="bad gateway"),
            httpx.Response(502, text="bad gateway"),
            httpx.Response(200),
        ]
    )
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=3, base_delay_s=0.001),
    )
    verdict = await inspect_tool_use(_call(), opts)
    assert verdict.kind == "allow"
    assert route.call_count == 3


@respx.mock
async def test_retries_exhausted_raises_last_error() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(502, text="bg"))
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=2, base_delay_s=0.001),
    )
    with pytest.raises(WardenTransportError) as exc:
        await inspect_tool_use(_call(), opts)
    assert exc.value.status == 502


@respx.mock
async def test_403_does_not_retry() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["nope"],
                "review_reasons": [],
                "intent_category": "x",
            },
        )
    )
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=5, base_delay_s=0.001),
    )
    verdict = await inspect_tool_use(_call(), opts)
    assert verdict.kind == "deny"
    assert route.call_count == 1


@respx.mock
async def test_401_does_not_retry() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(401, text="auth")
    )
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=5, base_delay_s=0.001),
    )
    with pytest.raises(WardenTransportError) as exc:
        await inspect_tool_use(_call(), opts)
    assert exc.value.status == 401
    assert route.call_count == 1


@respx.mock
async def test_network_failure_retries() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        side_effect=[httpx.ConnectError("boom"), httpx.Response(200)]
    )
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=2, base_delay_s=0.001),
    )
    verdict = await inspect_tool_use(_call(), opts)
    assert verdict.kind == "allow"
    assert route.call_count == 2


@respx.mock
async def test_max_attempts_1_disables_retry() -> None:
    route = respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(502, text="bg")
    )
    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=1, base_delay_s=0.001),
    )
    with pytest.raises(WardenTransportError):
        await inspect_tool_use(_call(), opts)
    assert route.call_count == 1
