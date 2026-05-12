"""Sync clients: `anthropic.Anthropic` / `openai.OpenAI` (non-async).

The wrap layer routes sync vs. async via
`inspect.iscoroutinefunction(create)`. Sync paths use `httpx.Client`
and `time.sleep`; behaviour mirrors the async path.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import (
    FAKE_ENDPOINT,
    FakeSyncAnthropicClient,
    FakeSyncAnthropicMessages,
    FakeSyncOpenAIChat,
    FakeSyncOpenAIClient,
    FakeSyncOpenAICompletions,
    make_anthropic_message_with_tool_use,
    make_openai_completion_with_tool_call,
)

from warden_ai.errors import WardenDenied, WardenTransportError
from warden_ai.options import WardenOptions
from warden_ai.wrap import warden_wrap


def _sync_anthropic(response: dict) -> FakeSyncAnthropicClient:
    return FakeSyncAnthropicClient(messages=FakeSyncAnthropicMessages(response=response))


def _sync_openai(response: dict) -> FakeSyncOpenAIClient:
    return FakeSyncOpenAIClient(
        chat=FakeSyncOpenAIChat(completions=FakeSyncOpenAICompletions(response=response))
    )


@respx.mock
def test_sync_anthropic_allow_returns_response() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(_sync_anthropic(make_anthropic_message_with_tool_use()), opts)
    result = client.messages.create(model="claude-x")
    assert result["stop_reason"] == "tool_use"


@respx.mock
def test_sync_anthropic_deny_raises_warden_denied() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["nope"],
                "review_reasons": [],
                "intent_category": "fs_write",
            },
        )
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(
        _sync_anthropic(make_anthropic_message_with_tool_use(tool_name="rm_rf")),
        opts,
    )
    with pytest.raises(WardenDenied) as exc:
        client.messages.create(model="claude-x")
    assert exc.value.tool_name == "rm_rf"
    assert exc.value.intent_category == "fs_write"


@respx.mock
def test_sync_openai_deny_raises() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["nope"],
                "review_reasons": [],
                "intent_category": "code_execution",
            },
        )
    )
    opts = WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)
    client = warden_wrap(
        _sync_openai(make_openai_completion_with_tool_call(name="exec_sql")), opts
    )
    with pytest.raises(WardenDenied):
        client.chat.completions.create(model="gpt-5")


@respx.mock
def test_sync_observe_transport_error_routes_to_callback() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(502))
    errors: list[str] = []

    def on_policy_error(err, ctx) -> None:  # type: ignore[no-untyped-def]
        errors.append(f"{ctx.tool_name}:{err.status}")

    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        mode="observe",
        timeout_s=2.0,
        on_policy_error=on_policy_error,
    )
    client = warden_wrap(
        _sync_anthropic(make_anthropic_message_with_tool_use()), opts
    )
    # Should NOT raise.
    result = client.messages.create(model="claude-x")
    assert result["stop_reason"] == "tool_use"
    assert errors == ["list_files:502"]


@respx.mock
def test_sync_enforce_5xx_propagates_after_retries_exhausted() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(502))
    from warden_ai.options import WardenRetryOptions

    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        mode="enforce",
        timeout_s=2.0,
        retry=WardenRetryOptions(max_attempts=1, base_delay_s=0.001),
    )
    client = warden_wrap(
        _sync_anthropic(make_anthropic_message_with_tool_use()), opts
    )
    with pytest.raises(WardenTransportError):
        client.messages.create(model="claude-x")
