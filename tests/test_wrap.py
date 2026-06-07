"""End-to-end wrap behaviour across enforce + observe, Anthropic + OpenAI."""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import (
    FAKE_ENDPOINT,
    FakeAnthropicClient,
    FakeAnthropicMessages,
    FakeOpenAIChat,
    FakeOpenAIClient,
    FakeOpenAICompletions,
    make_anthropic_message_with_tool_use,
    make_openai_completion_with_tool_call,
)

from clavenar_agent_sdk.errors import (
    ClavenarConfigError,
    ClavenarDenied,
    ClavenarPending,
    ClavenarTransportError,
)
from clavenar_agent_sdk.options import ClavenarOptions
from clavenar_agent_sdk.wrap import clavenar_wrap


def _anthropic_client(response: dict) -> FakeAnthropicClient:
    return FakeAnthropicClient(messages=FakeAnthropicMessages(response=response))


def _openai_client(response: dict) -> FakeOpenAIClient:
    return FakeOpenAIClient(
        chat=FakeOpenAIChat(completions=FakeOpenAICompletions(response=response))
    )


# ---- detection ------------------------------------------------------------


def test_wrap_rejects_non_client() -> None:
    with pytest.raises(ClavenarConfigError, match=r"messages\.create"):
        clavenar_wrap(object(), ClavenarOptions(endpoint=FAKE_ENDPOINT))


def test_wrap_rejects_bad_endpoint() -> None:
    with pytest.raises(ClavenarConfigError, match="endpoint"):
        clavenar_wrap(
            _anthropic_client(make_anthropic_message_with_tool_use()),
            ClavenarOptions(endpoint="not-a-url"),
        )


# ---- enforce mode ---------------------------------------------------------


@respx.mock
async def test_enforce_anthropic_allow_passes_through(opts: ClavenarOptions) -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    client = clavenar_wrap(_anthropic_client(make_anthropic_message_with_tool_use()), opts)
    result = await client.messages.create(model="claude-x")
    assert result["stop_reason"] == "tool_use"


@respx.mock
async def test_enforce_anthropic_deny_raises_clavenar_denied(opts: ClavenarOptions) -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["denied by policy"],
                "review_reasons": [],
                "intent_category": "code_execution",
            },
        )
    )
    client = clavenar_wrap(
        _anthropic_client(make_anthropic_message_with_tool_use(tool_name="sql_execute")),
        opts,
    )
    with pytest.raises(ClavenarDenied) as exc:
        await client.messages.create(model="claude-x")
    assert exc.value.tool_name == "sql_execute"
    assert exc.value.intent_category == "code_execution"


@respx.mock
async def test_enforce_anthropic_pending_raises_clavenar_pending(opts: ClavenarOptions) -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            202,
            json={
                "status": "pending",
                "correlation_id": "corr-pending",
                "review_reasons": ["needs human"],
            },
        )
    )
    client = clavenar_wrap(_anthropic_client(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(ClavenarPending) as exc:
        await client.messages.create(model="claude-x")
    assert exc.value.correlation_id == "corr-pending"


@respx.mock
async def test_enforce_openai_deny_raises(opts: ClavenarOptions) -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["openai-side deny"],
                "review_reasons": [],
                "intent_category": "fs_write",
            },
        )
    )
    client = clavenar_wrap(_openai_client(make_openai_completion_with_tool_call(name="rm_rf")), opts)
    with pytest.raises(ClavenarDenied):
        await client.chat.completions.create(model="gpt-5")


@respx.mock
async def test_enforce_transport_failure_propagates(opts: ClavenarOptions) -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(side_effect=httpx.ConnectError("boom"))
    client = clavenar_wrap(_anthropic_client(make_anthropic_message_with_tool_use()), opts)
    with pytest.raises(ClavenarTransportError):
        await client.messages.create(model="claude-x")


# ---- observe mode ---------------------------------------------------------


@respx.mock
async def test_observe_deny_passes_through_with_callback(opts_observe: ClavenarOptions) -> None:
    seen: list[str] = []

    async def on_verdict(verdict, ctx) -> None:  # type: ignore[no-untyped-def]
        seen.append(f"{verdict.kind}:{ctx.tool_name}")

    opts = ClavenarOptions(
        endpoint=FAKE_ENDPOINT,
        mode="observe",
        timeout_s=2.0,
        on_verdict=on_verdict,
    )
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["would_deny"],
                "review_reasons": [],
                "intent_category": "code_execution",
            },
        )
    )
    client = clavenar_wrap(_anthropic_client(make_anthropic_message_with_tool_use()), opts)
    result = await client.messages.create(model="claude-x")
    assert result["stop_reason"] == "tool_use"  # passes through
    assert seen == ["deny:list_files"]


@respx.mock
async def test_observe_transport_error_routes_to_on_policy_error(
    opts_observe: ClavenarOptions,
) -> None:
    errors: list[str] = []

    async def on_policy_error(err, ctx) -> None:  # type: ignore[no-untyped-def]
        errors.append(f"{ctx.tool_name}:{err.status}")

    opts = ClavenarOptions(
        endpoint=FAKE_ENDPOINT,
        mode="observe",
        timeout_s=2.0,
        on_policy_error=on_policy_error,
    )
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(502, text="bad gateway"))
    client = clavenar_wrap(_anthropic_client(make_anthropic_message_with_tool_use()), opts)
    # Should NOT raise — observe mode preserves the agent call.
    result = await client.messages.create(model="claude-x")
    assert result["stop_reason"] == "tool_use"
    assert errors == ["list_files:502"]


# ---- no-tool-use bypass ---------------------------------------------------


@respx.mock
async def test_text_only_response_bypasses_inspect() -> None:
    # No tool_use blocks — no /mcp call should be issued at all.
    route = respx.post(f"{FAKE_ENDPOINT}/mcp")
    text_only = {"content": [{"type": "text", "text": "hello"}]}
    client = clavenar_wrap(_anthropic_client(text_only), ClavenarOptions(endpoint=FAKE_ENDPOINT))
    await client.messages.create(model="claude-x")
    assert route.call_count == 0
