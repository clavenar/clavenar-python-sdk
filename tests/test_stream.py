"""Streaming wrappers for Anthropic + OpenAI.

The closing event (`content_block_stop` / `finish_reason='tool_calls'`)
is held until warden verdicts return; a denied call raises
mid-iteration before partner code can act on it.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from conftest import (
    FAKE_ENDPOINT,
    async_iter,
    make_anthropic_tool_use_events,
    make_openai_tool_call_chunks,
)

from warden_ai.errors import WardenDenied, WardenPending
from warden_ai.options import WardenOptions
from warden_ai.stream import wrap_anthropic_stream, wrap_openai_chat_stream


def _enforce() -> WardenOptions:
    return WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)


# ---- Anthropic streams ----------------------------------------------------


@respx.mock
async def test_anthropic_stream_allow_yields_all_events() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    events = make_anthropic_tool_use_events()
    out = []
    async for ev in wrap_anthropic_stream(async_iter(events), _enforce()):
        out.append(ev["type"])
    assert out == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_stop",
    ]


@respx.mock
async def test_anthropic_stream_deny_raises_before_content_block_stop() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["denied"],
                "review_reasons": [],
                "intent_category": "x",
            },
        )
    )
    events = make_anthropic_tool_use_events(tool_name="rm_rf")
    seen: list[str] = []
    with pytest.raises(WardenDenied) as exc:
        async for ev in wrap_anthropic_stream(async_iter(events), _enforce()):
            seen.append(ev["type"])
    assert exc.value.tool_name == "rm_rf"
    # content_block_stop must NOT have been yielded — partner never
    # saw the tool call as finalized.
    assert "content_block_stop" not in seen
    assert "content_block_start" in seen


@respx.mock
async def test_anthropic_stream_pending_raises_warden_pending() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            202,
            json={
                "status": "pending",
                "correlation_id": "corr-x",
                "review_reasons": ["needs human"],
            },
        )
    )
    events = make_anthropic_tool_use_events()
    with pytest.raises(WardenPending) as exc:
        async for _ in wrap_anthropic_stream(async_iter(events), _enforce()):
            pass
    assert exc.value.correlation_id == "corr-x"


@respx.mock
async def test_anthropic_stream_observe_deny_passes_through() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["would_deny"],
                "review_reasons": [],
                "intent_category": "x",
            },
        )
    )
    events = make_anthropic_tool_use_events()
    seen: list[str] = []
    seen_verdicts: list[str] = []

    async def on_verdict(verdict, ctx) -> None:  # type: ignore[no-untyped-def]
        seen_verdicts.append(f"{verdict.kind}:{ctx.tool_name}")

    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT, mode="observe", timeout_s=2.0, on_verdict=on_verdict
    )
    async for ev in wrap_anthropic_stream(async_iter(events), opts):
        seen.append(ev["type"])
    assert "content_block_stop" in seen
    assert seen_verdicts == ["deny:list_files"]


@respx.mock
async def test_anthropic_stream_text_only_passes_through() -> None:
    # No tool_use block → no inspection → no /mcp call.
    route = respx.post(f"{FAKE_ENDPOINT}/mcp")
    events = [
        {"type": "message_start"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    out = []
    async for ev in wrap_anthropic_stream(async_iter(events), _enforce()):
        out.append(ev["type"])
    assert route.call_count == 0
    assert len(out) == 5


# ---- OpenAI streams -------------------------------------------------------


@respx.mock
async def test_openai_stream_allow_yields_all_chunks() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(200))
    chunks = make_openai_tool_call_chunks()
    out = []
    async for c in wrap_openai_chat_stream(async_iter(chunks), _enforce()):
        out.append(c)
    assert len(out) == len(chunks)


@respx.mock
async def test_openai_stream_deny_raises_before_finish_chunk() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["denied"],
                "review_reasons": [],
                "intent_category": "x",
            },
        )
    )
    chunks = make_openai_tool_call_chunks(name="drop_table")
    seen: list[dict] = []
    with pytest.raises(WardenDenied) as exc:
        async for c in wrap_openai_chat_stream(async_iter(chunks), _enforce()):
            seen.append(c)
    assert exc.value.tool_name == "drop_table"
    # Last chunk (the finish_reason='tool_calls' one) must NOT have
    # been yielded.
    for c in seen:
        for choice in c["choices"]:
            assert choice.get("finish_reason") != "tool_calls"


@respx.mock
async def test_openai_stream_observe_transport_error_routes_to_callback() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(return_value=httpx.Response(502))
    chunks = make_openai_tool_call_chunks()
    errors: list[str] = []

    async def on_policy_error(err, ctx) -> None:  # type: ignore[no-untyped-def]
        errors.append(f"{ctx.tool_name}:{err.status}")

    opts = WardenOptions(
        endpoint=FAKE_ENDPOINT,
        mode="observe",
        timeout_s=2.0,
        on_policy_error=on_policy_error,
    )
    # Must not raise — observe contract.
    out = []
    async for c in wrap_openai_chat_stream(async_iter(chunks), opts):
        out.append(c)
    assert len(out) == len(chunks)
    assert errors == ["list_files:502"]
