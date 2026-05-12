"""Wrap an Anthropic / OpenAI client so every tool call is inspected
by warden before the caller sees it.

Detection is structural and runs once at wrap time:

- `client.messages.create` → Anthropic. Intercepts the response, walks
  `content[]` for `tool_use` blocks, normalizes, inspects.
- `client.chat.completions.create` → OpenAI. Intercepts the response,
  walks `choices[].message.tool_calls`, JSON-parses each `arguments`
  string, normalizes, inspects.

Both async (`AsyncAnthropic`, `AsyncOpenAI`) and sync (`Anthropic`,
`OpenAI`) clients are supported. Sync vs async is determined by
`inspect.iscoroutinefunction(create)`; the sync path uses
`httpx.Client` and `time.sleep`, the async path uses `httpx.AsyncClient`
and `asyncio.sleep`.

Streaming (`stream=True`) is intercepted: each event/chunk is passed
through in order, but the closing event (Anthropic `content_block_stop`,
OpenAI `finish_reason='tool_calls'`) is held until warden verdicts
land — a denied call raises mid-iteration before the partner can act
on it. See `stream.py`.
"""

from __future__ import annotations

import asyncio
import inspect as inspect_mod
from collections.abc import AsyncIterable, Iterable
from typing import Any, Literal
from urllib.parse import urlparse

from warden_ai._anthropic import extract_tool_uses
from warden_ai._openai import extract_tool_calls
from warden_ai.errors import (
    WardenConfigError,
    WardenDenied,
    WardenPending,
    WardenTransportError,
)
from warden_ai.options import WardenOptions, WardenVerdictContext
from warden_ai.stream import (
    wrap_anthropic_stream,
    wrap_anthropic_stream_sync,
    wrap_openai_chat_stream,
    wrap_openai_chat_stream_sync,
)
from warden_ai.transport import (
    NormalizedToolCall,
    inspect_tool_use,
    inspect_tool_use_sync,
    poll_pending_once,
    poll_pending_once_sync,
)

ClientKind = Literal["anthropic", "openai"]
ClientMode = Literal["async", "sync"]


def warden_wrap(client: Any, opts: WardenOptions) -> Any:
    """Wrap an async or sync Anthropic / OpenAI client.

    The wrap is in-place — we monkeypatch the `.create` attribute on
    the existing object rather than building a Proxy facade, because
    Python's attribute model doesn't have a clean object-proxy
    equivalent and partners typically pass the wrapped client by
    reference into their agent framework.
    """
    _validate_options(opts)
    kind, mode = _detect_client(client)
    if kind == "anthropic" and mode == "async":
        return _wrap_anthropic_async(client, opts)
    if kind == "anthropic" and mode == "sync":
        return _wrap_anthropic_sync(client, opts)
    if kind == "openai" and mode == "async":
        return _wrap_openai_async(client, opts)
    return _wrap_openai_sync(client, opts)


def _detect_client(client: Any) -> tuple[ClientKind, ClientMode]:
    if client is None:
        raise WardenConfigError("warden_wrap: client must not be None")

    messages = getattr(client, "messages", None)
    if messages is not None:
        create = getattr(messages, "create", None)
        if callable(create):
            mode: ClientMode = "async" if inspect_mod.iscoroutinefunction(create) else "sync"
            return "anthropic", mode

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None:
        create = getattr(completions, "create", None)
        if callable(create):
            mode = "async" if inspect_mod.iscoroutinefunction(create) else "sync"
            return "openai", mode

    raise WardenConfigError(
        "warden_wrap: client must expose messages.create (Anthropic) or "
        "chat.completions.create (OpenAI)"
    )


def _wrap_anthropic_async(client: Any, opts: WardenOptions) -> Any:
    inner = client.messages.create

    async def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        result = await _maybe_await(inner(*args, **kwargs))
        if kwargs.get("stream") is True or _is_async_iterable(result):
            return wrap_anthropic_stream(result, opts)
        calls = extract_tool_uses(result)
        await _inspect_all_async(calls, opts)
        return result

    client.messages.create = create_wrapped
    return client


def _wrap_anthropic_sync(client: Any, opts: WardenOptions) -> Any:
    inner = client.messages.create

    def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        if kwargs.get("stream") is True or _is_iterable_non_message(result):
            return wrap_anthropic_stream_sync(result, opts)
        calls = extract_tool_uses(result)
        _inspect_all_sync(calls, opts)
        return result

    client.messages.create = create_wrapped
    return client


def _wrap_openai_async(client: Any, opts: WardenOptions) -> Any:
    inner = client.chat.completions.create

    async def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        result = await _maybe_await(inner(*args, **kwargs))
        if kwargs.get("stream") is True or _is_async_iterable(result):
            return wrap_openai_chat_stream(result, opts)
        calls = extract_tool_calls(result)
        await _inspect_all_async(calls, opts)
        return result

    client.chat.completions.create = create_wrapped
    return client


def _wrap_openai_sync(client: Any, opts: WardenOptions) -> Any:
    inner = client.chat.completions.create

    def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        result = inner(*args, **kwargs)
        if kwargs.get("stream") is True or _is_iterable_non_message(result):
            return wrap_openai_chat_stream_sync(result, opts)
        calls = extract_tool_calls(result)
        _inspect_all_sync(calls, opts)
        return result

    client.chat.completions.create = create_wrapped
    return client


async def _inspect_all_async(
    calls: list[NormalizedToolCall], opts: WardenOptions
) -> None:
    """Inspect all tool calls concurrently via `asyncio.gather`; fire
    callbacks in submission order; raise on the first deny/pending.
    Observe-mode transport errors are caught per-call.
    """
    if not calls:
        return
    enforce = opts.mode == "enforce"

    async def one(call: NormalizedToolCall) -> tuple[NormalizedToolCall, Any]:
        try:
            verdict = await inspect_tool_use(call, opts)
            return call, verdict
        except WardenTransportError as e:
            if not enforce:
                return call, e
            raise

    results = await asyncio.gather(*(one(c) for c in calls))

    for call, result in results:
        ctx = WardenVerdictContext(
            tool_name=call.name,
            tool_use_id=call.id,
            tool_input=call.input,
        )
        if isinstance(result, WardenTransportError):
            if opts.on_policy_error is not None:
                await _maybe_await(opts.on_policy_error(result, ctx))
            continue
        verdict = result
        if opts.on_verdict is not None:
            await _maybe_await(opts.on_verdict(verdict, ctx))
        if not enforce:
            continue
        _raise_for_verdict_async(verdict, call, opts)


def _raise_for_verdict_async(
    verdict: Any, call: NormalizedToolCall, opts: WardenOptions
) -> None:
    if verdict.kind == "deny":
        raise WardenDenied(
            tool_name=call.name,
            reasons=verdict.reasons,
            review_reasons=verdict.review_reasons,
            intent_category=verdict.intent_category,
            correlation_id=verdict.correlation_id,
        )
    if verdict.kind == "pending":
        corr = verdict.correlation_id

        async def _poll(corr_id: str = corr) -> Any:
            return await poll_pending_once(corr_id, opts)

        raise WardenPending(
            tool_name=call.name,
            correlation_id=verdict.correlation_id,
            review_reasons=verdict.review_reasons,
            poll_once=_poll,
        )


def _inspect_all_sync(
    calls: list[NormalizedToolCall], opts: WardenOptions
) -> None:
    """Sync mirror of `_inspect_all_async`. Inspections run serially —
    `asyncio.gather` has no sync equivalent and threading for I/O here
    isn't worth the complexity for the 1-3 tool calls a typical turn
    emits.
    """
    if not calls:
        return
    enforce = opts.mode == "enforce"
    results: list[tuple[NormalizedToolCall, Any]] = []
    for c in calls:
        try:
            results.append((c, inspect_tool_use_sync(c, opts)))
        except WardenTransportError as e:
            if not enforce:
                results.append((c, e))
                continue
            raise
    for call, result in results:
        ctx = WardenVerdictContext(
            tool_name=call.name,
            tool_use_id=call.id,
            tool_input=call.input,
        )
        if isinstance(result, WardenTransportError):
            if opts.on_policy_error is not None:
                out = opts.on_policy_error(result, ctx)
                if asyncio.iscoroutine(out):
                    raise WardenConfigError(
                        "on_policy_error returned a coroutine but the client is "
                        "sync; use a sync callback for sync clients"
                    )
            continue
        verdict = result
        if opts.on_verdict is not None:
            out = opts.on_verdict(verdict, ctx)
            if asyncio.iscoroutine(out):
                raise WardenConfigError(
                    "on_verdict returned a coroutine but the client is sync; "
                    "use a sync callback for sync clients"
                )
        if not enforce:
            continue
        _raise_for_verdict_sync(verdict, call, opts)


def _raise_for_verdict_sync(
    verdict: Any, call: NormalizedToolCall, opts: WardenOptions
) -> None:
    if verdict.kind == "deny":
        raise WardenDenied(
            tool_name=call.name,
            reasons=verdict.reasons,
            review_reasons=verdict.review_reasons,
            intent_category=verdict.intent_category,
            correlation_id=verdict.correlation_id,
        )
    if verdict.kind == "pending":
        corr = verdict.correlation_id

        async def _poll(corr_id: str = corr) -> Any:
            return poll_pending_once_sync(corr_id, opts)

        raise WardenPending(
            tool_name=call.name,
            correlation_id=verdict.correlation_id,
            review_reasons=verdict.review_reasons,
            poll_once=_poll,
        )


async def _maybe_await(value: Any) -> Any:
    """The wrapped client's `.create` may be an `AsyncMock` (sync
    return), an unsuitable client, or a real coroutine. Accept all
    three — if `value` is awaitable, await it; otherwise pass through.
    """
    if asyncio.iscoroutine(value):
        return await value
    return value


def _is_async_iterable(v: Any) -> bool:
    return isinstance(v, AsyncIterable)


def _is_iterable_non_message(v: Any) -> bool:
    """Heuristic: a sync stream object is iterable but the Message /
    ChatCompletion response objects (which carry `content`, `choices`)
    are not. We don't want to treat a Message dict as a stream.
    """
    if isinstance(v, (str, bytes, dict)):
        return False
    if hasattr(v, "content") or hasattr(v, "choices"):
        return False
    return isinstance(v, Iterable)


def _validate_options(opts: WardenOptions) -> None:
    if not opts.endpoint:
        raise WardenConfigError("warden_wrap: opts.endpoint is required")
    parsed = urlparse(opts.endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise WardenConfigError(
            f"warden_wrap: opts.endpoint is not a valid URL: {opts.endpoint!r}"
        )
    if opts.timeout_s <= 0:
        raise WardenConfigError(
            f"warden_wrap: opts.timeout_s must be positive (got {opts.timeout_s})"
        )
    if opts.mode not in ("enforce", "observe"):
        raise WardenConfigError(
            f"warden_wrap: opts.mode must be 'enforce' or 'observe' (got {opts.mode!r})"
        )
    if opts.retry.max_attempts < 1:
        raise WardenConfigError(
            f"warden_wrap: opts.retry.max_attempts must be >= 1 "
            f"(got {opts.retry.max_attempts})"
        )
    if opts.retry.base_delay_s < 0:
        raise WardenConfigError(
            f"warden_wrap: opts.retry.base_delay_s must be >= 0 "
            f"(got {opts.retry.base_delay_s})"
        )
