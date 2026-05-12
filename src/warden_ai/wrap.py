"""Wrap an async Anthropic / OpenAI client so every tool call is
inspected by warden before the caller sees it.

Detection is structural and runs once at wrap time:

- `client.messages.create` → Anthropic. We intercept the response,
  walk `content[]` for `tool_use` blocks, normalize them, inspect.
- `client.chat.completions.create` → OpenAI. We intercept the
  response, walk `choices[].message.tool_calls`, JSON-parse the
  `arguments` string, normalize, inspect.

The MVP does NOT detect streaming clients — passing
`stream=True` to a wrapped client returns the raw stream unwrapped.
Streaming support lands in the feature-complete release
(Phase 2 sprint 2).
"""

from __future__ import annotations

import asyncio
from typing import Any
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
from warden_ai.transport import (
    NormalizedToolCall,
    inspect_tool_use,
    poll_pending_once,
)


def warden_wrap(client: Any, opts: WardenOptions) -> Any:
    """Wrap an async Anthropic or OpenAI client.

    Anthropic: install a hook on `client.messages.create`.
    OpenAI:    install a hook on `client.chat.completions.create`.

    Any other client shape raises `WardenConfigError`. The wrap is
    in-place — we monkeypatch the `.create` attribute on the existing
    object rather than building a Proxy facade, because Python's
    attribute model doesn't have a clean object-proxy equivalent and
    partners typically pass the wrapped client by reference into their
    agent framework.
    """
    _validate_options(opts)
    kind = _detect_client(client)
    if kind == "anthropic":
        return _wrap_anthropic(client, opts)
    return _wrap_openai(client, opts)


def _detect_client(client: Any) -> str:
    if client is None:
        raise WardenConfigError("warden_wrap: client must not be None")

    messages = getattr(client, "messages", None)
    if messages is not None and callable(getattr(messages, "create", None)):
        return "anthropic"

    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None and callable(getattr(completions, "create", None)):
        return "openai"

    raise WardenConfigError(
        "warden_wrap: client must expose messages.create (Anthropic) or "
        "chat.completions.create (OpenAI)"
    )


def _wrap_anthropic(client: Any, opts: WardenOptions) -> Any:
    inner = client.messages.create

    async def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            # MVP scope: streaming bypasses inspection. Documented in
            # README + raises a one-line warning on first use so a
            # partner relying on streaming notices.
            _warn_stream_not_inspected()
            return await _maybe_await(inner(*args, **kwargs))
        result = await _maybe_await(inner(*args, **kwargs))
        calls = extract_tool_uses(result)
        await _inspect_all(calls, opts)
        return result

    client.messages.create = create_wrapped
    return client


def _wrap_openai(client: Any, opts: WardenOptions) -> Any:
    inner = client.chat.completions.create

    async def create_wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream") is True:
            _warn_stream_not_inspected()
            return await _maybe_await(inner(*args, **kwargs))
        result = await _maybe_await(inner(*args, **kwargs))
        calls = extract_tool_calls(result)
        await _inspect_all(calls, opts)
        return result

    client.chat.completions.create = create_wrapped
    return client


async def _inspect_all(
    calls: list[NormalizedToolCall], opts: WardenOptions
) -> None:
    """Inspect every normalized tool call concurrently. Fire the
    verdict callback in submission order; (enforce only) raise on the
    first deny / pending.

    Mirrors the TS SDK's `inspectAllToolCalls`: calls run in parallel
    via `asyncio.gather`, but the consume-loop walks them in submission
    order so the first deny in `calls[]` is what raises, not the first
    deny to come back over the wire. Observe-mode transport errors are
    caught per-call and surfaced via `on_policy_error`.
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


async def _maybe_await(value: Any) -> Any:
    """The wrapped client's `.create` may be an `AsyncMock` (in tests
    using a sync return), an `Anthropic` sync client misused under
    async (returns a non-coroutine), or a real async coroutine. Accept
    all three — if `value` is awaitable, await it; otherwise pass
    through.
    """
    if asyncio.iscoroutine(value):
        return await value
    return value


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


_warned_stream = False


def _warn_stream_not_inspected() -> None:
    """One-time warning when a partner passes `stream=True` to a
    wrapped client. The MVP's documented contract is "non-streaming
    only"; streaming arrives in the feature-complete release.
    """
    global _warned_stream
    if _warned_stream:
        return
    _warned_stream = True
    import warnings

    warnings.warn(
        "warden_ai 0.1.0 MVP does not inspect streaming responses. "
        "The underlying agent call is passing through unchecked. "
        "Streaming support lands in the feature-complete release.",
        RuntimeWarning,
        stacklevel=3,
    )
