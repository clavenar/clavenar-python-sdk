"""Streaming wrappers for Anthropic + OpenAI, async and sync.

The contract mirrors the TS SDK's `stream.ts` one-for-one:

  1. Tool-call assembly is observed. Anthropic deltas with
     `type == "input_json_delta"` (the `partial_json` field) and
     OpenAI `tool_calls[i].function.arguments` deltas are buffered
     per tool until the call closes.
  2. The closing event (`content_block_stop` on Anthropic,
     `finish_reason == "tool_calls"` on OpenAI) is held while warden
     inspects. On deny in enforce mode we raise BEFORE yielding the
     closing event — partner code never sees a denied tool call as
     actionable.

`on_verdict` fires for every inspected tool call before any raise so
observe-mode telemetry stays consistent with the non-streaming path.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

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
    inspect_tool_use_sync,
    poll_pending_once,
    poll_pending_once_sync,
)


@dataclass
class _ToolBuf:
    id: str | None = None
    name: str | None = None
    args_buf: str = ""


@dataclass
class _ChoiceBufs:
    by_index: dict[int, _ToolBuf] = field(default_factory=dict)


async def wrap_anthropic_stream(
    upstream: AsyncIterable[Any],
    opts: WardenOptions,
) -> AsyncIterator[Any]:
    """Wrap an Anthropic async message stream. Yields each event in
    order. On `content_block_stop` for a tool_use block, inspects
    before yielding; a denied call raises mid-iteration.
    """
    bufs: dict[int, _ToolBuf] = {}
    enforce = opts.mode == "enforce"

    async for event in upstream:
        kind = _evt(event, "type")
        if kind == "content_block_start" and _anthropic_is_tool_use_block(event):
            block = _evt(event, "content_block")
            idx = _evt(event, "index")
            if isinstance(idx, int):
                bufs[idx] = _ToolBuf(
                    id=_evt(block, "id"),
                    name=_evt(block, "name"),
                )
            yield event
            continue
        if kind == "content_block_delta":
            idx = _evt(event, "index")
            delta = _evt(event, "delta")
            buf = bufs.get(idx) if isinstance(idx, int) else None
            if buf is not None and _evt(delta, "type") == "input_json_delta":
                partial = _evt(delta, "partial_json")
                if isinstance(partial, str):
                    buf.args_buf += partial
            yield event
            continue
        if kind == "content_block_stop":
            idx = _evt(event, "index")
            buf = bufs.pop(idx, None) if isinstance(idx, int) else None
            if buf is None:
                yield event
                continue
            call = _buf_to_call(buf, "Anthropic tool_use")
            await _inspect_and_maybe_raise(call, opts, enforce)
            yield event
            continue
        yield event


def wrap_anthropic_stream_sync(
    upstream: Iterable[Any],
    opts: WardenOptions,
) -> Iterator[Any]:
    """Sync mirror of `wrap_anthropic_stream`. Same semantics; raises
    `WardenDenied` / `WardenPending` mid-iteration on enforce deny.
    """
    bufs: dict[int, _ToolBuf] = {}
    enforce = opts.mode == "enforce"

    for event in upstream:
        kind = _evt(event, "type")
        if kind == "content_block_start" and _anthropic_is_tool_use_block(event):
            block = _evt(event, "content_block")
            idx = _evt(event, "index")
            if isinstance(idx, int):
                bufs[idx] = _ToolBuf(
                    id=_evt(block, "id"),
                    name=_evt(block, "name"),
                )
            yield event
            continue
        if kind == "content_block_delta":
            idx = _evt(event, "index")
            delta = _evt(event, "delta")
            buf = bufs.get(idx) if isinstance(idx, int) else None
            if buf is not None and _evt(delta, "type") == "input_json_delta":
                partial = _evt(delta, "partial_json")
                if isinstance(partial, str):
                    buf.args_buf += partial
            yield event
            continue
        if kind == "content_block_stop":
            idx = _evt(event, "index")
            buf = bufs.pop(idx, None) if isinstance(idx, int) else None
            if buf is None:
                yield event
                continue
            call = _buf_to_call(buf, "Anthropic tool_use")
            _inspect_and_maybe_raise_sync(call, opts, enforce)
            yield event
            continue
        yield event


async def wrap_openai_chat_stream(
    upstream: AsyncIterable[Any],
    opts: WardenOptions,
) -> AsyncIterator[Any]:
    """Wrap an OpenAI async chat-completion chunk stream. Tool deltas
    are accumulated per `(choice_index, tool_index)`. On a chunk with
    `finish_reason == "tool_calls"` for a choice, every assembled tool
    in that choice is inspected concurrently before the chunk is
    yielded.
    """
    bufs: dict[int, _ChoiceBufs] = {}
    enforce = opts.mode == "enforce"

    async for chunk in upstream:
        choices = _evt(chunk, "choices") or []
        to_inspect: list[int] = []
        for choice in choices:
            choice_idx = _evt(choice, "index")
            if not isinstance(choice_idx, int):
                continue
            delta = _evt(choice, "delta")
            deltas = _evt(delta, "tool_calls") if delta is not None else None
            if isinstance(deltas, list):
                for d in deltas:
                    _accumulate_openai(bufs, choice_idx, d)
            if _evt(choice, "finish_reason") == "tool_calls":
                to_inspect.append(choice_idx)
        for choice_idx in to_inspect:
            calls = _drain_openai_choice(bufs, choice_idx)
            await _inspect_choice_batch(calls, opts, enforce)
        yield chunk


def wrap_openai_chat_stream_sync(
    upstream: Iterable[Any],
    opts: WardenOptions,
) -> Iterator[Any]:
    """Sync mirror of `wrap_openai_chat_stream`."""
    bufs: dict[int, _ChoiceBufs] = {}
    enforce = opts.mode == "enforce"

    for chunk in upstream:
        choices = _evt(chunk, "choices") or []
        to_inspect: list[int] = []
        for choice in choices:
            choice_idx = _evt(choice, "index")
            if not isinstance(choice_idx, int):
                continue
            delta = _evt(choice, "delta")
            deltas = _evt(delta, "tool_calls") if delta is not None else None
            if isinstance(deltas, list):
                for d in deltas:
                    _accumulate_openai(bufs, choice_idx, d)
            if _evt(choice, "finish_reason") == "tool_calls":
                to_inspect.append(choice_idx)
        for choice_idx in to_inspect:
            calls = _drain_openai_choice(bufs, choice_idx)
            _inspect_choice_batch_sync(calls, opts, enforce)
        yield chunk


def _accumulate_openai(
    bufs: dict[int, _ChoiceBufs], choice_idx: int, d: Any
) -> None:
    cb = bufs.setdefault(choice_idx, _ChoiceBufs())
    tool_idx = _evt(d, "index", default=0)
    if not isinstance(tool_idx, int):
        return
    buf = cb.by_index.setdefault(tool_idx, _ToolBuf())
    d_id = _evt(d, "id")
    if isinstance(d_id, str):
        buf.id = d_id
    fn = _evt(d, "function")
    if fn is not None:
        d_name = _evt(fn, "name")
        if isinstance(d_name, str):
            buf.name = d_name
        d_args = _evt(fn, "arguments")
        if isinstance(d_args, str):
            buf.args_buf += d_args


def _drain_openai_choice(
    bufs: dict[int, _ChoiceBufs], choice_idx: int
) -> list[NormalizedToolCall]:
    cb = bufs.pop(choice_idx, None)
    if cb is None:
        return []
    out: list[NormalizedToolCall] = []
    for tool_idx, buf in cb.by_index.items():
        if buf.id is None or buf.name is None:
            raise WardenConfigError(
                "OpenAI stream chunk finished with finish_reason='tool_calls' "
                f"but tool_call buffer (choice {choice_idx}, tool {tool_idx}) "
                "is missing id or name"
            )
        out.append(_buf_to_call(buf, "OpenAI tool_call"))
    return out


def _buf_to_call(buf: _ToolBuf, label: str) -> NormalizedToolCall:
    if buf.id is None or buf.name is None:
        raise WardenConfigError(
            f"{label} buffer missing id or name at close"
        )
    if buf.args_buf == "":
        parsed: Any = {}
    else:
        try:
            parsed = json.loads(buf.args_buf)
        except json.JSONDecodeError as e:
            raise WardenConfigError(
                f"{label} {buf.id} ({buf.name}) streamed unparseable arguments: {e}"
            ) from e
    return NormalizedToolCall(id=buf.id, name=buf.name, input=parsed)


def _anthropic_is_tool_use_block(event: Any) -> bool:
    block = _evt(event, "content_block")
    if block is None:
        return False
    return bool(_evt(block, "type") == "tool_use")


def _evt(obj: Any, key: str, *, default: Any = None) -> Any:
    """Dual access: dict-key or attribute. Anthropic + OpenAI SDKs
    expose Pydantic models (attribute access); raw HTTP / fake streams
    might come as dicts.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def _inspect_and_maybe_raise(
    call: NormalizedToolCall, opts: WardenOptions, enforce: bool
) -> None:
    try:
        verdict = await inspect_tool_use(call, opts)
    except WardenTransportError as e:
        if not enforce:
            await _fire_policy_error(e, call, opts)
            return
        raise
    await _process_verdict(verdict, call, opts, enforce)


def _inspect_and_maybe_raise_sync(
    call: NormalizedToolCall, opts: WardenOptions, enforce: bool
) -> None:
    try:
        verdict = inspect_tool_use_sync(call, opts)
    except WardenTransportError as e:
        if not enforce:
            _fire_policy_error_sync(e, call, opts)
            return
        raise
    _process_verdict_sync(verdict, call, opts, enforce)


async def _inspect_choice_batch(
    calls: list[NormalizedToolCall], opts: WardenOptions, enforce: bool
) -> None:
    if not calls:
        return

    async def one(c: NormalizedToolCall) -> tuple[NormalizedToolCall, Any]:
        try:
            return c, await inspect_tool_use(c, opts)
        except WardenTransportError as e:
            if not enforce:
                return c, e
            raise

    results = await asyncio.gather(*(one(c) for c in calls))
    for call, result in results:
        if isinstance(result, WardenTransportError):
            await _fire_policy_error(result, call, opts)
            continue
        await _process_verdict(result, call, opts, enforce)


def _inspect_choice_batch_sync(
    calls: list[NormalizedToolCall], opts: WardenOptions, enforce: bool
) -> None:
    if not calls:
        return
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
        if isinstance(result, WardenTransportError):
            _fire_policy_error_sync(result, call, opts)
            continue
        _process_verdict_sync(result, call, opts, enforce)


async def _fire_policy_error(
    error: WardenTransportError, call: NormalizedToolCall, opts: WardenOptions
) -> None:
    if opts.on_policy_error is None:
        return
    ctx = WardenVerdictContext(
        tool_name=call.name, tool_use_id=call.id, tool_input=call.input
    )
    out = opts.on_policy_error(error, ctx)
    if asyncio.iscoroutine(out):
        await out


def _fire_policy_error_sync(
    error: WardenTransportError, call: NormalizedToolCall, opts: WardenOptions
) -> None:
    if opts.on_policy_error is None:
        return
    ctx = WardenVerdictContext(
        tool_name=call.name, tool_use_id=call.id, tool_input=call.input
    )
    out = opts.on_policy_error(error, ctx)
    if asyncio.iscoroutine(out):
        raise WardenConfigError(
            "on_policy_error returned a coroutine but the stream is sync; "
            "use a sync callback for sync clients"
        )


async def _process_verdict(
    verdict: Any, call: NormalizedToolCall, opts: WardenOptions, enforce: bool
) -> None:
    ctx = WardenVerdictContext(
        tool_name=call.name, tool_use_id=call.id, tool_input=call.input
    )
    if opts.on_verdict is not None:
        out = opts.on_verdict(verdict, ctx)
        if asyncio.iscoroutine(out):
            await out
    if not enforce:
        return
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


def _process_verdict_sync(
    verdict: Any, call: NormalizedToolCall, opts: WardenOptions, enforce: bool
) -> None:
    ctx = WardenVerdictContext(
        tool_name=call.name, tool_use_id=call.id, tool_input=call.input
    )
    if opts.on_verdict is not None:
        out = opts.on_verdict(verdict, ctx)
        if asyncio.iscoroutine(out):
            raise WardenConfigError(
                "on_verdict returned a coroutine but the stream is sync; "
                "use a sync callback for sync clients"
            )
    if not enforce:
        return
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

        # Sync clients won't await; expose the sync poller.
        async def _poll_async(corr_id: str = corr) -> Any:
            return poll_pending_once_sync(corr_id, opts)

        raise WardenPending(
            tool_name=call.name,
            correlation_id=verdict.correlation_id,
            review_reasons=verdict.review_reasons,
            poll_once=_poll_async,
        )
