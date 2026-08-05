"""Streaming wrappers for Anthropic + OpenAI, async and sync.

The contract mirrors the TS SDK's `stream.ts` one-for-one:

  1. Tool-call assembly is observed. Anthropic deltas with
     `type == "input_json_delta"` (the `partial_json` field) and
     OpenAI `tool_calls[i].function.arguments` deltas are buffered
     per tool until the call closes.
  2. The closing event (`content_block_stop` on Anthropic,
     `finish_reason == "tool_calls"` on OpenAI) is held while clavenar
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

from clavenar_agent_sdk.devmode import emit_deny_panel
from clavenar_agent_sdk.errors import (
    ClavenarConfigError,
    ClavenarDenied,
    ClavenarPending,
    ClavenarRateLimited,
    ClavenarTransportError,
)
from clavenar_agent_sdk.options import ClavenarOptions, ClavenarVerdictContext
from clavenar_agent_sdk.transport import (
    MAX_BATCH_REQUEST_BYTES,
    MAX_TOOL_ARGUMENT_BYTES,
    NormalizedToolCall,
    inspect_tool_use,
    inspect_tool_use_sync,
    inspect_tool_uses,
    inspect_tool_uses_sync,
    poll_pending_once,
    poll_pending_once_sync,
)

MAX_STREAM_BUFFERS = 128


@dataclass
class _ToolBuf:
    id: str | None = None
    name: str | None = None
    args_buf: str = ""
    args_bytes: int = 0


@dataclass
class _ChoiceBufs:
    by_index: dict[int, _ToolBuf] = field(default_factory=dict)


async def wrap_anthropic_stream(
    upstream: AsyncIterable[Any],
    opts: ClavenarOptions,
) -> AsyncIterator[Any]:
    """Wrap an Anthropic async message stream. Yields each event in
    order. On `content_block_stop` for a tool_use block, inspects
    before yielding; a denied call raises mid-iteration.
    """
    bufs: dict[int, _ToolBuf] = {}
    ignored: set[int] = set()
    enforce = opts.mode == "enforce"

    async for event in upstream:
        kind = _evt(event, "type")
        if kind == "content_block_start" and _anthropic_is_tool_use_block(event):
            block = _evt(event, "content_block")
            idx = _evt(event, "index")
            tool_id = _evt(block, "id")
            tool_name = _evt(block, "name")
            if (
                isinstance(idx, int)
                and not isinstance(idx, bool)
                and idx >= 0
                and isinstance(tool_id, str)
                and bool(tool_id)
                and isinstance(tool_name, str)
                and bool(tool_name)
            ):
                if idx in bufs or len(bufs) >= MAX_STREAM_BUFFERS:
                    bufs.pop(idx, None)
                    ignored.add(idx)
                    await _handle_stream_shape_error(
                        ClavenarTransportError(
                            "Anthropic stream opened a duplicate or excess tool-use buffer"
                        ),
                        opts,
                        enforce,
                        "Anthropic",
                    )
                else:
                    bufs[idx] = _ToolBuf(id=tool_id, name=tool_name)
            else:
                await _handle_stream_shape_error(
                    ClavenarTransportError(
                        "Anthropic stream tool_use start is missing a valid index, id, or name"
                    ),
                    opts,
                    enforce,
                    "Anthropic",
                )
                if isinstance(idx, int):
                    ignored.add(idx)
            yield event
            continue
        if kind == "content_block_delta":
            idx = _evt(event, "index")
            delta = _evt(event, "delta")
            buf = bufs.get(idx) if isinstance(idx, int) else None
            if buf is not None and _evt(delta, "type") == "input_json_delta":
                partial = _evt(delta, "partial_json")
                if isinstance(partial, str):
                    try:
                        _append_args(buf, partial, sum(item.args_bytes for item in bufs.values()))
                    except ClavenarTransportError as error:
                        await _handle_stream_shape_error(error, opts, enforce, "Anthropic")
                        if isinstance(idx, int):
                            bufs.pop(idx, None)
                            ignored.add(idx)
            yield event
            continue
        if kind == "content_block_stop":
            idx = _evt(event, "index")
            if isinstance(idx, int) and idx in ignored:
                ignored.remove(idx)
                yield event
                continue
            buf = bufs.pop(idx, None) if isinstance(idx, int) else None
            if buf is None:
                yield event
                continue
            try:
                call = _buf_to_call(buf, "Anthropic tool_use")
            except ClavenarTransportError as error:
                await _handle_stream_shape_error(error, opts, enforce, "Anthropic")
                yield event
                continue
            await _inspect_and_maybe_raise(call, opts, enforce)
            yield event
            continue
        yield event
    if bufs:
        await _handle_stream_shape_error(
            ClavenarTransportError(
                "Anthropic stream ended before an open tool_use block was closed"
            ),
            opts,
            enforce,
            "Anthropic",
        )


def wrap_anthropic_stream_sync(
    upstream: Iterable[Any],
    opts: ClavenarOptions,
) -> Iterator[Any]:
    """Sync mirror of `wrap_anthropic_stream`. Same semantics; raises
    `ClavenarDenied` / `ClavenarPending` mid-iteration on enforce deny.
    """
    bufs: dict[int, _ToolBuf] = {}
    ignored: set[int] = set()
    enforce = opts.mode == "enforce"

    for event in upstream:
        kind = _evt(event, "type")
        if kind == "content_block_start" and _anthropic_is_tool_use_block(event):
            block = _evt(event, "content_block")
            idx = _evt(event, "index")
            tool_id = _evt(block, "id")
            tool_name = _evt(block, "name")
            if (
                isinstance(idx, int)
                and not isinstance(idx, bool)
                and idx >= 0
                and isinstance(tool_id, str)
                and bool(tool_id)
                and isinstance(tool_name, str)
                and bool(tool_name)
            ):
                if idx in bufs or len(bufs) >= MAX_STREAM_BUFFERS:
                    bufs.pop(idx, None)
                    ignored.add(idx)
                    _handle_stream_shape_error_sync(
                        ClavenarTransportError(
                            "Anthropic stream opened a duplicate or excess tool-use buffer"
                        ),
                        opts,
                        enforce,
                        "Anthropic",
                    )
                else:
                    bufs[idx] = _ToolBuf(id=tool_id, name=tool_name)
            else:
                _handle_stream_shape_error_sync(
                    ClavenarTransportError(
                        "Anthropic stream tool_use start is missing a valid index, id, or name"
                    ),
                    opts,
                    enforce,
                    "Anthropic",
                )
                if isinstance(idx, int):
                    ignored.add(idx)
            yield event
            continue
        if kind == "content_block_delta":
            idx = _evt(event, "index")
            delta = _evt(event, "delta")
            buf = bufs.get(idx) if isinstance(idx, int) else None
            if buf is not None and _evt(delta, "type") == "input_json_delta":
                partial = _evt(delta, "partial_json")
                if isinstance(partial, str):
                    try:
                        _append_args(buf, partial, sum(item.args_bytes for item in bufs.values()))
                    except ClavenarTransportError as error:
                        _handle_stream_shape_error_sync(error, opts, enforce, "Anthropic")
                        if isinstance(idx, int):
                            bufs.pop(idx, None)
                            ignored.add(idx)
            yield event
            continue
        if kind == "content_block_stop":
            idx = _evt(event, "index")
            if isinstance(idx, int) and idx in ignored:
                ignored.remove(idx)
                yield event
                continue
            buf = bufs.pop(idx, None) if isinstance(idx, int) else None
            if buf is None:
                yield event
                continue
            try:
                call = _buf_to_call(buf, "Anthropic tool_use")
            except ClavenarTransportError as error:
                _handle_stream_shape_error_sync(error, opts, enforce, "Anthropic")
                yield event
                continue
            _inspect_and_maybe_raise_sync(call, opts, enforce)
            yield event
            continue
        yield event
    if bufs:
        _handle_stream_shape_error_sync(
            ClavenarTransportError(
                "Anthropic stream ended before an open tool_use block was closed"
            ),
            opts,
            enforce,
            "Anthropic",
        )


async def wrap_openai_chat_stream(
    upstream: AsyncIterable[Any],
    opts: ClavenarOptions,
) -> AsyncIterator[Any]:
    """Wrap an OpenAI async chat-completion chunk stream. Tool deltas
    are accumulated per `(choice_index, tool_index)`. On a chunk with
    `finish_reason == "tool_calls"` for a choice, every assembled tool
    in that choice is inspected concurrently before the chunk is
    yielded.
    """
    bufs: dict[int, _ChoiceBufs] = {}
    ignored: set[int] = set()
    enforce = opts.mode == "enforce"

    async for chunk in upstream:
        choices = _evt(chunk, "choices")
        if not isinstance(choices, list):
            await _handle_stream_shape_error(
                ClavenarTransportError("OpenAI stream choices is not a list"),
                opts,
                enforce,
                "OpenAI",
            )
            yield chunk
            continue
        to_inspect: list[int] = []
        for choice in choices:
            choice_idx = _evt(choice, "index")
            if not isinstance(choice_idx, int) or isinstance(choice_idx, bool) or choice_idx < 0:
                await _handle_stream_shape_error(
                    ClavenarTransportError("OpenAI stream choice has an invalid index"),
                    opts,
                    enforce,
                    "OpenAI",
                )
                continue
            delta = _evt(choice, "delta")
            deltas = _evt(delta, "tool_calls") if delta is not None else None
            if isinstance(deltas, list):
                for d in deltas:
                    if choice_idx in ignored:
                        continue
                    try:
                        _accumulate_openai(bufs, choice_idx, d)
                    except ClavenarTransportError as error:
                        bufs.pop(choice_idx, None)
                        ignored.add(choice_idx)
                        await _handle_stream_shape_error(error, opts, enforce, "OpenAI")
            if _evt(choice, "finish_reason") == "tool_calls":
                to_inspect.append(choice_idx)
        for choice_idx in to_inspect:
            if choice_idx in ignored:
                ignored.remove(choice_idx)
                continue
            try:
                calls = _drain_openai_choice(bufs, choice_idx)
            except ClavenarTransportError as error:
                await _handle_stream_shape_error(error, opts, enforce, "OpenAI")
                continue
            await _inspect_choice_batch(calls, opts, enforce)
        yield chunk
    if any(choice.by_index for choice in bufs.values()):
        await _handle_stream_shape_error(
            ClavenarTransportError(
                "OpenAI stream ended before buffered tool calls reached a terminal chunk"
            ),
            opts,
            enforce,
            "OpenAI",
        )


def wrap_openai_chat_stream_sync(
    upstream: Iterable[Any],
    opts: ClavenarOptions,
) -> Iterator[Any]:
    """Sync mirror of `wrap_openai_chat_stream`."""
    bufs: dict[int, _ChoiceBufs] = {}
    ignored: set[int] = set()
    enforce = opts.mode == "enforce"

    for chunk in upstream:
        choices = _evt(chunk, "choices")
        if not isinstance(choices, list):
            _handle_stream_shape_error_sync(
                ClavenarTransportError("OpenAI stream choices is not a list"),
                opts,
                enforce,
                "OpenAI",
            )
            yield chunk
            continue
        to_inspect: list[int] = []
        for choice in choices:
            choice_idx = _evt(choice, "index")
            if not isinstance(choice_idx, int) or isinstance(choice_idx, bool) or choice_idx < 0:
                _handle_stream_shape_error_sync(
                    ClavenarTransportError("OpenAI stream choice has an invalid index"),
                    opts,
                    enforce,
                    "OpenAI",
                )
                continue
            delta = _evt(choice, "delta")
            deltas = _evt(delta, "tool_calls") if delta is not None else None
            if isinstance(deltas, list):
                for d in deltas:
                    if choice_idx in ignored:
                        continue
                    try:
                        _accumulate_openai(bufs, choice_idx, d)
                    except ClavenarTransportError as error:
                        bufs.pop(choice_idx, None)
                        ignored.add(choice_idx)
                        _handle_stream_shape_error_sync(error, opts, enforce, "OpenAI")
            if _evt(choice, "finish_reason") == "tool_calls":
                to_inspect.append(choice_idx)
        for choice_idx in to_inspect:
            if choice_idx in ignored:
                ignored.remove(choice_idx)
                continue
            try:
                calls = _drain_openai_choice(bufs, choice_idx)
            except ClavenarTransportError as error:
                _handle_stream_shape_error_sync(error, opts, enforce, "OpenAI")
                continue
            _inspect_choice_batch_sync(calls, opts, enforce)
        yield chunk
    if any(choice.by_index for choice in bufs.values()):
        _handle_stream_shape_error_sync(
            ClavenarTransportError(
                "OpenAI stream ended before buffered tool calls reached a terminal chunk"
            ),
            opts,
            enforce,
            "OpenAI",
        )


def _accumulate_openai(bufs: dict[int, _ChoiceBufs], choice_idx: int, d: Any) -> None:
    cb = bufs.setdefault(choice_idx, _ChoiceBufs())
    tool_idx = _evt(d, "index")
    if not isinstance(tool_idx, int) or isinstance(tool_idx, bool) or tool_idx < 0:
        raise ClavenarTransportError("OpenAI stream tool_call delta has an invalid index")
    if tool_idx not in cb.by_index and sum(len(choice.by_index) for choice in bufs.values()) >= 128:
        raise ClavenarTransportError("OpenAI stream has more than 128 open tool-call buffers")
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
            _append_args(buf, d_args, _total_openai_bytes(bufs))


def _drain_openai_choice(bufs: dict[int, _ChoiceBufs], choice_idx: int) -> list[NormalizedToolCall]:
    cb = bufs.pop(choice_idx, None)
    if cb is None or not cb.by_index:
        raise ClavenarTransportError(
            "OpenAI stream finished with finish_reason='tool_calls' without tool buffers"
        )
    out: list[NormalizedToolCall] = []
    for tool_idx, buf in cb.by_index.items():
        if buf.id is None or buf.name is None:
            raise ClavenarTransportError(
                "OpenAI stream chunk finished with finish_reason='tool_calls' "
                f"but tool_call buffer (choice {choice_idx}, tool {tool_idx}) "
                "is missing id or name"
            )
        out.append(_buf_to_call(buf, "OpenAI tool_call"))
    return out


def _buf_to_call(buf: _ToolBuf, label: str) -> NormalizedToolCall:
    if buf.id is None or buf.name is None:
        raise ClavenarTransportError(f"{label} buffer missing id or name at close")
    if buf.args_buf == "":
        parsed: Any = {}
    else:
        try:
            parsed = json.loads(buf.args_buf)
        except json.JSONDecodeError as e:
            raise ClavenarTransportError(
                f"{label} {buf.id} ({buf.name}) streamed unparseable arguments: {e}"
            ) from e
    return NormalizedToolCall(id=buf.id, name=buf.name, input=parsed)


def _append_args(buf: _ToolBuf, chunk: str, total_before: int) -> None:
    chunk_bytes = len(chunk.encode("utf-8"))
    if buf.args_bytes + chunk_bytes > MAX_TOOL_ARGUMENT_BYTES:
        raise ClavenarTransportError(
            f"streamed tool arguments exceeded {MAX_TOOL_ARGUMENT_BYTES} bytes"
        )
    if total_before + chunk_bytes > MAX_BATCH_REQUEST_BYTES:
        raise ClavenarTransportError(
            f"streamed tool-call batch exceeded {MAX_BATCH_REQUEST_BYTES} bytes"
        )
    buf.args_buf += chunk
    buf.args_bytes += chunk_bytes


def _total_openai_bytes(bufs: dict[int, _ChoiceBufs]) -> int:
    return sum(tool.args_bytes for choice in bufs.values() for tool in choice.by_index.values())


async def _handle_stream_shape_error(
    error: ClavenarTransportError,
    opts: ClavenarOptions,
    enforce: bool,
    provider: str,
) -> None:
    if enforce:
        raise error
    call = NormalizedToolCall(
        id="<unknown>",
        name=f"<{provider.lower()}-stream>",
        input=None,
    )
    await _fire_policy_error(error, call, opts)


def _handle_stream_shape_error_sync(
    error: ClavenarTransportError,
    opts: ClavenarOptions,
    enforce: bool,
    provider: str,
) -> None:
    if enforce:
        raise error
    call = NormalizedToolCall(
        id="<unknown>",
        name=f"<{provider.lower()}-stream>",
        input=None,
    )
    _fire_policy_error_sync(error, call, opts)


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
    call: NormalizedToolCall, opts: ClavenarOptions, enforce: bool
) -> None:
    try:
        verdict = await inspect_tool_use(call, opts)
    except ClavenarTransportError as e:
        if not enforce:
            await _fire_policy_error(e, call, opts)
            return
        raise
    await _process_verdict(verdict, call, opts, enforce)


def _inspect_and_maybe_raise_sync(
    call: NormalizedToolCall, opts: ClavenarOptions, enforce: bool
) -> None:
    try:
        verdict = inspect_tool_use_sync(call, opts)
    except ClavenarTransportError as e:
        if not enforce:
            _fire_policy_error_sync(e, call, opts)
            return
        raise
    _process_verdict_sync(verdict, call, opts, enforce)


async def _inspect_choice_batch(
    calls: list[NormalizedToolCall], opts: ClavenarOptions, enforce: bool
) -> None:
    if not calls:
        return

    try:
        verdict = await inspect_tool_uses(calls, opts)
    except ClavenarTransportError as error:
        if enforce:
            raise
        for call in calls:
            await _fire_policy_error(error, call, opts)
        return
    for call in calls:
        await _process_verdict(verdict, call, opts, enforce)


def _inspect_choice_batch_sync(
    calls: list[NormalizedToolCall], opts: ClavenarOptions, enforce: bool
) -> None:
    if not calls:
        return
    try:
        verdict = inspect_tool_uses_sync(calls, opts)
    except ClavenarTransportError as error:
        if enforce:
            raise
        for call in calls:
            _fire_policy_error_sync(error, call, opts)
        return
    for call in calls:
        _process_verdict_sync(verdict, call, opts, enforce)


async def _fire_policy_error(
    error: ClavenarTransportError, call: NormalizedToolCall, opts: ClavenarOptions
) -> None:
    if opts.on_policy_error is None:
        return
    ctx = ClavenarVerdictContext(tool_name=call.name, tool_use_id=call.id, tool_input=call.input)
    out = opts.on_policy_error(error, ctx)
    if asyncio.iscoroutine(out):
        await out


def _fire_policy_error_sync(
    error: ClavenarTransportError, call: NormalizedToolCall, opts: ClavenarOptions
) -> None:
    if opts.on_policy_error is None:
        return
    ctx = ClavenarVerdictContext(tool_name=call.name, tool_use_id=call.id, tool_input=call.input)
    out = opts.on_policy_error(error, ctx)
    if asyncio.iscoroutine(out):
        raise ClavenarConfigError(
            "on_policy_error returned a coroutine but the stream is sync; "
            "use a sync callback for sync clients"
        )


async def _process_verdict(
    verdict: Any, call: NormalizedToolCall, opts: ClavenarOptions, enforce: bool
) -> None:
    ctx = ClavenarVerdictContext(tool_name=call.name, tool_use_id=call.id, tool_input=call.input)
    if opts.on_verdict is not None:
        out = opts.on_verdict(verdict, ctx)
        if asyncio.iscoroutine(out):
            await out
    if not enforce:
        return
    if verdict.kind == "deny":
        denied = ClavenarDenied(
            tool_name=call.name,
            reasons=verdict.reasons,
            review_reasons=verdict.review_reasons,
            intent_category=verdict.intent_category,
            layer=verdict.layer,
            correlation_id=verdict.correlation_id,
            detail=verdict.detail,
        )
        if opts.dev_mode:
            emit_deny_panel(denied)
        raise denied
    if verdict.kind == "pending":
        corr = verdict.correlation_id

        async def _poll(corr_id: str = corr) -> Any:
            return await poll_pending_once(corr_id, opts)

        raise ClavenarPending(
            tool_name=call.name,
            correlation_id=verdict.correlation_id,
            review_reasons=verdict.review_reasons,
            poll_once=_poll,
        )
    if verdict.kind == "rate_limited":
        raise ClavenarRateLimited(
            tool_name=call.name,
            code=verdict.code,
            reasons=verdict.reasons,
            retry_after_secs=verdict.retry_after_secs,
            layer=verdict.layer,
            correlation_id=verdict.correlation_id,
        )


def _process_verdict_sync(
    verdict: Any, call: NormalizedToolCall, opts: ClavenarOptions, enforce: bool
) -> None:
    ctx = ClavenarVerdictContext(tool_name=call.name, tool_use_id=call.id, tool_input=call.input)
    if opts.on_verdict is not None:
        out = opts.on_verdict(verdict, ctx)
        if asyncio.iscoroutine(out):
            raise ClavenarConfigError(
                "on_verdict returned a coroutine but the stream is sync; "
                "use a sync callback for sync clients"
            )
    if not enforce:
        return
    if verdict.kind == "deny":
        denied = ClavenarDenied(
            tool_name=call.name,
            reasons=verdict.reasons,
            review_reasons=verdict.review_reasons,
            intent_category=verdict.intent_category,
            layer=verdict.layer,
            correlation_id=verdict.correlation_id,
            detail=verdict.detail,
        )
        if opts.dev_mode:
            emit_deny_panel(denied)
        raise denied
    if verdict.kind == "pending":
        corr = verdict.correlation_id

        def _poll_sync(corr_id: str = corr) -> Any:
            return poll_pending_once_sync(corr_id, opts)

        raise ClavenarPending(
            tool_name=call.name,
            correlation_id=verdict.correlation_id,
            review_reasons=verdict.review_reasons,
            poll_once_sync=_poll_sync,
        )
    if verdict.kind == "rate_limited":
        raise ClavenarRateLimited(
            tool_name=call.name,
            code=verdict.code,
            reasons=verdict.reasons,
            retry_after_secs=verdict.retry_after_secs,
            layer=verdict.layer,
            correlation_id=verdict.correlation_id,
        )
