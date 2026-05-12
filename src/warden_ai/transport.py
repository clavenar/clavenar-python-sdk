"""HTTP transport for warden-lite. Submits one normalized tool call,
parses the verdict (allow / deny / pending), and surfaces correlation
ids for ledger lookups.

Both async and sync flavours live here. The sync flavour exists for
partners wrapping `anthropic.Anthropic` / `openai.OpenAI` (the
non-async SDK clients) — the wrap pattern can't transparently jump
out of a sync caller into an event loop, so a parallel sync transport
is the cleanest seam.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from warden_ai.errors import WardenTransportError
from warden_ai.options import WardenOptions

CORRELATION_HEADER = "x-warden-correlation-id"


@dataclass(frozen=True)
class NormalizedToolCall:
    """Provider-agnostic shape of one tool call ready for inspection."""

    id: str
    name: str
    input: Any


@dataclass(frozen=True)
class _Allow:
    correlation_id: str | None = None
    kind: Literal["allow"] = "allow"


@dataclass(frozen=True)
class _Deny:
    reasons: list[str]
    review_reasons: list[str]
    intent_category: str
    correlation_id: str | None = None
    kind: Literal["deny"] = "deny"


@dataclass(frozen=True)
class _Pending:
    correlation_id: str
    review_reasons: list[str]
    kind: Literal["pending"] = "pending"


WardenVerdict = _Allow | _Deny | _Pending


@dataclass(frozen=True)
class WardenPendingView:
    """`GET /pending/{id}` response shape — mirrors `PendingView` in warden-lite."""

    correlation_id: str
    agent_id: str
    tool_type: str
    method: str
    review_reasons: list[str]
    requested_at: str
    decided_at: str | None
    decision: Literal["allow", "deny"] | None
    decider_note: str | None


async def inspect_tool_use(
    tool_call: NormalizedToolCall,
    opts: WardenOptions,
    *,
    client: httpx.AsyncClient | None = None,
) -> WardenVerdict:
    """Submit one normalized tool call to warden-lite for inspection.

    Wire contract: `POST {endpoint}/mcp` with a JSON-RPC 2.0 envelope.
    Server: `warden-lite/src/proxy.rs::handle_mcp`.

    Retry semantics: network failures and 5xx retry up to
    `opts.retry.max_attempts` with jittered exponential backoff. 200,
    403, and other 4xx never retry. Pass `client` to share a connection
    pool across many inspections; omit to mint a single-shot one.
    """
    retry = opts.retry
    if retry.max_attempts < 1:
        raise WardenTransportError(
            f"retry.max_attempts must be >= 1, got {retry.max_attempts}"
        )
    last_err: WardenTransportError | None = None
    for attempt in range(retry.max_attempts):
        try:
            return await _inspect_single_attempt(tool_call, opts, client)
        except WardenTransportError as e:
            last_err = e
            if not _is_retriable(e) or attempt == retry.max_attempts - 1:
                raise
            await asyncio.sleep(_backoff_s(retry.base_delay_s, attempt))
    raise last_err or WardenTransportError("warden inspect: no attempts ran")


async def _inspect_single_attempt(
    tool_call: NormalizedToolCall,
    opts: WardenOptions,
    client: httpx.AsyncClient | None,
) -> WardenVerdict:
    body = _inspect_body(tool_call)
    headers = _inspect_headers(opts)
    url = _join_url(opts.endpoint, "/mcp")
    owned: httpx.AsyncClient | None = None
    if client is None:
        owned = httpx.AsyncClient(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = await client.post(url, json=body, headers=headers, timeout=opts.timeout_s)
        except httpx.TimeoutException as e:
            raise WardenTransportError(
                f"warden inspect timed out after {opts.timeout_s}s"
            ) from e
        except httpx.HTTPError as e:
            raise WardenTransportError(f"warden inspect failed: {e}") from e
    finally:
        if owned is not None:
            await owned.aclose()

    return _parse_inspect_response(response)


def inspect_tool_use_sync(
    tool_call: NormalizedToolCall,
    opts: WardenOptions,
    *,
    client: httpx.Client | None = None,
) -> WardenVerdict:
    """Sync mirror of `inspect_tool_use` for partners wrapping
    `anthropic.Anthropic` / `openai.OpenAI`.

    Same retry semantics as the async path, with `time.sleep` between
    attempts. Pass `client` to share a connection pool.
    """
    retry = opts.retry
    if retry.max_attempts < 1:
        raise WardenTransportError(
            f"retry.max_attempts must be >= 1, got {retry.max_attempts}"
        )
    last_err: WardenTransportError | None = None
    for attempt in range(retry.max_attempts):
        try:
            return _inspect_single_attempt_sync(tool_call, opts, client)
        except WardenTransportError as e:
            last_err = e
            if not _is_retriable(e) or attempt == retry.max_attempts - 1:
                raise
            time.sleep(_backoff_s(retry.base_delay_s, attempt))
    raise last_err or WardenTransportError("warden inspect: no attempts ran")


def _inspect_single_attempt_sync(
    tool_call: NormalizedToolCall,
    opts: WardenOptions,
    client: httpx.Client | None,
) -> WardenVerdict:
    body = _inspect_body(tool_call)
    headers = _inspect_headers(opts)
    url = _join_url(opts.endpoint, "/mcp")
    owned: httpx.Client | None = None
    if client is None:
        owned = httpx.Client(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = client.post(url, json=body, headers=headers, timeout=opts.timeout_s)
        except httpx.TimeoutException as e:
            raise WardenTransportError(
                f"warden inspect timed out after {opts.timeout_s}s"
            ) from e
        except httpx.HTTPError as e:
            raise WardenTransportError(f"warden inspect failed: {e}") from e
    finally:
        if owned is not None:
            owned.close()

    return _parse_inspect_response(response)


def _inspect_body(tool_call: NormalizedToolCall) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_call.name, "arguments": tool_call.input},
        "id": tool_call.id,
    }


def _inspect_headers(opts: WardenOptions) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **opts.extra_headers}
    if opts.token:
        headers["Authorization"] = f"Bearer {opts.token}"
    return headers


def _parse_inspect_response(response: httpx.Response) -> WardenVerdict:
    correlation_id = response.headers.get(CORRELATION_HEADER)

    if response.status_code == 200:
        return _Allow(correlation_id=correlation_id)

    if response.status_code == 403:
        payload = _parse_deny_body(response)
        return _Deny(
            reasons=payload["reasons"],
            review_reasons=payload["review_reasons"],
            intent_category=payload["intent_category"],
            correlation_id=correlation_id,
        )

    if response.status_code == 202:
        payload = _parse_pending_body(response)
        corr = correlation_id or payload["correlation_id"]
        if not corr:
            raise WardenTransportError(
                "warden 202 missing correlation id (header and body both empty)",
                status=202,
            )
        return _Pending(
            correlation_id=corr,
            review_reasons=payload["review_reasons"],
        )

    text = _safe_text(response)
    raise WardenTransportError(
        f"warden inspect: unexpected status {response.status_code}"
        + (f": {text}" if text else ""),
        status=response.status_code,
    )


def _is_retriable(e: WardenTransportError) -> bool:
    # No status → fetch itself rejected (DNS, ECONNREFUSED, abort). Retry.
    # 5xx → server error, retry. Everything else (401, 404, 400) is a
    # config error — retrying won't help.
    if e.status is None:
        return True
    return 500 <= e.status < 600


def _backoff_s(base_s: float, attempt: int) -> float:
    # Exponential with full jitter: random in [base*2^attempt/2, base*2^attempt].
    ceiling: float = base_s * (2 ** attempt)
    return float(ceiling * (0.5 + random.random() * 0.5))


async def poll_pending_once(
    correlation_id: str,
    opts: WardenOptions,
    *,
    client: httpx.AsyncClient | None = None,
) -> WardenPendingView:
    """Single `GET /pending/{correlation_id}` poll.

    Returns the parsed view; the caller's polling loop branches on
    `decision`. 404 and 401 are terminal and surface as
    `WardenTransportError`. 5xx + network failures also raise — the
    `WardenPending.resolve` loop catches and retries those between
    polls.
    """
    headers: dict[str, str] = dict(opts.extra_headers)
    if opts.token:
        headers["Authorization"] = f"Bearer {opts.token}"

    url = _join_url(opts.endpoint, f"/pending/{correlation_id}")
    owned: httpx.AsyncClient | None = None
    if client is None:
        owned = httpx.AsyncClient(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = await client.get(url, headers=headers, timeout=opts.timeout_s)
        except httpx.TimeoutException as e:
            raise WardenTransportError(
                f"warden poll timed out after {opts.timeout_s}s"
            ) from e
        except httpx.HTTPError as e:
            raise WardenTransportError(f"warden poll failed: {e}") from e
    finally:
        if owned is not None:
            await owned.aclose()

    if response.status_code == 200:
        return _parse_pending_view(response)
    text = _safe_text(response)
    raise WardenTransportError(
        f"warden poll: unexpected status {response.status_code}"
        + (f": {text}" if text else ""),
        status=response.status_code,
    )


def poll_pending_once_sync(
    correlation_id: str,
    opts: WardenOptions,
    *,
    client: httpx.Client | None = None,
) -> WardenPendingView:
    """Sync mirror of `poll_pending_once`. Used by `WardenPending.resolve_sync`."""
    headers: dict[str, str] = dict(opts.extra_headers)
    if opts.token:
        headers["Authorization"] = f"Bearer {opts.token}"

    url = _join_url(opts.endpoint, f"/pending/{correlation_id}")
    owned: httpx.Client | None = None
    if client is None:
        owned = httpx.Client(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = client.get(url, headers=headers, timeout=opts.timeout_s)
        except httpx.TimeoutException as e:
            raise WardenTransportError(
                f"warden poll timed out after {opts.timeout_s}s"
            ) from e
        except httpx.HTTPError as e:
            raise WardenTransportError(f"warden poll failed: {e}") from e
    finally:
        if owned is not None:
            owned.close()

    if response.status_code == 200:
        return _parse_pending_view(response)
    text = _safe_text(response)
    raise WardenTransportError(
        f"warden poll: unexpected status {response.status_code}"
        + (f": {text}" if text else ""),
        status=response.status_code,
    )


def _parse_deny_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as e:
        raise WardenTransportError(
            f"warden 403 with unparseable body: {e}", status=403
        ) from e
    if not isinstance(body, dict):
        raise WardenTransportError(
            f"warden 403 with unexpected body shape: {body!r}", status=403
        )
    if (
        body.get("error") != "security_violation"
        or not isinstance(body.get("reasons"), list)
        or not isinstance(body.get("review_reasons"), list)
        or not isinstance(body.get("intent_category"), str)
    ):
        raise WardenTransportError(
            f"warden 403 with unexpected body shape: {body!r}", status=403
        )
    return body


def _parse_pending_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as e:
        raise WardenTransportError(
            f"warden 202 with unparseable body: {e}", status=202
        ) from e
    if not isinstance(body, dict):
        raise WardenTransportError(
            f"warden 202 with unexpected body shape: {body!r}", status=202
        )
    if (
        body.get("status") != "pending"
        or not isinstance(body.get("correlation_id"), str)
        or not isinstance(body.get("review_reasons"), list)
    ):
        raise WardenTransportError(
            f"warden 202 with unexpected body shape: {body!r}", status=202
        )
    return body


def _parse_pending_view(response: httpx.Response) -> WardenPendingView:
    try:
        body = response.json()
    except ValueError as e:
        raise WardenTransportError(
            f"warden poll with unparseable body: {e}", status=response.status_code
        ) from e
    if not isinstance(body, dict):
        raise WardenTransportError(
            f"warden poll with unexpected body shape: {body!r}",
            status=response.status_code,
        )
    decision = body.get("decision")
    if decision not in (None, "allow", "deny"):
        raise WardenTransportError(
            f"warden poll: unrecognized decision {decision!r}",
            status=response.status_code,
        )
    return WardenPendingView(
        correlation_id=body["correlation_id"],
        agent_id=body["agent_id"],
        tool_type=body["tool_type"],
        method=body["method"],
        review_reasons=body["review_reasons"],
        requested_at=body["requested_at"],
        decided_at=body.get("decided_at"),
        decision=decision,
        decider_note=body.get("decider_note"),
    )


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:
        return ""


def _join_url(base: str, path: str) -> str:
    """Same `joinUrl` semantics as the TS SDK — drops a trailing slash
    on base and a leading slash on path. Does NOT use `urllib.parse.urljoin`
    because that drops the base path for absolute-looking paths.
    """
    b = base.rstrip("/")
    p = path.lstrip("/")
    return f"{b}/{p}"
