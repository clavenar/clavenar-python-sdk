"""HTTP transport for clavenar-lite. Submits one normalized tool call,
parses the verdict (allow / deny / pending / rate-limited), and surfaces
correlation ids for ledger lookups.

Both async and sync flavours live here. The sync flavour exists for
partners wrapping `anthropic.Anthropic` / `openai.OpenAI` (the
non-async SDK clients) — the wrap pattern can't transparently jump
out of a sync caller into an event loop, so a parallel sync transport
is the cleanest seam.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote, urlparse

import httpx

from clavenar_agent_sdk.errors import ClavenarConfigError, ClavenarTransportError
from clavenar_agent_sdk.options import ClavenarOptions

CORRELATION_HEADER = "x-clavenar-correlation-id"
DECISION_CONTRACT = "clavenar.decision/v1"
DECISION_CONTRACT_HEADER = "x-clavenar-decision-contract"
IDEMPOTENCY_ID_HEADER = "x-clavenar-idempotency-id"

MAX_RETRY_ATTEMPTS = 10
MAX_RETRY_DELAY_S = 60.0
MAX_TIMEOUT_S = 300.0
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ERROR_PREVIEW_BYTES = 4 * 1024
MAX_TOOL_ARGUMENT_BYTES = 1024 * 1024
MAX_BATCH_REQUEST_BYTES = 4 * 1024 * 1024
MAX_IDENTIFIER_BYTES = 1024

_RESERVED_HEADERS = {
    "authorization",
    "content-length",
    "content-type",
    DECISION_CONTRACT_HEADER,
    IDEMPOTENCY_ID_HEADER,
}


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
    layer: str | None = None
    correlation_id: str | None = None
    # Verbose-verdict per-detector breakdown, present only when the
    # gateway runs with CLAVENAR_PROXY_VERBOSE_VERDICTS=true. Shape:
    # {"detectors": [{"detector", "score", "flagged"?}], "degraded": [..]}.
    detail: dict[str, Any] | None = None
    kind: Literal["deny"] = "deny"


@dataclass(frozen=True)
class _Pending:
    correlation_id: str
    review_reasons: list[str]
    kind: Literal["pending"] = "pending"


@dataclass(frozen=True)
class _RateLimited:
    code: Literal["rate_limited", "quota_exceeded"]
    reasons: list[str]
    # Seconds to wait before retrying; None on quota_exceeded.
    retry_after_secs: int | None = None
    layer: str | None = None
    correlation_id: str | None = None
    kind: Literal["rate_limited"] = "rate_limited"


ClavenarVerdict = _Allow | _Deny | _Pending | _RateLimited


@dataclass(frozen=True)
class ClavenarPendingView:
    """`GET /pending/{id}` response shape — mirrors `PendingView` in clavenar-lite."""

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
    opts: ClavenarOptions,
    *,
    client: httpx.AsyncClient | None = None,
) -> ClavenarVerdict:
    """Submit one normalized tool call to clavenar-lite for inspection.

    Wire contract: `POST {endpoint}/mcp` with a JSON-RPC 2.0 envelope.
    Server: `clavenar-lite/src/proxy.rs::handle_mcp`.

    Retry semantics: network failures and 5xx retry up to
    `opts.retry.max_attempts` with jittered exponential backoff. 200,
    403, and 429 are verdicts and never retry (429 carries
    `retry_after_secs` for the caller to honor); other 4xx never retry
    either. Pass `client` to share a connection pool across many
    inspections; omit to mint a single-shot one.
    """
    validate_transport_options(opts)
    _validate_tool_call(tool_call)
    idempotency_id = str(uuid.uuid4())
    return await _inspect_decision(
        _inspect_body(tool_call, idempotency_id), idempotency_id, opts, client
    )


async def inspect_tool_uses(
    tool_calls: list[NormalizedToolCall],
    opts: ClavenarOptions,
    *,
    client: httpx.AsyncClient | None = None,
) -> ClavenarVerdict:
    """Submit one ordered atomic decision for a complete provider turn."""
    validate_transport_options(opts)
    idempotency_id = str(uuid.uuid4())
    return await _inspect_decision(
        _atomic_batch_body(tool_calls, idempotency_id), idempotency_id, opts, client
    )


async def _inspect_decision(
    body: dict[str, Any],
    idempotency_id: str,
    opts: ClavenarOptions,
    client: httpx.AsyncClient | None,
) -> ClavenarVerdict:
    retry = opts.retry
    request_bytes = _serialize_request(body)
    last_err: ClavenarTransportError | None = None
    for attempt in range(retry.max_attempts):
        try:
            return await _inspect_single_attempt(request_bytes, idempotency_id, opts, client)
        except ClavenarTransportError as e:
            last_err = e
            if not _is_retriable(e) or attempt == retry.max_attempts - 1:
                raise
            await asyncio.sleep(_backoff_s(retry.base_delay_s, attempt))
    raise last_err or ClavenarTransportError("clavenar inspect: no attempts ran")


async def _inspect_single_attempt(
    body: bytes,
    idempotency_id: str,
    opts: ClavenarOptions,
    client: httpx.AsyncClient | None,
) -> ClavenarVerdict:
    if client is not None and opts.transport_profile is not None:
        raise ClavenarTransportError(
            "transport_profile cannot be combined with an injected HTTP client"
        )
    headers = _inspect_headers(opts, idempotency_id)
    url = _join_url(opts.endpoint, "/mcp")
    owned: httpx.AsyncClient | None = None
    if client is None:
        if opts.transport_profile is not None:
            client = opts.transport_profile.async_client()
        else:
            owned = httpx.AsyncClient(timeout=opts.timeout_s)
            client = owned
    timeout_s = _request_timeout(opts)
    try:
        response = await _request_bounded_async(
            client,
            "POST",
            url,
            headers=headers,
            content=body,
            timeout_s=timeout_s,
            operation="inspect",
        )
    finally:
        if owned is not None:
            await owned.aclose()

    return _parse_inspect_response(response)


def inspect_tool_use_sync(
    tool_call: NormalizedToolCall,
    opts: ClavenarOptions,
    *,
    client: httpx.Client | None = None,
) -> ClavenarVerdict:
    """Sync mirror of `inspect_tool_use` for partners wrapping
    `anthropic.Anthropic` / `openai.OpenAI`.

    Same retry semantics as the async path, with `time.sleep` between
    attempts. Pass `client` to share a connection pool.
    """
    validate_transport_options(opts)
    _validate_tool_call(tool_call)
    idempotency_id = str(uuid.uuid4())
    return _inspect_decision_sync(
        _inspect_body(tool_call, idempotency_id), idempotency_id, opts, client
    )


def inspect_tool_uses_sync(
    tool_calls: list[NormalizedToolCall],
    opts: ClavenarOptions,
    *,
    client: httpx.Client | None = None,
) -> ClavenarVerdict:
    """Sync mirror of :func:`inspect_tool_uses`."""
    validate_transport_options(opts)
    idempotency_id = str(uuid.uuid4())
    return _inspect_decision_sync(
        _atomic_batch_body(tool_calls, idempotency_id), idempotency_id, opts, client
    )


def _inspect_decision_sync(
    body: dict[str, Any],
    idempotency_id: str,
    opts: ClavenarOptions,
    client: httpx.Client | None,
) -> ClavenarVerdict:
    retry = opts.retry
    request_bytes = _serialize_request(body)
    last_err: ClavenarTransportError | None = None
    for attempt in range(retry.max_attempts):
        try:
            return _inspect_single_attempt_sync(request_bytes, idempotency_id, opts, client)
        except ClavenarTransportError as e:
            last_err = e
            if not _is_retriable(e) or attempt == retry.max_attempts - 1:
                raise
            time.sleep(_backoff_s(retry.base_delay_s, attempt))
    raise last_err or ClavenarTransportError("clavenar inspect: no attempts ran")


def _inspect_single_attempt_sync(
    body: bytes,
    idempotency_id: str,
    opts: ClavenarOptions,
    client: httpx.Client | None,
) -> ClavenarVerdict:
    if client is not None and opts.transport_profile is not None:
        raise ClavenarTransportError(
            "transport_profile cannot be combined with an injected HTTP client"
        )
    headers = _inspect_headers(opts, idempotency_id)
    url = _join_url(opts.endpoint, "/mcp")
    owned: httpx.Client | None = None
    if client is None:
        if opts.transport_profile is not None:
            client = opts.transport_profile.client()
        else:
            owned = httpx.Client(timeout=opts.timeout_s)
            client = owned
    timeout_s = _request_timeout(opts)
    try:
        response = _request_bounded_sync(
            client,
            "POST",
            url,
            headers=headers,
            content=body,
            timeout_s=timeout_s,
            operation="inspect",
        )
    finally:
        if owned is not None:
            owned.close()

    return _parse_inspect_response(response)


async def _request_bounded_async(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout_s: float,
    operation: str,
    content: bytes | None = None,
) -> httpx.Response:
    try:
        async with client.stream(
            method,
            url,
            headers=headers,
            content=content,
            timeout=timeout_s,
        ) as response:
            limit = _response_limit(response.status_code)
            _reject_oversized_content_length(response, limit, operation)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > limit:
                    raise ClavenarTransportError(
                        f"clavenar {operation}: response exceeded {limit} bytes",
                        status=response.status_code,
                    )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=response.request,
            )
    except ClavenarTransportError:
        raise
    except httpx.TimeoutException as error:
        raise ClavenarTransportError(
            f"clavenar {operation} timed out after {timeout_s}s"
        ) from error
    except httpx.HTTPError as error:
        raise ClavenarTransportError(f"clavenar {operation} failed: {error}") from error


def _request_bounded_sync(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout_s: float,
    operation: str,
    content: bytes | None = None,
) -> httpx.Response:
    try:
        with client.stream(
            method,
            url,
            headers=headers,
            content=content,
            timeout=timeout_s,
        ) as response:
            limit = _response_limit(response.status_code)
            _reject_oversized_content_length(response, limit, operation)
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > limit:
                    raise ClavenarTransportError(
                        f"clavenar {operation}: response exceeded {limit} bytes",
                        status=response.status_code,
                    )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(body),
                request=response.request,
            )
    except ClavenarTransportError:
        raise
    except httpx.TimeoutException as error:
        raise ClavenarTransportError(
            f"clavenar {operation} timed out after {timeout_s}s"
        ) from error
    except httpx.HTTPError as error:
        raise ClavenarTransportError(f"clavenar {operation} failed: {error}") from error


def _response_limit(status: int) -> int:
    return MAX_RESPONSE_BYTES if status in {200, 202, 403, 429} else MAX_ERROR_PREVIEW_BYTES


def _reject_oversized_content_length(response: httpx.Response, limit: int, operation: str) -> None:
    value = response.headers.get("content-length")
    if value is None:
        return
    try:
        size = int(value)
    except ValueError:
        return
    if size > limit:
        raise ClavenarTransportError(
            f"clavenar {operation}: response exceeded {limit} bytes",
            status=response.status_code,
        )


def _serialize_request(body: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClavenarTransportError(
            f"clavenar inspect request is not valid JSON: {error}"
        ) from error
    if len(encoded) > MAX_BATCH_REQUEST_BYTES:
        raise ClavenarTransportError(
            f"clavenar inspect request exceeded {MAX_BATCH_REQUEST_BYTES} bytes"
        )
    return encoded


def _validate_tool_call(tool_call: NormalizedToolCall) -> None:
    if not tool_call.id or not tool_call.name:
        raise ClavenarTransportError("tool call requires non-empty id and name")
    for label, value in (("id", tool_call.id), ("name", tool_call.name)):
        if len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise ClavenarTransportError(f"tool call {label} exceeded {MAX_IDENTIFIER_BYTES} bytes")
    try:
        encoded = json.dumps(
            tool_call.input,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClavenarTransportError(
            f"tool call {tool_call.id} arguments are not valid JSON: {error}"
        ) from error
    if len(encoded) > MAX_TOOL_ARGUMENT_BYTES:
        raise ClavenarTransportError(
            f"tool call {tool_call.id} arguments exceeded {MAX_TOOL_ARGUMENT_BYTES} bytes"
        )


def _inspect_body(tool_call: NormalizedToolCall, idempotency_id: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_call.name, "arguments": tool_call.input},
        "id": idempotency_id,
    }


def _atomic_batch_body(tool_calls: list[NormalizedToolCall], idempotency_id: str) -> dict[str, Any]:
    if not 1 <= len(tool_calls) <= 128:
        raise ClavenarTransportError("atomic decision batch must contain 1..128 calls")
    ids = [call.id for call in tool_calls]
    if any(not call.id or not call.name for call in tool_calls) or len(ids) != len(set(ids)):
        raise ClavenarTransportError(
            "atomic decision batch requires unique non-empty call ids and names"
        )
    for call in tool_calls:
        _validate_tool_call(call)
    return {
        "jsonrpc": "2.0",
        "id": idempotency_id,
        "method": "clavenar/tools.batch",
        "params": {
            "name": "clavenar.atomic-batch",
            "arguments": {
                "contract": "clavenar.atomic-tool-call-batch/v1",
                "calls": [
                    {"id": call.id, "name": call.name, "arguments": call.input}
                    for call in tool_calls
                ],
            },
        },
    }


def _inspect_headers(opts: ClavenarOptions, idempotency_id: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        DECISION_CONTRACT_HEADER: DECISION_CONTRACT,
        IDEMPOTENCY_ID_HEADER: idempotency_id,
        **opts.extra_headers,
    }
    token = _transport_token(opts)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _transport_token(opts: ClavenarOptions) -> str | None:
    if opts.token is not None and opts.transport_profile is not None:
        raise ClavenarTransportError(
            "token cannot be combined with transport_profile token acquisition"
        )
    return opts.transport_profile.token() if opts.transport_profile is not None else opts.token


def _request_timeout(opts: ClavenarOptions) -> float:
    return (
        opts.transport_profile.request_timeout_s
        if opts.transport_profile is not None
        else opts.timeout_s
    )


def _parse_inspect_response(response: httpx.Response) -> ClavenarVerdict:
    correlation_id = response.headers.get(CORRELATION_HEADER)
    contract = response.headers.get(DECISION_CONTRACT_HEADER)
    if contract is not None and contract != DECISION_CONTRACT:
        raise ClavenarTransportError(
            f"clavenar inspect: unsupported decision contract {contract!r}",
            status=response.status_code,
        )

    if response.status_code == 200:
        if response.content.strip():
            try:
                payload = response.json()
            except ValueError as error:
                raise ClavenarTransportError(
                    f"clavenar 200 with unparseable body: {error}", status=200
                ) from error
            legacy_allow = payload == {"verdict": "allow"}
            contract_allow = (
                isinstance(payload, dict)
                and set(payload) == {"contract", "decision", "correlation_id", "executable"}
                and payload.get("contract") == DECISION_CONTRACT
                and payload.get("decision") == "allow"
                and isinstance(payload.get("correlation_id"), str)
                and bool(payload["correlation_id"])
                and payload.get("executable") is False
            )
            if not legacy_allow and not contract_allow:
                raise ClavenarTransportError(
                    f"clavenar 200 with unexpected body shape: {_safe_repr(payload)}",
                    status=200,
                )
            if contract_allow:
                body_correlation_id = payload["correlation_id"]
                if correlation_id and correlation_id != body_correlation_id:
                    raise ClavenarTransportError(
                        "clavenar 200 correlation id header/body mismatch", status=200
                    )
                correlation_id = correlation_id or body_correlation_id
        return _Allow(correlation_id=correlation_id)

    if response.status_code == 403:
        payload = _parse_deny_body(response)
        return _Deny(
            reasons=payload["reasons"],
            review_reasons=payload["review_reasons"],
            intent_category=payload["intent_category"],
            layer=payload.get("layer"),
            correlation_id=correlation_id,
            detail=payload.get("detail"),
        )

    if response.status_code == 202:
        payload = _parse_pending_body(response)
        body_correlation_id = payload["correlation_id"]
        if correlation_id and body_correlation_id and correlation_id != body_correlation_id:
            raise ClavenarTransportError(
                "clavenar 202 correlation id header/body mismatch",
                status=202,
            )
        corr = correlation_id or body_correlation_id
        if not corr:
            raise ClavenarTransportError(
                "clavenar 202 missing correlation id (header and body both empty)",
                status=202,
            )
        return _Pending(
            correlation_id=corr,
            review_reasons=payload["review_reasons"],
        )

    if response.status_code == 429:
        payload = _parse_rate_limit_body(response)
        return _RateLimited(
            code=payload["code"],
            reasons=payload["reasons"],
            retry_after_secs=payload["retry_after_secs"],
            layer=payload["layer"],
            correlation_id=correlation_id or payload["correlation_id"],
        )

    text = _safe_text(response)
    raise ClavenarTransportError(
        f"clavenar inspect: unexpected status {response.status_code}"
        + (f": {text}" if text else ""),
        status=response.status_code,
    )


def _is_retriable(e: ClavenarTransportError) -> bool:
    # No status → fetch itself rejected (DNS, ECONNREFUSED, abort). Retry.
    # 5xx → server error, retry. Everything else (401, 404, 400) is a
    # config error — retrying won't help.
    if e.status is None:
        return True
    return 500 <= e.status < 600


def _backoff_s(base_s: float, attempt: int) -> float:
    # Exponential with full jitter: random in [base*2^attempt/2, base*2^attempt].
    ceiling: float = min(MAX_RETRY_DELAY_S, base_s * (2**attempt))
    return float(ceiling * (0.5 + random.random() * 0.5))


async def poll_pending_once(
    correlation_id: str,
    opts: ClavenarOptions,
    *,
    client: httpx.AsyncClient | None = None,
) -> ClavenarPendingView:
    validate_transport_options(opts)
    _validate_identifier(correlation_id, "pending correlation id")
    if client is not None and opts.transport_profile is not None:
        raise ClavenarTransportError(
            "transport_profile cannot be combined with an injected HTTP client"
        )
    """Single `GET /pending/{correlation_id}` poll.

    Returns the parsed view; the caller's polling loop branches on
    `decision`. 404 and 401 are terminal and surface as
    `ClavenarTransportError`. 5xx + network failures also raise — the
    `ClavenarPending.resolve` loop catches and retries those between
    polls.
    """
    headers: dict[str, str] = dict(opts.extra_headers)
    token = _transport_token(opts)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = _join_url(opts.endpoint, f"/pending/{quote(correlation_id, safe='')}")
    owned: httpx.AsyncClient | None = None
    if client is None:
        if opts.transport_profile is not None:
            client = opts.transport_profile.async_client()
        else:
            owned = httpx.AsyncClient(timeout=opts.timeout_s)
            client = owned
    timeout_s = _request_timeout(opts)
    try:
        response = await _request_bounded_async(
            client,
            "GET",
            url,
            headers=headers,
            timeout_s=timeout_s,
            operation="poll",
        )
    finally:
        if owned is not None:
            await owned.aclose()

    if response.status_code == 200:
        return _parse_pending_view(response, correlation_id)
    text = _safe_text(response)
    raise ClavenarTransportError(
        f"clavenar poll: unexpected status {response.status_code}" + (f": {text}" if text else ""),
        status=response.status_code,
    )


def poll_pending_once_sync(
    correlation_id: str,
    opts: ClavenarOptions,
    *,
    client: httpx.Client | None = None,
) -> ClavenarPendingView:
    validate_transport_options(opts)
    _validate_identifier(correlation_id, "pending correlation id")
    if client is not None and opts.transport_profile is not None:
        raise ClavenarTransportError(
            "transport_profile cannot be combined with an injected HTTP client"
        )
    """Sync mirror of `poll_pending_once`. Used by `ClavenarPending.resolve_sync`."""
    headers: dict[str, str] = dict(opts.extra_headers)
    token = _transport_token(opts)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = _join_url(opts.endpoint, f"/pending/{quote(correlation_id, safe='')}")
    owned: httpx.Client | None = None
    if client is None:
        if opts.transport_profile is not None:
            client = opts.transport_profile.client()
        else:
            owned = httpx.Client(timeout=opts.timeout_s)
            client = owned
    timeout_s = _request_timeout(opts)
    try:
        response = _request_bounded_sync(
            client,
            "GET",
            url,
            headers=headers,
            timeout_s=timeout_s,
            operation="poll",
        )
    finally:
        if owned is not None:
            owned.close()

    if response.status_code == 200:
        return _parse_pending_view(response, correlation_id)
    text = _safe_text(response)
    raise ClavenarTransportError(
        f"clavenar poll: unexpected status {response.status_code}" + (f": {text}" if text else ""),
        status=response.status_code,
    )


def _parse_deny_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as e:
        raise ClavenarTransportError(f"clavenar 403 with unparseable body: {e}", status=403) from e
    if not isinstance(body, dict) or not isinstance(body.get("error"), str):
        raise ClavenarTransportError(
            f"clavenar 403 with unexpected body shape: {body!r}", status=403
        )
    # The shared envelope (lite + full-edition proxy) uses several error
    # codes / layers and omits empty review_reasons / absent
    # intent_category. Normalise so the caller sees the always-present
    # fields; keep `layer` when reported.
    return {
        "error": body["error"],
        "reasons": body["reasons"] if isinstance(body.get("reasons"), list) else [],
        "review_reasons": body["review_reasons"]
        if isinstance(body.get("review_reasons"), list)
        else [],
        "intent_category": body["intent_category"]
        if isinstance(body.get("intent_category"), str)
        else "",
        "layer": body["layer"] if isinstance(body.get("layer"), str) else None,
        # Optional verbose-verdict breakdown; kept as-is when a dict.
        "detail": body["detail"] if isinstance(body.get("detail"), dict) else None,
    }


def _parse_pending_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as e:
        raise ClavenarTransportError(f"clavenar 202 with unparseable body: {e}", status=202) from e
    if not isinstance(body, dict):
        raise ClavenarTransportError(
            f"clavenar 202 with unexpected body shape: {body!r}", status=202
        )
    if (
        body.get("status") != "pending"
        or not isinstance(body.get("correlation_id"), str)
        or not isinstance(body.get("review_reasons"), list)
    ):
        raise ClavenarTransportError(
            f"clavenar 202 with unexpected body shape: {body!r}", status=202
        )
    return body


def _parse_rate_limit_body(response: httpx.Response) -> dict[str, Any]:
    # Lenient like the deny parser: only the string `error` code is
    # required; the verdict falls back to `rate_limited` when the body
    # omits it (both codes ride HTTP 429).
    try:
        body = response.json()
    except ValueError as e:
        raise ClavenarTransportError(f"clavenar 429 with unparseable body: {e}", status=429) from e
    if not isinstance(body, dict) or not isinstance(body.get("error"), str):
        raise ClavenarTransportError(
            f"clavenar 429 with unexpected body shape: {body!r}", status=429
        )
    reasons = body.get("reasons")
    return {
        "code": "quota_exceeded" if body.get("verdict") == "quota_exceeded" else "rate_limited",
        "reasons": [s for s in reasons if isinstance(s, str)] if isinstance(reasons, list) else [],
        "retry_after_secs": body["retry_after_secs"]
        if isinstance(body.get("retry_after_secs"), int)
        and not isinstance(body.get("retry_after_secs"), bool)
        and body["retry_after_secs"] >= 0
        else None,
        "layer": body["layer"] if isinstance(body.get("layer"), str) else None,
        "correlation_id": body["correlation_id"]
        if isinstance(body.get("correlation_id"), str)
        else None,
    }


def _parse_pending_view(
    response: httpx.Response, expected_correlation_id: str
) -> ClavenarPendingView:
    try:
        body = response.json()
    except ValueError as e:
        raise ClavenarTransportError(
            f"clavenar poll with unparseable body: {e}", status=response.status_code
        ) from e
    if not isinstance(body, dict):
        raise ClavenarTransportError(
            f"clavenar poll with unexpected body shape: {body!r}",
            status=response.status_code,
        )
    required_strings = ("correlation_id", "agent_id", "tool_type", "method", "requested_at")
    if any(not isinstance(body.get(field), str) for field in required_strings) or not isinstance(
        body.get("review_reasons"), list
    ):
        raise ClavenarTransportError(
            f"clavenar poll with unexpected body shape: {_safe_repr(body)}",
            status=response.status_code,
        )
    if body["correlation_id"] != expected_correlation_id:
        raise ClavenarTransportError(
            "clavenar poll returned a different correlation id",
            status=response.status_code,
        )
    decision = body.get("decision")
    if decision not in (None, "allow", "deny"):
        raise ClavenarTransportError(
            f"clavenar poll: unrecognized decision {decision!r}",
            status=response.status_code,
        )
    if body.get("decided_at") is not None and not isinstance(body.get("decided_at"), str):
        raise ClavenarTransportError(
            "clavenar poll: decided_at must be a string or null",
            status=response.status_code,
        )
    if body.get("decider_note") is not None and not isinstance(body.get("decider_note"), str):
        raise ClavenarTransportError(
            "clavenar poll: decider_note must be a string or null",
            status=response.status_code,
        )
    return ClavenarPendingView(
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
        return response.content[:MAX_ERROR_PREVIEW_BYTES].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _safe_repr(value: Any) -> str:
    rendered = repr(value)
    return rendered[:MAX_ERROR_PREVIEW_BYTES]


def validate_transport_options(opts: ClavenarOptions) -> None:
    """Validate the complete transport policy before any network I/O."""
    if not isinstance(opts.endpoint, str) or not opts.endpoint:
        raise ClavenarConfigError("clavenar endpoint is required")
    try:
        parsed = urlparse(opts.endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ClavenarConfigError(f"clavenar endpoint is invalid: {error}") from error
    if parsed.scheme not in {"http", "https"} or hostname is None or port == 0:
        raise ClavenarConfigError("clavenar endpoint must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ClavenarConfigError("clavenar endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ClavenarConfigError("clavenar endpoint must not contain a query or fragment")

    if opts.token is not None and opts.transport_profile is not None:
        raise ClavenarConfigError(
            "token cannot be combined with transport_profile token acquisition"
        )
    if opts.token is not None and (
        not opts.token.strip() or "\r" in opts.token or "\n" in opts.token
    ):
        raise ClavenarConfigError("clavenar token must be non-empty and single-line")
    has_credentials = opts.token is not None or opts.transport_profile is not None
    if has_credentials and parsed.scheme != "https":
        is_explicit_loopback = False
        try:
            address = ipaddress.ip_address(hostname)
            is_explicit_loopback = address in {
                ipaddress.ip_address("127.0.0.1"),
                ipaddress.ip_address("::1"),
            }
        except ValueError:
            pass
        if not opts.allow_insecure_loopback or not is_explicit_loopback:
            raise ClavenarConfigError(
                "clavenar credentials require HTTPS; set allow_insecure_loopback only "
                "for an explicit 127.0.0.1 or ::1 development endpoint"
            )

    if (
        isinstance(opts.timeout_s, bool)
        or not isinstance(opts.timeout_s, (int, float))
        or not math.isfinite(opts.timeout_s)
        or not 0 < opts.timeout_s <= MAX_TIMEOUT_S
    ):
        raise ClavenarConfigError(f"clavenar timeout_s must be finite and in (0, {MAX_TIMEOUT_S}]")
    if (
        isinstance(opts.retry.max_attempts, bool)
        or not isinstance(opts.retry.max_attempts, int)
        or not 1 <= opts.retry.max_attempts <= MAX_RETRY_ATTEMPTS
    ):
        raise ClavenarConfigError(f"retry.max_attempts must be in [1, {MAX_RETRY_ATTEMPTS}]")
    if (
        isinstance(opts.retry.base_delay_s, bool)
        or not isinstance(opts.retry.base_delay_s, (int, float))
        or not math.isfinite(opts.retry.base_delay_s)
        or not 0 <= opts.retry.base_delay_s <= MAX_RETRY_DELAY_S
    ):
        raise ClavenarConfigError(
            f"retry.base_delay_s must be finite and in [0, {MAX_RETRY_DELAY_S}]"
        )
    if opts.mode not in {"enforce", "observe"}:
        raise ClavenarConfigError("clavenar mode must be 'enforce' or 'observe'")
    for name, value in opts.extra_headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ClavenarConfigError("extra header names and values must be strings")
        lower_name = name.lower()
        if not name or lower_name in _RESERVED_HEADERS:
            raise ClavenarConfigError(f"extra header {name!r} is reserved or empty")
        if any(char in name or char in value for char in ("\r", "\n")):
            raise ClavenarConfigError("extra header names and values must be single-line")


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ClavenarConfigError(f"{label} must be non-empty")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise ClavenarConfigError(f"{label} exceeded {MAX_IDENTIFIER_BYTES} bytes")


def _join_url(base: str, path: str) -> str:
    """Same `joinUrl` semantics as the TS SDK — drops a trailing slash
    on base and a leading slash on path. Does NOT use `urllib.parse.urljoin`
    because that drops the base path for absolute-looking paths.
    """
    b = base.rstrip("/")
    p = path.lstrip("/")
    return f"{b}/{p}"
