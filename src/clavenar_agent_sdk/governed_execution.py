"""Verified, recoverable governed execution for registered provider effects."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import ipaddress
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import httpx

from clavenar_agent_sdk.errors import (
    ClavenarConfigError,
    ClavenarRecoveryRequired,
    ClavenarTransportError,
)

DECISION_CONTRACT = "clavenar.decision/v1"
EXECUTION_CONTRACT = "clavenar.execution/v1"
DURABLE_EXECUTION_CONTRACT = "clavenar.sdk-durable-intent-outbox/v1"

MAX_RETRY_ATTEMPTS = 10
MAX_RETRY_DELAY_S = 60.0
MAX_TIMEOUT_S = 300.0
MAX_RESPONSE_BYTES = 1 << 20
MAX_ERROR_PREVIEW_BYTES = 4 << 10
MAX_SAFE_INTEGER = 9_007_199_254_740_991
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class PreparedToolRequest:
    """Serializable exact tool request with identity allocated before network access."""

    idempotency_id: str
    name: str
    arguments: Any

    @classmethod
    def new(cls, name: str, arguments: Any) -> PreparedToolRequest:
        return cls.restore(str(uuid.uuid4()), name, arguments)

    @classmethod
    def restore(cls, idempotency_id: str, name: str, arguments: Any) -> PreparedToolRequest:
        prepared = cls(idempotency_id=idempotency_id, name=name, arguments=arguments)
        _validate_prepared(prepared)
        return prepared


@dataclass(frozen=True)
class ToolExecutionRequest:
    authorization_id: str
    idempotency_id: str
    executor_id: str
    execution_payload: Any


@dataclass(frozen=True)
class ExecutionEffect:
    result: Any
    effect_id: str


@dataclass(frozen=True)
class ExecutionState:
    """Recoverable durable state for one stable idempotency id."""

    intent: JsonObject | None = None
    completion: JsonObject | None = None


@dataclass(frozen=True)
class GovernedExecutionOutcome:
    result: Any
    effect_id: str
    idempotency_id: str
    receipt: JsonObject


class AsyncDurableExecutionStore(Protocol):
    """Atomic intent and completion/outbox authority owned by the application."""

    async def load_execution(self, idempotency_id: str) -> ExecutionState: ...

    async def commit_intent(self, intent: JsonObject) -> None: ...

    async def commit_completion_and_enqueue_receipt(self, completion: JsonObject) -> None: ...


class SyncDurableExecutionStore(Protocol):
    """Synchronous form of the recoverable durable execution store."""

    def load_execution(self, idempotency_id: str) -> ExecutionState: ...

    def commit_intent(self, intent: JsonObject) -> None: ...

    def commit_completion_and_enqueue_receipt(self, completion: JsonObject) -> None: ...


AsyncExecutor = Callable[[ToolExecutionRequest], Awaitable[ExecutionEffect]]
AsyncEffectRecoverer = Callable[[JsonObject], Awaitable[ExecutionEffect | None]]
AsyncAuthorizationVerifier = Callable[[JsonObject], Awaitable[None] | None]
AsyncReceiptSigner = Callable[[JsonObject], Awaitable[dict[str, str]]]
SyncExecutor = Callable[[ToolExecutionRequest], ExecutionEffect]
SyncEffectRecoverer = Callable[[JsonObject], ExecutionEffect | None]
SyncAuthorizationVerifier = Callable[[JsonObject], None]
SyncReceiptSigner = Callable[[JsonObject], dict[str, str]]


@dataclass(frozen=True)
class AsyncGovernedExecutionOptions:
    endpoint: str
    executor_id: str
    executor: AsyncExecutor
    durable_store: AsyncDurableExecutionStore
    verify_authorization: AsyncAuthorizationVerifier
    sign_receipt: AsyncReceiptSigner
    recover_effect: AsyncEffectRecoverer | None = None
    token: str | None = None
    allow_insecure_loopback: bool = False
    timeout_s: float = 10.0
    finalization_timeout_s: float = 30.0
    max_attempts: int = 3
    base_delay_s: float = 0.1
    client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class SyncGovernedExecutionOptions:
    endpoint: str
    executor_id: str
    executor: SyncExecutor
    durable_store: SyncDurableExecutionStore
    verify_authorization: SyncAuthorizationVerifier
    sign_receipt: SyncReceiptSigner
    recover_effect: SyncEffectRecoverer | None = None
    token: str | None = None
    allow_insecure_loopback: bool = False
    timeout_s: float = 10.0
    finalization_timeout_s: float = 30.0
    max_attempts: int = 3
    base_delay_s: float = 0.1
    client: httpx.Client | None = None


async def execute_tool(
    name: str, arguments: Any, opts: AsyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    return await execute_prepared_tool(PreparedToolRequest.new(name, arguments), opts)


async def execute_prepared_tool(
    prepared: PreparedToolRequest, opts: AsyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    _validate_prepared(prepared)
    _validate_options(opts)
    body = _tool_body(prepared)
    state = await opts.durable_store.load_execution(prepared.idempotency_id)
    if not isinstance(state, ExecutionState):
        raise ClavenarConfigError("durable store returned an invalid execution state")
    if state.completion is not None:
        return await _recovered_completion_async(prepared, body, state, opts)
    if state.intent is not None:
        intent = cast(JsonObject, _clone_json(state.intent))
        auth = await _validate_stored_intent_async(intent, prepared, body, opts)
        if opts.recover_effect is None:
            raise ClavenarRecoveryRequired(prepared.idempotency_id)
        effect = await opts.recover_effect(cast(JsonObject, _clone_json(intent)))
        if effect is None:
            raise ClavenarRecoveryRequired(prepared.idempotency_id)
        return await _complete_async(intent["authorization"], auth, effect, opts)

    signed = await _request_authorization_async(body, prepared.idempotency_id, opts)
    auth = _validate_authorization(signed, prepared, body)
    await _verify_async(opts.verify_authorization, signed, stored=False)
    intent = _execution_intent(signed, auth, opts.executor_id)
    await opts.durable_store.commit_intent(cast(JsonObject, _clone_json(intent)))
    effect = await opts.executor(_execution_request(auth, opts.executor_id))
    return await _complete_async(signed, auth, effect, opts)


def execute_tool_sync(
    name: str, arguments: Any, opts: SyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    return execute_prepared_tool_sync(PreparedToolRequest.new(name, arguments), opts)


def execute_prepared_tool_sync(
    prepared: PreparedToolRequest, opts: SyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    _validate_prepared(prepared)
    _validate_options(opts)
    body = _tool_body(prepared)
    state = opts.durable_store.load_execution(prepared.idempotency_id)
    if not isinstance(state, ExecutionState):
        raise ClavenarConfigError("durable store returned an invalid execution state")
    if state.completion is not None:
        return _recovered_completion_sync(prepared, body, state, opts)
    if state.intent is not None:
        intent = cast(JsonObject, _clone_json(state.intent))
        auth = _validate_stored_intent_sync(intent, prepared, body, opts)
        if opts.recover_effect is None:
            raise ClavenarRecoveryRequired(prepared.idempotency_id)
        effect = opts.recover_effect(cast(JsonObject, _clone_json(intent)))
        if effect is None:
            raise ClavenarRecoveryRequired(prepared.idempotency_id)
        return _complete_sync(intent["authorization"], auth, effect, opts)

    signed = _request_authorization_sync(body, prepared.idempotency_id, opts)
    auth = _validate_authorization(signed, prepared, body)
    _verify_sync(opts.verify_authorization, signed, stored=False)
    intent = _execution_intent(signed, auth, opts.executor_id)
    opts.durable_store.commit_intent(cast(JsonObject, _clone_json(intent)))
    effect = opts.executor(_execution_request(auth, opts.executor_id))
    return _complete_sync(signed, auth, effect, opts)


async def _complete_async(
    signed: JsonObject,
    auth: JsonObject,
    effect: ExecutionEffect,
    opts: AsyncGovernedExecutionOptions,
) -> GovernedExecutionOutcome:
    stable_effect = _stable_effect(effect)
    unsigned = _unsigned_receipt(signed, auth, stable_effect)
    signature = await _critical_async(
        opts.sign_receipt(cast(JsonObject, _clone_json(unsigned))),
        opts.finalization_timeout_s,
        "receipt signing",
    )
    completion, receipt = _completion(unsigned, signature, auth, stable_effect, opts.executor_id)
    await _critical_async(
        opts.durable_store.commit_completion_and_enqueue_receipt(
            cast(JsonObject, _clone_json(completion))
        ),
        opts.finalization_timeout_s,
        "durable completion",
    )
    return GovernedExecutionOutcome(
        result=_clone_json(stable_effect.result),
        effect_id=stable_effect.effect_id,
        idempotency_id=cast(str, auth["idempotency_id"]),
        receipt=receipt,
    )


def _complete_sync(
    signed: JsonObject,
    auth: JsonObject,
    effect: ExecutionEffect,
    opts: SyncGovernedExecutionOptions,
) -> GovernedExecutionOutcome:
    stable_effect = _stable_effect(effect)
    unsigned = _unsigned_receipt(signed, auth, stable_effect)
    signature = _critical_sync(
        lambda: opts.sign_receipt(cast(JsonObject, _clone_json(unsigned))),
        opts.finalization_timeout_s,
        "receipt signing",
    )
    completion, receipt = _completion(unsigned, signature, auth, stable_effect, opts.executor_id)
    _critical_sync(
        lambda: opts.durable_store.commit_completion_and_enqueue_receipt(
            cast(JsonObject, _clone_json(completion))
        ),
        opts.finalization_timeout_s,
        "durable completion",
    )
    return GovernedExecutionOutcome(
        result=_clone_json(stable_effect.result),
        effect_id=stable_effect.effect_id,
        idempotency_id=cast(str, auth["idempotency_id"]),
        receipt=receipt,
    )


async def _validate_stored_intent_async(
    intent: JsonObject,
    prepared: PreparedToolRequest,
    body: JsonObject,
    opts: AsyncGovernedExecutionOptions,
) -> JsonObject:
    auth = _validate_stored_intent(intent, prepared, body, opts.executor_id)
    await _verify_async(opts.verify_authorization, cast(JsonObject, intent["authorization"]), True)
    return auth


def _validate_stored_intent_sync(
    intent: JsonObject,
    prepared: PreparedToolRequest,
    body: JsonObject,
    opts: SyncGovernedExecutionOptions,
) -> JsonObject:
    auth = _validate_stored_intent(intent, prepared, body, opts.executor_id)
    _verify_sync(opts.verify_authorization, cast(JsonObject, intent["authorization"]), True)
    return auth


def _validate_stored_intent(
    intent: JsonObject,
    prepared: PreparedToolRequest,
    body: JsonObject,
    executor_id: str,
) -> JsonObject:
    if (
        intent.get("contract") != DURABLE_EXECUTION_CONTRACT
        or intent.get("stage") != "execution.intent"
        or intent.get("idempotency_id") != prepared.idempotency_id
        or intent.get("executor_id") != executor_id
        or not isinstance(intent.get("authorization"), dict)
    ):
        raise ClavenarConfigError("stored execution intent does not match the prepared request")
    signed = cast(JsonObject, intent["authorization"])
    auth = _validate_authorization(signed, prepared, body)
    bindings = {
        "authorization_id": "authorization_id",
        "tenant": "tenant",
        "workload_id": "agent_id",
        "workload_spiffe": "agent_spiffe",
        "payload_sha256": "payload_sha256",
    }
    if any(intent.get(left) != auth.get(right) for left, right in bindings.items()):
        raise ClavenarConfigError("stored execution intent changed an authorization binding")
    return auth


async def _recovered_completion_async(
    prepared: PreparedToolRequest,
    body: JsonObject,
    state: ExecutionState,
    opts: AsyncGovernedExecutionOptions,
) -> GovernedExecutionOutcome:
    if state.intent is None or state.completion is None:
        raise ClavenarConfigError("durable completion is missing its execution intent")
    auth = await _validate_stored_intent_async(state.intent, prepared, body, opts)
    return _validate_completion(
        state.completion,
        cast(JsonObject, state.intent["authorization"]),
        auth,
        prepared,
        opts.executor_id,
    )


def _recovered_completion_sync(
    prepared: PreparedToolRequest,
    body: JsonObject,
    state: ExecutionState,
    opts: SyncGovernedExecutionOptions,
) -> GovernedExecutionOutcome:
    if state.intent is None or state.completion is None:
        raise ClavenarConfigError("durable completion is missing its execution intent")
    auth = _validate_stored_intent_sync(state.intent, prepared, body, opts)
    return _validate_completion(
        state.completion,
        cast(JsonObject, state.intent["authorization"]),
        auth,
        prepared,
        opts.executor_id,
    )


def _validate_completion(
    completion: JsonObject,
    signed: JsonObject,
    auth: JsonObject,
    prepared: PreparedToolRequest,
    executor_id: str,
) -> GovernedExecutionOutcome:
    if (
        completion.get("contract") != DURABLE_EXECUTION_CONTRACT
        or completion.get("stage") != "execution.completed"
        or completion.get("authorization_id") != auth["authorization_id"]
        or completion.get("idempotency_id") != prepared.idempotency_id
        or completion.get("executor_id") != executor_id
        or not isinstance(completion.get("effect_id"), str)
        or not completion["effect_id"]
        or not isinstance(completion.get("receipt"), dict)
    ):
        raise ClavenarConfigError("stored execution completion is invalid")
    receipt = cast(JsonObject, completion["receipt"])
    signature = receipt.get("workload_signature")
    result_sha256 = _sha256(completion.get("actual_result"))
    if (
        completion.get("actual_result_sha256") != result_sha256
        or receipt.get("result_sha256") != result_sha256
        or receipt.get("authorization_id") != auth["authorization_id"]
        or receipt.get("idempotency_id") != prepared.idempotency_id
        or receipt.get("effect_id") != completion["effect_id"]
        or receipt.get("contract") != EXECUTION_CONTRACT
        or receipt.get("stage") != "execution.completed"
        or receipt.get("correlation_id") != auth["correlation_id"]
        or receipt.get("agent_id") != auth["agent_id"]
        or receipt.get("agent_spiffe") != auth["agent_spiffe"]
        or receipt.get("tenant") != auth["tenant"]
        or receipt.get("credential_fingerprint") != auth["credential_fingerprint"]
        or receipt.get("method") != auth["method"]
        or receipt.get("payload_sha256") != auth["payload_sha256"]
        or _canonical(receipt.get("authorization")) != _canonical(signed)
        or not isinstance(signature, dict)
        or signature.get("credential_fingerprint") != auth["credential_fingerprint"]
        or not signature.get("algorithm")
        or not signature.get("value")
    ):
        raise ClavenarConfigError("stored execution completion failed integrity validation")
    return GovernedExecutionOutcome(
        result=_clone_json(completion.get("actual_result")),
        effect_id=cast(str, completion["effect_id"]),
        idempotency_id=prepared.idempotency_id,
        receipt=cast(JsonObject, _clone_json(receipt)),
    )


async def _request_authorization_async(
    body: JsonObject, idempotency_id: str, opts: AsyncGovernedExecutionOptions
) -> JsonObject:
    last_error: ClavenarTransportError | None = None
    for attempt in range(opts.max_attempts):
        try:
            return await _request_authorization_once_async(body, idempotency_id, opts)
        except ClavenarTransportError as error:
            last_error = error
            if not _retryable(error) or attempt + 1 == opts.max_attempts:
                raise
            await asyncio.sleep(min(opts.base_delay_s * (2**attempt), MAX_RETRY_DELAY_S))
    raise last_error or ClavenarTransportError("governed authorization made no attempt")


async def _request_authorization_once_async(
    body: JsonObject, idempotency_id: str, opts: AsyncGovernedExecutionOptions
) -> JsonObject:
    headers = _decision_headers(idempotency_id, opts.token)
    owned: httpx.AsyncClient | None = None
    client = opts.client
    if client is None:
        owned = httpx.AsyncClient(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = await client.post(
                _join_url(opts.endpoint, "/mcp"),
                content=_canonical(body).encode(),
                headers=headers,
                timeout=opts.timeout_s,
            )
        except httpx.HTTPError as error:
            raise ClavenarTransportError(f"governed authorization failed: {error}") from error
    finally:
        if owned is not None:
            await owned.aclose()
    return _authorization_response(response)


def _request_authorization_sync(
    body: JsonObject, idempotency_id: str, opts: SyncGovernedExecutionOptions
) -> JsonObject:
    last_error: ClavenarTransportError | None = None
    for attempt in range(opts.max_attempts):
        try:
            return _request_authorization_once_sync(body, idempotency_id, opts)
        except ClavenarTransportError as error:
            last_error = error
            if not _retryable(error) or attempt + 1 == opts.max_attempts:
                raise
            time.sleep(min(opts.base_delay_s * (2**attempt), MAX_RETRY_DELAY_S))
    raise last_error or ClavenarTransportError("governed authorization made no attempt")


def _request_authorization_once_sync(
    body: JsonObject, idempotency_id: str, opts: SyncGovernedExecutionOptions
) -> JsonObject:
    headers = _decision_headers(idempotency_id, opts.token)
    owned: httpx.Client | None = None
    client = opts.client
    if client is None:
        owned = httpx.Client(timeout=opts.timeout_s)
        client = owned
    try:
        try:
            response = client.post(
                _join_url(opts.endpoint, "/mcp"),
                content=_canonical(body).encode(),
                headers=headers,
                timeout=opts.timeout_s,
            )
        except httpx.HTTPError as error:
            raise ClavenarTransportError(f"governed authorization failed: {error}") from error
    finally:
        if owned is not None:
            owned.close()
    return _authorization_response(response)


def _authorization_response(response: httpx.Response) -> JsonObject:
    content = response.content
    if len(content) > MAX_RESPONSE_BYTES:
        raise ClavenarTransportError(
            f"governed authorization response exceeds {MAX_RESPONSE_BYTES} bytes",
            status=response.status_code,
        )
    if response.status_code != 200:
        error_text = _bounded_error_text(content)
        raise ClavenarTransportError(
            f"governed authorization returned {response.status_code}: {error_text}",
            status=response.status_code,
        )
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, ValueError) as error:
        raise ClavenarTransportError(
            f"governed authorization returned invalid JSON: {error}", status=200
        ) from error
    if not isinstance(payload, dict):
        raise ClavenarTransportError("governed authorization must be an object", status=200)
    return cast(JsonObject, payload)


def _validate_authorization(
    signed: JsonObject,
    prepared: PreparedToolRequest,
    body: JsonObject,
) -> JsonObject:
    signature = signed.get("identity_signature")
    auth_value = signed.get("authorization")
    if not isinstance(signature, dict) or not signature:
        raise ClavenarConfigError("authorization is missing a valid identity signature")
    if not isinstance(auth_value, dict):
        raise ClavenarConfigError("authorization is missing")
    auth = cast(JsonObject, auth_value)
    if auth.get("contract") != EXECUTION_CONTRACT or auth.get("stage") != "authorization":
        raise ClavenarConfigError("invalid governed execution authorization contract")
    if auth.get("idempotency_id") != prepared.idempotency_id:
        raise ClavenarConfigError("authorization changed the idempotency identity")
    for field in ("authorization_id", "correlation_id"):
        if not _is_uuid(auth.get(field)):
            raise ClavenarConfigError(f"authorization contains invalid {field}")
    required = (
        "agent_id",
        "agent_spiffe",
        "tenant",
        "credential_fingerprint",
        "brain_version",
    )
    if any(not isinstance(auth.get(field), str) or not auth[field] for field in required) or not (
        _is_sha256(auth.get("payload_sha256")) and _is_sha256(auth.get("brain_evidence_sha256"))
    ):
        raise ClavenarConfigError("authorization is missing an execution binding")
    if not isinstance(auth.get("decision_principal"), dict) or not isinstance(
        auth.get("policy_bundle"), dict
    ):
        raise ClavenarConfigError("authorization contains invalid decision evidence")
    if auth.get("method") != "tools/call" or auth.get("tool_name") != prepared.name:
        raise ClavenarConfigError("authorization changed the tool binding")
    payload = auth.get("execution_payload")
    if not isinstance(payload, dict) or set(payload) != {"jsonrpc", "id", "method", "params"}:
        raise ClavenarConfigError(
            "authorization execution payload changed a protected request binding"
        )
    params = payload.get("params")
    if (
        payload.get("jsonrpc") != "2.0"
        or payload.get("id") != prepared.idempotency_id
        or payload.get("method") != "tools/call"
        or not isinstance(params, dict)
        or set(params) != {"name", "arguments"}
        or params.get("name") != prepared.name
    ):
        raise ClavenarConfigError(
            "authorization execution payload changed a protected request binding"
        )
    if auth.get("payload_sha256") != _sha256(payload):
        raise ClavenarConfigError("authorization payload digest does not match execution payload")
    if auth.get("modification_diff") is None and _canonical(payload) != _canonical(body):
        raise ClavenarConfigError("authorization changed an unmodified execution payload")
    if auth.get("modification_diff") is not None:
        _canonical(auth["modification_diff"])
    return auth


def _execution_intent(signed: JsonObject, auth: JsonObject, executor_id: str) -> JsonObject:
    return {
        "contract": DURABLE_EXECUTION_CONTRACT,
        "stage": "execution.intent",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "tenant": auth["tenant"],
        "workload_id": auth["agent_id"],
        "workload_spiffe": auth["agent_spiffe"],
        "payload_sha256": auth["payload_sha256"],
        "executor_id": executor_id,
        "authorization": _clone_json(signed),
    }


def _execution_request(auth: JsonObject, executor_id: str) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        authorization_id=cast(str, auth["authorization_id"]),
        idempotency_id=cast(str, auth["idempotency_id"]),
        executor_id=executor_id,
        execution_payload=_clone_json(auth.get("execution_payload")),
    )


def _unsigned_receipt(signed: JsonObject, auth: JsonObject, effect: ExecutionEffect) -> JsonObject:
    if not isinstance(effect, ExecutionEffect) or not effect.effect_id.strip():
        raise ClavenarConfigError("executor returned an invalid effect")
    result_sha256 = _sha256(effect.result)
    fields = (
        "authorization_id",
        "idempotency_id",
        "correlation_id",
        "agent_id",
        "agent_spiffe",
        "tenant",
        "credential_fingerprint",
        "method",
        "payload_sha256",
    )
    if any(field not in auth for field in fields):
        raise ClavenarConfigError("authorization is missing a receipt binding")
    return {
        "contract": EXECUTION_CONTRACT,
        "stage": "execution.completed",
        **{field: auth[field] for field in fields},
        "authorization": _clone_json(signed),
        "result_sha256": result_sha256,
        "effect_id": effect.effect_id,
    }


def _stable_effect(effect: ExecutionEffect) -> ExecutionEffect:
    if not isinstance(effect, ExecutionEffect) or not isinstance(effect.effect_id, str):
        raise ClavenarConfigError("executor returned an invalid effect")
    if not effect.effect_id.strip():
        raise ClavenarConfigError("executor returned an invalid effect")
    return ExecutionEffect(result=_clone_json(effect.result), effect_id=effect.effect_id)


def _completion(
    unsigned: JsonObject,
    signature: dict[str, str],
    auth: JsonObject,
    effect: ExecutionEffect,
    executor_id: str,
) -> tuple[JsonObject, JsonObject]:
    signature = _clone_json(signature)
    if not isinstance(signature, dict) or not all(
        isinstance(signature.get(field), str) and signature[field]
        for field in ("algorithm", "credential_fingerprint", "value")
    ):
        raise ClavenarConfigError("receipt signer returned an invalid workload signature")
    if signature["credential_fingerprint"] != auth["credential_fingerprint"]:
        raise ClavenarConfigError("receipt signer credential does not match the authorization")
    receipt = {**unsigned, "workload_signature": _clone_json(signature)}
    completion: JsonObject = {
        "contract": DURABLE_EXECUTION_CONTRACT,
        "stage": "execution.completed",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "executor_id": executor_id,
        "actual_result": _clone_json(effect.result),
        "actual_result_sha256": unsigned["result_sha256"],
        "effect_id": effect.effect_id,
        "receipt": receipt,
    }
    return completion, receipt


async def _verify_async(
    verifier: AsyncAuthorizationVerifier, signed: JsonObject, stored: bool
) -> None:
    try:
        result = verifier(cast(JsonObject, _clone_json(signed)))
        if inspect.isawaitable(result):
            await result
    except Exception as error:
        prefix = "stored authorization" if stored else "authorization"
        raise ClavenarConfigError(f"{prefix} signature verification failed: {error}") from error


def _verify_sync(verifier: SyncAuthorizationVerifier, signed: JsonObject, stored: bool) -> None:
    try:
        verifier(cast(JsonObject, _clone_json(signed)))
    except Exception as error:
        prefix = "stored authorization" if stored else "authorization"
        raise ClavenarConfigError(f"{prefix} signature verification failed: {error}") from error


async def _critical_async(awaitable: Awaitable[Any], timeout_s: float, operation: str) -> Any:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout_s)
    except asyncio.CancelledError:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        finally:
            raise
    except TimeoutError as error:
        raise ClavenarTransportError(f"{operation} timed out after {timeout_s}s") from error


def _critical_sync(operation: Callable[[], Any], timeout_s: float, name: str) -> Any:
    """Bound a sync finalizer without abandoning it midway through its durable write."""

    future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def run() -> None:
        try:
            future.set_result(operation())
        except BaseException as error:
            future.set_exception(error)

    thread = threading.Thread(
        target=run,
        name="clavenar-governed-finalization",
        daemon=True,
    )
    thread.start()
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as error:
        raise ClavenarTransportError(f"{name} timed out after {timeout_s}s") from error


def _tool_body(prepared: PreparedToolRequest) -> JsonObject:
    return {
        "jsonrpc": "2.0",
        "id": prepared.idempotency_id,
        "method": "tools/call",
        "params": {"name": prepared.name, "arguments": prepared.arguments},
    }


def _decision_headers(idempotency_id: str, token: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "x-clavenar-decision-contract": DECISION_CONTRACT,
        "x-clavenar-idempotency-id": idempotency_id,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_prepared(prepared: PreparedToolRequest) -> None:
    if not prepared.name.strip():
        raise ClavenarConfigError("tool name must not be empty")
    if not _is_uuid(prepared.idempotency_id):
        raise ClavenarConfigError("idempotency id must be a canonical UUID")
    _canonical(prepared.arguments)


def _validate_options(opts: AsyncGovernedExecutionOptions | SyncGovernedExecutionOptions) -> None:
    _validate_endpoint(opts.endpoint, bool(opts.token), opts.allow_insecure_loopback)
    if not opts.executor_id.strip():
        raise ClavenarConfigError("executor_id is required")
    if (
        not callable(opts.executor)
        or not callable(opts.sign_receipt)
        or not callable(opts.verify_authorization)
    ):
        raise ClavenarConfigError(
            "recoverable durable store, executor, receipt signer, "
            "and authorization verifier are required"
        )
    if not isinstance(opts.max_attempts, int) or not 1 <= opts.max_attempts <= MAX_RETRY_ATTEMPTS:
        raise ClavenarConfigError(f"max_attempts must be between 1 and {MAX_RETRY_ATTEMPTS}")
    for name, value, maximum in (
        ("timeout_s", opts.timeout_s, MAX_TIMEOUT_S),
        ("finalization_timeout_s", opts.finalization_timeout_s, MAX_TIMEOUT_S),
    ):
        if not math.isfinite(value) or not 0 < value <= maximum:
            raise ClavenarConfigError(f"{name} must be between 0 and {maximum}")
    if not math.isfinite(opts.base_delay_s) or not 0 <= opts.base_delay_s <= MAX_RETRY_DELAY_S:
        raise ClavenarConfigError(f"base_delay_s must be between 0 and {MAX_RETRY_DELAY_S}")


def _validate_endpoint(
    endpoint: str, sends_credentials: bool, allow_insecure_loopback: bool
) -> None:
    try:
        parsed = urlsplit(endpoint)
        _ = parsed.port
    except ValueError as error:
        raise ClavenarConfigError("endpoint must be a valid absolute URL") from error
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.netloc:
        raise ClavenarConfigError("endpoint must be a valid absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ClavenarConfigError("endpoint must not contain user info, a query, or a fragment")
    if any(ord(char) < 32 for char in endpoint):
        raise ClavenarConfigError("endpoint must not contain control characters")
    if sends_credentials and parsed.scheme != "https":
        try:
            loopback = ipaddress.ip_address(parsed.hostname) in {
                ipaddress.ip_address("127.0.0.1"),
                ipaddress.ip_address("::1"),
            }
        except ValueError:
            loopback = False
        if not allow_insecure_loopback or not loopback:
            raise ClavenarConfigError(
                "credentials require https; plaintext is available only for an explicitly "
                "enabled loopback development endpoint"
            )


def _canonical(value: Any) -> str:
    """Canonical JSON shared with the TypeScript SDK's finite safe-number subset."""

    return _canonical_value(value, set())


def _clone_json(value: Any) -> Any:
    return json.loads(_canonical(value))


def _canonical_value(value: Any, ancestors: set[int]) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ClavenarConfigError("JSON integers must be safely representable")
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ClavenarConfigError("JSON numbers must be finite")
        return _ecmascript_number(value)
    if not isinstance(value, (list, dict)):
        raise ClavenarConfigError(f"value is not JSON serializable: {type(value).__name__}")
    identity = id(value)
    if identity in ancestors:
        raise ClavenarConfigError("value contains a JSON cycle")
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            return "[" + ",".join(_canonical_value(entry, ancestors) for entry in value) + "]"
        if any(not isinstance(key, str) for key in value):
            raise ClavenarConfigError("JSON object keys must be strings")
        keys = sorted(value, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{_canonical_value(value[key], ancestors)}"
                for key in keys
            )
            + "}"
        )
    finally:
        ancestors.remove(identity)


def _ecmascript_number(value: float) -> str:
    if value == 0:
        return "0"
    raw = repr(value).lower()
    if "e" not in raw:
        return raw[:-2] if raw.endswith(".0") else raw
    coefficient, exponent_text = raw.split("e")
    exponent = int(exponent_text)
    sign = ""
    if coefficient.startswith("-"):
        sign, coefficient = "-", coefficient[1:]
    digits = coefficient.replace(".", "")
    decimal_position = 1 + exponent
    absolute = abs(value)
    if 1e-6 <= absolute < 1e21:
        if decimal_position <= 0:
            return sign + "0." + "0" * (-decimal_position) + digits
        if decimal_position >= len(digits):
            return sign + digits + "0" * (decimal_position - len(digits))
        return sign + digits[:decimal_position] + "." + digits[decimal_position:]
    normalized = digits[0]
    if len(digits) > 1:
        normalized += "." + digits[1:].rstrip("0")
    exponent_sign = "+" if exponent >= 0 else "-"
    return f"{sign}{normalized}e{exponent_sign}{abs(exponent)}"


def _sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode()).hexdigest()}"


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _retryable(error: ClavenarTransportError) -> bool:
    return error.status is None or 500 <= error.status < 600


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def _bounded_error_text(content: bytes) -> str:
    preview = content[:MAX_ERROR_PREVIEW_BYTES].decode("utf-8", errors="replace")
    preview = " ".join(preview.split())
    return preview + ("..." if len(content) > MAX_ERROR_PREVIEW_BYTES else "")
