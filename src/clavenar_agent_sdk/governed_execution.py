"""Side-effect-free decision plus durable registered-executor execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx

from clavenar_agent_sdk.errors import ClavenarConfigError, ClavenarTransportError

DECISION_CONTRACT = "clavenar.decision/v1"
EXECUTION_CONTRACT = "clavenar.execution/v1"
DURABLE_EXECUTION_CONTRACT = "clavenar.sdk-durable-intent-outbox/v1"

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
class GovernedExecutionOutcome:
    result: Any
    effect_id: str
    idempotency_id: str
    receipt: JsonObject


class AsyncDurableExecutionStore(Protocol):
    async def commit_intent(self, intent: JsonObject) -> None: ...

    async def commit_completion_and_enqueue_receipt(self, completion: JsonObject) -> None: ...


class SyncDurableExecutionStore(Protocol):
    def commit_intent(self, intent: JsonObject) -> None: ...

    def commit_completion_and_enqueue_receipt(self, completion: JsonObject) -> None: ...


AsyncExecutor = Callable[[ToolExecutionRequest], Awaitable[ExecutionEffect]]
AsyncReceiptSigner = Callable[[JsonObject], Awaitable[dict[str, str]]]
SyncExecutor = Callable[[ToolExecutionRequest], ExecutionEffect]
SyncReceiptSigner = Callable[[JsonObject], dict[str, str]]


@dataclass(frozen=True)
class AsyncGovernedExecutionOptions:
    endpoint: str
    executor_id: str
    executor: AsyncExecutor
    durable_store: AsyncDurableExecutionStore
    sign_receipt: AsyncReceiptSigner
    token: str | None = None
    timeout_s: float = 10.0
    max_attempts: int = 3
    base_delay_s: float = 0.1
    client: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class SyncGovernedExecutionOptions:
    endpoint: str
    executor_id: str
    executor: SyncExecutor
    durable_store: SyncDurableExecutionStore
    sign_receipt: SyncReceiptSigner
    token: str | None = None
    timeout_s: float = 10.0
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
    _validate_options(opts.endpoint, opts.executor_id, opts.max_attempts)
    body = _tool_body(prepared)
    signed = await _request_authorization_async(body, prepared.idempotency_id, opts)
    auth, intent = _authorization_and_intent(signed, prepared, body, opts.executor_id)
    await opts.durable_store.commit_intent(intent)
    effect = await opts.executor(_execution_request(auth, opts.executor_id))
    completion, receipt = await _completion_async(signed, auth, effect, opts)
    await opts.durable_store.commit_completion_and_enqueue_receipt(completion)
    return GovernedExecutionOutcome(
        result=effect.result,
        effect_id=effect.effect_id,
        idempotency_id=prepared.idempotency_id,
        receipt=receipt,
    )


def execute_tool_sync(
    name: str, arguments: Any, opts: SyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    return execute_prepared_tool_sync(PreparedToolRequest.new(name, arguments), opts)


def execute_prepared_tool_sync(
    prepared: PreparedToolRequest, opts: SyncGovernedExecutionOptions
) -> GovernedExecutionOutcome:
    _validate_prepared(prepared)
    _validate_options(opts.endpoint, opts.executor_id, opts.max_attempts)
    body = _tool_body(prepared)
    signed = _request_authorization_sync(body, prepared.idempotency_id, opts)
    auth, intent = _authorization_and_intent(signed, prepared, body, opts.executor_id)
    opts.durable_store.commit_intent(intent)
    effect = opts.executor(_execution_request(auth, opts.executor_id))
    completion, receipt = _completion_sync(signed, auth, effect, opts)
    opts.durable_store.commit_completion_and_enqueue_receipt(completion)
    return GovernedExecutionOutcome(
        result=effect.result,
        effect_id=effect.effect_id,
        idempotency_id=prepared.idempotency_id,
        receipt=receipt,
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
            await asyncio.sleep(opts.base_delay_s * (2**attempt))
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
                json=body,
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
            time.sleep(opts.base_delay_s * (2**attempt))
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
                json=body,
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
    if response.status_code != 200:
        raise ClavenarTransportError(
            f"governed authorization returned {response.status_code}: {response.text}",
            status=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise ClavenarTransportError(
            f"governed authorization returned invalid JSON: {error}", status=200
        ) from error
    if not isinstance(payload, dict):
        raise ClavenarTransportError("governed authorization must be an object", status=200)
    return cast(JsonObject, payload)


def _authorization_and_intent(
    signed: JsonObject,
    prepared: PreparedToolRequest,
    body: JsonObject,
    executor_id: str,
) -> tuple[JsonObject, JsonObject]:
    auth_value = signed.get("authorization")
    if not isinstance(auth_value, dict):
        raise ClavenarConfigError("authorization is missing")
    auth = cast(JsonObject, auth_value)
    if auth.get("contract") != EXECUTION_CONTRACT or auth.get("stage") != "authorization":
        raise ClavenarConfigError("invalid governed execution authorization contract")
    if auth.get("idempotency_id") != prepared.idempotency_id:
        raise ClavenarConfigError("authorization changed the idempotency identity")
    for field in ("authorization_id", "correlation_id"):
        try:
            uuid.UUID(str(auth.get(field)))
        except ValueError as error:
            raise ClavenarConfigError(f"authorization contains invalid {field}") from error
    if auth.get("modification_diff") is None and _canonical(
        auth.get("execution_payload")
    ) != _canonical(body):
        raise ClavenarConfigError("authorization changed an unmodified execution payload")
    required = (
        "authorization_id",
        "idempotency_id",
        "tenant",
        "agent_id",
        "agent_spiffe",
        "payload_sha256",
    )
    if any(not isinstance(auth.get(field), str) or not auth[field] for field in required):
        raise ClavenarConfigError("authorization is missing an execution binding")
    intent: JsonObject = {
        "contract": DURABLE_EXECUTION_CONTRACT,
        "stage": "execution.intent",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "tenant": auth["tenant"],
        "workload_id": auth["agent_id"],
        "workload_spiffe": auth["agent_spiffe"],
        "payload_sha256": auth["payload_sha256"],
        "executor_id": executor_id,
        "authorization": signed,
    }
    return auth, intent


def _execution_request(auth: JsonObject, executor_id: str) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        authorization_id=cast(str, auth["authorization_id"]),
        idempotency_id=cast(str, auth["idempotency_id"]),
        executor_id=executor_id,
        execution_payload=auth.get("execution_payload"),
    )


async def _completion_async(
    signed: JsonObject,
    auth: JsonObject,
    effect: ExecutionEffect,
    opts: AsyncGovernedExecutionOptions,
) -> tuple[JsonObject, JsonObject]:
    unsigned = _unsigned_receipt(signed, auth, effect)
    signature = await opts.sign_receipt(unsigned)
    return _completion(unsigned, signature, auth, effect, opts.executor_id)


def _completion_sync(
    signed: JsonObject,
    auth: JsonObject,
    effect: ExecutionEffect,
    opts: SyncGovernedExecutionOptions,
) -> tuple[JsonObject, JsonObject]:
    unsigned = _unsigned_receipt(signed, auth, effect)
    signature = opts.sign_receipt(unsigned)
    return _completion(unsigned, signature, auth, effect, opts.executor_id)


def _unsigned_receipt(signed: JsonObject, auth: JsonObject, effect: ExecutionEffect) -> JsonObject:
    if not effect.effect_id:
        raise ClavenarConfigError("executor returned an empty effect id")
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
        "authorization": signed,
        "result_sha256": result_sha256,
        "effect_id": effect.effect_id,
    }


def _completion(
    unsigned: JsonObject,
    signature: dict[str, str],
    auth: JsonObject,
    effect: ExecutionEffect,
    executor_id: str,
) -> tuple[JsonObject, JsonObject]:
    if not all(signature.get(field) for field in ("algorithm", "credential_fingerprint", "value")):
        raise ClavenarConfigError("receipt signer returned an invalid workload signature")
    receipt = {**unsigned, "workload_signature": signature}
    completion: JsonObject = {
        "contract": DURABLE_EXECUTION_CONTRACT,
        "stage": "execution.completed",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "executor_id": executor_id,
        "actual_result": effect.result,
        "actual_result_sha256": unsigned["result_sha256"],
        "effect_id": effect.effect_id,
        "receipt": receipt,
    }
    return completion, receipt


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
    try:
        uuid.UUID(prepared.idempotency_id)
    except ValueError as error:
        raise ClavenarConfigError("idempotency id must be a UUID") from error


def _validate_options(endpoint: str, executor_id: str, max_attempts: int) -> None:
    if not endpoint or not executor_id.strip():
        raise ClavenarConfigError("endpoint and executor_id are required")
    if max_attempts < 1:
        raise ClavenarConfigError("max_attempts must be >= 1")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ClavenarConfigError(f"value is not JSON serializable: {error}") from error


def _sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode()).hexdigest()}"


def _retryable(error: ClavenarTransportError) -> bool:
    return error.status is None or 500 <= error.status < 600


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"
