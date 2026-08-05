from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from clavenar_agent_sdk.governed_execution import (
    AsyncGovernedExecutionOptions,
    ExecutionEffect,
    ExecutionState,
    PreparedToolRequest,
    ToolExecutionRequest,
    execute_prepared_tool,
)

ENDPOINT = "https://gateway.example"
PREPARED = PreparedToolRequest.restore(
    "cfcc8767-4c73-41cc-8ece-b855863924c4", "payments.transfer", {"amount": 100}
)


def _authorization() -> dict[str, Any]:
    return {
        "authorization": {
            "contract": "clavenar.execution/v1",
            "stage": "authorization",
            "authorization_id": "354c33ed-e5d3-4af7-a1b8-b009d50b0bc5",
            "idempotency_id": PREPARED.idempotency_id,
            "correlation_id": "c1a28e4c-a17d-5b3d-884b-e5b627f762c2",
            "agent_id": "payments-agent",
            "agent_spiffe": "spiffe://clavenar.local/tenant/acme/agent/payments-agent/instance/one",
            "tenant": "acme",
            "credential_fingerprint": "sha256:" + "1" * 64,
            "method": "tools/call",
            "tool_name": PREPARED.name,
            "execution_payload": {
                "jsonrpc": "2.0",
                "id": PREPARED.idempotency_id,
                "method": "tools/call",
                "params": {"name": PREPARED.name, "arguments": PREPARED.arguments},
            },
            "payload_sha256": (
                "sha256:269123e546c75ec2df26ce4a52baeab92e58afdfabcb111c3e9069a37f78f1c5"
            ),
            "decision_principal": {"subject": "system:policy-brain"},
            "modification_diff": None,
            "policy_bundle": {"schema_version": 1},
            "brain_version": "brain-fixture",
            "brain_evidence_sha256": "sha256:" + "3" * 64,
        },
        "identity_signature": {"algorithm": "Ed25519", "key_id": "identity:v1"},
    }


class Store:
    def __init__(self, order: list[str], fail_intent: bool = False) -> None:
        self.order = order
        self.fail_intent = fail_intent
        self.intent: dict[str, Any] | None = None
        self.completion: dict[str, Any] | None = None

    async def commit_intent(self, intent: dict[str, Any]) -> None:
        self.order.append("intent")
        if self.fail_intent:
            raise RuntimeError("store unavailable")
        self.intent = intent

    async def load_execution(self, _idempotency_id: str) -> ExecutionState:
        return ExecutionState(intent=self.intent, completion=self.completion)

    async def commit_completion_and_enqueue_receipt(self, completion: dict[str, Any]) -> None:
        self.order.append("completion")
        self.completion = completion


@respx.mock
async def test_governed_execution_orders_intent_effect_completion() -> None:
    route = respx.post(f"{ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200, json=_authorization())
    )
    order: list[str] = []
    store = Store(order)

    async def executor(request: ToolExecutionRequest) -> ExecutionEffect:
        order.append("effect")
        assert request.idempotency_id == PREPARED.idempotency_id
        return ExecutionEffect(result={"ok": True}, effect_id="provider-operation-123")

    async def signer(unsigned: dict[str, Any]) -> dict[str, str]:
        unsigned["authorization_id"] = "mutated-by-signer"
        return {
            "algorithm": "ES256",
            "credential_fingerprint": "sha256:" + "1" * 64,
            "value": "signed",
        }

    result = await execute_prepared_tool(
        PREPARED,
        AsyncGovernedExecutionOptions(
            endpoint=ENDPOINT,
            executor_id="payments-provider",
            executor=executor,
            durable_store=store,
            verify_authorization=lambda signed: signed["authorization"].update(
                {"tool_name": "mutated-by-verifier"}
            ),
            sign_receipt=signer,
        ),
    )
    assert order == ["intent", "effect", "completion"]
    assert result.result == {"ok": True}
    assert (
        result.receipt["authorization_id"] == _authorization()["authorization"]["authorization_id"]
    )
    assert result.receipt["authorization"]["authorization"]["tool_name"] == PREPARED.name
    assert store.completion is not None
    assert store.completion["actual_result_sha256"] == (
        "sha256:4062edaf750fb8074e7e83e0c9028c94e32468a8b6f1614774328ef045150f93"
    )
    request = route.calls[0].request
    assert request.headers["x-clavenar-decision-contract"] == "clavenar.decision/v1"
    assert request.headers["x-clavenar-idempotency-id"] == PREPARED.idempotency_id


@respx.mock
async def test_intent_failure_invokes_no_executor() -> None:
    respx.post(f"{ENDPOINT}/mcp").mock(return_value=httpx.Response(200, json=_authorization()))
    called = False

    async def executor(_: ToolExecutionRequest) -> ExecutionEffect:
        nonlocal called
        called = True
        return ExecutionEffect(result={}, effect_id="unexpected")

    async def signer(_: dict[str, Any]) -> dict[str, str]:
        return {}

    with pytest.raises(RuntimeError, match="store unavailable"):
        await execute_prepared_tool(
            PREPARED,
            AsyncGovernedExecutionOptions(
                endpoint=ENDPOINT,
                executor_id="payments-provider",
                executor=executor,
                durable_store=Store([], fail_intent=True),
                verify_authorization=lambda _signed: None,
                sign_receipt=signer,
            ),
        )
    assert not called


@respx.mock
async def test_executor_failure_is_never_retried() -> None:
    route = respx.post(f"{ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200, json=_authorization())
    )
    calls = 0

    async def executor(_: ToolExecutionRequest) -> ExecutionEffect:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider response lost")

    async def signer(_: dict[str, Any]) -> dict[str, str]:
        raise AssertionError("signer must not run")

    with pytest.raises(RuntimeError, match="provider response lost"):
        await execute_prepared_tool(
            PREPARED,
            AsyncGovernedExecutionOptions(
                endpoint=ENDPOINT,
                executor_id="payments-provider",
                executor=executor,
                durable_store=Store([]),
                verify_authorization=lambda _signed: None,
                sign_receipt=signer,
                max_attempts=3,
                base_delay_s=0.001,
            ),
        )
    assert route.call_count == 1
    assert calls == 1


@respx.mock
async def test_signature_verification_fails_before_intent() -> None:
    respx.post(f"{ENDPOINT}/mcp").mock(return_value=httpx.Response(200, json=_authorization()))
    store = Store([])
    called = False

    async def executor(_: ToolExecutionRequest) -> ExecutionEffect:
        nonlocal called
        called = True
        return ExecutionEffect({}, "unexpected")

    def verifier(_: dict[str, Any]) -> None:
        raise RuntimeError("unknown identity key")

    async def signer(_: dict[str, Any]) -> dict[str, str]:
        raise AssertionError("signer must not run")

    with pytest.raises(Exception, match="authorization signature verification failed"):
        await execute_prepared_tool(
            PREPARED,
            AsyncGovernedExecutionOptions(
                endpoint=ENDPOINT,
                executor_id="payments-provider",
                executor=executor,
                durable_store=store,
                verify_authorization=verifier,
                sign_receipt=signer,
            ),
        )
    assert store.intent is None
    assert not called


async def test_persisted_intent_never_replays_without_conclusive_recovery() -> None:
    signed = _authorization()
    auth = signed["authorization"]
    store = Store([])
    store.intent = {
        "contract": "clavenar.sdk-durable-intent-outbox/v1",
        "stage": "execution.intent",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "tenant": auth["tenant"],
        "workload_id": auth["agent_id"],
        "workload_spiffe": auth["agent_spiffe"],
        "payload_sha256": auth["payload_sha256"],
        "executor_id": "payments-provider",
        "authorization": signed,
    }
    called = False

    async def executor(_: ToolExecutionRequest) -> ExecutionEffect:
        nonlocal called
        called = True
        return ExecutionEffect({}, "unexpected")

    async def signer(_: dict[str, Any]) -> dict[str, str]:
        raise AssertionError("signer must not run")

    from clavenar_agent_sdk import ClavenarRecoveryRequired

    with pytest.raises(ClavenarRecoveryRequired):
        await execute_prepared_tool(
            PREPARED,
            AsyncGovernedExecutionOptions(
                endpoint=ENDPOINT,
                executor_id="payments-provider",
                executor=executor,
                durable_store=store,
                verify_authorization=lambda _signed: None,
                sign_receipt=signer,
            ),
        )
    assert not called


async def test_recoverer_finalizes_without_replaying_executor() -> None:
    signed = _authorization()
    auth = signed["authorization"]
    store = Store([])
    store.intent = {
        "contract": "clavenar.sdk-durable-intent-outbox/v1",
        "stage": "execution.intent",
        "authorization_id": auth["authorization_id"],
        "idempotency_id": auth["idempotency_id"],
        "tenant": auth["tenant"],
        "workload_id": auth["agent_id"],
        "workload_spiffe": auth["agent_spiffe"],
        "payload_sha256": auth["payload_sha256"],
        "executor_id": "payments-provider",
        "authorization": signed,
    }

    async def executor(_: ToolExecutionRequest) -> ExecutionEffect:
        raise AssertionError("executor must not replay")

    async def recover(_: dict[str, Any]) -> ExecutionEffect:
        return ExecutionEffect({"ok": True}, "provider-operation-123")

    async def signer(_: dict[str, Any]) -> dict[str, str]:
        return {
            "algorithm": "ES256",
            "credential_fingerprint": auth["credential_fingerprint"],
            "value": "signed",
        }

    outcome = await execute_prepared_tool(
        PREPARED,
        AsyncGovernedExecutionOptions(
            endpoint=ENDPOINT,
            executor_id="payments-provider",
            executor=executor,
            recover_effect=recover,
            durable_store=store,
            verify_authorization=lambda _signed: None,
            sign_receipt=signer,
        ),
    )
    assert outcome.effect_id == "provider-operation-123"
    assert store.completion is not None
