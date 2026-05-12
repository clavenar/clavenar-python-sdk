"""OpenAI Realtime helper tests against a respx-mocked warden-lite."""

from __future__ import annotations

import httpx
import respx

from warden_ai.options import WardenOptions
from warden_ai.realtime import (
    inspect_realtime_function_call,
    is_realtime_function_call_done,
    normalize_realtime_function_call,
)

FAKE_ENDPOINT = "http://warden-lite.test"


def _done_event() -> dict[str, object]:
    return {
        "type": "response.function_call_arguments.done",
        "response_id": "resp_1",
        "item_id": "item_1",
        "output_index": 0,
        "call_id": "call_abc",
        "name": "wire_transfer",
        "arguments": '{"to":"acct-9","amount":250}',
    }


def test_is_done_matches_terminal_event() -> None:
    assert is_realtime_function_call_done(_done_event()) is True


def test_is_done_rejects_audio_delta() -> None:
    assert (
        is_realtime_function_call_done(
            {"type": "response.audio.delta", "delta": "aGVsbG8="}
        )
        is False
    )


def test_is_done_rejects_in_flight_delta() -> None:
    assert (
        is_realtime_function_call_done(
            {
                "type": "response.function_call_arguments.delta",
                "call_id": "call_abc",
                "delta": '{"to":"ac',
            }
        )
        is False
    )


def test_is_done_rejects_malformed_done_event() -> None:
    assert (
        is_realtime_function_call_done(
            {"type": "response.function_call_arguments.done"}
        )
        is False
    )


def test_normalize_parses_json_arguments() -> None:
    call = normalize_realtime_function_call(_done_event())
    assert call.id == "call_abc"
    assert call.name == "wire_transfer"
    assert call.input == {"to": "acct-9", "amount": 250}


def test_normalize_falls_back_to_raw_string_on_invalid_json() -> None:
    evt = _done_event()
    evt["arguments"] = "not-json"
    call = normalize_realtime_function_call(evt)
    assert call.input == "not-json"


@respx.mock
async def test_inspect_returns_allow_on_200() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(200, json={})
    )
    verdict = await inspect_realtime_function_call(
        _done_event(),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "allow"


@respx.mock
async def test_inspect_forwards_deny_on_403() -> None:
    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": "security_violation",
                "reasons": ["wire_transfer requires approval"],
                "review_reasons": [],
                "intent_category": "PolicyDeny",
            },
        )
    )
    verdict = await inspect_realtime_function_call(
        _done_event(),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    assert verdict.kind == "deny"
    assert "approval" in " ".join(verdict.reasons)  # type: ignore[union-attr]


@respx.mock
async def test_inspect_uses_call_id_as_envelope_id() -> None:
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.read() and {"body": request.read().decode()})
        return httpx.Response(200, json={})

    respx.post(f"{FAKE_ENDPOINT}/mcp").mock(side_effect=_handler)
    await inspect_realtime_function_call(
        _done_event(),
        WardenOptions(endpoint=FAKE_ENDPOINT),
    )
    body = captured["body"]
    assert isinstance(body, str)
    assert '"id":"call_abc"' in body or '"id": "call_abc"' in body
    assert '"wire_transfer"' in body
