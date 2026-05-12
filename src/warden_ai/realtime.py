"""OpenAI Realtime helpers — gate tool calls a voice / streaming
agent emits over the Realtime websocket.

The Realtime API is websocket-based; there is no `client.method()` for
`warden_wrap` to intercept. Drain the WS event stream yourself and run
each ``response.function_call_arguments.done`` event through
:func:`inspect_realtime_function_call` before dispatching your
handler. Deltas can be ignored — by the time the ``done`` event
arrives the model has committed and the full argument payload is
present.

Source: https://platform.openai.com/docs/api-reference/realtime-server-events
"""

from __future__ import annotations

import json
from typing import Any

from warden_ai.options import WardenOptions
from warden_ai.transport import NormalizedToolCall, WardenVerdict, inspect_tool_use


def is_realtime_function_call_done(evt: dict[str, Any]) -> bool:
    """True when ``evt`` is the terminal arg event for a tool call.

    Server events the API emits for in-flight tool calls include a
    sequence of ``response.function_call_arguments.delta`` events
    followed by exactly one ``response.function_call_arguments.done``.
    Only the ``done`` event is inspected — by then the model has
    committed and the full argument payload is present.
    """
    return (
        evt.get("type") == "response.function_call_arguments.done"
        and isinstance(evt.get("call_id"), str)
        and isinstance(evt.get("arguments"), str)
        and isinstance(evt.get("name"), str)
    )


def normalize_realtime_function_call(evt: dict[str, Any]) -> NormalizedToolCall:
    """Build a :class:`NormalizedToolCall` from one ``done`` event.

    The ``arguments`` field is a JSON-encoded string in the Realtime
    contract; we parse it on the way through. On parse failure the
    call's ``input`` lands as the raw string so warden can still
    inspect the *attempt* — a malformed-args policy is a legitimate
    use case and the helper shouldn't swallow it.
    """
    try:
        input_val: Any = json.loads(evt["arguments"])
    except json.JSONDecodeError:
        input_val = evt["arguments"]
    return NormalizedToolCall(id=evt["call_id"], name=evt["name"], input=input_val)


async def inspect_realtime_function_call(
    evt: dict[str, Any],
    opts: WardenOptions,
) -> WardenVerdict:
    """One-shot inspection helper.

    Equivalent to
    ``inspect_tool_use(normalize_realtime_function_call(evt), opts)``
    but gives partners a single import to point at from their WS
    message pump.
    """
    return await inspect_tool_use(normalize_realtime_function_call(evt), opts)
