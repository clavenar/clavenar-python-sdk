"""OpenAI Realtime + warden — gate tool calls a voice / streaming
agent emits over the Realtime websocket.

The Realtime API is websocket-based; there's no ``client.method()``
for :func:`warden_wrap` to intercept. Drain the server-event stream
yourself; for each ``response.function_call_arguments.done`` event,
run it through :func:`inspect_realtime_function_call` before
dispatching the handler. Deltas are ignored — by the time ``done``
arrives, the model has committed and the full argument payload is
present.

Usage:
    pip install warden-ai websockets
    python examples/openai_realtime_recipe.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from warden_ai import (
    WardenOptions,
    inspect_realtime_function_call,
    is_realtime_function_call_done,
)


class _StubRealtimeWebSocket:
    """Stand-in for a real Realtime WS so this recipe stays
    dependency-free. In production wire the ``websockets`` package
    against ``wss://api.openai.com/v1/realtime?model=...``.
    """

    async def send(self, payload: str) -> None:
        print(f"→ ws.send: {payload}")


async def main() -> None:
    options = WardenOptions(
        endpoint=os.environ.get("WARDEN_LITE_URL", "http://localhost:8088"),
        token=os.environ.get("WARDEN_LITE_TOKEN", "demo-token"),
        mode="enforce",
    )

    ws = _StubRealtimeWebSocket()

    # Stub events: response.output_item.added announces the call,
    # deltas accumulate the args, done is the terminal event warden
    # inspects.
    events: list[dict[str, Any]] = [
        {"type": "session.created", "session": {"id": "sess_demo"}},
        {
            "type": "response.output_item.added",
            "response_id": "resp_1",
            "item": {
                "type": "function_call",
                "call_id": "call_w1",
                "name": "wire_transfer",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_1",
            "call_id": "call_w1",
            "delta": '{"to":"ac',
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_1",
            "call_id": "call_w1",
            "delta": 'ct-9","amount":250}',
        },
        {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "call_id": "call_w1",
            "name": "wire_transfer",
            "arguments": '{"to":"acct-9","amount":250}',
        },
    ]

    for evt in events:
        if not is_realtime_function_call_done(evt):
            continue

        verdict = await inspect_realtime_function_call(evt, options)

        if verdict.kind == "allow":
            print(f"allow: {evt['name']}({evt['arguments']}) — dispatch handler")
            # Real handler runs here; reply via
            # conversation.item.create / function_call_output.
            continue

        if verdict.kind == "deny":
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": evt["call_id"],
                            "output": (
                                "[warden] denied: "
                                + " ; ".join(verdict.reasons)
                            ),
                        },
                    }
                )
            )
            continue

        # Pending: surface a placeholder so the model isn't stranded.
        print(f"pending: {verdict.correlation_id} — awaiting operator decide")
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": evt["call_id"],
                        "output": "[warden] awaiting human approval",
                    },
                }
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
