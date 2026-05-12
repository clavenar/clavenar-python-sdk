"""LlamaIndex + warden — gate every FunctionTool call.

LlamaIndex agents register tools as ``FunctionTool`` objects with a
plain Python callable. Wrap the callable with the helper below
before handing it to LlamaIndex; the wrapped callable consults
warden, then runs your handler on green / approved-pending.

Usage:
    pip install warden-ai llama-index
    python examples/llamaindex_recipe.py
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import secrets
from collections.abc import Awaitable, Callable
from typing import Any

from warden_ai import (
    NormalizedToolCall,
    WardenDenied,
    WardenOptions,
    WardenPending,
    inspect_tool_use,
)

ToolCallable = Callable[..., Any | Awaitable[Any]]


def warden_function_tool(
    options: WardenOptions,
    name: str,
    fn: ToolCallable,
) -> ToolCallable:
    """Return an async callable LlamaIndex's FunctionTool can wrap.
    The returned callable consults warden before invoking ``fn``;
    on deny it returns an explanatory string LlamaIndex surfaces
    to the model.
    """

    async def gated(**kwargs: Any) -> Any:
        try:
            await inspect_tool_use(
                NormalizedToolCall(
                    id=secrets.token_hex(4),
                    name=name,
                    input=kwargs,
                ),
                options,
            )
        except WardenPending as pending:
            try:
                await pending.resolve()
            except WardenDenied as decided:
                return (
                    f"[warden] {name} denied by operator: "
                    f"{' ; '.join(decided.reasons)}"
                )
        except WardenDenied as denied:
            return f"[warden] {name} denied: {' ; '.join(denied.reasons)}"
        result = fn(**kwargs)
        if inspect.iscoroutine(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result)

    gated.__name__ = name
    gated.__doc__ = fn.__doc__
    return gated


# ---- Example wiring ------------------------------------------------


async def fetch_user(userId: str) -> dict[str, Any]:
    """Fetch a user record by id."""
    return {"userId": userId, "name": f"user-{userId}"}


async def wire_transfer(to: str, amount: float) -> dict[str, Any]:
    """Send a wire transfer."""
    return {"to": to, "amount": amount, "ok": True}


async def main() -> None:
    options = WardenOptions(
        endpoint=os.environ.get("WARDEN_LITE_URL", "http://localhost:8088"),
        token=os.environ.get("WARDEN_LITE_TOKEN", "demo-token"),
        mode="enforce",
    )

    # Real wiring (commented to keep this file dependency-free):
    #
    #   from llama_index.core.tools import FunctionTool
    #   fetch_tool = FunctionTool.from_defaults(
    #       fn=warden_function_tool(options, "fetch_user", fetch_user)
    #   )
    #   wire_tool = FunctionTool.from_defaults(
    #       fn=warden_function_tool(options, "wire_transfer", wire_transfer)
    #   )
    #   agent = ReActAgent.from_tools([fetch_tool, wire_tool], llm=llm)
    #   await agent.aquery("Send $250 to acct-9")

    gated_fetch = warden_function_tool(options, "fetch_user", fetch_user)
    gated_wire = warden_function_tool(options, "wire_transfer", wire_transfer)

    print("→ fetch_user")
    print(" ", await gated_fetch(userId="alice"))
    print("\n→ wire_transfer")
    print(" ", await gated_wire(to="acct-9", amount=250))


if __name__ == "__main__":
    asyncio.run(main())
