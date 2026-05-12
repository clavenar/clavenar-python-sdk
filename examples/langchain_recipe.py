"""LangChain (Python) + warden — gate every tool the agent runs.

The load-bearing pattern is `warden_tool(name, description, func)`:
returns a LangChain-shaped StructuredTool whose `func` consults
warden before delegating to your real handler. Drop this helper
into your existing LangChain agent and every tool call routes
through the proxy without further changes.

LangChain has many tool-registration shapes (StructuredTool,
@tool, agent_toolkits). The pattern below uses the lowest-common-
denominator dict shape so the helper transplants into any of them
with a trivial constructor swap.

Usage:
    pip install warden-ai langchain
    python examples/langchain_recipe.py
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


def _new_tool_use_id() -> str:
    return secrets.token_hex(4)


def warden_tool(
    options: WardenOptions,
    name: str,
    description: str,
    func: Callable[[dict[str, Any]], Awaitable[Any]],
) -> dict[str, Any]:
    """Wrap a LangChain-shaped tool so its `func` consults warden
    before running. Returns the same shape LangChain consumes:
    ``{"name", "description", "func"}``.
    """

    async def gated(arg_str: str) -> str:
        try:
            args = json.loads(arg_str)
        except json.JSONDecodeError:
            args = arg_str
        try:
            await inspect_tool_use(
                NormalizedToolCall(
                    id=_new_tool_use_id(),
                    name=name,
                    input=args,
                ),
                options,
            )
        except WardenPending as pending:
            try:
                await pending.resolve()
            except WardenDenied as decided:
                return f"[warden] denied by operator: {' ; '.join(decided.reasons)}"
        except WardenDenied as denied:
            return f"[warden] denied: {' ; '.join(denied.reasons)}"
        result = func(args) if not inspect.iscoroutinefunction(func) else await func(args)
        return result if isinstance(result, str) else json.dumps(result)

    return {"name": name, "description": description, "func": gated}


# ---- Example wiring ------------------------------------------------


async def _fetch_user(args: dict[str, Any]) -> dict[str, Any]:
    return {"userId": args["userId"], "name": f"user-{args['userId']}"}


async def _wire_transfer(args: dict[str, Any]) -> dict[str, Any]:
    return {"to": args["to"], "amount": args["amount"], "ok": True}


async def main() -> None:
    options = WardenOptions(
        endpoint=os.environ.get("WARDEN_LITE_URL", "http://localhost:8088"),
        token=os.environ.get("WARDEN_LITE_TOKEN", "demo-token"),
        mode="enforce",
    )

    tools = [
        warden_tool(options, "fetch_user", "Fetch a user record.", _fetch_user),
        warden_tool(options, "wire_transfer", "Send a wire transfer.", _wire_transfer),
    ]

    # Stand-in for `agent_executor.ainvoke(...)` — runs each tool
    # directly so the wiring is visible.
    for tool in tools:
        sample = (
            '{"userId":"alice"}'
            if tool["name"] == "fetch_user"
            else '{"to":"acct-9","amount":250}'
        )
        print(f"\n→ {tool['name']}: {sample}")
        print(" ", await tool["func"](sample))


if __name__ == "__main__":
    asyncio.run(main())
