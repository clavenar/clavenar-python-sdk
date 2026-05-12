# Changelog

All notable changes to `warden-ai` (Python) are recorded here.

## 0.1.0 — 2026-05-12

Initial MVP release.

- Wraps async Anthropic + OpenAI Python clients.
- Inspects every tool call (`tool_use` / `tool_calls`) in parallel
  before the agent loop sees the response.
- Verdicts: allow / deny (`WardenDenied`) / pending (`WardenPending`
  with `resolve()` polling).
- Modes: enforce (default) and observe with `on_policy_error`.
- Transport: `httpx.AsyncClient` against `warden-lite`'s `/mcp` and
  `/pending/{id}`.
- 19 unit tests, ruff clean, mypy `--strict` clean.

### Not in 0.1.0 (planned for feature-complete release)

- Streaming responses (`stream=True`) pass through unchecked with a
  one-time `RuntimeWarning`.
- Retries on transport failure — single attempt today.
- Sync clients (`anthropic.Anthropic`, `openai.OpenAI`).
- LangChain / LlamaIndex / Mastra adapter recipes (sprint 3).

These ship in 0.2.0 / 0.3.0 alongside the TS SDK 0.3.0 surface.
