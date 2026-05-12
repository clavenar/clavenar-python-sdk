# Changelog

All notable changes to `warden-ai` (Python) are recorded here.

## 0.2.0 — 2026-05-12

Feature-complete release. Reaches 1:1 parity with the TS SDK at
`@vanteguardlabs/warden-ai-sdk@0.3.0`.

### Added

- **Streaming inspection** for both providers. `stream=True` is
  intercepted; the closing event (Anthropic `content_block_stop`,
  OpenAI `finish_reason="tool_calls"`) is held until warden returns a
  verdict. A denied tool raises mid-iteration before partner code can
  act on it. Supports both async (`AsyncAnthropic`, `AsyncOpenAI`)
  and sync streams.
- **Sync clients** — `anthropic.Anthropic` / `openai.OpenAI` (non-
  async) wrap the same way. Detection is via
  `inspect.iscoroutinefunction` on `create`. Sync paths use
  `httpx.Client` and `time.sleep`; observe-mode and pending semantics
  match the async path.
- **Retries** — transient (5xx, network) failures retry up to
  `WardenRetryOptions.max_attempts` (default 3) with jittered
  exponential backoff (`base_delay_s=0.1` by default). 200, 403, 202,
  and 4xx other than 5xx never retry. `max_attempts=1` disables
  retries entirely.
- **`WardenRetryOptions`** type exported from the public API.
- **Parallel tool_use observability**: when a multi-tool turn comes
  back, all inspections kick off concurrently via `asyncio.gather`.
  Verdict callbacks fire in submission order so the first deny in
  `tool_calls[]` is the one that raises, deterministically (the same
  pattern as the TS SDK's `inspectAllToolCalls`).
- **Streaming wrappers** exported for direct use:
  `wrap_anthropic_stream`, `wrap_anthropic_stream_sync`,
  `wrap_openai_chat_stream`, `wrap_openai_chat_stream_sync`.
- **Sync transport helpers** exported: `inspect_tool_use_sync` and
  `poll_pending_once_sync`.

### Changed

- The MVP's one-time `RuntimeWarning` for `stream=True` is gone —
  streaming is now inspected.
- README + CHANGELOG no longer carry a "what this MVP does NOT do
  yet" section.

### Migration notes

- `WardenOptions` gained a `retry` field with a default `(3, 0.1)`
  policy. Existing code that constructed `WardenOptions` by keyword
  is unaffected; positional construction was never recommended.
- Sync-mode callbacks (`on_verdict`, `on_policy_error`) must be sync
  functions when wrapping a sync client. Passing an `async def`
  callback to a sync wrap raises `WardenConfigError` at fire time.

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
