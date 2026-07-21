# Changelog

All notable changes to `clavenar-agent-sdk` (Python) are recorded here.

## Unreleased

## 1.3.0 — 2026-07-21

### Changed

- Automatic retries are explicitly confined to the side-effect-free decision
  transport with one stable pre-network idempotency ID. Registered executor
  failures remain single-attempt, and the shared retry-separation fixture is
  packaged for cross-language conformance.

## 1.2.0 — 2026-07-21

### Added

- Async and sync governed-execution APIs with serializable prepared requests,
  a registered executor, durable intent/completion storage, workload receipt
  signing, and actual provider-result return.
- The shared `clavenar.sdk-cross-language/v1` conformance fixture.

### Changed

- Inspection explicitly selects `clavenar.decision/v1` with a canonical UUID
  allocated before the first attempt and retained across safe retries.
  Multi-tool turns use one ordered atomic decision rather than independent
  sibling requests.

### Added

- **429 rate-limit verdicts.** An HTTP 429 from `POST /mcp` now parses
  as a rate-limited verdict — `rate_limited` (request-velocity gate,
  carries `retry_after_secs`) or `quota_exceeded` (per-tenant spend
  gate) — instead of collapsing into a generic transport error.
  Enforce mode raises the new `ClavenarRateLimited` (`tool_name`,
  `code`, `reasons`, `retry_after_secs`, `layer`, `correlation_id`);
  observe mode passes the response through and fires `on_verdict` with
  the `rate_limited` verdict. A 429 is a verdict, never retried by the
  backoff loop. Covers async, sync, and streaming paths.

## 1.1.0 — 2026-06-08

### Added

- **Dev-mode deny rendering.** `ClavenarOptions(dev_mode=True)` prints a
  per-detector deny panel to stderr on `ClavenarDenied` when the gateway
  runs with verbose verdicts; `render_deny_panel(err)` returns the same
  panel as a string. Dev/staging only — detailed denials are an attacker
  oracle.

## 1.0.0 — 2026-06-07

Renamed for the org's by-language SDK family. **Breaking:** the
distribution is now `clavenar-agent-sdk` (was `clavenar-ai`) and the
import module is `clavenar_agent_sdk` (was `clavenar_ai`); the GitHub
repository is `clavenar-python-sdk`. Update `pip install` and your
imports — the public API is otherwise unchanged from 0.2.0.

## 0.2.0 — 2026-05-12

Feature-complete release. Reaches 1:1 parity with the TS SDK at
`@vanteguardlabs/clavenar-ai-sdk@0.3.0`.

### Added

- **Streaming inspection** for both providers. `stream=True` is
  intercepted; the closing event (Anthropic `content_block_stop`,
  OpenAI `finish_reason="tool_calls"`) is held until clavenar returns a
  verdict. A denied tool raises mid-iteration before partner code can
  act on it. Supports both async (`AsyncAnthropic`, `AsyncOpenAI`)
  and sync streams.
- **Sync clients** — `anthropic.Anthropic` / `openai.OpenAI` (non-
  async) wrap the same way. Detection is via
  `inspect.iscoroutinefunction` on `create`. Sync paths use
  `httpx.Client` and `time.sleep`; observe-mode and pending semantics
  match the async path.
- **Retries** — transient (5xx, network) failures retry up to
  `ClavenarRetryOptions.max_attempts` (default 3) with jittered
  exponential backoff (`base_delay_s=0.1` by default). 200, 403, 202,
  and 4xx other than 5xx never retry. `max_attempts=1` disables
  retries entirely.
- **`ClavenarRetryOptions`** type exported from the public API.
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

- `ClavenarOptions` gained a `retry` field with a default `(3, 0.1)`
  policy. Existing code that constructed `ClavenarOptions` by keyword
  is unaffected; positional construction was never recommended.
- Sync-mode callbacks (`on_verdict`, `on_policy_error`) must be sync
  functions when wrapping a sync client. Passing an `async def`
  callback to a sync wrap raises `ClavenarConfigError` at fire time.

## 0.1.0 — 2026-05-12

Initial MVP release.

- Wraps async Anthropic + OpenAI Python clients.
- Inspects every tool call (`tool_use` / `tool_calls`) in parallel
  before the agent loop sees the response.
- Verdicts: allow / deny (`ClavenarDenied`) / pending (`ClavenarPending`
  with `resolve()` polling).
- Modes: enforce (default) and observe with `on_policy_error`.
- Transport: `httpx.AsyncClient` against `clavenar-lite`'s `/mcp` and
  `/pending/{id}`.
- 19 unit tests, ruff clean, mypy `--strict` clean.
