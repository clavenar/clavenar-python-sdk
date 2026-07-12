# TS ↔ Python parity

Both agent SDKs are clients of the same wire contract (the public
`clavenar-specs` TECH_SPEC). Each `POST /mcp` outcome maps to the same
observable behaviour in both; the Python column applies to the async
and sync (`_sync`) paths alike.

| HTTP | Verdict | Retried | TS (`@clavenar/agent-sdk`) | Python (`clavenar-agent-sdk`) |
|---|---|---|---|---|
| 200 | allow | never | response passes through | response passes through |
| 403 | deny | never | enforce: throw `ClavenarDenied`; observe: `onVerdict` | enforce: raise `ClavenarDenied`; observe: `on_verdict` |
| 202 | pending | never | throw `ClavenarPending`, `resolve()` polls | raise `ClavenarPending`, `resolve()` polls |
| 429 | rate_limited / quota_exceeded | never | enforce: throw `ClavenarRateLimited` (`code`, `reasons`, `retryAfterSecs`, `layer`, `correlationId`); observe: `onVerdict` | enforce: raise `ClavenarRateLimited` (`code`, `reasons`, `retry_after_secs`, `layer`, `correlation_id`); observe: `on_verdict` |
| other 4xx | — | never | throw `ClavenarTransportError` | raise `ClavenarTransportError` |
| 5xx / network | — | jittered backoff, 3 attempts | throw `ClavenarTransportError` after retries | raise `ClavenarTransportError` after retries |

429 parsing is lenient in both SDKs: a string `error` code is required;
`verdict` falls back to `rate_limited` unless it is exactly
`quota_exceeded`; `reasons` defaults to `[]`; `retry_after_secs` is
optional (set on `rate_limited` only). The correlation id prefers the
`X-Clavenar-Correlation-Id` response header, falling back to the body's
`correlation_id`. A malformed 429 body (no string `error`) raises the
transport error with status 429.
