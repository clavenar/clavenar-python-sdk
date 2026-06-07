# Security Policy

`clavenar-ai` (Python) is the client half of a security product. We take
vulnerability reports seriously and aim to acknowledge every report
within 72 hours.

## Reporting a vulnerability

Email **vanteguardlabs@gmail.com** with:

- A description of the issue and the impact you observed.
- Steps to reproduce. A minimal proof-of-concept is appreciated but not
  required if the issue is structural.
- Affected file path, commit hash, and (if applicable) the
  `ClavenarOptions` configuration that reproduces the issue.
- Whether you would like public credit in the disclosure announcement.

PGP/GPG: not yet available. If you need an encrypted channel, mention it
in your initial email and we will arrange one.

## Scope

In scope:

- The `clavenar_ai` package: client-side request shaping, the wrap
  surface (`clavenar_wrap`, `clavenar_messages`), inspection request
  signing, retry / pending poll loops, and the
  `ClavenarDenied` / `ClavenarPending` / `ClavenarTransportError` raise
  contract.
- Transport security between the SDK and the inspect endpoint
  (`endpoint` URL handling, TLS verification posture, header
  forwarding via `extra_headers`).
- The streaming intercept (Anthropic `content_block_stop` /
  OpenAI `finish_reason="tool_calls"`) — verdict-before-tool ordering.
- Sync vs async client detection in `clavenar_wrap`.

Out of scope:

- Authentication against upstream model providers (Anthropic API key,
  OpenAI API key). Those flow through the upstream SDKs unchanged.
- Sandboxing the Python runtime itself. The SDK runs in your process;
  arbitrary Python is out of our trust boundary by construction.
- Issues in `httpx`, `anthropic`, or `openai` upstream — please report
  to those projects directly. We track CVE advisories that affect
  pinned versions in `CHANGELOG.md`.
- Findings against the demo flow on `demo.clavenar.com`
  when caused by demo-specific configuration (the demo accepts
  `X-Clavenar-Demo-Prefix` headers visitors mint themselves).

## Safe harbor

We will not pursue civil or criminal action against researchers who:

- Make a good-faith effort to avoid privacy violations, destruction of
  data, and interruption or degradation of our services.
- Only interact with accounts they own or with explicit permission of
  the account holder.
- Give us reasonable time to respond before disclosing publicly.
- Do not exploit a security issue beyond what is necessary to confirm
  it.

## Response targets

- **72 hours**: acknowledgement of the report.
- **7 days**: triage outcome (accepted / duplicate / out-of-scope) and a
  CVE assignment plan if applicable.
- **90 days**: public disclosure, coordinated with the reporter.

We may extend the disclosure window for issues that require a
coordinated multi-language fix (the TypeScript SDK at
`clavenar-typescript-sdk` shares the same wire contract); we will tell you in
advance and explain why.
