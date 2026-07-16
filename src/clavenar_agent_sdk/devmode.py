"""Developer-mode rendering of a denied tool call's verbose-verdict detail.

When `ClavenarOptions.dev_mode` is on, the SDK writes a readable panel
(per-detector scores, degraded lanes, reasons, correlation id) to stderr
before raising `ClavenarDenied`. Pure render + a best-effort emit.
"""

from __future__ import annotations

import contextlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clavenar_agent_sdk.errors import ClavenarDenied


def render_deny_panel(denied: ClavenarDenied) -> str:
    """Render a denied tool call as a readable console panel string.

    Falls back to a hint when the gateway didn't include `detail`
    (verbose-verdicts off).
    """
    lines: list[str] = [f"━━ clavenar denied: {denied.tool_name} ━━"]

    meta: list[str] = []
    if denied.layer:
        meta.append(f"layer={denied.layer}")
    if denied.intent_category:
        meta.append(f"intent={denied.intent_category}")
    if denied.correlation_id:
        meta.append(f"correlation={denied.correlation_id}")
    if meta:
        lines.append("  " + "  ".join(meta))

    if denied.reasons:
        lines.append("  reasons:")
        lines.extend(f"    - {r}" for r in denied.reasons)

    detail = denied.detail or {}
    detectors = detail.get("detectors") if isinstance(detail, dict) else None
    if isinstance(detectors, list) and detectors:
        lines.append("  detectors:")
        for d in detectors:
            if not isinstance(d, dict):
                continue
            name = str(d.get("detector", "?"))
            try:
                score = f"{float(d.get('score', 0)):.2f}"
            except (TypeError, ValueError):
                score = "?"
            flag = "  ⚠ flagged" if d.get("flagged") is True else ""
            lines.append(f"    {name:<22}{score}{flag}")
        degraded = detail.get("degraded")
        if isinstance(degraded, list) and degraded:
            lines.append(f"  degraded: {', '.join(str(x) for x in degraded)}")
    else:
        lines.append(
            "  (no per-detector detail — run the gateway with CLAVENAR_PROXY_VERBOSE_VERDICTS=true)"
        )

    if denied.correlation_id:
        lines.append(
            f"  trace: look up correlation {denied.correlation_id} in the console audit trail"
        )
    return "\n".join(lines)


def emit_deny_panel(denied: ClavenarDenied) -> None:
    """Write the dev-mode deny panel to stderr. Best-effort — never raises."""
    with contextlib.suppress(Exception):
        print(render_deny_panel(denied), file=sys.stderr)
