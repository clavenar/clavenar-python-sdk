"""Wrap your async Anthropic / OpenAI client with Agent Warden inspection.

MVP scope: async clients only, non-streaming. Streaming, retries,
parallel tool_use, WardenPending.resolve, and onPolicyError follow in
the feature-complete release (Phase 2 sprint 2).
"""

from warden_ai.errors import (
    WardenConfigError,
    WardenDenied,
    WardenPending,
    WardenTransportError,
)
from warden_ai.options import WardenOptions, WardenVerdictContext
from warden_ai.transport import (
    NormalizedToolCall,
    WardenVerdict,
    inspect_tool_use,
    poll_pending_once,
)
from warden_ai.wrap import warden_wrap

__version__ = "0.1.0"

__all__ = [
    "NormalizedToolCall",
    "WardenConfigError",
    "WardenDenied",
    "WardenOptions",
    "WardenPending",
    "WardenTransportError",
    "WardenVerdict",
    "WardenVerdictContext",
    "__version__",
    "inspect_tool_use",
    "poll_pending_once",
    "warden_wrap",
]
