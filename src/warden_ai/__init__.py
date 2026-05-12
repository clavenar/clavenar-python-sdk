"""Wrap your Anthropic / OpenAI client with Agent Warden inspection.

Supports async (`AsyncAnthropic`, `AsyncOpenAI`) and sync
(`Anthropic`, `OpenAI`) clients, with non-streaming and streaming
responses. Tool calls are inspected by warden-lite before the partner
sees them; a denied call raises `WardenDenied` (mid-iteration for
streams), a parked call raises `WardenPending` with an `await
.resolve()` helper that blocks until an operator decides.
"""

from warden_ai.errors import (
    WardenConfigError,
    WardenDenied,
    WardenPending,
    WardenTransportError,
)
from warden_ai.options import (
    WardenOptions,
    WardenRetryOptions,
    WardenVerdictContext,
)
from warden_ai.realtime import (
    inspect_realtime_function_call,
    is_realtime_function_call_done,
    normalize_realtime_function_call,
)
from warden_ai.stream import (
    wrap_anthropic_stream,
    wrap_anthropic_stream_sync,
    wrap_openai_chat_stream,
    wrap_openai_chat_stream_sync,
)
from warden_ai.transport import (
    NormalizedToolCall,
    WardenVerdict,
    inspect_tool_use,
    inspect_tool_use_sync,
    poll_pending_once,
    poll_pending_once_sync,
)
from warden_ai.wrap import warden_wrap

__version__ = "0.2.0"

__all__ = [
    "NormalizedToolCall",
    "WardenConfigError",
    "WardenDenied",
    "WardenOptions",
    "WardenPending",
    "WardenRetryOptions",
    "WardenTransportError",
    "WardenVerdict",
    "WardenVerdictContext",
    "__version__",
    "inspect_realtime_function_call",
    "inspect_tool_use",
    "inspect_tool_use_sync",
    "is_realtime_function_call_done",
    "normalize_realtime_function_call",
    "poll_pending_once",
    "poll_pending_once_sync",
    "warden_wrap",
    "wrap_anthropic_stream",
    "wrap_anthropic_stream_sync",
    "wrap_openai_chat_stream",
    "wrap_openai_chat_stream_sync",
]
