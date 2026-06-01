"""Wrap your Anthropic / OpenAI client with Clavenar inspection.

Supports async (`AsyncAnthropic`, `AsyncOpenAI`) and sync
(`Anthropic`, `OpenAI`) clients, with non-streaming and streaming
responses. Tool calls are inspected by clavenar-lite before the partner
sees them; a denied call raises `ClavenarDenied` (mid-iteration for
streams), a parked call raises `ClavenarPending` with an `await
.resolve()` helper that blocks until an operator decides.
"""

from clavenar_ai.errors import (
    ClavenarConfigError,
    ClavenarDenied,
    ClavenarPending,
    ClavenarTransportError,
)
from clavenar_ai.options import (
    ClavenarOptions,
    ClavenarRetryOptions,
    ClavenarVerdictContext,
)
from clavenar_ai.realtime import (
    inspect_realtime_function_call,
    is_realtime_function_call_done,
    normalize_realtime_function_call,
)
from clavenar_ai.stream import (
    wrap_anthropic_stream,
    wrap_anthropic_stream_sync,
    wrap_openai_chat_stream,
    wrap_openai_chat_stream_sync,
)
from clavenar_ai.transport import (
    NormalizedToolCall,
    ClavenarVerdict,
    inspect_tool_use,
    inspect_tool_use_sync,
    poll_pending_once,
    poll_pending_once_sync,
)
from clavenar_ai.wrap import clavenar_wrap

__version__ = "0.2.0"

__all__ = [
    "NormalizedToolCall",
    "ClavenarConfigError",
    "ClavenarDenied",
    "ClavenarOptions",
    "ClavenarPending",
    "ClavenarRetryOptions",
    "ClavenarTransportError",
    "ClavenarVerdict",
    "ClavenarVerdictContext",
    "__version__",
    "inspect_realtime_function_call",
    "inspect_tool_use",
    "inspect_tool_use_sync",
    "is_realtime_function_call_done",
    "normalize_realtime_function_call",
    "poll_pending_once",
    "poll_pending_once_sync",
    "clavenar_wrap",
    "wrap_anthropic_stream",
    "wrap_anthropic_stream_sync",
    "wrap_openai_chat_stream",
    "wrap_openai_chat_stream_sync",
]
