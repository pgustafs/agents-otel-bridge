"""OpenTelemetry tracing bridge for OpenAI Agents SDK."""

from .bridge import (
    setup_otel_tracing,
    OTelBridgeProcessor,
    EnhancedOTelBridgeProcessor,
)

__version__ = "0.0.2"
__all__ = [
    "setup_otel_tracing",
    "OTelBridgeProcessor",
    "EnhancedOTelBridgeProcessor",
]
