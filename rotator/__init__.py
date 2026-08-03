"""SmartRotator — multi-provider, multi-key, round-robin LLM gateway."""

from .router import Rotator
from .providers import (
    AllProvidersExhausted,
    ChatMessage,
    ChatResult,
    ImageInput,
    ProviderError,
    RateLimitError,
)

__all__ = [
    "Rotator",
    "ChatMessage",
    "ChatResult",
    "ImageInput",
    "ProviderError",
    "RateLimitError",
    "AllProvidersExhausted",
]
__version__ = "0.1.0"
