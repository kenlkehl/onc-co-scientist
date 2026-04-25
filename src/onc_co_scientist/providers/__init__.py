"""LLM provider abstraction used by internal steps (synthetic generation, hypothesis matching).

Providers are NOT used to run the agentic benchmark loop itself - that is
performed by the user's external harness of choice. They are used only for
internal LLM-assisted work inside this package.
"""

from .base import ChatMessage, ChatResponse, LLMProvider
from .registry import ProviderConfig, get_provider

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMProvider",
    "ProviderConfig",
    "get_provider",
]
