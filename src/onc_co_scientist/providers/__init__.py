"""LLM provider abstraction used by internal steps (synthetic generation, hypothesis matching).

Providers support internal LLM-assisted steps and the expected/surprising research
controller. Native transport clients also serve the structured research runner.
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
