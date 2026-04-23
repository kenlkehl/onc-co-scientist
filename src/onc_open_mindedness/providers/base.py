"""LLMProvider protocol and shared message types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model_id: str
    raw: object = field(default=None, repr=False)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal chat-style interface shared by all providers."""

    @property
    def model_id(self) -> str: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> ChatResponse: ...
