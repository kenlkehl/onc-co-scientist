"""Anthropic-on-Vertex provider.

Uses the ``anthropic[vertex]`` SDK (AnthropicVertex client). Google Cloud
credentials must be configured in the environment (ADC) and the GCP region +
project must be provided via config or env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import ChatMessage, ChatResponse


@dataclass(frozen=True)
class AnthropicVertexConfig:
    model_id: str = "claude-sonnet-4-6"
    region: str | None = None  # defaults to $CLOUD_ML_REGION
    project_id: str | None = None  # defaults to $ANTHROPIC_VERTEX_PROJECT_ID
    max_retries: int = 2


class AnthropicVertexProvider:
    """Thin adapter around ``anthropic.AnthropicVertex``."""

    def __init__(self, config: AnthropicVertexConfig) -> None:
        self._config = config
        self._client = self._build_client()

    def _build_client(self):  # pragma: no cover - requires network/credentials
        try:
            from anthropic import AnthropicVertex  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "anthropic-vertex provider requires the 'anthropic-vertex' extra: "
                "pip install 'onc-co-scientist[anthropic-vertex]'"
            ) from exc

        region = self._config.region or os.getenv("CLOUD_ML_REGION")
        project_id = self._config.project_id or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
        kwargs: dict[str, object] = {"max_retries": self._config.max_retries}
        if region:
            kwargs["region"] = region
        if project_id:
            kwargs["project_id"] = project_id
        return AnthropicVertex(**kwargs)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> ChatResponse:  # pragma: no cover - requires network/credentials
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.messages.create(
            model=self._config.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=api_messages,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return ChatResponse(text=text, model_id=self._config.model_id, raw=response)
