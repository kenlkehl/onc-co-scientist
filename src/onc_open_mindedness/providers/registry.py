"""Provider registry keyed by config dicts.

A config dict has the shape::

    {"kind": "anthropic_vertex", "model_id": "claude-sonnet-4-6", ...}
    {"kind": "vllm_openai", "model_id": "meta-llama/...", "base_url": "..."}
"""

from __future__ import annotations

from typing import Any

from .anthropic_vertex import AnthropicVertexConfig, AnthropicVertexProvider
from .base import LLMProvider
from .vllm_openai import VLLMConfig, VLLMProvider

ProviderConfig = dict[str, Any]


def get_provider(config: ProviderConfig) -> LLMProvider:
    kind = config.get("kind")
    if kind == "anthropic_vertex":
        return AnthropicVertexProvider(
            AnthropicVertexConfig(
                model_id=config.get("model_id", "claude-sonnet-4-6"),
                region=config.get("region"),
                project_id=config.get("project_id"),
                max_retries=int(config.get("max_retries", 2)),
            )
        )
    if kind == "vllm_openai":
        if "model_id" not in config:
            raise ValueError("vllm_openai provider config requires a 'model_id'.")
        return VLLMProvider(
            VLLMConfig(
                model_id=config["model_id"],
                base_url=config.get("base_url", "http://localhost:8000/v1"),
                api_key=config.get("api_key", "EMPTY"),
                timeout_s=float(config.get("timeout_s", 120.0)),
            )
        )
    raise ValueError(
        f"Unknown provider kind {kind!r}. Supported: 'anthropic_vertex', 'vllm_openai'."
    )
