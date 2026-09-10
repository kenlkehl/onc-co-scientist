"""Claude on Vertex: shared text and native-tool transport using ADC.

Native assistant blocks (including signed thinking) are retained verbatim. The
OpenAI-shaped envelope is only an internal interface to the structured runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .base import ChatMessage, ChatResponse


@dataclass(frozen=True)
class AnthropicVertexConfig:
    model_id: str = "claude-sonnet-4-6"
    region: str | None = None
    project_id: str | None = None
    max_retries: int = 2
    timeout_s: float = 120.0
    reasoning_effort: str | None = None

    def __post_init__(self):
        if self.reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Claude effort must be low, medium, high, xhigh, or max")
        if self.timeout_s <= 0 or self.max_retries < 0:
            raise ValueError("Positive timeout_s and nonnegative max_retries required")


class AnthropicVertexClient:
    def __init__(self, config: AnthropicVertexConfig):
        self.config = config
        self.region = (
            config.region
            or os.getenv("CLOUD_ML_REGION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or "global"
        )
        self.project_id = (
            config.project_id
            or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        self._client = self._build_client()
        # The SDK can resolve the project from ADC.
        self.project_id = self.project_id or getattr(self._client, "project_id", None)
        host = (
            "aiplatform.googleapis.com"
            if self.region == "global"
            else (f"{self.region}-aiplatform.googleapis.com")
        )
        self.base_url = (
            f"https://{host}/v1/projects/{self.project_id}/locations/{self.region}"
            "/publishers/anthropic/models"
        )

    def _build_client(self):  # pragma: no cover - credential setup
        try:
            from anthropic import AnthropicVertex
        except ImportError as exc:
            raise RuntimeError(
                "Install the Vertex SDK: pip install -U 'onc-co-scientist[anthropic-vertex]'"
            ) from exc
        kwargs: dict[str, Any] = {
            "region": self.region,
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout_s,
        }
        if self.project_id:
            kwargs["project_id"] = self.project_id
        return AnthropicVertex(**kwargs)

    @staticmethod
    def _history(messages):
        system, history = [], []
        for message in messages:
            role = message["role"]
            if role == "system":
                system.append(message["content"])
                continue
            if role == "tool":
                role = "user"
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message.get("content") or "",
                    }
                ]
            elif role == "assistant" and "anthropic_content" in message:
                blocks = message["anthropic_content"]
            else:
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls") or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "input": json.loads(call["function"]["arguments"]),
                        }
                    )
            if role not in {"user", "assistant"} or not blocks:
                raise ValueError(f"Unsupported or empty Claude message: {role}")
            # Parallel tool results must be adjacent in a single user message.
            if history and history[-1]["role"] == role:
                history[-1]["content"].extend(blocks)
            else:
                history.append({"role": role, "content": list(blocks)})
        return "\n\n".join(system), history

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_completion_tokens: int | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
    ) -> dict:
        if service_tier is not None:
            raise ValueError("Claude Vertex does not support OpenAI service tiers")
        effort = reasoning_effort or self.config.reasoning_effort
        if effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Claude reasoning effort")
        system, history = self._history(messages)
        limit = max_completion_tokens if max_completion_tokens is not None else max_tokens
        if limit < 1:
            raise ValueError("max_tokens must be positive")
        body: dict[str, Any] = {"model": model, "messages": history, "max_tokens": limit}
        if system:
            body["system"] = system
        # Opus 4.7+ rejects sampling parameters. Our protocol's temperature=0 is
        # omitted for these models; the effective settings are logged below.
        modern_opus = model.startswith(("claude-opus-5", "claude-opus-4-7", "claude-opus-4-8"))
        if temperature is not None and not modern_opus:
            body["temperature"] = temperature
        if effort:
            body["extra_body"] = {"output_config": {"effort": effort}}
        if tools:
            if tool_choice != "auto":
                raise ValueError("Claude research tools require tool_choice=auto")
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
            body["tool_choice"] = {"type": "auto"}
        # Streaming supports long generations without the SDK's non-streaming
        # timeout guard. get_final_message aggregates native content and usage.
        with self._client.messages.stream(**body) as stream:
            response = stream.get_final_message()
        raw = response.model_dump(mode="json", exclude_none=True)
        parts = raw.get("content", [])
        calls = [
            {
                "id": p["id"],
                "type": "function",
                "function": {
                    "name": p["name"],
                    "arguments": json.dumps(p["input"]),
                },
            }
            for p in parts
            if p["type"] == "tool_use"
        ]
        text = "".join(p["text"] for p in parts if p["type"] == "text")
        usage = raw.get("usage", {})
        error = None
        reason = raw.get("stop_reason")
        if reason == "max_tokens":
            error = "Output token limit reached: Claude response truncated; increase max_tokens"
        elif reason not in {"end_turn", "tool_use", "stop_sequence"}:
            error = f"Claude generation stopped: {reason}"
        elif not text and not calls:
            error = "Claude returned no text or tool calls"
        if usage.get("output_tokens") is None or usage.get("input_tokens") is None:
            error = "Claude response omitted token usage"
        returned_model = raw.get("model", model)
        if returned_model != model:
            error = f"Claude returned a different model: {returned_model!r}, requested {model!r}"
        return {
            "model": returned_model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": calls,
                        "anthropic_content": parts,
                    },
                    "finish_reason": "length"
                    if reason == "max_tokens"
                    else ("tool_calls" if calls else "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": sum(
                    usage.get(k, 0) or 0
                    for k in (
                        "input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                ),
                # Anthropic output_tokens already includes thinking; do not add it twice.
                "completion_tokens": usage.get("output_tokens"),
            },
            "anthropic_request": body,
            "anthropic_response": raw,
            "adapter_error": error,
        }


class AnthropicVertexProvider:
    def __init__(self, config: AnthropicVertexConfig):
        self._config = config
        self._client = AnthropicVertexClient(config)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def chat_for_retry(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
        final_retry: bool = False,
    ) -> ChatResponse:
        # Never silently change effort/model on a repair. Return failure usage to
        # the matrix coordinator so truncated generations remain in accounting.
        payload = [{"role": "system", "content": system}] if system else []
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        raw = self._client.complete(
            model=self.model_id, messages=payload, temperature=temperature, max_tokens=max_tokens
        )
        return ChatResponse(
            text=raw["choices"][0]["message"]["content"], model_id=raw["model"], raw=raw
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> ChatResponse:
        response = self.chat_for_retry(
            messages, temperature=temperature, max_tokens=max_tokens, system=system
        )
        if response.raw.get("adapter_error"):
            raise RuntimeError(response.raw["adapter_error"])
        return response
