"""Local vLLM provider via its OpenAI-compatible chat completions endpoint.

Point at any ``vllm serve`` deployment that exposes ``/v1/chat/completions``.
This is the path to reach open-weights models (Llama, Qwen, etc.) for
hypothesis-matching and downstream Aim 2.2 intervention work.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ChatMessage, ChatResponse


@dataclass(frozen=True)
class VLLMConfig:
    model_id: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM accepts any non-empty string by default
    timeout_s: float = 120.0
    reasoning_effort: str | None = None
    service_tier: str | None = None
    disable_thinking_on_final_retry: bool = False


class VLLMProvider:
    """Thin adapter around ``openai.OpenAI`` pointed at a vLLM endpoint."""

    def __init__(self, config: VLLMConfig) -> None:
        self._config = config
        self._client = self._build_client()

    def _build_client(self):  # pragma: no cover - requires network
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "vllm-openai provider requires the 'vllm-openai' extra: "
                "pip install 'onc-co-scientist[vllm-openai]'"
            ) from exc
        return OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout_s,
        )

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
        disable_thinking: bool = False,
    ) -> ChatResponse:  # pragma: no cover - requires network
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend({"role": m.role, "content": m.content} for m in messages)
        options = {}
        if disable_thinking:
            options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if self._config.reasoning_effort is not None:
            options["reasoning_effort"] = self._config.reasoning_effort
        if self._config.service_tier is not None:
            options["service_tier"] = self._config.service_tier
        response = self._client.chat.completions.create(
            model=self._config.model_id,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **options,
        )
        text = response.choices[0].message.content or ""
        return ChatResponse(text=text, model_id=self._config.model_id, raw=response)

    def chat_for_retry(self, messages, *, final_retry=False, **kwargs):
        return self.chat(
            messages,
            disable_thinking=final_retry and self._config.disable_thinking_on_final_retry,
            **kwargs,
        )
