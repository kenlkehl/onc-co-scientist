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
    disable_thinking_on_final_retry: bool = True
    sampling_profile: str | None = "auto"
    json_object_output: bool = False

    def __post_init__(self):
        if self.sampling_profile not in (None, "auto", "qwen3_8", "gemma4"):
            raise ValueError("Unknown vLLM sampling profile")


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

    def generation_options(self, *, final_retry=False, truncation_failures=0):
        """Explicit options for execution and the durable request audit."""
        disabled = final_retry and truncation_failures >= 2 and self._config.disable_thinking_on_final_retry
        profile = self._config.sampling_profile
        if profile == "auto":
            name = self.model_id.lower()
            profile = "gemma4" if "gemma-4" in name or "gemma4" in name else "qwen3_8" if "qwen3.8" in name else None
        options = {}
        extra = {}
        if profile == "gemma4":
            options.update(temperature=1.0, top_p=0.95)
            extra.update(top_k=64)
            extra["chat_template_kwargs"] = {"enable_thinking": not disabled}
        elif profile == "qwen3_8":
            options.update(temperature=0.7 if disabled else 1.0,
                           top_p=0.8 if disabled else 0.95,
                           presence_penalty=1.5 if disabled else 0.0)
            extra.update(top_k=20, min_p=0.0, repetition_penalty=1.0)
            extra["chat_template_kwargs"] = {"enable_thinking": not disabled}
        elif disabled:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            options["extra_body"] = extra
        if self._config.json_object_output:
            options["response_format"] = {"type": "json_object"}
        return options

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
        disable_thinking: bool = False,
        generation_options: dict | None = None,
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
        options.update(generation_options if generation_options is not None else self.generation_options())
        effective_temperature = options.pop("temperature", temperature)
        response = self._client.chat.completions.create(
            model=self._config.model_id,
            messages=api_messages,
            temperature=effective_temperature,
            max_tokens=max_tokens,
            **options,
        )
        text = response.choices[0].message.content or ""
        raw = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(raw, dict):
            raw["generation_options"] = {"temperature": effective_temperature, **options}
            if response.choices[0].finish_reason == "length":
                raw["adapter_error"] = "Output token limit reached; incomplete generation rejected"
                text = ""
        return ChatResponse(text=text, model_id=self._config.model_id, raw=raw)

    def chat_for_retry(self, messages, *, final_retry=False, truncation_failures=0, **kwargs):
        return self.chat(
            messages,
            disable_thinking=final_retry and truncation_failures >= 2 and self._config.disable_thinking_on_final_retry,
            generation_options=self.generation_options(final_retry=final_retry, truncation_failures=truncation_failures),
            **kwargs,
        )
