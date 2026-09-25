"""Fingerprint-checked CAA transport for the shared discovery controller."""

from __future__ import annotations

from dataclasses import dataclass

from .vllm_openai import VLLMConfig, VLLMProvider


@dataclass(frozen=True)
class CAAConfig(VLLMConfig):
    server_fingerprint: str = ""
    sampling_profile: str | None = "gemma4"
    json_object_output: bool = False

    def __post_init__(self):
        super().__post_init__()
        if len(self.server_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in self.server_fingerprint
        ):
            raise ValueError("CAA requires the frozen /v1/caa server fingerprint")
        if self.json_object_output:
            raise ValueError("CAA uses shared JSON stage prompts, not constrained decoding")


class CAAProvider(VLLMProvider):
    def generation_options(self, **kwargs):
        options = super().generation_options(**kwargs)
        options.setdefault("extra_body", {})["caa_expected_fingerprint"] = (
            self._config.server_fingerprint
        )
        return options

    def chat(self, *args, **kwargs):
        response = super().chat(*args, **kwargs)
        raw = response.raw
        provenance = (raw.get("caa_provenance") or {}) if isinstance(raw, dict) else {}
        if provenance.get("fingerprint") != self._config.server_fingerprint:
            raise RuntimeError("CAA response has missing or changed serving provenance")
        if provenance.get("arm", {}).get("model_id") != self.model_id:
            raise RuntimeError("CAA response arm does not match the requested model alias")
        usage = raw.get("usage") or {}
        if any(not isinstance(usage.get(k), int) or usage[k] < 0
               for k in ("prompt_tokens", "completion_tokens")):
            raise RuntimeError("CAA response is missing generated-token accounting")
        return response
