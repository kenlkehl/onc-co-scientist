"""Fingerprint-checked CAA transport for the shared discovery controller."""

from __future__ import annotations

from dataclasses import dataclass

from .base import ProviderInfrastructureError
from .vllm_openai import VLLMConfig, VLLMProvider


@dataclass(frozen=True)
class CAAConfig(VLLMConfig):
    timeout_s: float = 21600.0
    max_retries: int = 0
    server_fingerprint: str = ""
    sampling_profile: str | None = "gemma4"
    json_object_output: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.max_retries != 0:
            raise ValueError("CAA transport retries must be disabled; failures halt the campaign")
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
        try:
            response = super().chat(*args, **kwargs)
        except Exception as exc:
            # No hidden HTTP retries and no scientific repair prompts for a
            # crashed, unreachable, timed-out or misconfigured inference server.
            raise ProviderInfrastructureError(f"CAA request failed: {exc}") from exc
        raw = response.raw
        provenance = (raw.get("caa_provenance") or {}) if isinstance(raw, dict) else {}
        if (not isinstance(provenance, dict)
                or provenance.get("fingerprint") != self._config.server_fingerprint):
            raise ProviderInfrastructureError(
                "CAA response has missing or changed serving provenance"
            )
        arm = provenance.get("arm") or {}
        if not isinstance(arm, dict) or arm.get("model_id") != self.model_id:
            raise ProviderInfrastructureError(
                "CAA response arm does not match the requested model alias"
            )
        usage = raw.get("usage") or {}
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            raise ProviderInfrastructureError("CAA response is missing generated-token accounting")
        return response
