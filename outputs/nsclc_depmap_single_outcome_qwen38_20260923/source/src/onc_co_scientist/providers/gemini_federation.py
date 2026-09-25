"""Audited, tool-free Vertex transport for federated scientific workflows only."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..harness.durable_io import atomic_write_json
from .base import ChatResponse, ProviderInfrastructureError
from .federated_prompt import FederatedPromptLayout
from .gemini_vertex import GeminiVertexClient, GeminiVertexConfig


@dataclass(frozen=True)
class GeminiFederationConfig(GeminiVertexConfig):
    audit_dir: str = ""
    resume_audit: bool = False
    max_retries: int = 8


def now():
    return datetime.now(UTC).isoformat()


def usage_receipt(raw):
    u = raw.get("usageMetadata", {})
    if "promptTokenCount" not in u or not (
        "candidatesTokenCount" in u or "thoughtsTokenCount" in u
    ):
        raise ProviderInfrastructureError("Gemini response omitted token usage")
    inp = u["promptTokenCount"]
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    cached = u.get("cachedContentTokenCount", 0)
    if not all(type(n) is int and n >= 0 for n in (inp, out, cached)) or cached > inp:
        raise ProviderInfrastructureError("Invalid Gemini token accounting")
    return dict(input_tokens=inp, output_tokens=out, cached_input_tokens=cached,
                cache_write_input_tokens=0, reasoning_output_tokens=u.get("thoughtsTokenCount", 0))


class GeminiFederationProvider:
    def __init__(self, config: GeminiFederationConfig):
        if not config.audit_dir:
            raise ValueError("Federated Gemini requires a durable audit directory")
        self.config = config
        self.client = GeminiVertexClient(config)
        self.root = Path(config.audit_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.layout = FederatedPromptLayout(str(self.root))
        self.lock = threading.Lock()

    @property
    def model_id(self):
        return self.config.model_id

    def request_body(self, messages, system, max_tokens):
        wire, _, metadata = self.layout.render(messages, system,
            "Follow the supplied scientific task and return its requested JSON record. "
            "All evidence is supplied in the messages; an external controller executes analyses. "
            "No external tools are available.")
        systems, contents = [], []
        for m in wire:
            value = m["content"]
            parts = [{"text": value}] if isinstance(value, str) else [{"text": p["text"]} for p in value]
            if m["role"] in {"system", "developer"}:
                systems.extend(parts)
            else:
                role = "model" if m["role"] == "assistant" else "user"
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append(dict(role=role, parts=parts))
        body = dict(systemInstruction=dict(parts=systems), contents=contents,
                    generationConfig=dict(maxOutputTokens=min(max_tokens, 65536),
                        thinkingConfig=dict(thinkingLevel=(self.config.reasoning_effort or "medium").upper()),
                        responseMimeType="application/json"))
        # No sampling knobs (unsupported by 3.8), tools, retrieval, attachments, or code execution.
        return body, metadata

    def _send(self, body, call):
        import requests
        from google.auth.transport.requests import Request

        url = f"{self.client.base_url}/publishers/google/models/{self.model_id}:generateContent"
        existing = list(call.glob("attempt-*"))
        offset = max([int(p.name.split("-")[1]) for p in existing] or [0])
        force_refresh = False
        for index in range(self.config.max_retries + 1):
            attempt = call / f"attempt-{offset + index + 1:04d}"
            attempt.mkdir()
            atomic_write_json(call / "activity.json", dict(status="requesting", updated_at=now(), attempt=attempt.name))
            started = time.monotonic()
            try:
                credentials = self.client.credentials
                if force_refresh or not credentials.valid:
                    credentials.refresh(Request())
                    force_refresh = False
                headers = {"Content-Type": "application/json"}
                credentials.apply(headers)
                response = requests.post(url, json=body, headers=headers, timeout=(30, self.config.timeout_s))
                code = response.status_code
                atomic_write_json(attempt / "http.json", dict(status=code, ended_at=now()))
                if code == 200:
                    raw = response.json()
                    atomic_write_json(attempt / "response.json", raw)
                    receipt = usage_receipt(raw)
                    # Global standard pay-as-you-go introductory prices through 2026-12-31.
                    cost = ((receipt["input_tokens"] - receipt["cached_input_tokens"]) * .75
                            + receipt["cached_input_tokens"] * .075 + receipt["output_tokens"] * 3.75) / 1e6
                    atomic_write_json(attempt / "metrics.json", dict(backend="gemini_vertex", usage=receipt,
                        estimated_cost_usd=cost, duration_seconds=time.monotonic()-started, ended_at=now()))
                    return raw, receipt, offset + index + 1, cost
                try:
                    message = response.json().get("error", {}).get("message", "API error")
                except ValueError:
                    message = "API error"
                error = f"Gemini HTTP {code}: {message}"
                retryable = code in {401, 408, 429, 500, 502, 503, 504}
                force_refresh = code == 401
                retry_after = response.headers.get("Retry-After", "0")
                delay = max(min(60, 2**index), min(float(retry_after), 120) if retry_after.isdigit() else 0)
            except ProviderInfrastructureError:
                raise
            except Exception as exc:
                # Never persist authentication headers or credential objects.
                error = f"Gemini transport/authentication error: {type(exc).__name__}"
                retryable, delay = True, min(60, 2**index)
            atomic_write_json(attempt / "error.json", dict(error=error, retryable=retryable, ended_at=now()))
            if not retryable or index == self.config.max_retries:
                atomic_write_json(call / "activity.json", dict(status="interrupted", error=error, updated_at=now()))
                raise ProviderInfrastructureError(error)
            atomic_write_json(call / "activity.json", dict(status="retry_wait", error=error,
                retry_in_seconds=delay, updated_at=now()))
            time.sleep(delay)
        raise AssertionError("unreachable")

    def chat(self, messages, *, temperature=0.0, max_tokens=65536, system=None):
        # A fresh direct call is always an independent sample, even for identical text.
        return self.chat_for_call(messages, call_identity=uuid.uuid4().hex,
            temperature=temperature, max_tokens=max_tokens, system=system)

    def chat_for_call(self, messages, *, call_identity, temperature=0.0, max_tokens=65536, system=None):
        with self.lock:
            body, layout = self.request_body(messages, system, max_tokens)
            # The coordinator identity includes participant, stage, attempt and scope.
            # Equal prompts from different peers must never share a sampled completion.
            identity = hashlib.sha256(json.dumps([call_identity, body], sort_keys=True).encode()).hexdigest()
            call = self.root / ("call-" + identity)
            call.mkdir(exist_ok=True)
            request_path = call / "request.json"
            if request_path.exists() and json.loads(request_path.read_text()) != body:
                raise ProviderInfrastructureError("Gemini request identity collision")
            atomic_write_json(request_path, body)
            atomic_write_json(call / "identity.json", dict(participant_call_identity=call_identity,
                request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()))
            atomic_write_json(call / "layout.json", layout)
            cached = call / "completion.json"
            if cached.exists():
                saved = json.loads(cached.read_text())
                return ChatResponse(text=saved["text"], model_id=self.model_id, raw=saved["raw"])
            raw, usage, attempts, cost = self._send(body, call)
            candidates = raw.get("candidates") or []
            candidate = candidates[0] if candidates else {}
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            finish = candidate.get("finishReason")
            error = None
            if any("functionCall" in p for p in parts):
                error = "Unexpected tool call in tool-free Gemini response"
            elif finish == "MAX_TOKENS":
                error = "Output token limit reached in Gemini response"
            elif finish != "STOP" or not text:
                error = f"Gemini returned no usable answer: {finish}"
            metrics = dict(backend="gemini_vertex", usage=usage, infrastructure_attempts=attempts,
                model_version=raw.get("modelVersion"), estimated_cost_usd=cost)
            atomic_write_json(call / "metrics.json", metrics)
            value = dict(metrics=metrics, adapter_error=error)
            atomic_write_json(cached, dict(text=text, raw=value))
            atomic_write_json(call / "activity.json", dict(status="completed", updated_at=now(), error=error))
            return ChatResponse(text=text, model_id=self.model_id, raw=value)
