"""Gemini on Vertex AI, using ADC and the native generateContent API.

The completion facade adapts the repository's common message/tool contract;
requests go directly to Google, without an OpenAI server or API key.
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from .base import ChatMessage, ChatResponse


@dataclass(frozen=True)
class GeminiVertexConfig:
    model_id: str = "gemini-3.8-flash"
    project_id: str | None = None
    location: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 2
    reasoning_effort: str | None = None


class GeminiAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Gemini HTTP {status_code}: {message}")
        self.status_code = status_code


class Completion(SimpleNamespace):
    """Small SDK-independent response object for the existing agent controller."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        def unpack(value):
            if isinstance(value, SimpleNamespace):
                return {k: unpack(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value

        return unpack(self)


def _object(value):
    if isinstance(value, dict):
        return Completion(**{k: _object(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_object(v) for v in value]
    return value


class GeminiVertexClient:
    def __init__(self, config: GeminiVertexConfig, *, credentials=None):
        if config.timeout_s <= 0 or config.max_retries < 0:
            raise ValueError("Gemini timeout must be positive and retries nonnegative")
        if config.reasoning_effort not in {None, "low", "medium", "high"}:
            raise ValueError("Gemini thinking level must be low, medium, high, or omitted")
        self.config = config
        detected_project = None
        if credentials is None:
            try:
                import google.auth
            except ImportError as exc:
                raise RuntimeError(
                    "Install 'onc-co-scientist[gemini-vertex]' for Gemini ADC"
                ) from exc
            credentials, detected_project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        self.credentials = credentials
        self.project_id = (
            config.project_id
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("ANTHROPIC_VERTEX_PROJECT_ID")
            or detected_project
            or getattr(credentials, "quota_project_id", None)
        )
        self.location = config.location or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
        if not self.project_id:
            raise ValueError("Set GOOGLE_CLOUD_PROJECT or pass project_id for Gemini Vertex AI")
        for value in (self.project_id, self.location):
            if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
                raise ValueError("Invalid Vertex project or location")
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else (f"{self.location}-aiplatform.googleapis.com")
        )
        self.base_url = f"https://{host}/v1/projects/{self.project_id}/locations/{self.location}"
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, *, timeout: float):
        return GeminiVertexClient(
            replace(
                self.config, timeout_s=timeout, project_id=self.project_id, location=self.location
            ),
            credentials=self.credentials,
        )

    def generate(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            raise ValueError("Use a Gemini publisher model ID, e.g. gemini-3.8-flash")
        from google.auth.transport.requests import Request

        deadline = time.monotonic() + self.config.timeout_s
        url = f"{self.base_url}/publishers/google/models/{model}:generateContent"
        for attempt in range(self.config.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Gemini request deadline exceeded")
            if not self.credentials.valid:
                auth_request = Request()
                self.credentials.refresh(
                    lambda *a, transport=auth_request, limit=remaining, **kw: transport(
                        *a, **{**kw, "timeout": limit}
                    )
                )
            headers = {"Content-Type": "application/json"}
            self.credentials.apply(headers)
            request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gemini request deadline exceeded during authentication")
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Do not include request headers/credentials in diagnostics.
                raw = exc.read().decode(errors="replace")
                try:
                    message = json.loads(raw).get("error", {}).get("message", "API error")
                except ValueError:
                    message = "API error"
                error = GeminiAPIError(exc.code, message)
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise error from None
            except (urllib.error.URLError, TimeoutError) as exc:
                error = RuntimeError(f"Gemini transport failed: {type(exc).__name__}")
            if attempt == self.config.max_retries:
                raise error
            delay = min(2**attempt, max(0, deadline - time.monotonic()))
            time.sleep(delay)
        raise RuntimeError("Gemini request failed")

    def create(self, **kwargs: Any) -> Completion:
        return _object(self.complete(**kwargs))

    def complete(self, *, model: str, messages: list[dict[str, Any]], **options) -> dict[str, Any]:
        if options.get("service_tier"):
            raise ValueError("OpenAI service tiers are not supported by Gemini Vertex")
        extra = options.get("extra_body") or {}
        if extra.get("chat_template_kwargs") or extra.get("structured_outputs"):
            raise ValueError("vLLM template settings are not supported by Gemini")
        if extra.get("repetition_penalty", 1.0) != 1.0:
            raise ValueError("Gemini does not support repetition_penalty")
        contents, systems, call_names = [], [], {}
        for message in messages:
            role, text = message["role"], message.get("content")
            if role in {"system", "developer"}:
                if text:
                    systems.append({"text": text})
                continue
            if role == "tool":
                name = message.get("name") or call_names.get(message.get("tool_call_id"))
                if not name:
                    raise ValueError("Tool response has no corresponding function call")
                parts = [{"functionResponse": {"name": name, "response": {"result": text}}}]
            else:
                for call in message.get("tool_calls") or []:
                    call_names[call["id"]] = call["function"]["name"]
                if "gemini_parts" in message:
                    parts = copy.deepcopy(message["gemini_parts"])
                else:
                    parts = [{"text": text}] if text else []
                    for call in message.get("tool_calls") or []:
                        fn = call["function"]
                        parts.append(
                            {
                                "functionCall": {
                                    "name": fn["name"],
                                    "args": json.loads(fn["arguments"]),
                                }
                            }
                        )
            if parts:
                contents.append(
                    {"role": "model" if role == "assistant" else "user", "parts": parts}
                )
        config = {
            "maxOutputTokens": options.get("max_completion_tokens", options.get("max_tokens", 4096))
        }
        for source, target in (("temperature", "temperature"), ("top_p", "topP"), ("seed", "seed")):
            if source in options:
                config[target] = options[source]
        if extra.get("top_k", -1) > 0:
            config["topK"] = extra["top_k"]
        effort = options.get("reasoning_effort") or self.config.reasoning_effort
        if effort:
            if effort not in {"low", "medium", "high"}:
                raise ValueError("Gemini thinking level must be low, medium, or high")
            config["thinkingConfig"] = {"thinkingLevel": effort.upper()}
        fmt = options.get("response_format") or {}
        if fmt.get("type") == "json_schema":
            config.update(
                responseMimeType="application/json", responseJsonSchema=fmt["json_schema"]["schema"]
            )
        elif fmt.get("type") == "json_object":
            config["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": config}
        if systems:
            body["systemInstruction"] = {"parts": systems}
        if options.get("tools"):
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parametersJsonSchema": t["function"]["parameters"],
                        }
                        for t in options["tools"]
                    ]
                }
            ]
            choice = options.get("tool_choice", "auto")
            fc = {
                "mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(
                    choice if isinstance(choice, str) else "required", "ANY"
                )
            }
            if isinstance(choice, dict):
                fc["allowedFunctionNames"] = [choice["function"]["name"]]
            body["toolConfig"] = {"functionCallingConfig": fc}
        raw = self.generate(model, body)
        candidates = raw.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {raw.get('promptFeedback', {})}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [
            {
                "id": "gemini_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": p["functionCall"]["name"],
                    "arguments": json.dumps(p["functionCall"].get("args", {})),
                },
            }
            for p in parts
            if "functionCall" in p
        ]
        reason = candidate.get("finishReason")
        if reason not in {"STOP", "MAX_TOKENS"}:
            raise RuntimeError(f"Gemini generation stopped: {reason}")
        usage = raw.get("usageMetadata", {})
        if "candidatesTokenCount" not in usage and "thoughtsTokenCount" not in usage:
            raise RuntimeError("Gemini response omitted generated token usage")
        return {
            "model": raw.get("modelVersion", model),
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": calls,
                        "gemini_parts": parts,
                    },
                    "finish_reason": "length"
                    if reason == "MAX_TOKENS"
                    else ("tool_calls" if calls else "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0)
                + usage.get("thoughtsTokenCount", 0),
                "completion_tokens_details": {
                    "reasoning_tokens": usage.get("thoughtsTokenCount", 0)
                },
            },
            "gemini_response": raw,
        }


class GeminiVertexProvider:
    def __init__(self, config: GeminiVertexConfig):
        self._config = config
        self._client = GeminiVertexClient(config)

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
    ) -> ChatResponse:
        payload = [{"role": "system", "content": system}] if system else []
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        raw = self._client.complete(
            model=self.model_id, messages=payload, temperature=temperature, max_tokens=max_tokens
        )
        choice = raw["choices"][0]
        if choice["finish_reason"] == "length":
            raise RuntimeError("Gemini response truncated; increase max_tokens")
        text = choice["message"]["content"]
        if not text:
            raise RuntimeError("Gemini returned no text")
        return ChatResponse(text=text, model_id=raw["model"], raw=raw)
