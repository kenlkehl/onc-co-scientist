"""Native Biomni completions using the existing Azure auth and quota adapter.

Only authentication and admission are shared with CodexCLIProvider. No Codex CLI,
research instructions, or controller loop is invoked by this transport.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
from pathlib import Path

from ..providers.codex_cli import CodexCLIConfig, CodexCLIProvider
from .transport import http_json, without_reasoning, write_json

NATIVE_STEP_INSTRUCTIONS = """Biomni transport protocol for this response:
Return exactly one JSON object with action and content, matching the supplied schema.
This replaces only the XML formatting instructions; all task and tool instructions
still apply. action='execute' means content is literal Python source for Biomni's
real persistent Python interpreter, without XML tags or Markdown fences. Emitting
this object submits the code for execution AFTER this response ends. The next
message will contain Biomni's actual <observation> result. Do not simulate execution,
invent observations, or wait for a tool result within this response. Reasoning or
describing code does not run it. Emit one command and yield to the interpreter.
action='solution' means content is the final answer and ends the native workflow;
use it only when the task is finished, grounded in actual observations. Earlier
assistant commands in the dialogue use native XML and observations use <observation>.
Do not put <execute>, </execute>, <solution>, or </solution> in content itself.
"""

NATIVE_STEP_FORMAT = {
    "type": "json_schema",
    "name": "biomni_native_step",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["execute", "solution"]},
            "content": {"type": "string"},
        },
        "required": ["action", "content"],
        "additionalProperties": False,
    },
}


def native_step_response(result):
    """Decode one visible command; never recover code from private reasoning."""
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        # RunBroker rejects length-truncated responses before any native execution.
        return result

    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate native step field")
            value[key] = item
        return value

    step = json.loads(choice["message"]["content"], object_pairs_hook=unique_keys)
    if not isinstance(step, dict) or set(step) != {"action", "content"}:
        raise ValueError("Native step must contain exactly action and content")
    action, content = step["action"], step["content"]
    if action not in ("execute", "solution") or not isinstance(content, str) or not content.strip():
        raise ValueError("Invalid native step action or empty content")
    if re.search(r"</?(?:execute|solution)>", content, re.IGNORECASE):
        raise ValueError("Nested native control tags are not allowed")
    if action == "execute" and content.lstrip().startswith("```"):
        raise ValueError("Native execute content must be literal code")
    return {
        **result,
        "choices": [
            {
                **choice,
                "message": {"role": "assistant", "content": f"<{action}>\n{content}\n</{action}>"},
            }
        ],
    }


def response_usage(raw, *, use_chat=False):
    usage = raw.get("usage") or {}
    if use_chat:
        return usage
    return {
        target: usage[origin]
        for origin, target in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
            ("input_tokens_details", "prompt_tokens_details"),
            ("output_tokens_details", "completion_tokens_details"),
        )
        if origin in usage
    }


class AzureNativeTransport(CodexCLIProvider):
    def __init__(self, config):
        self.config = config


def azure_provider(spec, audit):
    return AzureNativeTransport(
        CodexCLIConfig(
            model_id=spec.model,
            reasoning_effort=spec.reasoning_effort,
            backend="azure",
            azure_endpoint=spec.base_url,
            azure_cli_executable=spec.azure_cli_executable,
            azure_pacing_dir=spec.azure_pacing_dir,
            audit_dir=str(audit),
        )
    )


def responses_request(body):
    """Translate Biomni's text dialogue without adding another agent or tools."""
    if body.get("tools") or body.get("functions"):
        raise ValueError("Native Biomni Azure transport expects code-based tool execution")
    messages = []
    for message in body["messages"]:
        if message["role"] not in {"system", "developer", "user", "assistant"}:
            raise ValueError("Unsupported native dialogue role")
        if message.get("tool_calls") or message.get("function_call"):
            raise ValueError("Unexpected API function call in native code dialogue")
        content = message.get("content") or ""
        if isinstance(content, list):
            if any(b.get("type") != "text" for b in content):
                raise ValueError("Only text content is supported by the native Azure adapter")
            content = "\n".join(b["text"] for b in content)
        messages.append({"role": message["role"], "content": content})
    request = {
        "model": body["model"],
        "input": messages,
        "reasoning": {"effort": body["reasoning_effort"]},
        "max_output_tokens": body["max_tokens"],
        "store": False,
        "stream": False,
        "truncation": "disabled",
    }
    if body.get("_biomni_native_step"):
        messages.append({"role": "developer", "content": NATIVE_STEP_INSTRUCTIONS})
        request["text"] = {"format": NATIVE_STEP_FORMAT}
    return request


def chat_response(raw):
    content = []
    for item in raw.get("output", []):
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message":
            raise ValueError("Unexpected non-message Azure output")
        for block in item.get("content", []):
            if block.get("type") == "refusal":
                raise ValueError("Azure model refused the native request")
            if block.get("type") == "output_text":
                content.append(block["text"])
    incomplete = raw.get("status") == "incomplete"
    if raw.get("status") not in {"completed", "incomplete"}:
        raise ValueError("Azure did not complete the native request")
    if not content and not incomplete:
        raise ValueError("Azure returned no native assistant text")
    return {
        "id": raw.get("id"),
        "object": "chat.completion",
        "created": raw.get("created_at", int(time.time())),
        "model": raw.get("model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "length" if incomplete else "stop",
                "message": {"role": "assistant", "content": "\n".join(content)},
            }
        ],
        "usage": response_usage(raw),
    }


def chat_request(body):
    """Use the same text dialogue with Biomni's native Chat Completions protocol."""
    request = responses_request(body)
    wire = {
        "model": request["model"],
        "messages": request["input"],
        "reasoning_effort": request["reasoning"]["effort"],
        "max_completion_tokens": request["max_output_tokens"],
        "stream": False,
    }
    if body.get("_biomni_native_step"):
        wire["response_format"] = {
            "type": "json_schema",
            "json_schema": {k: v for k, v in NATIVE_STEP_FORMAT.items() if k != "type"},
        }
    return wire


def native_chat_response(raw):
    choices = raw.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "length"}:
        raise ValueError("Unexpected Azure Chat Completions result")
    message = choices[0].get("message", {})
    if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
        raise ValueError("Azure did not return a native text dialogue response")
    return {**raw, "choices": [{**choices[0], "message": without_reasoning(message)}]}


def azure_complete(spec, body, record, path):
    audit = Path(path).parent.parent / "azure_transport" / Path(path).stem
    audit.mkdir(parents=True, exist_ok=True)
    provider = azure_provider(spec, audit)
    use_chat = spec.azure_api == "chat_completions"
    request = chat_request(body) if use_chat else responses_request(body)
    route = "/chat/completions" if use_chat else "/responses"
    record.update(
        backend="azure",
        wire_api=spec.azure_api,
        native_protocol="structured" if body.get("_biomni_native_step") else "text",
        wire_request=request,
        effective_max_tokens=spec.max_tokens,
        requested_max_tokens=spec.max_tokens,
        completion_policy="fixed",
        token_count_source="provider_usage",
        context_enforcement="service_context_limit" if use_chat else "service_truncation_disabled",
        transport_attempts=[],
    )
    write_json(path, record)
    started = time.monotonic()
    auth_retries = 0
    while True:
        attempt = audit / f"attempt-{len(record['transport_attempts']) + 1:04d}"
        attempt.mkdir()
        dialogue = request["messages"] if use_chat else request["input"]
        with provider.request_slot(attempt, json.dumps(dialogue), spec.max_tokens) as lease:
            remaining = spec.request_timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Azure transport deadline exceeded during quota admission")
            key = provider.environment()["OCS_AZURE_ACCESS_TOKEN"]
            item = {"started_at": time.time(), "status": "inflight"}
            record["transport_attempts"].append(item)
            write_json(path, record)
            try:
                raw = http_json(
                    spec.base_url.rstrip("/") + route, request, timeout=remaining, key=key
                )
            except urllib.error.HTTPError as exc:
                # Authentication/error bodies may contain service internals. Keep only
                # status and retry hints; never persist tokens or request headers.
                item.update(status="rejected", http_status=exc.code)
                write_json(path, record)
                if exc.code == 429:
                    hint = exc.headers.get("retry-after", "")
                    hint_ms = exc.headers.get("retry-after-ms", "")
                    lease.throttled([], f"retry-after: {hint} retry-after-ms: {hint_ms}")
                    continue
                if exc.code in {401, 403} and auth_retries < provider.config.azure_auth_retries:
                    auth_retries += 1
                    time.sleep(provider.auth_retry_delay(auth_retries - 1))
                    continue
                raise RuntimeError(f"Azure native request rejected (HTTP {exc.code})") from None
            # Ambiguous network errors are never automatically replayed.
            item["status"] = "completed"
            lease.succeeded()
            record["raw_response"] = raw
            # Preserve billed usage even when refusal/schema validation rejects output.
            record["provider_usage"] = response_usage(raw, use_chat=use_chat)
            record["prompt_token_count"] = (raw.get("usage") or {}).get(
                "prompt_tokens" if use_chat else "input_tokens"
            )
            write_json(path, record)
            result = native_chat_response(raw) if use_chat else chat_response(raw)
            return native_step_response(result) if body.get("_biomni_native_step") else result
