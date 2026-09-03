#!/usr/bin/env python3
"""Bridge the experiment harness to an OpenAI-compatible vLLM chat server.

The server does not need native function-calling support.  The model selects a
schema-constrained controller action (run Python or finish), and this adapter
executes Python inside a bubblewrap sandbox with a read-only public workspace,
a writable per-session scratch directory, and no network.  Accepted stage
artifacts are retained as bounded conversation history keyed by harness
``session_id`` so persistent and non-persistent workflows remain distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from onc_co_scientist.harness.runtime import AgentArtifact, AgentResponse, AgentUsage

try:
    from scripts.codex_cli_json_adapter import (
        _artifact_from_text,
        _required_bool,
        _required_string,
        _validate_artifact_contract,
        _write_json_atomic,
        artifact_schema,
    )
except ModuleNotFoundError:  # Direct execution puts ``scripts/`` on sys.path.
    from codex_cli_json_adapter import (  # type: ignore[no-redef]
        _artifact_from_text,
        _required_bool,
        _required_string,
        _validate_artifact_contract,
        _write_json_atomic,
        artifact_schema,
    )


ADAPTER_PROTOCOL_VERSION = 5
DEFAULT_API_TIMEOUT_SECONDS = 7_200.0
DEFAULT_MAX_TOKENS = 100_000
DECISION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Sandboxed scientific-agent controller decision",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["python", "final"]},
        "python_code": {"type": "string"},
        "purpose": {"type": "string", "minLength": 1},
    },
    "required": ["action", "python_code", "purpose"],
}
PYTHON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "python_code": {"type": "string", "minLength": 1},
        "purpose": {"type": "string", "minLength": 1},
    },
    "required": ["python_code", "purpose"],
}
FINISH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"purpose": {"type": "string", "minLength": 1}},
    "required": ["purpose"],
}

SYSTEM_PROMPT = """You are a scientific analysis agent in a controlled experiment.
Use only the task prompt, prior accepted stage artifacts supplied in this conversation,
and results returned by the sandboxed Python controller. Never infer or seek evaluator
files, scoring keys, parent directories, sibling workspaces, credentials, or network
resources.

On controller-decision turns, return the required JSON decision object. Choose action
"python" and provide self-contained Python code when you need to inspect data, calculate
statistics, verify prior work, or read public documentation. Python starts in the public
workspace. It can read that workspace and installed scientific packages, can write only
to the scratch directory named in the task prompt, and has no network. Print concise,
information-rich results; do not print whole datasets. Choose action "final" only when
you have enough reproducible evidence for the requested stage. Keep controller reasoning
brief, never repeat an intermediate thought, and emit the decision promptly.

After choosing "final", you will receive a separate request for the exact stage artifact.
On artifact turns, return only the artifact required by the user's stage contract. Treat
the artifact JSON schema as authoritative. Preserve null and negative findings, never
invent quantitative support, and make every supported_claim_indices entry refer to an
existing claim whose supported field is true.
"""

NATIVE_TOOL_SYSTEM_PROMPT = """You are a scientific analysis agent in a controlled
experiment. Use only the task prompt, prior accepted stage artifacts supplied in this
conversation, and results returned by the sandboxed Python controller. Never infer or
seek evaluator files, scoring keys, parent directories, sibling workspaces, credentials,
or network resources.

On controller turns, call exactly one provided function. Call run_python with
self-contained code when you need to inspect data, calculate statistics, verify prior
work, or read public documentation. Python starts in the public workspace, can read that
workspace and installed scientific packages, can write only to the scratch directory
named in the task prompt, and has no network. Print concise, information-rich results;
do not print whole datasets. Call finish_stage only when you have enough reproducible
evidence for the requested stage. Keep controller reasoning brief, never repeat an
intermediate thought, and emit exactly one function call promptly.

On artifact turns, call submit_stage_artifact exactly once with the complete artifact
required by the user's stage contract. Treat its parameter schema as authoritative.
Preserve null and negative findings, never invent quantitative support, and make every
supported_claim_indices entry refer to an existing claim whose supported field is true.
"""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("vLLM adapter exhausted its total call deadline.")
    return remaining


def _session_path(request: dict[str, Any]) -> Path:
    scratch_dir = Path(_required_string(request, "scratch_dir"))
    digest = _sha256(_required_string(request, "session_id"))
    return scratch_dir.parent / ".vllm_sessions" / f"{digest}.json"


def _session_settings(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    return {
        "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
        "model_id": model_id,
        "base_url": args.base_url.rstrip("/"),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_decision_tokens": args.max_decision_tokens,
        "max_tokens": args.max_tokens,
        "max_history_chars": args.max_history_chars,
        "max_controller_decisions": args.max_controller_decisions,
        "thinking_mode": args.thinking_mode,
        "interaction_mode": args.interaction_mode,
    }


def _load_session(
    path: Path,
    *,
    session_id: str,
    settings: dict[str, Any],
) -> list[dict[str, str]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("harness_session_sha256") != _sha256(session_id):
        raise RuntimeError("Refusing to resume a vLLM session with a mismatched session ID.")
    if payload.get("settings") != settings:
        raise RuntimeError("Refusing to resume a vLLM session with changed model settings.")
    history = payload.get("history")
    if not isinstance(history, list):
        raise RuntimeError("vLLM session history is not an array.")
    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            raise RuntimeError("vLLM session history contains a non-object entry.")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise RuntimeError("vLLM session history contains an invalid message.")
        normalized.append({"role": role, "content": content})
    return normalized


def _trim_history(history: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    trimmed = list(history)
    while len(trimmed) > 2 and sum(len(item["content"]) for item in trimmed) > max_chars:
        del trimmed[:2]
    return trimmed


def _save_session(
    path: Path,
    *,
    session_id: str,
    settings: dict[str, Any],
    history: list[dict[str, str]],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "harness_session_sha256": _sha256(session_id),
            "settings": settings,
            "history": history,
        },
    )


def _build_client(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment validation covers this.
        raise RuntimeError(
            "The vLLM adapter requires the OpenAI Python client in its runtime environment."
        ) from exc
    return OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.api_timeout_seconds,
        max_retries=0,
    )


def _response_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("The vLLM client returned an unsupported response object.")


def _client_request_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Move vLLM-only request fields under the OpenAI SDK's extra_body hook."""
    payload = dict(request_payload)
    extra_body = {
        key: payload.pop(key)
        for key in (
            "chat_template_kwargs",
            "structured_outputs",
            "top_k",
            "repetition_penalty",
        )
        if key in payload
    }
    if extra_body:
        payload["extra_body"] = extra_body
    return payload


def _retryable_api_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    try:
        code = int(status)
    except (TypeError, ValueError):
        return True
    return code in {408, 409, 429} or code >= 500


def _completion_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("The vLLM response contained no choices.")
    content = getattr(choices[0].message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The vLLM response contained no assistant content.")
    return content.strip()


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        max(0, int(getattr(usage, "prompt_tokens", 0) or 0)),
        max(0, int(getattr(usage, "completion_tokens", 0) or 0)),
    )


def _finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return reason if isinstance(reason, str) else None


def _attempt_seed(
    *, seed_namespace: str, turn_number: int, schema_name: str, attempt_number: int
) -> int:
    seed_material = (
        f"{seed_namespace}:{turn_number}:{schema_name}:attempt:{attempt_number}"
    )
    return int(_sha256(seed_material)[:8], 16) & 0x7FFFFFFF


def _recovery_prompt(
    *,
    error: Exception,
    finish_reason: str | None,
    allowed_functions: tuple[str, ...] | None = None,
) -> str:
    if allowed_functions:
        choices = " or ".join(f"`{name}`" for name in allowed_functions)
        response_instruction = (
            f"Call exactly one available function: {choices}. Do not call any other "
            "function and do not emit the next phase's artifact."
        )
    else:
        response_instruction = "Emit exactly one object that satisfies the required schema."
    if finish_reason == "length":
        return (
            "GENERATION RECOVERY: The previous response exhausted its token budget "
            "before emitting one valid required response. Do not resume, quote, or "
            "repeat its reasoning. Use the evidence already available. "
            f"{response_instruction} Respond immediately. "
            f"Validation error: {error}"
        )
    return (
        "SCHEMA RECOVERY: The previous response was rejected for this exact reason: "
        f"{error}. Do not repeat the rejected response or extended reasoning. "
        f"{response_instruction} Respond immediately."
    )


def _chat_completion(
    *,
    client: Any,
    args: argparse.Namespace,
    model_id: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    turn_dir: Path,
    turn_number: int,
    deadline: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any], int, int, float, int]:
    base_request_payload: dict[str, Any] = {
        "model": model_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }
    turn_dir.mkdir(parents=True, exist_ok=False)

    total_duration = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    recovery: str | None = None
    disable_thinking_for_recovery = False
    for attempt_number in range(1, args.max_api_retries + 2):
        attempt_dir = turn_dir / f"attempt_{attempt_number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_messages = list(messages)
        if recovery is not None:
            attempt_messages.append({"role": "user", "content": recovery})
        seed = _attempt_seed(
            seed_namespace=args.seed_namespace,
            turn_number=turn_number,
            schema_name=schema_name,
            attempt_number=attempt_number,
        )
        request_payload = {
            **base_request_payload,
            "messages": attempt_messages,
            "seed": seed,
        }
        if args.thinking_mode != "server-default":
            request_payload["chat_template_kwargs"] = {
                "enable_thinking": (
                    args.thinking_mode == "enabled"
                    and not disable_thinking_for_recovery
                )
            }
        if attempt_number == 1:
            _write_json_atomic(turn_dir / "request.json", request_payload)
        _write_json_atomic(attempt_dir / "request.json", request_payload)
        started = time.monotonic()
        try:
            timeout = min(args.api_timeout_seconds, _remaining(deadline))
            response = client.with_options(timeout=timeout).chat.completions.create(
                **_client_request_payload(request_payload)
            )
        except Exception as exc:
            duration = time.monotonic() - started
            total_duration += duration
            _write_json_atomic(
                attempt_dir / "error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_seconds": duration,
                    "retryable": _retryable_api_error(exc),
                    "seed": seed,
                },
            )
            if attempt_number > args.max_api_retries or not _retryable_api_error(exc):
                raise
            delay = min(float(2 ** (attempt_number - 1)), _remaining(deadline))
            time.sleep(delay)
            continue

        duration = time.monotonic() - started
        total_duration += duration
        payload = _response_payload(response)
        _write_json_atomic(attempt_dir / "response.json", payload)
        input_tokens, output_tokens = _usage(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        finish_reason = _finish_reason(response)
        parse_error: Exception | None = None
        content = ""
        try:
            content = _completion_content(response)
            _schema_object_from_text(content, schema=schema, label=schema_name)
        except (RuntimeError, ValueError) as exc:
            parse_error = exc
        if parse_error is not None:
            _write_json_atomic(
                attempt_dir / "error.json",
                {
                    "error_type": type(parse_error).__name__,
                    "error": str(parse_error),
                    "duration_seconds": duration,
                    "retryable": True,
                    "finish_reason": finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "seed": seed,
                },
            )
            if attempt_number > args.max_api_retries:
                raise parse_error
            recovery = _recovery_prompt(
                error=parse_error,
                finish_reason=finish_reason,
            )
            disable_thinking_for_recovery = finish_reason == "length"
            delay = min(float(2 ** (attempt_number - 1)), _remaining(deadline))
            time.sleep(delay)
            continue
        (attempt_dir / "content.json").write_text(content + "\n", encoding="utf-8")
        _write_json_atomic(
            attempt_dir / "success.json",
            {
                "duration_seconds": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "seed": seed,
                "finish_reason": finish_reason,
            },
        )
        return (
            content,
            payload,
            total_input_tokens,
            total_output_tokens,
            total_duration,
            attempt_number,
        )
    raise AssertionError("unreachable")


def _function_tool(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": parameters,
        },
    }


def _controller_tools() -> list[dict[str, Any]]:
    return [
        _function_tool(
            "run_python",
            "Run self-contained Python in the isolated scientific-analysis sandbox.",
            PYTHON_TOOL_SCHEMA,
        ),
        _function_tool(
            "finish_stage",
            "Stop using tools and proceed to the required stage artifact.",
            FINISH_TOOL_SCHEMA,
        ),
    ]


def _artifact_tool(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _function_tool(
            "submit_stage_artifact",
            "Submit the complete stage artifact required by the experiment contract.",
            schema,
        )
    ]


def _native_tool_call(response: Any) -> tuple[dict[str, str], dict[str, Any]]:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("The vLLM response contained no choices.")
    message = choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise RuntimeError("The vLLM response did not contain exactly one tool call.")
    tool_call = tool_calls[0]
    call_id = getattr(tool_call, "id", None)
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("The vLLM tool call omitted its ID.")
    if not isinstance(name, str) or not name:
        raise RuntimeError("The vLLM tool call omitted its function name.")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, separators=(",", ":"))
    if not isinstance(arguments, str) or not arguments.strip():
        raise RuntimeError("The vLLM tool call omitted its arguments.")
    normalized = {"id": call_id, "name": name, "arguments": arguments.strip()}
    assistant_message = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": normalized["arguments"]},
            }
        ],
    }
    return normalized, assistant_message


def _native_tool_completion(
    *,
    client: Any,
    args: argparse.Namespace,
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
    schema_name: str,
    turn_dir: Path,
    turn_number: int,
    deadline: float,
    max_tokens: int,
    content_fallback_schema: dict[str, Any] | None = None,
    content_fallback_name: str | None = None,
    content_fallback_validator: Callable[[str], None] | None = None,
    allow_unavailable_fallback_tool: bool = False,
) -> tuple[
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    int,
    int,
    float,
    int,
    str,
]:
    base_request_payload: dict[str, Any] = {
        "model": model_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
    }
    tool_schemas = {
        str(tool["function"]["name"]): tool["function"]["parameters"]
        for tool in tools
    }
    turn_dir.mkdir(parents=True, exist_ok=False)

    total_duration = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    recovery: str | None = None
    disable_thinking_for_recovery = False
    for attempt_number in range(1, args.max_api_retries + 2):
        attempt_dir = turn_dir / f"attempt_{attempt_number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_messages = list(messages)
        if recovery is not None:
            attempt_messages.append({"role": "user", "content": recovery})
        seed = _attempt_seed(
            seed_namespace=args.seed_namespace,
            turn_number=turn_number,
            schema_name=schema_name,
            attempt_number=attempt_number,
        )
        request_payload: dict[str, Any] = {
            **base_request_payload,
            "messages": attempt_messages,
            "seed": seed,
        }
        if args.thinking_mode != "server-default":
            request_payload["chat_template_kwargs"] = {
                "enable_thinking": (
                    args.thinking_mode == "enabled"
                    and not disable_thinking_for_recovery
                )
            }
        if attempt_number == 1:
            _write_json_atomic(turn_dir / "request.json", request_payload)
        _write_json_atomic(attempt_dir / "request.json", request_payload)
        started = time.monotonic()
        try:
            timeout = min(args.api_timeout_seconds, _remaining(deadline))
            response = client.with_options(timeout=timeout).chat.completions.create(
                **_client_request_payload(request_payload)
            )
        except Exception as exc:
            duration = time.monotonic() - started
            total_duration += duration
            _write_json_atomic(
                attempt_dir / "error.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_seconds": duration,
                    "retryable": _retryable_api_error(exc),
                    "seed": seed,
                },
            )
            if attempt_number > args.max_api_retries or not _retryable_api_error(exc):
                raise
            delay = min(float(2 ** (attempt_number - 1)), _remaining(deadline))
            time.sleep(delay)
            continue

        duration = time.monotonic() - started
        total_duration += duration
        payload = _response_payload(response)
        _write_json_atomic(attempt_dir / "response.json", payload)
        input_tokens, output_tokens = _usage(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        finish_reason = _finish_reason(response)
        response_mode = "native_tool"
        parse_error: Exception | None = None
        tool_call: dict[str, str] | None = None
        try:
            tool_call, assistant_message = _native_tool_call(response)
            tool_schema = tool_schemas.get(tool_call["name"])
            if tool_schema is None:
                if (
                    allow_unavailable_fallback_tool
                    and content_fallback_schema is not None
                    and content_fallback_name is not None
                    and tool_call["name"] == content_fallback_name
                ):
                    _schema_object_from_text(
                        tool_call["arguments"],
                        schema=content_fallback_schema,
                        label=f"out-of-phase {content_fallback_name} tool call",
                    )
                    if content_fallback_validator is not None:
                        content_fallback_validator(tool_call["arguments"])
                    response_mode = "validated_native_artifact_fallback"
                else:
                    raise ValueError(
                        "vLLM invoked an unavailable native tool: "
                        f"{tool_call['name']}"
                    )
            else:
                _schema_object_from_text(
                    tool_call["arguments"],
                    schema=tool_schema,
                    label=f"{tool_call['name']} tool call",
                )
        except (RuntimeError, ValueError) as exc:
            parse_error = exc
            if (
                tool_call is None
                and content_fallback_schema is not None
                and content_fallback_name is not None
            ):
                try:
                    fallback = _schema_object_from_content(
                        _native_message_content(response),
                        schema=content_fallback_schema,
                        label=f"{content_fallback_name} content fallback",
                    )
                    arguments = json.dumps(
                        fallback, ensure_ascii=False, separators=(",", ":")
                    )
                    if content_fallback_validator is not None:
                        content_fallback_validator(arguments)
                except (RuntimeError, ValueError) as fallback_exc:
                    parse_error = RuntimeError(
                        f"{exc} Validated content fallback was rejected: {fallback_exc}"
                    )
                else:
                    tool_call = {
                        "id": f"content-fallback-{turn_number}-{attempt_number}",
                        "name": content_fallback_name,
                        "arguments": arguments,
                    }
                    assistant_message = {
                        "role": "assistant",
                        "content": _native_message_content(response),
                    }
                    response_mode = "validated_content_fallback"
                    parse_error = None
        if parse_error is not None:
            _write_json_atomic(
                attempt_dir / "error.json",
                {
                    "error_type": type(parse_error).__name__,
                    "error": str(parse_error),
                    "duration_seconds": duration,
                    "retryable": True,
                    "finish_reason": finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "seed": seed,
                },
            )
            if attempt_number > args.max_api_retries:
                raise parse_error
            recovery = _recovery_prompt(
                error=parse_error,
                finish_reason=finish_reason,
                allowed_functions=tuple(sorted(tool_schemas)),
            )
            disable_thinking_for_recovery = finish_reason == "length"
            delay = min(float(2 ** (attempt_number - 1)), _remaining(deadline))
            time.sleep(delay)
            continue
        (attempt_dir / "content.json").write_text(
            tool_call["arguments"] + "\n", encoding="utf-8"
        )
        _write_json_atomic(
            attempt_dir / "success.json",
            {
                "duration_seconds": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "seed": seed,
                "finish_reason": finish_reason,
                "tool_call_id": tool_call["id"],
                "tool_name": tool_call["name"],
                "response_mode": response_mode,
                **(
                    {
                        "content_fallback_schema_sha256": _sha256(
                            json.dumps(
                                content_fallback_schema,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                    }
                    if response_mode
                    in {
                        "validated_content_fallback",
                        "validated_native_artifact_fallback",
                    }
                    else {}
                ),
            },
        )
        return (
            tool_call,
            assistant_message,
            payload,
            total_input_tokens,
            total_output_tokens,
            total_duration,
            attempt_number,
            response_mode,
        )
    raise AssertionError("unreachable")


def _analysis_python_roots(source: Path) -> tuple[Path, Path, Path]:
    requested = source.expanduser().absolute()
    executable = requested.resolve(strict=True)
    environment_root = requested.parent.parent.resolve(strict=True)
    python_root = executable.parent.parent.resolve(strict=True)
    site_packages = next(
        (
            path.resolve(strict=True)
            for path in (environment_root / "lib").glob("python*/site-packages")
            if path.is_dir()
        ),
        None,
    )
    if site_packages is None:
        raise ValueError(f"No site-packages directory found under {environment_root}.")
    return requested, environment_root, python_root


def _resource_limits(python_timeout_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (python_timeout_seconds + 5,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256,) * 2)


def _sandbox_command(
    *,
    bwrap: str,
    python: Path,
    environment_root: Path,
    python_root: Path,
    workspace: Path,
    scratch_dir: Path,
    code_path: Path,
) -> list[str]:
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-net",
        "--unshare-uts",
        "--clearenv",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    roots: list[Path] = []
    for raw in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
        path = Path(raw)
        if path.exists():
            roots.append(path)
    roots.extend((environment_root, python_root, workspace))
    # Preserve standard symlink destinations such as /lib64. Resolving them
    # before mounting removes the ELF interpreter path inside the new root.
    for root in dict.fromkeys(roots):
        command.extend(["--ro-bind", str(root), str(root)])
    command.extend(
        [
            "--bind",
            str(scratch_dir),
            str(scratch_dir),
            "--chdir",
            str(workspace),
            "--setenv",
            "HOME",
            str(scratch_dir),
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PATH",
            f"{python.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            str(python),
            str(code_path),
        ]
    )
    return command


def _read_tool_log(path: Path, max_chars: int) -> tuple[str, bool, int]:
    size = path.stat().st_size if path.exists() else 0
    with path.open("rb") as stream:
        raw = stream.read(max_chars * 4 + 1)
    text = raw.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars or size > len(raw)
    if len(text) > max_chars:
        text = text[:max_chars]
    if truncated:
        text += "\n...[tool output truncated by adapter]"
    return text, truncated, size


def _run_python_tool(
    *,
    code: str,
    args: argparse.Namespace,
    workspace: Path,
    scratch_dir: Path,
    tool_dir: Path,
    tool_number: int,
) -> dict[str, Any]:
    if not code.strip():
        raise ValueError("A python action must provide non-empty python_code.")
    python, environment_root, python_root = _analysis_python_roots(args.analysis_python)
    tool_dir.mkdir(parents=True, exist_ok=False)
    audit_code = tool_dir / "python_code.py"
    audit_code.write_text(code.rstrip() + "\n", encoding="utf-8")
    execution_code = scratch_dir / f"vllm_tool_{os.getpid()}_{tool_number:04d}.py"
    execution_code.write_text(code.rstrip() + "\n", encoding="utf-8")
    stdout_path = tool_dir / "stdout.log"
    stderr_path = tool_dir / "stderr.log"
    command = _sandbox_command(
        bwrap=args.bwrap,
        python=python,
        environment_root=environment_root,
        python_root=python_root,
        workspace=workspace,
        scratch_dir=scratch_dir,
        code_path=execution_code,
    )
    _write_json_atomic(tool_dir / "command.json", command)
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=args.python_timeout_seconds,
                check=False,
                preexec_fn=lambda: _resource_limits(args.python_timeout_seconds),
            )
            returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    duration = time.monotonic() - started
    stdout_text, stdout_truncated, stdout_bytes = _read_tool_log(
        stdout_path, args.max_tool_output_chars
    )
    stderr_text, stderr_truncated, stderr_bytes = _read_tool_log(
        stderr_path, args.max_tool_output_chars
    )
    result = {
        "tool": "python",
        "tool_number": tool_number,
        "status": "timeout" if timed_out else ("ok" if returncode == 0 else "error"),
        "returncode": returncode,
        "duration_seconds": duration,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    _write_json_atomic(tool_dir / "result.json", result)
    return result


def _schema_object_from_text(
    raw: str, *, schema: dict[str, Any], label: str
) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"vLLM {label} was not valid JSON.") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"vLLM {label} failed at {location}: {error.message}")
    assert isinstance(payload, dict)
    return payload


def _native_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("The vLLM response contained no choices.")
    content = getattr(choices[0].message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The vLLM response contained no non-empty text fallback.")
    return content


def _schema_object_from_content(
    raw: str, *, schema: dict[str, Any], label: str
) -> dict[str, Any]:
    """Accept one bare JSON object or one JSON Markdown fence, then enforce schema."""
    text = raw.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in {"```", "```json"}:
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError(f"vLLM {label} had an unterminated Markdown fence.")
        text = "\n".join(lines[1:-1]).strip()
    return _schema_object_from_text(text, schema=schema, label=label)


def _decision_from_text(raw: str) -> dict[str, str]:
    payload = _schema_object_from_text(
        raw, schema=DECISION_SCHEMA, label="controller decision"
    )
    return {key: str(payload[key]) for key in ("action", "python_code", "purpose")}


def _native_controller_decision(tool_call: dict[str, str]) -> dict[str, str]:
    name = tool_call["name"]
    if name == "run_python":
        payload = _schema_object_from_text(
            tool_call["arguments"], schema=PYTHON_TOOL_SCHEMA, label="Python tool call"
        )
        return {
            "action": "python",
            "python_code": str(payload["python_code"]),
            "purpose": str(payload["purpose"]),
        }
    if name == "finish_stage":
        payload = _schema_object_from_text(
            tool_call["arguments"], schema=FINISH_TOOL_SCHEMA, label="finish tool call"
        )
        return {
            "action": "final",
            "python_code": "",
            "purpose": str(payload["purpose"]),
        }
    raise ValueError(f"vLLM returned an unknown controller tool: {name}")


def _repair_prompt(error: ValueError) -> str:
    return (
        "CONTROLLER CONTRACT REPAIR. Your prior stage artifact was rejected for this exact "
        f"reason: {error}. Return one complete replacement artifact now. Reuse the completed "
        "analysis and tool evidence; do not request or rerun tools. Ensure every "
        "supported_claim_indices entry is unique, in range, and points to a claim whose "
        "supported field is true."
    )


def run_adapter(args: argparse.Namespace, *, client: Any | None = None) -> AgentResponse:
    request = json.loads(args.request_file.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Request file must contain one JSON object.")
    request_id = _required_string(request, "request_id")
    session_id = _required_string(request, "session_id")
    model_id = _required_string(request, "model_id")
    require_final_answer = _required_bool(request, "require_final_answer")
    prompt = _required_string(request, "prompt")
    workspace = Path(_required_string(request, "workspace")).resolve(strict=True)
    scratch_dir = Path(_required_string(request, "scratch_dir")).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    call_dir = args.output.parent
    call_dir.mkdir(parents=True, exist_ok=True)
    turns_dir = call_dir / "vllm_turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    audit_path = call_dir / "vllm_call.json"
    deadline = time.monotonic() + args.timeout_seconds
    started = time.monotonic()

    settings = _session_settings(args, model_id)
    session_path = _session_path(request)
    history = _load_session(session_path, session_id=session_id, settings=settings)
    session_action = "resumed" if history else "started"
    stage_schema = artifact_schema(require_final_answer=require_final_answer)
    schema_hash = _sha256(json.dumps(stage_schema, sort_keys=True, separators=(",", ":")))
    args.seed_namespace = f"{request_id}:{_sha256(prompt)}"
    root_audit: dict[str, Any] = {
        "schema_version": 1,
        "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
        "status": "starting",
        "request_id": request_id,
        "model_id": model_id,
        "base_url": args.base_url,
        "prompt_sha256": _sha256(prompt),
        "output_schema_sha256": schema_hash,
        "require_final_answer": require_final_answer,
        "session_action": session_action,
        "session_record": str(session_path),
        "history_messages_loaded": len(history),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_decision_tokens": args.max_decision_tokens,
        "max_tokens": args.max_tokens,
        "max_tool_calls": args.max_tool_calls,
        "max_controller_decisions": args.max_controller_decisions,
        "max_contract_repairs": args.max_contract_repairs,
        "max_api_retries": args.max_api_retries,
        "thinking_mode": args.thinking_mode,
        "interaction_mode": args.interaction_mode,
        "turns": [],
        "tool_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    def update_audit(status: str, **extra: Any) -> None:
        root_audit.update(
            {
                "status": status,
                "duration_seconds": time.monotonic() - started,
                **extra,
            }
        )
        _write_json_atomic(audit_path, root_audit)

    update_audit("starting")
    if client is None:
        client = _build_client(args)
    system_prompt = (
        NATIVE_TOOL_SYSTEM_PROMPT
        if args.interaction_mode == "native-tools"
        else SYSTEM_PROMPT
    )
    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": prompt})
    model_turn = 0
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    raw_text = ""
    artifact: AgentArtifact | None = None

    def validate_stage_fallback(content: str) -> None:
        candidate = _artifact_from_text(content, schema=stage_schema)
        _validate_artifact_contract(
            candidate, require_final_answer=require_final_answer
        )

    try:
        controller_decisions = 0
        while (
            tool_calls < args.max_tool_calls
            and controller_decisions < args.max_controller_decisions
        ):
            controller_decisions += 1
            model_turn += 1
            decision_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "CONTROLLER TURN ONLY: Call exactly one available function: "
                        "run_python or finish_stage. Do not call submit_stage_artifact and "
                        "do not emit the stage artifact on this turn. Inspect or verify with "
                        "sandboxed Python when needed. If the evidence is already sufficient, "
                        "call finish_stage now; a separate artifact turn will follow."
                    ),
                },
            ]
            turn_dir = turns_dir / f"model_turn_{model_turn:04d}_decision"
            tool_call: dict[str, str] | None = None
            fallback_artifact: AgentArtifact | None = None
            if args.interaction_mode == "native-tools":
                (
                    tool_call,
                    assistant_message,
                    _,
                    used_in,
                    used_out,
                    duration,
                    attempts,
                    response_mode,
                ) = _native_tool_completion(
                    client=client,
                    args=args,
                    model_id=model_id,
                    messages=decision_messages,
                    tools=_controller_tools(),
                    tool_choice="required",
                    schema_name="scientific_agent_controller_tools",
                    turn_dir=turn_dir,
                    turn_number=model_turn,
                    deadline=deadline,
                    max_tokens=args.max_decision_tokens,
                    content_fallback_schema=stage_schema,
                    content_fallback_name="submit_stage_artifact",
                    content_fallback_validator=validate_stage_fallback,
                    allow_unavailable_fallback_tool=(
                        tool_calls >= args.min_tool_calls
                    ),
                )
                content = tool_call["arguments"]
                if response_mode in {
                    "validated_content_fallback",
                    "validated_native_artifact_fallback",
                }:
                    fallback_artifact = _artifact_from_text(
                        content, schema=stage_schema
                    )
                    _validate_artifact_contract(
                        fallback_artifact,
                        require_final_answer=require_final_answer,
                    )
                    decision = {
                        "action": "final",
                        "python_code": "",
                        "purpose": (
                            "schema-valid artifact returned during controller turn"
                        ),
                    }
                else:
                    decision = _native_controller_decision(tool_call)
            else:
                content, _, used_in, used_out, duration, attempts = _chat_completion(
                    client=client,
                    args=args,
                    model_id=model_id,
                    messages=decision_messages,
                    schema=DECISION_SCHEMA,
                    schema_name="scientific_agent_decision",
                    turn_dir=turn_dir,
                    turn_number=model_turn,
                    deadline=deadline,
                    max_tokens=args.max_decision_tokens,
                )
                assistant_message = {"role": "assistant", "content": content}
                decision = _decision_from_text(content)
                response_mode = "json_schema"
            input_tokens += used_in
            output_tokens += used_out
            record = {
                "model_turn": model_turn,
                "kind": "decision",
                "action": decision["action"],
                "response_mode": response_mode,
                "duration_seconds": duration,
                "api_attempts": attempts,
                "input_tokens": used_in,
                "output_tokens": used_out,
                "directory": str(turn_dir.relative_to(call_dir)),
            }
            root_audit["turns"].append(record)
            root_audit["input_tokens"] = input_tokens
            root_audit["output_tokens"] = output_tokens
            update_audit("running")
            messages.extend(
                [
                    {"role": "user", "content": decision_messages[-1]["content"]},
                    assistant_message,
                ]
            )
            if fallback_artifact is not None and tool_calls >= args.min_tool_calls:
                record["kind"] = "content_artifact"
                record["status"] = "accepted"
                artifact = fallback_artifact
                raw_text = content
                update_audit("running")
                break
            if decision["action"] == "final":
                if tool_calls >= args.min_tool_calls:
                    if tool_call is not None and response_mode == "native_tool":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": tool_call["name"],
                                "content": "Accepted. Proceed to the final stage artifact.",
                            }
                        )
                    break
                reminder = (
                    "You must inspect the public workspace with sandboxed Python at "
                    "least once before finalizing. Choose a python action next."
                )
                if tool_call is not None and response_mode == "native_tool":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["name"],
                            "content": reminder,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": reminder})
                continue

            tool_calls += 1
            tool_dir = call_dir / "vllm_tools" / f"tool_{tool_calls:04d}"
            result = _run_python_tool(
                code=decision["python_code"],
                args=args,
                workspace=workspace,
                scratch_dir=scratch_dir,
                tool_dir=tool_dir,
                tool_number=tool_calls,
            )
            root_audit["tool_calls"] = tool_calls
            root_audit["turns"].append(
                {
                    "kind": "python_tool",
                    "tool_number": tool_calls,
                    "status": result["status"],
                    "duration_seconds": result["duration_seconds"],
                    "directory": str(tool_dir.relative_to(call_dir)),
                }
            )
            update_audit("running")
            tool_result = "SANDBOXED PYTHON RESULT\n" + json.dumps(result)
            if tool_call is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": tool_result,
                    }
                )
            else:
                messages.append({"role": "user", "content": tool_result})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The controller decision or Python-tool budget is exhausted. Produce the "
                        "best evidence-grounded stage artifact now without requesting more tools."
                    ),
                }
            )

        if artifact is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "FINAL ARTIFACT: Return only the complete stage artifact required by the "
                        "original task. Do not wrap it in a controller action or Markdown."
                    ),
                }
            )
        validation_error: ValueError | None = None
        repair_attempts = (
            range(args.max_contract_repairs + 1) if artifact is None else range(0)
        )
        for repair_number in repair_attempts:
            model_turn += 1
            kind = "artifact" if repair_number == 0 else "contract_repair"
            turn_dir = turns_dir / f"model_turn_{model_turn:04d}_{kind}"
            schema_name = (
                "scientific_synthesis_artifact"
                if require_final_answer
                else "scientific_stage_artifact"
            )
            artifact_tool_call: dict[str, str] | None = None
            if args.interaction_mode == "native-tools":
                (
                    artifact_tool_call,
                    assistant_message,
                    _,
                    used_in,
                    used_out,
                    duration,
                    attempts,
                    response_mode,
                ) = _native_tool_completion(
                    client=client,
                    args=args,
                    model_id=model_id,
                    messages=messages,
                    tools=_artifact_tool(stage_schema),
                    tool_choice={
                        "type": "function",
                        "function": {"name": "submit_stage_artifact"},
                    },
                    schema_name=schema_name,
                    turn_dir=turn_dir,
                    turn_number=model_turn,
                    deadline=deadline,
                    max_tokens=args.max_tokens,
                    content_fallback_schema=stage_schema,
                    content_fallback_name="submit_stage_artifact",
                )
                if artifact_tool_call["name"] != "submit_stage_artifact":
                    raise ValueError(
                        "vLLM invoked an unexpected artifact tool: "
                        f"{artifact_tool_call['name']}"
                    )
                content = artifact_tool_call["arguments"]
            else:
                content, _, used_in, used_out, duration, attempts = _chat_completion(
                    client=client,
                    args=args,
                    model_id=model_id,
                    messages=messages,
                    schema=stage_schema,
                    schema_name=schema_name,
                    turn_dir=turn_dir,
                    turn_number=model_turn,
                    deadline=deadline,
                    max_tokens=args.max_tokens,
                )
                assistant_message = {"role": "assistant", "content": content}
                response_mode = "json_schema"
            input_tokens += used_in
            output_tokens += used_out
            record = {
                "model_turn": model_turn,
                "kind": kind,
                "response_mode": response_mode,
                "duration_seconds": duration,
                "api_attempts": attempts,
                "input_tokens": used_in,
                "output_tokens": used_out,
                "directory": str(turn_dir.relative_to(call_dir)),
            }
            root_audit["turns"].append(record)
            root_audit["input_tokens"] = input_tokens
            root_audit["output_tokens"] = output_tokens
            raw_text = content
            messages.append(assistant_message)
            try:
                candidate = _artifact_from_text(content, schema=stage_schema)
                _validate_artifact_contract(
                    candidate, require_final_answer=require_final_answer
                )
            except ValueError as exc:
                validation_error = exc
                record["status"] = "contract_rejected"
                record["validation_error"] = str(exc)
                update_audit("repair_pending", last_contract_error=str(exc))
                if repair_number >= args.max_contract_repairs:
                    raise ValueError(
                        "vLLM artifact remained invalid after "
                        f"{repair_number + 1} attempt(s): {exc}"
                    ) from exc
                if (
                    artifact_tool_call is not None
                    and response_mode == "native_tool"
                ):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": artifact_tool_call["id"],
                            "name": artifact_tool_call["name"],
                            "content": "Artifact rejected: " + str(exc),
                        }
                    )
                messages.append({"role": "user", "content": _repair_prompt(exc)})
                continue
            artifact = candidate
            validation_error = None
            record["status"] = "accepted"
            break

        if artifact is None:
            raise RuntimeError(
                "vLLM adapter ended without an accepted artifact: "
                f"{validation_error or 'unknown validation failure'}"
            )

        accepted_json = artifact.model_dump_json()
        new_history = [
            *history,
            {"role": "user", "content": "SESSION STAGE REQUEST\n" + prompt},
            {
                "role": "assistant",
                "content": "SESSION ACCEPTED STAGE ARTIFACT\n" + accepted_json,
            },
        ]
        new_history = _trim_history(new_history, args.max_history_chars)
        _save_session(
            session_path,
            session_id=session_id,
            settings=settings,
            history=new_history,
        )
        duration = time.monotonic() - started
        response = AgentResponse(
            request_id=request_id,
            artifact=artifact,
            usage=AgentUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                duration_seconds=duration,
            ),
            raw_text=raw_text,
            runtime_metadata={
                "adapter": "vllm-cli-json",
                "adapter_protocol_version": ADAPTER_PROTOCOL_VERSION,
                "base_url": args.base_url,
                "model_id": model_id,
                "thinking_mode": args.thinking_mode,
                "interaction_mode": args.interaction_mode,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "max_decision_tokens": args.max_decision_tokens,
                "max_artifact_tokens": args.max_tokens,
                "session_action": session_action,
                "session_sha256": _sha256(session_id),
                "history_messages_loaded": len(history),
                "history_messages_saved": len(new_history),
                "model_turns": model_turn,
                "tool_calls": tool_calls,
                "contract_repair_count": sum(
                    turn.get("kind") == "contract_repair"
                    for turn in root_audit["turns"]
                ),
                "audit_file": audit_path.name,
            },
        )
        _write_json_atomic(args.output, response.model_dump(mode="json"))
        update_audit(
            "accepted",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            model_turns=model_turn,
            history_messages_saved=len(new_history),
        )
        return response
    except Exception as exc:
        update_audit(
            "error",
            error_type=type(exc).__name__,
            error=str(exc),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            model_turns=model_turn,
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--analysis-python", type=Path, required=True)
    parser.add_argument("--bwrap", default="bwrap")
    parser.add_argument("--timeout-seconds", type=int, default=14_380)
    parser.add_argument(
        "--api-timeout-seconds", type=float, default=DEFAULT_API_TIMEOUT_SECONDS
    )
    parser.add_argument("--python-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-api-retries", type=int, default=2)
    parser.add_argument("--max-contract-repairs", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--max-controller-decisions", type=int, default=40)
    parser.add_argument("--max-tool-output-chars", type=int, default=40_000)
    parser.add_argument("--max-history-chars", type=int, default=180_000)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum completion tokens per vLLM request, including reasoning tokens "
            "and the visible response."
        ),
    )
    parser.add_argument(
        "--max-decision-tokens",
        type=int,
        default=None,
        help=(
            "Maximum completion tokens for controller decisions. The default uses "
            "--max-tokens; artifact and repair turns always use --max-tokens."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="vLLM top-k sampling; -1 disables the cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="vLLM repetition penalty applied to prompt and generated tokens.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("server-default", "enabled", "disabled"),
        default="server-default",
        help=(
            "Whether to override the model chat template's enable_thinking flag. "
            "The default leaves the vLLM server/model setting unchanged."
        ),
    )
    parser.add_argument(
        "--interaction-mode",
        choices=("json-schema", "native-tools"),
        default="json-schema",
        help=(
            "Use schema-constrained JSON controller responses or the server's native "
            "function-call protocol for controller actions and artifact submission."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_decision_tokens is None:
        args.max_decision_tokens = args.max_tokens
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not 0 < args.api_timeout_seconds < args.timeout_seconds:
        parser.error("--api-timeout-seconds must be positive and below the total timeout")
    for name in ("max_api_retries", "max_contract_repairs", "min_tool_calls"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_tool_calls < 1 or args.min_tool_calls > args.max_tool_calls:
        parser.error("tool-call limits are inconsistent")
    if args.max_controller_decisions < args.max_tool_calls:
        parser.error("--max-controller-decisions must be at least --max-tool-calls")
    if (
        args.max_tokens < 1
        or args.max_decision_tokens < 1
        or args.max_history_chars < 1
        or args.max_tool_output_chars < 1
    ):
        parser.error("token, history, and tool-output limits must be positive")
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        parser.error("sampling settings are out of range")
    if args.top_k != -1 and args.top_k < 1:
        parser.error("--top-k must be -1 or positive")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be positive")
    discovered_bwrap = shutil.which(args.bwrap)
    if discovered_bwrap is None:
        parser.error(f"bubblewrap executable not found: {args.bwrap}")
    args.bwrap = discovered_bwrap
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        response = run_adapter(args)
    except Exception as exc:
        print(f"vLLM CLI-JSON adapter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
