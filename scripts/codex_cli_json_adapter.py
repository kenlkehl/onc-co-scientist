#!/usr/bin/env python3
"""Bridge the generic harness file contract to stateful ``codex exec`` calls.

The harness starts this program once per agent call.  Codex sessions therefore
cannot live in Python memory.  A small record under the run's shared scratch
directory maps each harness ``session_id`` to the persisted Codex thread ID.
Repeated calls use ``codex exec resume``; new calls intentionally omit
``--ephemeral`` so that persistent and deliberative workflows retain context.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from onc_co_scientist.harness.runtime import (
    AgentArtifact,
    AgentResponse,
    AgentUsage,
)

SYNTHESIS_FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "conclusion": {"type": "string", "minLength": 1},
        "supported_claim_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
    },
    "required": ["conclusion", "supported_claim_indices"],
}

ARTIFACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Oncology co-scientist stage artifact",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "handoff": {"type": "string", "minLength": 1},
        "hypotheses": {"type": "array", "items": {"type": "string"}},
        "analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "method": {"type": "string"},
                    "result": {"type": "string"},
                },
                "required": ["method", "result"],
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "exposure": {"type": "string", "minLength": 1},
                    "outcome": {"type": "string", "minLength": 1},
                    "direction": {
                        "type": "string",
                        "enum": ["positive", "negative", "null", "uncertain"],
                    },
                    "subgroup": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "variable": {"type": "string", "minLength": 1},
                                "operator": {
                                    "type": "string",
                                    "enum": [
                                        "eq",
                                        "ne",
                                        "lt",
                                        "le",
                                        "gt",
                                        "ge",
                                        "in",
                                        "not_in",
                                    ],
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "integer"},
                                        {"type": "number"},
                                        {"type": "boolean"},
                                    ]
                                },
                            },
                            "required": ["variable", "operator", "value"],
                        },
                    },
                    "comparator": {"type": "string"},
                    "effect_estimate": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "effect_unit": {"type": "string"},
                    "p_value": {
                        "anyOf": [
                            {"type": "number", "minimum": 0, "maximum": 1},
                            {"type": "null"},
                        ]
                    },
                    "subgroup_n": {
                        "anyOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "null"},
                        ]
                    },
                    "exposed_n": {
                        "anyOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "null"},
                        ]
                    },
                    "comparator_n": {
                        "anyOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "null"},
                        ]
                    },
                    "supported": {"type": "boolean"},
                    "confidence": {
                        "anyOf": [
                            {"type": "number", "minimum": 0, "maximum": 1},
                            {"type": "null"},
                        ]
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "exposure",
                    "outcome",
                    "direction",
                    "subgroup",
                    "comparator",
                    "effect_estimate",
                    "effect_unit",
                    "p_value",
                    "subgroup_n",
                    "exposed_n",
                    "comparator_n",
                    "supported",
                    "confidence",
                    "evidence",
                ],
            },
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "minority_report": {"type": "string"},
        "final_answer": {
            "anyOf": [
                SYNTHESIS_FINAL_ANSWER_SCHEMA,
                {"type": "string", "minLength": 1},
                {"type": "null"},
            ]
        },
    },
    "required": [
        "summary",
        "handoff",
        "hypotheses",
        "analyses",
        "claims",
        "evidence",
        "concerns",
        "minority_report",
        "final_answer",
    ],
}


def artifact_schema(*, require_final_answer: bool) -> dict[str, Any]:
    """Return the exact generation contract for one harness stage."""

    schema = copy.deepcopy(ARTIFACT_SCHEMA)
    schema["title"] = (
        "Oncology co-scientist synthesis artifact"
        if require_final_answer
        else "Oncology co-scientist intermediate-stage artifact"
    )
    schema["properties"]["final_answer"] = (
        copy.deepcopy(SYNTHESIS_FINAL_ANSWER_SCHEMA)
        if require_final_answer
        else {"type": "null"}
    )
    return schema

TOOL_ITEM_TYPES = {
    "apply_patch",
    "collab_agent_tool_call",
    "command_execution",
    "computer_initialize_state",
    "dynamic_tool_call",
    "file_change",
    "function_call",
    "image_generation",
    "local_shell_call",
    "mcp_tool_call",
    "tool_call",
    "web_search",
}


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except (AttributeError, PermissionError, OSError):
        process.kill()


def run_subprocess_in_group(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Codex in a group that can be fully reaped after a timeout."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@dataclass(frozen=True)
class CodexEventStats:
    """Auditable per-turn statistics extracted from Codex JSONL output."""

    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    item_types: tuple[str, ...] = ()


def _integer(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _event_objects(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def parse_codex_events(raw: str) -> CodexEventStats:
    """Parse usage and deduplicated tool calls from ``codex exec --json`` output."""

    input_tokens = 0
    output_tokens = 0
    item_types: set[str] = set()
    tool_items: set[tuple[str, str]] = set()

    for event_index, event in enumerate(_event_objects(raw)):
        containers = [event]
        payload = event.get("payload")
        if isinstance(payload, dict):
            containers.append(payload)

        for container in containers:
            for usage_key in ("usage", "token_usage", "total_token_usage"):
                usage = container.get(usage_key)
                if not isinstance(usage, dict):
                    continue
                input_tokens = max(
                    input_tokens,
                    _integer(usage, "input_tokens", "inputTokens", "input"),
                )
                output_tokens = max(
                    output_tokens,
                    _integer(usage, "output_tokens", "outputTokens", "output"),
                )

            item = container.get("item")
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            if not item_type:
                continue
            item_types.add(item_type)
            if item_type in TOOL_ITEM_TYPES or item_type.endswith("_tool_call"):
                item_id = str(item.get("id") or item.get("call_id") or event_index)
                tool_items.add((item_id, item_type))

        event_type = str(event.get("type") or "").strip()
        if event_type in TOOL_ITEM_TYPES or event_type.endswith("_tool_call"):
            event_id = str(event.get("id") or event.get("call_id") or event_index)
            tool_items.add((event_id, event_type))
            item_types.add(event_type)

    return CodexEventStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=len(tool_items),
        item_types=tuple(sorted(item_types)),
    )


def extract_codex_thread_id(raw: str) -> str | None:
    """Extract the persisted thread ID from current or legacy Codex event shapes."""

    for event in _event_objects(raw):
        candidates = [event]
        for key in ("payload", "data"):
            value = event.get(key)
            if isinstance(value, dict):
                candidates.append(value)

        type_values = {str(event.get("type") or "").lower().replace(".", "_").replace("-", "_")}
        for candidate in candidates[1:]:
            type_values.add(
                str(candidate.get("type") or "").lower().replace(".", "_").replace("-", "_")
            )
        if not type_values.intersection({"thread_started", "codex_thread_started", "session_meta"}):
            continue

        for candidate in candidates:
            for key in ("thread_id", "threadId", "session_id", "id"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _required_string(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Request field {key!r} must be a non-empty string.")
    return value


def _required_bool(request: dict[str, Any], key: str) -> bool:
    value = request.get(key)
    if type(value) is not bool:
        raise ValueError(f"Request field {key!r} must be a boolean.")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _session_record_path(request: dict[str, Any]) -> Path:
    scratch_dir = Path(_required_string(request, "scratch_dir"))
    session_digest = _sha256(_required_string(request, "session_id"))
    return scratch_dir.parent / ".codex_sessions" / f"{session_digest}.json"


def _load_session(
    path: Path,
    *,
    session_id: str,
    model_id: str,
    reasoning_effort: str,
    service_tier: str,
) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "harness_session_sha256": _sha256(session_id),
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Refusing to resume incompatible Codex session: {key} mismatch.")
    thread_id = payload.get("codex_thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise RuntimeError("Codex session record does not contain a valid thread ID.")
    return thread_id.strip()


def _save_session(
    path: Path,
    *,
    session_id: str,
    thread_id: str,
    model_id: str,
    reasoning_effort: str,
    service_tier: str,
) -> None:
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "harness_session_sha256": _sha256(session_id),
            "codex_thread_id": thread_id,
            "model_id": model_id,
            "reasoning_effort": reasoning_effort,
            "service_tier": service_tier,
        },
    )


def _artifact_from_text(raw: str, *, schema: dict[str, Any]) -> AgentArtifact:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex final response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Codex final response must be one JSON object.")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(
            f"Codex final response violated the stage schema at {location}: {error.message}"
        )
    return AgentArtifact.model_validate(payload)


def _validate_artifact_contract(
    artifact: AgentArtifact,
    *,
    require_final_answer: bool,
) -> None:
    """Defend the stage contract even if a runtime ignores its output schema."""

    final_answer = artifact.final_answer
    if not require_final_answer:
        if final_answer is not None:
            raise ValueError("An intermediate-stage artifact must set final_answer to null.")
        return

    if not isinstance(final_answer, dict):
        raise ValueError("A synthesis artifact must return final_answer as an object.")
    expected_keys = {"conclusion", "supported_claim_indices"}
    if set(final_answer) != expected_keys:
        raise ValueError(
            "A synthesis final_answer must contain exactly conclusion and "
            "supported_claim_indices."
        )
    conclusion = final_answer["conclusion"]
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise ValueError("A synthesis final_answer conclusion must be non-empty.")
    indices = final_answer["supported_claim_indices"]
    if not isinstance(indices, list):
        raise ValueError("supported_claim_indices must be an array.")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("supported_claim_indices must contain non-negative integers.")
    if len(indices) != len(set(indices)):
        raise ValueError("supported_claim_indices must not contain duplicates.")
    for index in indices:
        if index >= len(artifact.claims):
            raise ValueError(
                f"supported_claim_indices references absent claim index {index}."
            )
        if not artifact.claims[index].supported:
            raise ValueError(
                f"supported_claim_indices references unsupported claim index {index}."
            )


def _analysis_python_runtime(source: Path) -> tuple[Path, Path, Path]:
    """Resolve a virtual-environment Python and its read-only library roots."""

    requested = source.expanduser().absolute()
    executable = requested.resolve(strict=True)
    match = re.fullmatch(r"python(?P<version>\d+\.\d+)", executable.name)
    if match is None:
        raise ValueError(
            "--analysis-python must resolve to a versioned CPython executable such as python3.13."
        )
    python_home = executable.parent.parent
    site_packages = (
        requested.parent.parent / "lib" / f"python{match.group('version')}" / "site-packages"
    ).resolve(strict=True)
    if not site_packages.is_dir():
        raise ValueError(f"Python site-packages path is not a directory: {site_packages}")

    return executable, python_home, site_packages


def _analysis_guard_root() -> Path:
    """Return the read-only startup hook used by analysis Python processes."""

    guard_root = Path(__file__).resolve().with_name("analysis_python_guard")
    if not (guard_root / "sitecustomize.py").is_file():
        raise ValueError(f"Analysis Python guard is missing: {guard_root}")
    return guard_root


def _codex_runtime_root(source: str) -> Path:
    """Return the non-secret installation root needed by Codex's sandbox helper."""

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        discovered = shutil.which(source)
        if discovered is None:
            raise ValueError(f"Codex executable not found: {source}")
        candidate = Path(discovered)
    executable = candidate.absolute().resolve(strict=True)
    if executable.name == "codex.js" and executable.parent.name == "bin":
        return executable.parent.parent
    return executable.parent


def _codex_command(
    *,
    codex: str,
    model_id: str,
    reasoning_effort: str,
    service_tier: str,
    workspace: Path,
    scratch_dir: Path,
    schema_path: Path,
    last_message_path: Path,
    read_roots: tuple[Path, ...],
    shell_environment: dict[str, str],
    thread_id: str | None,
) -> list[str]:
    filesystem_permissions = {
        ":root": "deny",
        ":minimal": "read",
        ":slash_tmp": "deny",
        str(workspace): "write",
        str(scratch_dir): "write",
        **{str(path): "read" for path in read_roots},
    }
    permission_entries = ",".join(
        f"{json.dumps(path)}={json.dumps(access)}"
        for path, access in filesystem_permissions.items()
    )
    permission_profile = f"{{filesystem={{{permission_entries}}},network={{enabled=false}}}}"
    environment_entries = ",".join(
        f"{json.dumps(key)}={json.dumps(value)}" for key, value in shell_environment.items()
    )
    environment_policy = f'{{inherit="none",set={{{environment_entries}}}}}'
    common = [
        "--model",
        model_id,
        "-c",
        f"model_reasoning_effort={reasoning_effort}",
        "-c",
        f'service_tier="{service_tier}"',
        "-c",
        "approval_policy=never",
        "-c",
        "allow_login_shell=false",
        "-c",
        f"permissions.nsclc_eval={permission_profile}",
        "-c",
        'default_permissions="nsclc_eval"',
        "-c",
        f"shell_environment_policy={environment_policy}",
        "--strict-config",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(last_message_path),
    ]
    if thread_id is not None:
        return [codex, "exec", "resume", *common, thread_id, "-"]
    return [
        codex,
        "exec",
        *common,
        "--color",
        "never",
        "-C",
        str(workspace),
        "--add-dir",
        str(scratch_dir),
        "-",
    ]


def _audit_payload(
    *,
    request: dict[str, Any],
    command: list[str],
    model_id: str,
    reasoning_effort: str,
    service_tier: str,
    session_action: str,
    prompt: str,
    prompt_kind: str,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "request_id": _required_string(request, "request_id"),
        "original_prompt_sha256": _sha256(_required_string(request, "prompt")),
        "prompt_sha256": _sha256(prompt),
        "prompt_kind": prompt_kind,
        "attempt_number": attempt_number,
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "require_final_answer": _required_bool(request, "require_final_answer"),
        "session_action": session_action,
        "command": command,
        "stdin": "<PROMPT_ON_STDIN>",
    }


def _contract_repair_prompt(error: ValueError) -> str:
    """Ask the same Codex thread to correct only its rejected artifact."""

    return (
        "The controller rejected your previous structured artifact. Correct that artifact "
        "in this same session and return one complete replacement matching the supplied "
        "output schema. Reuse the analysis already completed; do not rerun tools or perform "
        "new scientific analysis unless correction is impossible without it.\n\n"
        f"Controller validation error: {type(error).__name__}: {error}\n\n"
        "In particular, rebuild supported_claim_indices so every entry is a unique "
        "non-negative integer, is less than len(claims), and refers only to a claim whose "
        "supported field is true. Return only the corrected structured artifact."
    )


def _supported_claim_indices_schema(allowed_indices: list[int]) -> dict[str, Any]:
    """Return a narrow correction contract tied to the artifact's actual claims."""

    item_schema: dict[str, Any] = {"type": "integer"}
    if allowed_indices:
        item_schema["enum"] = allowed_indices
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Supported claim index correction",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "supported_claim_indices": {
                "type": "array",
                "items": item_schema,
                **({} if allowed_indices else {"maxItems": 0}),
            }
        },
        "required": ["supported_claim_indices"],
    }


def _supported_claim_indices_repair_prompt(
    artifact: AgentArtifact,
    error: ValueError,
) -> str:
    """Ask for only the invalid cross-reference field, not a rewritten artifact."""

    final_answer = artifact.final_answer if isinstance(artifact.final_answer, dict) else {}
    rejected = final_answer.get("supported_claim_indices")
    allowed = [index for index, claim in enumerate(artifact.claims) if claim.supported]
    return (
        "The controller accepted every field of your prior artifact except "
        "final_answer.supported_claim_indices. Return exactly one JSON object containing "
        "only supported_claim_indices. Do not rewrite any other field. Do not rerun tools "
        "or perform new scientific analysis. Preserve your intended selection where possible. "
        "Each returned index must be unique and must come from the allowed list below.\n\n"
        f"Controller validation error: {type(error).__name__}: {error}\n"
        f"Number of claims: {len(artifact.claims)}\n"
        f"Allowed indices (claims whose supported field is true): {json.dumps(allowed)}\n"
        f"Previously rejected indices: {json.dumps(rejected)}\n\n"
        'Return only: {"supported_claim_indices": [...]}'
    )


def _supported_claim_indices_patch_from_text(
    raw: str,
    *,
    schema: dict[str, Any],
) -> list[int]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex index correction was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Codex index correction must be one JSON object.")
    errors = list(Draft202012Validator(schema).iter_errors(payload))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(
            f"Codex index correction violated its schema at {location}: {error.message}"
        )
    indices = payload["supported_claim_indices"]
    if len(indices) != len(set(indices)):
        raise ValueError("supported_claim_indices must not contain duplicates.")
    return indices


def _is_supported_claim_indices_error(error: ValueError) -> bool:
    return "supported_claim_indices" in str(error)


def _transport_retry_reason(stdout: str, stderr: str) -> str | None:
    """Classify only errors that are plausibly transient controller transport faults."""

    diagnostic = f"{stderr}\n{stdout}".lower()
    patterns = (
        (
            "codex_backend_404",
            r"(?:backend-api/codex|/codex/(?:responses|models)).{0,500}404"
            r"|404.{0,500}(?:backend-api/codex|/codex/(?:responses|models))",
        ),
        ("unexpected_404", r"unexpected status(?: code)? 404(?: not found)?"),
        (
            "retryable_http_status",
            r"(?:status(?: code)?|http)[^\n]{0,80}"
            r"\b(?:408|409|425|429|500|502|503|504)\b",
        ),
        (
            "websocket_transport",
            r"websocket[^\n]{0,160}(?:fail|error|closed|disconnect|handshake|timeout)",
        ),
        (
            "response_stream",
            r"stream (?:disconnected|closed|ended)[^\n]{0,120}"
            r"(?:completion|response|turn)",
        ),
        (
            "connection_failure",
            r"(?:connection (?:reset|refused|aborted)|failed to connect|"
            r"temporary failure in name resolution|network is unreachable)",
        ),
        ("request_timeout", r"(?:request|transport|connection)[^\n]{0,100}(?:timed out|timeout)"),
    )
    for reason, pattern in patterns:
        if re.search(pattern, diagnostic, flags=re.DOTALL):
            return reason
    return None


def _resource_retry_reason(stdout: str, stderr: str, returncode: int) -> str | None:
    """Recognize bounded-analysis failures that are safe to retry correctively."""

    diagnostic = f"{stderr}\n{stdout}".lower()
    patterns = (
        ("analysis_memory_limit", r"ocs_analysis_resource_limit"),
        ("python_memory_error", r"\bmemoryerror\b"),
        (
            "allocation_failure",
            r"(?:unable|failed) to allocate [^\n]{0,160}(?:mib|gib|bytes?)"
            r"|cannot allocate memory|std::bad_alloc",
        ),
        (
            "out_of_memory",
            r"\bout of memory\b|oom-kill|oom killer|killed process [^\n]{0,160}oom",
        ),
        (
            "analysis_process_sigkill",
            r"(?:exit(?:ed)?(?: with)?(?: code)?|returncode|exit_code)[\"': =-]*137\b"
            r"|killed by signal 9|signal: 9 \(sigkill\)",
        ),
        (
            "analysis_process_sigsegv",
            r"(?:exit(?:ed)?(?: with)?(?: code)?|returncode|exit_code)[\"': =-]*139\b"
            r"|segmentation fault|signal: 11 \(sigsegv\)",
        ),
    )
    for reason, pattern in patterns:
        if re.search(pattern, diagnostic, flags=re.DOTALL):
            return reason
    signal_reasons = {
        -9: "codex_process_sigkill",
        137: "codex_process_sigkill",
        -11: "codex_process_sigsegv",
        139: "codex_process_sigsegv",
        -6: "codex_process_sigabrt",
        134: "codex_process_sigabrt",
    }
    return signal_reasons.get(returncode)


def _transport_retry_prompt(task_prompt: str, reason: str) -> str:
    return (
        "A transient controller transport failure interrupted the preceding attempt "
        f"({reason}). Complete the pending task below in this same session. Reuse any "
        "work already completed, but ensure the requested structured response is returned.\n\n"
        "--- pending task ---\n"
        f"{task_prompt}"
    )


def _resource_retry_prompt(task_prompt: str, reason: str) -> str:
    return (
        "The preceding attempt was interrupted by a controlled analysis-resource "
        f"failure ({reason}). Continue the pending task in this session, but do not "
        "rerun the failed analysis unchanged. First inspect all array/data-frame shapes "
        "and broadcasting operations; use explicit column dimensions (for example, "
        "vector[:, None]), shape assertions, compact dtypes, or chunked calculations as "
        "appropriate. Reuse completed work and return the requested structured response.\n\n"
        "--- pending task ---\n"
        f"{task_prompt}"
    )


def _experiment_root_for_call(call_dir: Path) -> Path:
    for ancestor in call_dir.parents:
        if ancestor.name == "runs":
            return ancestor.parent
    # Unit tests and standalone adapter probes use <root>/run/calls/<call>.
    return call_dir.parent.parent.parent


def _circuit_paths(call_dir: Path) -> tuple[Path, Path]:
    root = _experiment_root_for_call(call_dir)
    return root / ".codex_transport_circuit.json", root / ".codex_transport_circuit.lock"


def _read_circuit_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _locked_circuit_update(
    state_path: Path,
    lock_path: Path,
    update: Any,
) -> dict[str, Any]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _read_circuit_state(state_path)
        state = update(state)
        _write_json_atomic(state_path, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return state


def _register_transport_failure(
    *,
    call_dir: Path,
    reason: str,
    threshold: int,
    cooldown_seconds: float,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    state_path, lock_path = _circuit_paths(call_dir)
    now = time.time()

    def update(state: dict[str, Any]) -> dict[str, Any]:
        streak = int(state.get("failure_streak", 0) or 0) + 1
        open_until = float(state.get("open_until_epoch", 0.0) or 0.0)
        if threshold > 0 and streak >= threshold:
            open_until = max(open_until, now + cooldown_seconds, now + retry_delay_seconds)
        return {
            "schema_version": 1,
            "failure_streak": streak,
            "open_until_epoch": open_until,
            "last_failure_epoch": now,
            "last_failure_reason": reason,
        }

    return _locked_circuit_update(state_path, lock_path, update)


def _register_transport_success(call_dir: Path) -> None:
    state_path, lock_path = _circuit_paths(call_dir)
    now = time.time()

    def update(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "failure_streak": 0,
            "open_until_epoch": 0.0,
            "last_success_epoch": now,
            "last_failure_epoch": state.get("last_failure_epoch"),
            "last_failure_reason": state.get("last_failure_reason"),
        }

    _locked_circuit_update(state_path, lock_path, update)


def _transport_open_until(call_dir: Path) -> float:
    state_path, lock_path = _circuit_paths(call_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        state = _read_circuit_state(state_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return float(state.get("open_until_epoch", 0.0) or 0.0)


def _runtime_retry_delay(args: argparse.Namespace, retry_number: int, request_id: str) -> float:
    base = min(
        args.runtime_retry_max_seconds,
        args.runtime_retry_initial_seconds * (2 ** max(0, retry_number - 1)),
    )
    # Stable 0-20% jitter prevents six workers from retrying on the same instant.
    digest = hashlib.sha256(f"{request_id}:{retry_number}".encode()).digest()
    jitter = digest[0] / 255 * 0.2
    return base * (1.0 + jitter)


def _copy_attempt_outputs(attempt_dir: Path, call_dir: Path) -> None:
    """Expose the latest attempt at the legacy call-root paths without losing history."""

    for name in (
        "codex_agent_artifact.schema.json",
        "codex_last_message.json",
        "codex_events.jsonl",
        "codex_stderr.log",
    ):
        source = attempt_dir / name
        destination = call_dir / name
        if source.exists():
            shutil.copy2(source, destination)
        else:
            destination.unlink(missing_ok=True)


def run_adapter(args: argparse.Namespace) -> AgentResponse:
    request = json.loads(args.request_file.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("Request file must contain one JSON object.")

    request_id = _required_string(request, "request_id")
    session_id = _required_string(request, "session_id")
    model_id = _required_string(request, "model_id")
    request_reasoning_effort = _required_string(request, "reasoning_effort")
    if request_reasoning_effort != args.reasoning_effort:
        raise RuntimeError(
            "Refusing a reasoning-effort mismatch between the harness request "
            f"({request_reasoning_effort}) and adapter CLI ({args.reasoning_effort})."
        )
    require_final_answer = _required_bool(request, "require_final_answer")
    prompt = _required_string(request, "prompt")
    workspace = Path(_required_string(request, "workspace")).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError(f"Workspace is not a directory: {workspace}")
    scratch_dir = Path(_required_string(request, "scratch_dir")).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    read_roots = [path.resolve(strict=True) for path in args.read_root]
    # Codex launches its packaged native binary again inside bubblewrap when a
    # shell tool runs.  The user auth/config directory remains denied; only the
    # immutable installation tree is exposed to the model sandbox.
    read_roots.append(_codex_runtime_root(args.codex))
    python_executable: Path | None = None
    python_home: Path | None = None
    python_site_packages: Path | None = None
    if args.analysis_python is not None:
        python_executable, python_home, python_site_packages = _analysis_python_runtime(
            args.analysis_python
        )
        read_roots.extend((python_home, python_site_packages))
    shell_environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(scratch_dir),
    }
    if python_executable is not None and python_site_packages is not None:
        python_path_entries = [str(python_site_packages)]
        if args.analysis_memory_limit_mb > 0:
            guard_root = _analysis_guard_root()
            read_roots.append(guard_root)
            python_path_entries.insert(0, str(guard_root))
        shell_environment.update(
            {
                "PATH": f"{python_executable.parent}:/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONPATH": os.pathsep.join(python_path_entries),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        if args.analysis_memory_limit_mb > 0:
            shell_environment["OCS_ANALYSIS_MEMORY_LIMIT_MB"] = str(
                args.analysis_memory_limit_mb
            )

    call_dir = args.output.parent
    call_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = call_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    audit_path = call_dir / "codex_call.json"
    stage_schema = artifact_schema(require_final_answer=require_final_answer)
    output_schema_sha256 = _sha256(
        json.dumps(stage_schema, sort_keys=True, separators=(",", ":"))
    )

    session_path = _session_record_path(request)
    thread_id = _load_session(
        session_path,
        session_id=session_id,
        model_id=model_id,
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
    )
    initial_session_action = "resumed" if thread_id is not None else "started"
    root_audit: dict[str, Any] = {
        "schema_version": 4,
        "request_id": request_id,
        "prompt_sha256": _sha256(prompt),
        "model_id": model_id,
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
        "require_final_answer": require_final_answer,
        "session_action": initial_session_action,
        "stdin": "<PROMPT_ON_STDIN>",
        "output_schema_sha256": output_schema_sha256,
        "max_contract_repairs": args.max_contract_repairs,
        "max_runtime_retries": args.max_runtime_retries,
        "runtime_retry_initial_seconds": args.runtime_retry_initial_seconds,
        "runtime_retry_max_seconds": args.runtime_retry_max_seconds,
        "runtime_retry_budget_seconds": args.runtime_retry_budget_seconds,
        "transport_circuit_failure_threshold": args.transport_circuit_failure_threshold,
        "transport_circuit_cooldown_seconds": args.transport_circuit_cooldown_seconds,
        "analysis_memory_limit_mb": args.analysis_memory_limit_mb,
        "attempt_count": 0,
        "contract_repair_count": 0,
        "runtime_retry_count": 0,
        "transport_retry_count": 0,
        "resource_retry_count": 0,
        "attempts": [],
        "status": "starting",
    }
    _write_json_atomic(audit_path, root_audit)

    overall_started = time.monotonic()
    deadline = overall_started + args.timeout_seconds
    child_env = os.environ.copy()
    child_env["TMPDIR"] = str(scratch_dir)
    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0
    all_item_types: set[str] = set()
    attempt_records: list[dict[str, Any]] = []
    task_prompt = prompt
    task_kind = "original"
    task_schema = stage_schema
    next_prompt_kind = "original"
    prior_validation_error: str | None = None
    pending_artifact: AgentArtifact | None = None
    retry_not_before_epoch = 0.0
    retry_for_attempt: int | None = None
    last_retry_kind: str | None = None
    last_retry_reason: str | None = None
    runtime_retry_window_started: float | None = None
    contract_repair_count = 0
    runtime_retry_count = 0
    transport_retry_count = 0
    resource_retry_count = 0
    artifact: AgentArtifact | None = None
    raw_text = ""
    final_command: list[str] = []

    def update_root_audit(status: str, **details: Any) -> None:
        root_audit.update(
            {
                "status": status,
                "attempt_count": len(attempt_records),
                "contract_repair_count": contract_repair_count,
                "runtime_retry_count": runtime_retry_count,
                "transport_retry_count": transport_retry_count,
                "resource_retry_count": resource_retry_count,
                "attempts": attempt_records,
                "command": final_command,
                "duration_seconds": time.monotonic() - overall_started,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "tool_calls": total_tool_calls,
                "item_types": sorted(all_item_types),
            }
        )
        if thread_id is not None:
            root_audit["codex_thread_id"] = thread_id
        root_audit.update(details)
        _write_json_atomic(audit_path, root_audit)

    def adopt_thread_id(stdout: str) -> str | None:
        nonlocal thread_id
        observed_thread_id = extract_codex_thread_id(stdout)
        if thread_id is None and observed_thread_id is not None:
            thread_id = observed_thread_id
            _save_session(
                session_path,
                session_id=session_id,
                thread_id=thread_id,
                model_id=model_id,
                reasoning_effort=args.reasoning_effort,
                service_tier=args.service_tier,
            )
        elif (
            thread_id is not None
            and observed_thread_id is not None
            and observed_thread_id != thread_id
        ):
            return "thread_id_mismatch"
        return None

    attempt_number = 0
    while True:
        while True:
            circuit_open_until = (
                _transport_open_until(call_dir)
                if args.transport_circuit_failure_threshold > 0
                else 0.0
            )
            wake_epoch = max(retry_not_before_epoch, circuit_open_until)
            wait_seconds = wake_epoch - time.time()
            if wait_seconds <= 0:
                break
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                update_root_audit("timeout", error="total_timeout_during_transport_backoff")
                raise TimeoutError(
                    f"Codex call exhausted its total {args.timeout_seconds}s deadline "
                    "during transport backoff."
                )
            update_root_audit(
                "transport_backoff",
                transport_backoff_until_epoch=wake_epoch,
                transport_backoff_remaining_seconds=max(0.0, wait_seconds),
                last_transport_reason=(
                    last_retry_reason if last_retry_kind == "transport" else None
                ),
            )
            time.sleep(min(wait_seconds, remaining_seconds, 30.0))

        attempt_number += 1
        prompt_kind = next_prompt_kind
        if prompt_kind == "transport_retry" and thread_id is not None:
            current_prompt = _transport_retry_prompt(
                task_prompt, last_retry_reason or "transport_error"
            )
        elif prompt_kind == "resource_retry":
            current_prompt = _resource_retry_prompt(
                task_prompt, last_retry_reason or "analysis_resource_failure"
            )
        else:
            current_prompt = task_prompt
        attempt_dir = attempts_dir / f"attempt_{attempt_number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        schema_path = attempt_dir / "codex_agent_artifact.schema.json"
        last_message_path = attempt_dir / "codex_last_message.json"
        events_path = attempt_dir / "codex_events.jsonl"
        stderr_path = attempt_dir / "codex_stderr.log"
        attempt_audit_path = attempt_dir / "codex_call.json"
        _write_json_atomic(schema_path, task_schema)
        last_message_path.unlink(missing_ok=True)

        attempt_session_action = "resumed" if thread_id is not None else "started"
        command = _codex_command(
            codex=args.codex,
            model_id=model_id,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
            workspace=workspace,
            scratch_dir=scratch_dir,
            schema_path=schema_path,
            last_message_path=last_message_path,
            read_roots=tuple(dict.fromkeys(read_roots)),
            shell_environment=shell_environment,
            thread_id=thread_id,
        )
        final_command = command
        attempt_audit = _audit_payload(
            request=request,
            command=command,
            model_id=model_id,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
            session_action=attempt_session_action,
            prompt=current_prompt,
            prompt_kind=prompt_kind,
            attempt_number=attempt_number,
        )
        attempt_audit["schema_version"] = 4
        attempt_audit["task_kind"] = task_kind
        attempt_audit["output_schema_sha256"] = _sha256(
            json.dumps(task_schema, sort_keys=True, separators=(",", ":"))
        )
        if task_kind != "original" and prior_validation_error is not None:
            attempt_audit["repair_for_error"] = prior_validation_error
        if prompt_kind in {"transport_retry", "resource_retry"}:
            attempt_audit["retry_for_attempt"] = retry_for_attempt
            attempt_audit["runtime_retry_number"] = runtime_retry_count
            attempt_audit["runtime_retry_kind"] = last_retry_kind
            attempt_audit["runtime_retry_reason"] = last_retry_reason
            if prompt_kind == "transport_retry":
                attempt_audit["transport_retry_number"] = transport_retry_count
                attempt_audit["transport_retry_reason"] = last_retry_reason
            else:
                attempt_audit["resource_retry_number"] = resource_retry_count
                attempt_audit["resource_retry_reason"] = last_retry_reason
        attempt_audit["status"] = "running"
        _write_json_atomic(attempt_audit_path, attempt_audit)
        _copy_attempt_outputs(attempt_dir, call_dir)
        update_root_audit("running")

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            events_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            attempt_audit.update(
                {
                    "status": "timeout",
                    "duration_seconds": 0.0,
                    "error": "total_timeout_before_launch",
                }
            )
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "status": "timeout",
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit("timeout", error="total_timeout_before_launch")
            raise TimeoutError(
                f"Codex call exhausted its total {args.timeout_seconds}s deadline."
            )

        attempt_started = time.monotonic()
        try:
            completed = run_subprocess_in_group(
                command,
                cwd=workspace,
                env=child_env,
                timeout=remaining_seconds,
                input_text=current_prompt,
            )
        except subprocess.TimeoutExpired as exc:
            attempt_duration = time.monotonic() - attempt_started
            stdout = (
                exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode()
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            events_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")
            stats = parse_codex_events(stdout)
            total_input_tokens += stats.input_tokens
            total_output_tokens += stats.output_tokens
            total_tool_calls += stats.tool_calls
            all_item_types.update(stats.item_types)
            thread_error = adopt_thread_id(stdout)
            attempt_audit.update(
                {
                    "status": "timeout",
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "item_types": list(stats.item_types),
                    "error": "timeout",
                }
            )
            if thread_id is not None:
                attempt_audit["codex_thread_id"] = thread_id
            if thread_error is not None:
                attempt_audit["thread_error"] = thread_error
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "timeout",
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit("timeout", error="timeout")
            raise TimeoutError(
                f"Codex call exceeded its total {args.timeout_seconds}s deadline."
            ) from exc

        attempt_duration = time.monotonic() - attempt_started
        events_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        stats = parse_codex_events(completed.stdout)
        total_input_tokens += stats.input_tokens
        total_output_tokens += stats.output_tokens
        total_tool_calls += stats.tool_calls
        all_item_types.update(stats.item_types)
        thread_error = adopt_thread_id(completed.stdout)
        attempt_audit.update(
            {
                "returncode": completed.returncode,
                "duration_seconds": attempt_duration,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "tool_calls": stats.tool_calls,
                "item_types": list(stats.item_types),
            }
        )

        if thread_id is not None:
            attempt_audit["codex_thread_id"] = thread_id
        if thread_error is not None:
            attempt_audit.update(
                {"status": "runtime_error", "error": thread_error}
            )
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "runtime_error",
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit("runtime_error", error=thread_error)
            raise RuntimeError("Resumed Codex call emitted a different thread ID.")

        if completed.returncode != 0:
            transport_reason = _transport_retry_reason(
                completed.stdout, completed.stderr
            )
            resource_reason = (
                None
                if transport_reason is not None
                else _resource_retry_reason(
                    completed.stdout,
                    completed.stderr,
                    completed.returncode,
                )
            )
            retry_kind = (
                "transport"
                if transport_reason is not None
                else ("resource" if resource_reason is not None else None)
            )
            retry_reason = transport_reason or resource_reason
            if retry_reason is not None and runtime_retry_window_started is None:
                runtime_retry_window_started = time.monotonic()
            retry_budget_used = (
                0.0
                if runtime_retry_window_started is None
                else time.monotonic() - runtime_retry_window_started
            )
            retry_budget_available = (
                args.runtime_retry_budget_seconds == 0
                or retry_budget_used < args.runtime_retry_budget_seconds
            )
            can_retry = (
                retry_reason is not None
                and runtime_retry_count < args.max_runtime_retries
                and retry_budget_available
            )
            retry_delay = (
                _runtime_retry_delay(
                    args,
                    runtime_retry_count + 1,
                    request_id,
                )
                if can_retry
                else 0.0
            )
            if retry_kind == "transport" and retry_reason is not None:
                circuit_state = _register_transport_failure(
                    call_dir=call_dir,
                    reason=retry_reason,
                    threshold=args.transport_circuit_failure_threshold,
                    cooldown_seconds=args.transport_circuit_cooldown_seconds,
                    retry_delay_seconds=retry_delay,
                )
            else:
                circuit_state = {}
            enough_deadline = deadline - time.monotonic() > retry_delay
            if can_retry and enough_deadline:
                runtime_retry_count += 1
                if retry_kind == "transport":
                    transport_retry_count += 1
                else:
                    resource_retry_count += 1
                retry_status = f"{retry_kind}_retry_pending"
                attempt_audit.update(
                    {
                        "status": retry_status,
                        "error": "nonzero_exit",
                        "runtime_retry_kind": retry_kind,
                        "runtime_retry_reason": retry_reason,
                        "next_runtime_retry_number": runtime_retry_count,
                        "retry_delay_seconds": retry_delay,
                    }
                )
                if retry_kind == "transport":
                    attempt_audit.update(
                        {
                            "transport_retry_reason": retry_reason,
                            "next_transport_retry_number": transport_retry_count,
                            "circuit_failure_streak": circuit_state.get(
                                "failure_streak"
                            ),
                            "circuit_open_until_epoch": circuit_state.get(
                                "open_until_epoch"
                            ),
                        }
                    )
                else:
                    attempt_audit.update(
                        {
                            "resource_retry_reason": retry_reason,
                            "next_resource_retry_number": resource_retry_count,
                        }
                    )
                _write_json_atomic(attempt_audit_path, attempt_audit)
                _copy_attempt_outputs(attempt_dir, call_dir)
                attempt_record = {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": retry_status,
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "runtime_retry_kind": retry_kind,
                    "runtime_retry_reason": retry_reason,
                    "retry_delay_seconds": retry_delay,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
                attempt_record[f"{retry_kind}_retry_reason"] = retry_reason
                attempt_records.append(attempt_record)
                root_retry_details = {
                    "returncode": completed.returncode,
                    "last_runtime_retry_kind": retry_kind,
                    "last_runtime_retry_reason": retry_reason,
                    "next_retry_delay_seconds": retry_delay,
                }
                root_retry_details[f"last_{retry_kind}_reason"] = retry_reason
                update_root_audit(
                    retry_status,
                    **root_retry_details,
                )
                retry_not_before_epoch = time.time() + retry_delay
                retry_for_attempt = attempt_number
                last_retry_kind = retry_kind
                last_retry_reason = retry_reason
                next_prompt_kind = f"{retry_kind}_retry"
                continue

            exhaustion = "nonretryable_runtime_error"
            if retry_reason is not None and runtime_retry_count >= args.max_runtime_retries:
                exhaustion = "runtime_retry_limit_exhausted"
            elif retry_reason is not None and not retry_budget_available:
                exhaustion = "runtime_retry_budget_exhausted"
            elif retry_reason is not None and not enough_deadline:
                exhaustion = "runtime_retry_deadline_exhausted"
            attempt_audit.update(
                {
                    "status": "runtime_error",
                    "error": exhaustion,
                    "runtime_retry_kind": retry_kind,
                    "runtime_retry_reason": retry_reason,
                }
            )
            if retry_kind is not None:
                attempt_audit[f"{retry_kind}_retry_reason"] = retry_reason
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "runtime_error",
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit(
                "runtime_error",
                returncode=completed.returncode,
                error=exhaustion,
                last_runtime_retry_kind=retry_kind,
                last_runtime_retry_reason=retry_reason,
            )
            diagnostic = completed.stderr[-1000:] or completed.stdout[-1000:]
            raise RuntimeError(f"Codex exited {completed.returncode}: {diagnostic}")

        if args.transport_circuit_failure_threshold > 0:
            _register_transport_success(call_dir)
        runtime_retry_window_started = None

        observed_thread_id = extract_codex_thread_id(completed.stdout)
        if thread_id is None:
            if observed_thread_id is None:
                attempt_audit.update(
                    {"status": "runtime_error", "error": "missing_thread_id"}
                )
                _write_json_atomic(attempt_audit_path, attempt_audit)
                _copy_attempt_outputs(attempt_dir, call_dir)
                attempt_records.append(
                    {
                        "attempt_number": attempt_number,
                        "prompt_kind": prompt_kind,
                        "task_kind": task_kind,
                        "status": "runtime_error",
                        "returncode": completed.returncode,
                        "duration_seconds": attempt_duration,
                        "input_tokens": stats.input_tokens,
                        "output_tokens": stats.output_tokens,
                        "tool_calls": stats.tool_calls,
                        "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                    }
                )
                update_root_audit("runtime_error", error="missing_thread_id")
                raise RuntimeError("New Codex call did not emit a persisted thread ID.")
            # adopt_thread_id() normally persisted this before return-code handling.
            thread_id = observed_thread_id
        elif observed_thread_id is not None and observed_thread_id != thread_id:
            attempt_audit.update(
                {"status": "runtime_error", "error": "thread_id_mismatch"}
            )
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "runtime_error",
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit("runtime_error", error="thread_id_mismatch")
            raise RuntimeError("Resumed Codex call emitted a different thread ID.")

        attempt_audit["codex_thread_id"] = thread_id
        if not last_message_path.exists():
            attempt_audit.update(
                {"status": "runtime_error", "error": "missing_last_message"}
            )
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "runtime_error",
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            update_root_audit("runtime_error", error="missing_last_message")
            raise RuntimeError("Codex completed without writing --output-last-message.")

        raw_text = last_message_path.read_text(encoding="utf-8")
        candidate_artifact: AgentArtifact | None = None
        try:
            if task_kind == "supported_claim_indices_repair":
                if pending_artifact is None:
                    raise RuntimeError("Index repair started without a source artifact.")
                indices = _supported_claim_indices_patch_from_text(
                    raw_text,
                    schema=task_schema,
                )
                merged_payload = pending_artifact.model_dump(mode="json")
                final_answer = merged_payload.get("final_answer")
                if not isinstance(final_answer, dict):
                    raise RuntimeError("Index repair source lacks a final-answer object.")
                final_answer["supported_claim_indices"] = indices
                candidate_artifact = AgentArtifact.model_validate(merged_payload)
                artifact = candidate_artifact
                raw_text = json.dumps(merged_payload, indent=2) + "\n"
                (attempt_dir / "codex_merged_artifact.json").write_text(
                    raw_text,
                    encoding="utf-8",
                )
                attempt_audit["merged_artifact_file"] = "codex_merged_artifact.json"
            else:
                candidate_artifact = _artifact_from_text(raw_text, schema=stage_schema)
                artifact = candidate_artifact
            _validate_artifact_contract(
                artifact,
                require_final_answer=require_final_answer,
            )
        except ValueError as exc:
            prior_validation_error = f"{type(exc).__name__}: {exc}"
            attempt_audit.update(
                {
                    "status": "contract_rejected",
                    "validation_error": prior_validation_error,
                }
            )
            _write_json_atomic(attempt_audit_path, attempt_audit)
            _copy_attempt_outputs(attempt_dir, call_dir)
            attempt_records.append(
                {
                    "attempt_number": attempt_number,
                    "prompt_kind": prompt_kind,
                    "task_kind": task_kind,
                    "status": "contract_rejected",
                    "returncode": completed.returncode,
                    "duration_seconds": attempt_duration,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "tool_calls": stats.tool_calls,
                    "validation_error": prior_validation_error,
                    "audit_file": str(attempt_audit_path.relative_to(call_dir)),
                }
            )
            if contract_repair_count < args.max_contract_repairs:
                contract_repair_count += 1
                update_root_audit(
                    "repair_pending",
                    returncode=completed.returncode,
                    last_contract_error=prior_validation_error,
                )
                repair_source = candidate_artifact or pending_artifact
                if (
                    require_final_answer
                    and repair_source is not None
                    and (
                        task_kind == "supported_claim_indices_repair"
                        or _is_supported_claim_indices_error(exc)
                    )
                ):
                    pending_artifact = repair_source
                    task_kind = "supported_claim_indices_repair"
                    task_schema = _supported_claim_indices_schema(
                        [
                            index
                            for index, claim in enumerate(repair_source.claims)
                            if claim.supported
                        ]
                    )
                    task_prompt = _supported_claim_indices_repair_prompt(
                        repair_source,
                        exc,
                    )
                else:
                    pending_artifact = None
                    task_kind = "contract_repair"
                    task_schema = stage_schema
                    task_prompt = _contract_repair_prompt(exc)
                next_prompt_kind = task_kind
                retry_not_before_epoch = 0.0
                retry_for_attempt = None
                continue
            update_root_audit(
                "contract_rejected",
                returncode=completed.returncode,
                error="contract_repair_limit_exhausted",
                last_contract_error=prior_validation_error,
            )
            raise ValueError(
                "Codex artifact remained invalid after "
                f"{attempt_number} physical attempt(s): {prior_validation_error}"
            ) from exc

        attempt_audit["status"] = "accepted"
        _write_json_atomic(attempt_audit_path, attempt_audit)
        _copy_attempt_outputs(attempt_dir, call_dir)
        attempt_records.append(
            {
                "attempt_number": attempt_number,
                "prompt_kind": prompt_kind,
                "task_kind": task_kind,
                "status": "accepted",
                "returncode": completed.returncode,
                "duration_seconds": attempt_duration,
                "input_tokens": stats.input_tokens,
                "output_tokens": stats.output_tokens,
                "tool_calls": stats.tool_calls,
                "audit_file": str(attempt_audit_path.relative_to(call_dir)),
            }
        )
        # A narrow correction attempt intentionally emits only a patch. Expose a
        # complete, stage-schema-valid artifact at the canonical call-root path.
        _write_json_atomic(call_dir / "codex_agent_artifact.schema.json", stage_schema)
        (call_dir / "codex_last_message.json").write_text(raw_text, encoding="utf-8")
        update_root_audit("accepted", returncode=completed.returncode)
        break

    if artifact is None or thread_id is None:
        raise RuntimeError("Codex adapter ended without an accepted artifact and session.")

    duration = time.monotonic() - overall_started
    return AgentResponse(
        request_id=request_id,
        artifact=artifact,
        usage=AgentUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            tool_calls=total_tool_calls,
            duration_seconds=duration,
        ),
        raw_text=raw_text,
        runtime_metadata={
            "adapter": "codex-cli-json",
            "codex": args.codex,
            "model_id": model_id,
            "reasoning_effort": args.reasoning_effort,
            "service_tier": args.service_tier,
            "session_action": initial_session_action,
            "codex_thread_id": thread_id,
            "attempt_count": len(attempt_records),
            "contract_repair_count": contract_repair_count,
            "runtime_retry_count": runtime_retry_count,
            "transport_retry_count": transport_retry_count,
            "resource_retry_count": resource_retry_count,
            "analysis_memory_limit_mb": args.analysis_memory_limit_mb,
            "attempt_audit_files": [record["audit_file"] for record in attempt_records],
            "events_file": "codex_events.jsonl",
            "stderr_file": "codex_stderr.log",
            "audit_file": audit_path.name,
            "item_types": sorted(all_item_types),
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", default="codex")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument(
        "--service-tier",
        choices=("default", "fast"),
        default="default",
        help="Codex service tier; 'fast' requests priority processing.",
    )
    parser.add_argument(
        "--max-contract-repairs",
        type=int,
        default=0,
        help=(
            "Maximum same-session correction turns after JSON/schema/semantic "
            "artifact validation failures."
        ),
    )
    parser.add_argument(
        "--max-runtime-retries",
        type=int,
        default=0,
        help=(
            "Maximum retries for classified transient transport failures or "
            "controlled analysis-resource failures."
        ),
    )
    parser.add_argument(
        "--runtime-retry-initial-seconds",
        type=float,
        default=15.0,
        help="Initial exponential-backoff delay for transient transport failures.",
    )
    parser.add_argument(
        "--runtime-retry-max-seconds",
        type=float,
        default=900.0,
        help="Maximum per-retry transport backoff delay.",
    )
    parser.add_argument(
        "--runtime-retry-budget-seconds",
        type=float,
        default=0.0,
        help="Retry-window time budget; zero uses only the total call deadline.",
    )
    parser.add_argument(
        "--transport-circuit-failure-threshold",
        type=int,
        default=0,
        help="Shared consecutive-failure count that opens the experiment transport gate.",
    )
    parser.add_argument(
        "--transport-circuit-cooldown-seconds",
        type=float,
        default=120.0,
        help="Minimum shared pause after the transport circuit opens.",
    )
    parser.add_argument(
        "--read-root",
        action="append",
        default=[],
        type=Path,
        help="Additional non-sensitive directory that model tools may read (repeatable).",
    )
    parser.add_argument(
        "--analysis-python",
        type=Path,
        help=(
            "Virtual-environment Python whose resolved executable, runtime, and "
            "site-packages are exposed read-only to isolated model subprocesses."
        ),
    )
    parser.add_argument(
        "--analysis-memory-limit-mb",
        type=int,
        default=0,
        help=(
            "Per-analysis-Python address-space ceiling in MiB; zero disables the "
            "startup guard."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    if args.max_contract_repairs < 0:
        parser.error("--max-contract-repairs must be non-negative")
    if args.max_runtime_retries < 0:
        parser.error("--max-runtime-retries must be non-negative")
    if args.analysis_memory_limit_mb < 0:
        parser.error("--analysis-memory-limit-mb must be non-negative")
    if args.analysis_memory_limit_mb > 0 and args.analysis_python is None:
        parser.error("--analysis-memory-limit-mb requires --analysis-python")
    if args.runtime_retry_initial_seconds < 0:
        parser.error("--runtime-retry-initial-seconds must be non-negative")
    if args.runtime_retry_max_seconds < args.runtime_retry_initial_seconds:
        parser.error(
            "--runtime-retry-max-seconds must be at least "
            "--runtime-retry-initial-seconds"
        )
    if args.runtime_retry_budget_seconds < 0:
        parser.error("--runtime-retry-budget-seconds must be non-negative")
    if args.transport_circuit_failure_threshold < 0:
        parser.error("--transport-circuit-failure-threshold must be non-negative")
    if args.transport_circuit_cooldown_seconds < 0:
        parser.error("--transport-circuit-cooldown-seconds must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        response = run_adapter(args)
        _write_json_atomic(args.output, response.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports a concise diagnostic
        print(f"codex-cli-json adapter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
