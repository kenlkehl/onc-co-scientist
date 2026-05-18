"""OpenAI-compatible local server for CAA-steered Transformers models.

The module keeps FastAPI and Transformers imports lazy. Pure protocol helpers
are intentionally importable in the normal test environment, while
``create_app``/``serve`` require the runtime environment that hosts the model.
"""

from __future__ import annotations

import ast
import json
import re
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interventions import VectorBundle

DEFAULT_GEMMA31B_MODEL_PATH = (
    "/data1/ken/models/models--google--gemma-4-31b-it/"
    "snapshots/145dc2508c480a64b47242f160d286cff94a2343"
)
DEFAULT_GEMMA31B_VECTOR_PATH = "data/caa/gemma4_31b_clinical_pubmed_layers20_30_40.npz"
DEFAULT_CAA_CONCEPT = "paradigm_orthogonalized"
DEFAULT_CAA_LAYER = 40
DEFAULT_ALIAS_PREFIX = "gemma4-31b"
MAX_SYSTEM_CONTEXT_CHARS = 4000
MAX_TOOL_PROMPT_CHARS = 6000
MAX_TOOL_DESCRIPTION_CHARS = 220
MAX_TOOL_ARGUMENTS = 12
MIN_EXEC_YIELD_TIME_MS = 30_000


@dataclass(frozen=True)
class CaaModelAlias:
    """One exposed OpenAI model id and its steering configuration."""

    model_id: str
    arm: str
    concept: str = DEFAULT_CAA_CONCEPT
    layer: int = DEFAULT_CAA_LAYER
    scale: float | None = None
    mode: str = "add"

    @property
    def is_control(self) -> bool:
        return self.scale is None


ARM_SCALES: dict[str, float | None] = {
    "control": None,
    "neg002": -0.02,
    "neg005": -0.05,
    "neg010": -0.10,
}


def make_caa_model_aliases(
    *,
    alias_prefix: str = DEFAULT_ALIAS_PREFIX,
    layer: int = DEFAULT_CAA_LAYER,
    concept: str = DEFAULT_CAA_CONCEPT,
) -> dict[str, CaaModelAlias]:
    """Build the fixed four-arm alias map for one served model family."""

    prefix = alias_prefix.strip().rstrip("-")
    if not prefix:
        raise ValueError("alias_prefix must not be empty.")
    aliases: dict[str, CaaModelAlias] = {}
    for arm, scale in ARM_SCALES.items():
        model_id = f"{prefix}-control" if arm == "control" else f"{prefix}-caa-l{layer}-{arm}"
        aliases[model_id] = CaaModelAlias(
            model_id=model_id,
            arm=arm,
            concept=concept,
            layer=layer,
            scale=scale,
        )
    return aliases


CAA_MODEL_ALIASES: dict[str, CaaModelAlias] = make_caa_model_aliases()


class UnknownModelError(ValueError):
    """Raised when a request references a model alias the server does not expose."""


def resolve_model_alias(
    model: str | None,
    aliases: dict[str, CaaModelAlias] | None = None,
) -> CaaModelAlias:
    """Return the steering configuration for an exposed model alias."""

    if not model:
        raise UnknownModelError("Request is missing required field 'model'.")
    alias_map = aliases or CAA_MODEL_ALIASES
    try:
        return alias_map[model]
    except KeyError as exc:
        valid = ", ".join(sorted(alias_map))
        raise UnknownModelError(f"Unknown model alias {model!r}. Valid aliases: {valid}.") from exc


def models_payload(aliases: dict[str, CaaModelAlias] | None = None) -> dict[str, Any]:
    """OpenAI-compatible ``/v1/models`` payload."""

    now = int(time.time())
    alias_map = aliases or CAA_MODEL_ALIASES
    return {
        "object": "list",
        "data": [
            {
                "id": alias.model_id,
                "object": "model",
                "created": now,
                "owned_by": "onc-co-scientist",
            }
            for alias in alias_map.values()
        ],
    }


def response_error(message: str, *, status: int = 400, error_type: str = "invalid_request_error"):
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": status,
        }
    }


def build_generation_messages(
    payload: dict[str, Any],
    *,
    chat_messages: bool = False,
    compact_agent_context: bool = False,
    include_tool_instructions: bool = True,
) -> list[dict[str, Any]]:
    """Normalize Responses or Chat Completions payloads into chat messages."""

    messages: list[dict[str, Any]] = []
    instructions = _as_text(payload.get("instructions"))
    if compact_agent_context:
        instructions = compact_context_text(instructions, role="system")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    raw_messages = payload.get("messages") if chat_messages else payload.get("input")
    if raw_messages is None:
        raw_messages = "" if chat_messages else payload.get("prompt", "")

    if isinstance(raw_messages, str):
        if raw_messages:
            messages.append({"role": "user", "content": raw_messages})
    elif isinstance(raw_messages, list):
        messages.extend(
            _messages_from_input_list(
                raw_messages,
                compact_agent_context=compact_agent_context,
            )
        )
    else:
        messages.append({"role": "user", "content": _as_text(raw_messages)})

    tools = payload.get("tools") if include_tool_instructions else None
    if tools:
        tool_instructions = render_tool_instructions(tools, compact=compact_agent_context)
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + tool_instructions
        else:
            messages.insert(0, {"role": "system", "content": tool_instructions})
    return messages


def normalize_tools_for_chat_template(tools: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-style tools for Hugging Face chat templates."""

    if not isinstance(tools, list):
        return []
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if isinstance(tool.get("function"), dict):
            normalized.append(tool)
            continue
        name = _as_text(tool.get("name"))
        if not name:
            continue
        function = {
            "name": name,
            "description": _as_text(tool.get("description")),
            "parameters": tool.get("parameters") or {},
        }
        normalized.append({"type": tool.get("type") or "function", "function": function})
    return normalized


def render_tool_instructions(tools: Any, *, compact: bool = False) -> str:
    """Render a compact tool schema block for a model without native tool support."""

    if compact:
        tool_lines = _compact_tool_lines(tools)
        body = "\n".join(tool_lines) if tool_lines else "- No named tools were provided."
        rendered = (
            "Tool use is available. If a tool is needed, respond with only JSON in this shape:\n"
            '{"tool_calls":[{"name":"tool_name","arguments":{}}]}\n'
            "Do not wrap tool-call JSON in markdown. Do not emit Python dicts, role fields, "
            "or placeholder tool calls. Every tool call must include a real name and arguments. "
            "Required arguments are marked with *.\n"
            "If a tool result says a session is still running, poll that session with the "
            "available polling tool before treating the command as complete.\n"
            "Do not stop with a plan or future-tense status update; if more work remains, "
            "make the next tool call instead.\n"
            "Available tools:\n"
            + body
        )
        if len(rendered) > MAX_TOOL_PROMPT_CHARS:
            rendered = (
                rendered[: MAX_TOOL_PROMPT_CHARS - 46].rstrip()
                + "\n- Additional tool details omitted."
            )
        return rendered

    return (
        "Tool use is available. If a tool is needed, respond with only JSON in this shape:\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{}}]}\n'
        "Do not wrap tool-call JSON in markdown. Do not emit Python dicts, role fields, "
        "or placeholder tool calls. Every tool call must include a real name and arguments. "
        "If a tool result says a session is still "
        "running, poll that session with the available polling tool before treating the "
        "command as complete. Do not stop with a plan or future-tense status update; if "
        "more work remains, make the next tool call instead. Available tools:\n"
        + json.dumps(tools, indent=2, sort_keys=True)
    )


def compact_context_text(text: str, *, role: str) -> str:
    """Optionally trim overlong agent context before handing it to a local model."""

    stripped = text.strip()
    if not stripped:
        return ""
    if role in {"system", "developer"} and len(stripped) > MAX_SYSTEM_CONTEXT_CHARS:
        return stripped[: MAX_SYSTEM_CONTEXT_CHARS - 32].rstrip() + "\n[context truncated]"
    return stripped


def _compact_tool_lines(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        text = _as_text(tools)
        return [f"- {text[:MAX_TOOL_DESCRIPTION_CHARS]}"] if text else []

    lines: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            text = _as_text(tool)
            if text:
                lines.append(f"- {text[:MAX_TOOL_DESCRIPTION_CHARS]}")
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = _as_text(tool.get("name") or function.get("name") or tool.get("type"))
        if not name:
            continue
        description = _as_text(tool.get("description") or function.get("description"))
        if len(description) > MAX_TOOL_DESCRIPTION_CHARS:
            description = description[: MAX_TOOL_DESCRIPTION_CHARS - 3].rstrip() + "..."
        parameters = tool.get("parameters") or function.get("parameters") or {}
        args = _compact_tool_arguments(parameters)
        signature = f"{name}({', '.join(args)})" if args else name
        line = f"- {signature}"
        if description:
            line += f": {description}"
        lines.append(line)
    return lines


def _compact_tool_arguments(parameters: Any) -> list[str]:
    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return []
    required = set(parameters.get("required") or [])
    args = []
    for idx, (name, schema) in enumerate(properties.items()):
        if idx >= MAX_TOOL_ARGUMENTS:
            args.append("...")
            break
        type_name = _json_schema_type(schema)
        marker = "*" if name in required else ""
        args.append(f"{name}{marker}: {type_name}")
    return args


def _json_schema_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "any"
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = "|".join(_as_text(item) for item in raw_type)
    if raw_type:
        return _as_text(raw_type)
    if "enum" in schema:
        return "enum"
    if "anyOf" in schema or "oneOf" in schema:
        return "union"
    return "any"


def build_responses_payload(
    *,
    request_payload: dict[str, Any],
    generated_text: str,
    model_alias: CaaModelAlias | None = None,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI Responses API object."""

    alias = model_alias or resolve_model_alias(_as_text(request_payload.get("model")))
    response_id = _id("resp")
    created = int(time.time())
    tool_calls = parse_tool_calls(generated_text) if request_payload.get("tools") else []
    output: list[dict[str, Any]]
    output_text = ""
    if tool_calls:
        output = []
        for call in tool_calls:
            item = {
                "id": _id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("call_id") or _id("call"),
                "name": call["name"],
                "arguments": call["arguments"],
            }
            output.append(item)
    else:
        output_text = generated_text
        output = [
            {
                "id": _id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": generated_text,
                        "annotations": [],
                    }
                ],
            }
        ]
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": request_payload.get("instructions"),
        "max_output_tokens": _int_or_none(
            request_payload.get("max_output_tokens") or request_payload.get("max_tokens")
        ),
        "model": alias.model_id,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": bool(request_payload.get("parallel_tool_calls", True)),
        "temperature": request_payload.get("temperature"),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools", []),
        "top_p": request_payload.get("top_p"),
        "truncation": request_payload.get("truncation", "disabled"),
        "usage": None,
    }


def combine_previous_response_messages(
    previous_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine stored Responses state with the current turn.

    Responses clients commonly send a follow-up request with only
    ``previous_response_id`` plus new input items such as function-call output.
    The local server has to reconstruct the prior conversational context before
    handing the turn to a stateless Transformers model.
    """

    if not previous_messages:
        return current_messages
    current_system = [message for message in current_messages if message.get("role") == "system"]
    current_rest = [message for message in current_messages if message.get("role") != "system"]
    if current_system:
        previous_rest = [
            message for message in previous_messages if message.get("role") != "system"
        ]
        return current_system + previous_rest + current_rest
    return previous_messages + current_rest


def messages_from_response_output(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render Responses output items back into chat messages for local state."""

    messages: list[dict[str, Any]] = []
    for item in output:
        item_type = item.get("type")
        if item_type == "function_call":
            name = _as_text(item.get("name"))
            arguments = _as_text(item.get("arguments"))
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": _as_text(item.get("call_id")) or _id("call"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
        elif item_type == "message":
            role = _as_text(item.get("role")) or "assistant"
            content = _content_to_text(item.get("content", ""))
            if content:
                messages.append({"role": role, "content": content})
    return messages


def responses_sse_events(response_payload: dict[str, Any]) -> Iterable[str]:
    """Emit valid SSE events after a buffered generation completes."""

    created_payload = {**response_payload, "status": "in_progress", "output": [], "output_text": ""}
    sequence = 0
    yield _sse(
        "response.created",
        {
            "type": "response.created",
            "sequence_number": sequence,
            "response": created_payload,
        },
    )
    for output_index, item in enumerate(response_payload.get("output", [])):
        sequence += 1
        yield _sse(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": item,
                "response_id": response_payload["id"],
            },
        )
        if item.get("type") == "message":
            content = item.get("content") or []
            if content:
                sequence += 1
                yield _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "sequence_number": sequence,
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "part": content[0],
                        "response_id": response_payload["id"],
                    },
                )
            text = _message_text(item)
            sequence += 1
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                    "response_id": response_payload["id"],
                },
            )
            sequence += 1
            yield _sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "sequence_number": sequence,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                    "response_id": response_payload["id"],
                },
            )
        sequence += 1
        yield _sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": item,
                "response_id": response_payload["id"],
            },
        )
    sequence += 1
    yield _sse(
        "response.completed",
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": response_payload,
        },
    )
    yield "data: [DONE]\n\n"


def build_chat_completion_payload(
    *,
    request_payload: dict[str, Any],
    generated_text: str,
    model_alias: CaaModelAlias | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Chat Completions object for compatibility clients."""

    alias = model_alias or resolve_model_alias(_as_text(request_payload.get("model")))
    tool_calls = parse_tool_calls(generated_text) if request_payload.get("tools") else []
    message: dict[str, Any] = {"role": "assistant"}
    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": call.get("call_id") or _id("call"),
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in tool_calls
        ]
    else:
        message["content"] = generated_text
    return {
        "id": _id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias.model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": None,
    }


def chat_sse_events(chat_payload: dict[str, Any]) -> Iterable[str]:
    choice = chat_payload["choices"][0]
    message = choice["message"]
    base = {
        "id": chat_payload["id"],
        "object": "chat.completion.chunk",
        "created": chat_payload["created"],
        "model": chat_payload["model"],
    }
    yield _sse(
        "chat.completion.chunk",
        {
            **base,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
    )
    if message.get("tool_calls"):
        yield _sse(
            "chat.completion.chunk",
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": message["tool_calls"]},
                        "finish_reason": None,
                    }
                ],
            },
        )
    else:
        yield _sse(
            "chat.completion.chunk",
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": message.get("content", "")},
                        "finish_reason": None,
                    }
                ],
            },
        )
    yield _sse(
        "chat.completion.chunk",
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
        },
    )
    yield "data: [DONE]\n\n"


def parse_tool_calls(text: str) -> list[dict[str, str]]:
    """Parse explicit JSON tool-call output into Responses function calls."""

    parsed = _parse_json_candidate(text)
    if parsed is None:
        return _parse_text_tool_calls(text)
    raw_calls: Any
    if isinstance(parsed, list):
        raw_calls = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            raw_calls = parsed["tool_calls"]
        elif isinstance(parsed.get("function_calls"), list):
            raw_calls = parsed["function_calls"]
        elif isinstance(parsed.get("calls"), list):
            raw_calls = parsed["calls"]
        elif parsed.get("type") in {"function_call", "tool_call"} or "name" in parsed:
            raw_calls = [parsed]
        else:
            return _parse_text_tool_calls(text)
    else:
        return _parse_text_tool_calls(text)

    calls: list[dict[str, str]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if isinstance(function, dict):
            name = _as_text(function.get("name"))
            arguments = function.get("arguments", raw.get("arguments", {}))
        else:
            name = _as_text(raw.get("name") or raw.get("tool_name"))
            arguments = raw.get("arguments", raw.get("args", {}))
        if not name:
            continue
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments = _normalize_tool_arguments(name, arguments)
            arguments_text = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
        calls.append(
            {
                "call_id": _as_text(raw.get("call_id") or raw.get("id")) or _id("call"),
                "name": name,
                "arguments": arguments_text,
            }
        )
    return calls or _parse_text_tool_calls(text)


def _parse_text_tool_calls(text: str) -> list[dict[str, str]]:
    """Parse local-model fallback text like ``Tool call requested: name({...})``."""

    calls: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    pattern = re.compile(r"Tool call requested:\s*([A-Za-z_][\w.-]*)\((.*)\)\s*$")
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        name = match.group(1)
        raw_arguments = match.group(2).strip()
        arguments: Any
        if not raw_arguments:
            arguments = {}
        else:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                try:
                    arguments = ast.literal_eval(raw_arguments)
                except (SyntaxError, ValueError):
                    arguments = raw_arguments
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments = _normalize_tool_arguments(name, arguments)
            arguments_text = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
        key = (name, arguments_text)
        if key in seen:
            continue
        seen.add(key)
        calls.append({"call_id": _id("call"), "name": name, "arguments": arguments_text})
    return calls


def _normalize_tool_arguments(name: str, arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    normalized = dict(arguments)
    if name == "exec_command":
        sandbox_permissions = normalized.get("sandbox_permissions")
        if sandbox_permissions not in {
            None,
            "use_default",
            "require_escalated",
            "with_additional_permissions",
        }:
            normalized.pop("sandbox_permissions", None)
        yield_time_ms = _int_or_none(normalized.get("yield_time_ms"))
        if yield_time_ms is None or yield_time_ms < MIN_EXEC_YIELD_TIME_MS:
            normalized["yield_time_ms"] = MIN_EXEC_YIELD_TIME_MS
    return normalized


class CAAInferenceEngine:
    """Buffered, single-process Transformers inference engine."""

    def __init__(
        self,
        *,
        model_path: str,
        vector_file: Path,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        default_max_new_tokens: int = 4096,
        enable_thinking: bool = False,
        cache_dir: Path | None = None,
        compact_agent_context: bool = False,
        aliases: dict[str, CaaModelAlias] | None = None,
    ) -> None:
        self.model_path = model_path
        self.vector_file = vector_file
        self.dtype = dtype
        self.device_map = device_map
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.default_max_new_tokens = default_max_new_tokens
        self.enable_thinking = enable_thinking
        self.cache_dir = cache_dir
        self.compact_agent_context = compact_agent_context
        self.aliases = aliases or CAA_MODEL_ALIASES
        self.processor = None
        self.model = None
        self.vector_bundle: VectorBundle | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._response_histories: dict[str, list[dict[str, Any]]] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        with self._load_lock:
            if self.loaded:
                return
            from .interventions.caa import load_transformers_text_model

            self.vector_bundle = VectorBundle.load(self.vector_file)
            self.processor, self.model = load_transformers_text_model(
                self.model_path,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
            )

    def generate_for_request(
        self,
        payload: dict[str, Any],
        *,
        chat_messages: bool = False,
    ) -> tuple[str, CaaModelAlias, list[dict[str, Any]]]:
        self.load()
        alias = resolve_model_alias(_as_text(payload.get("model")), self.aliases)
        template_tools = normalize_tools_for_chat_template(payload.get("tools"))
        use_native_tools = False
        if template_tools:
            from .interventions.caa import processor_supports_tools

            assert self.processor is not None
            use_native_tools = processor_supports_tools(self.processor, template_tools)
        messages = build_generation_messages(
            payload,
            chat_messages=chat_messages,
            compact_agent_context=self.compact_agent_context,
            include_tool_instructions=not use_native_tools,
        )
        previous_response_id = _as_text(payload.get("previous_response_id"))
        if previous_response_id:
            with self._history_lock:
                previous_messages = list(self._response_histories.get(previous_response_id, []))
            messages = combine_previous_response_messages(previous_messages, messages)
        max_new_tokens = (
            _int_or_none(payload.get("max_output_tokens") or payload.get("max_tokens"))
            or self.default_max_new_tokens
        )
        temperature = _float_or_default(payload.get("temperature"), 0.2)
        top_p = _float_or_default(payload.get("top_p"), 0.95)
        top_k = _int_or_none(payload.get("top_k")) or 64
        assert self.processor is not None
        assert self.model is not None
        with self._generate_lock:
            if alias.is_control:
                from .interventions.caa import generate_messages_unsteered

                text = generate_messages_unsteered(
                    messages=messages,
                    tools=template_tools if use_native_tools else None,
                    processor=self.processor,
                    model=self.model,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    enable_thinking=self.enable_thinking,
                )
            else:
                from .interventions.caa import generate_messages_with_vector

                assert self.vector_bundle is not None
                text = generate_messages_with_vector(
                    messages=messages,
                    tools=template_tools if use_native_tools else None,
                    vector_bundle=self.vector_bundle,
                    concept=alias.concept,
                    layers=[alias.layer],
                    processor=self.processor,
                    model=self.model,
                    mode=alias.mode,  # type: ignore[arg-type]
                    scale=float(alias.scale),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    enable_thinking=self.enable_thinking,
                )
        return text, alias, messages

    def remember_response(
        self,
        response_id: str,
        generation_messages: list[dict[str, Any]],
        response_payload: dict[str, Any],
    ) -> None:
        output_messages = messages_from_response_output(response_payload.get("output") or [])
        with self._history_lock:
            self._response_histories[response_id] = generation_messages + output_messages


def create_app(engine: CAAInferenceEngine, *, load_on_startup: bool = True):
    """Create the FastAPI app. FastAPI is imported only when serving."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on optional runtime env
        raise RuntimeError(
            "CAA serving requires fastapi and uvicorn in the active Python environment. "
            "Install them in ~/oldtorch or run with an environment that provides them."
        ) from exc

    app = FastAPI(title="Oncology Co-Scientist CAA Server")

    if load_on_startup:

        @app.on_event("startup")
        def _load_model() -> None:
            engine.load()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": engine.loaded,
            "models": list(engine.aliases),
        }

    @app.get("/v1/models")
    def models():
        return models_payload(engine.aliases)

    async def responses(request):
        payload = await request.json()
        try:
            text, alias, generation_messages = engine.generate_for_request(
                payload,
                chat_messages=False,
            )
            response_payload = build_responses_payload(
                request_payload=payload,
                generated_text=text,
                model_alias=alias,
            )
            engine.remember_response(response_payload["id"], generation_messages, response_payload)
        except UnknownModelError as exc:
            return JSONResponse(response_error(str(exc), status=404), status_code=404)
        except Exception as exc:
            return JSONResponse(
                response_error(str(exc), status=500, error_type="server_error"),
                status_code=500,
            )
        if payload.get("stream"):
            return StreamingResponse(
                responses_sse_events(response_payload),
                media_type="text/event-stream",
            )
        return response_payload

    responses.__annotations__["request"] = Request
    app.post("/v1/responses")(responses)

    async def chat_completions(request):
        payload = await request.json()
        try:
            text, alias, _generation_messages = engine.generate_for_request(
                payload,
                chat_messages=True,
            )
            chat_payload = build_chat_completion_payload(
                request_payload=payload,
                generated_text=text,
                model_alias=alias,
            )
        except UnknownModelError as exc:
            return JSONResponse(response_error(str(exc), status=404), status_code=404)
        except Exception as exc:
            return JSONResponse(
                response_error(str(exc), status=500, error_type="server_error"),
                status_code=500,
            )
        if payload.get("stream"):
            return StreamingResponse(
                chat_sse_events(chat_payload),
                media_type="text/event-stream",
            )
        return chat_payload

    chat_completions.__annotations__["request"] = Request
    app.post("/v1/chat/completions")(chat_completions)

    return app


def serve(
    *,
    model_path: str = DEFAULT_GEMMA31B_MODEL_PATH,
    vector_file: Path = Path(DEFAULT_GEMMA31B_VECTOR_PATH),
    host: str = "127.0.0.1",
    port: int = 8765,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    default_max_new_tokens: int = 4096,
    enable_thinking: bool = False,
    cache_dir: Path | None = None,
    compact_agent_context: bool = False,
    alias_prefix: str = DEFAULT_ALIAS_PREFIX,
    steering_layer: int = DEFAULT_CAA_LAYER,
    concept: str = DEFAULT_CAA_CONCEPT,
) -> None:
    """Run the local CAA OpenAI-compatible server."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional runtime env
        raise RuntimeError(
            "CAA serving requires uvicorn in the active Python environment. "
            "Install it in ~/oldtorch or run with an environment that provides it."
        ) from exc

    engine = CAAInferenceEngine(
        model_path=model_path,
        vector_file=vector_file,
        dtype=dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        default_max_new_tokens=default_max_new_tokens,
        enable_thinking=enable_thinking,
        cache_dir=cache_dir,
        compact_agent_context=compact_agent_context,
        aliases=make_caa_model_aliases(
            alias_prefix=alias_prefix,
            layer=steering_layer,
            concept=concept,
        ),
    )
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port)


def _messages_from_input_list(
    items: list[Any],
    *,
    compact_agent_context: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": _as_text(item)})
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            name = _as_text(item.get("name"))
            arguments = _as_text(item.get("arguments"))
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": _as_text(item.get("call_id")) or _id("call"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )
            continue
        if item_type == "function_call_output":
            call_id = _as_text(item.get("call_id"))
            output = _as_text(item.get("output"))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }
            )
            continue
        if item.get("role") == "assistant" and isinstance(item.get("tool_calls"), list):
            message = {
                "role": "assistant",
                "content": item.get("content"),
                "tool_calls": item["tool_calls"],
            }
            messages.append(message)
            continue
        if item.get("role") == "tool":
            content = _content_to_text(item.get("content", item.get("text", "")))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _as_text(item.get("tool_call_id") or item.get("call_id")),
                    "content": content,
                }
            )
            continue
        role = _as_text(item.get("role") or ("assistant" if item_type == "message" else "user"))
        content = _content_to_text(item.get("content", item.get("text", "")))
        if compact_agent_context:
            content = compact_context_text(content, role=role)
        messages.append({"role": role or "user", "content": content})
    return messages


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"input_text", "output_text", "text"} or "text" in part:
                    parts.append(_as_text(part.get("text")))
        return "\n".join(p for p in parts if p)
    return _as_text(content)


def _message_text(item: dict[str, Any]) -> str:
    pieces = []
    for part in item.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "output_text":
            pieces.append(_as_text(part.get("text")))
    return "".join(pieces)


def _parse_json_candidate(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [stripped]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (SyntaxError, ValueError):
            pass
        for idx, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[idx:])
                return parsed
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate[idx:])
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except (SyntaxError, ValueError):
                    continue
    return None


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
