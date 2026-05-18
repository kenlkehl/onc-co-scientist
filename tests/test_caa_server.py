from __future__ import annotations

import json

from onc_co_scientist.caa_server import (
    CAA_MODEL_ALIASES,
    build_generation_messages,
    build_responses_payload,
    combine_previous_response_messages,
    make_caa_model_aliases,
    messages_from_response_output,
    models_payload,
    parse_tool_calls,
    render_tool_instructions,
    resolve_model_alias,
    responses_sse_events,
)


def test_caa_model_aliases_map_to_fixed_arms() -> None:
    assert resolve_model_alias("gemma4-31b-control").is_control

    neg005 = resolve_model_alias("gemma4-31b-caa-l40-neg005")
    assert neg005.concept == "paradigm_orthogonalized"
    assert neg005.layer == 40
    assert neg005.scale == -0.05
    assert neg005.mode == "add"


def test_models_payload_returns_all_aliases() -> None:
    payload = models_payload()
    ids = {item["id"] for item in payload["data"]}
    assert ids == set(CAA_MODEL_ALIASES)


def test_dynamic_model_aliases_support_smaller_gemma_family() -> None:
    aliases = make_caa_model_aliases(alias_prefix="gemma4-e4b", layer=29)
    assert set(aliases) == {
        "gemma4-e4b-control",
        "gemma4-e4b-caa-l29-neg002",
        "gemma4-e4b-caa-l29-neg005",
        "gemma4-e4b-caa-l29-neg010",
    }

    neg010 = resolve_model_alias("gemma4-e4b-caa-l29-neg010", aliases)
    assert neg010.layer == 29
    assert neg010.scale == -0.10

    payload = models_payload(aliases)
    assert {item["id"] for item in payload["data"]} == set(aliases)


def test_responses_input_normalization_accepts_string_arrays_and_instructions() -> None:
    messages = build_generation_messages(
        {
            "instructions": "System context",
            "input": "Analyze this dataset",
        }
    )
    assert messages == [
        {"role": "system", "content": "System context"},
        {"role": "user", "content": "Analyze this dataset"},
    ]

    messages = build_generation_messages(
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "First"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "tool result",
                },
            ]
        }
    )
    assert messages[0] == {"role": "user", "content": "First"}
    assert messages[1]["role"] == "user"
    assert "tool result" in messages[1]["content"]


def test_previous_response_messages_are_preserved_for_tool_followups() -> None:
    first_response = build_responses_payload(
        request_payload={
            "model": "gemma4-31b-control",
            "input": "Read the task brief.",
            "tools": [{"type": "function", "name": "exec_command"}],
        },
        generated_text=json.dumps(
            {"tool_calls": [{"name": "exec_command", "arguments": {"cmd": "cat brief.md"}}]}
        ),
    )
    previous_messages = build_generation_messages(
        {
            "model": "gemma4-31b-control",
            "input": "Read the task brief.",
            "tools": [{"type": "function", "name": "exec_command"}],
        }
    ) + messages_from_response_output(first_response["output"])
    current_messages = build_generation_messages(
        {
            "instructions": "Current system instructions.",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": first_response["output"][0]["call_id"],
                    "output": "brief contents",
                }
            ],
            "tools": [{"type": "function", "name": "exec_command"}],
        }
    )

    combined = combine_previous_response_messages(previous_messages, current_messages)

    assert combined[0]["role"] == "system"
    assert "Current system instructions" in combined[0]["content"]
    assert {"role": "user", "content": "Read the task brief."} in combined
    assert any(
        message["role"] == "assistant" and "Tool call requested: exec_command" in message["content"]
        for message in combined
    )
    assert any("brief contents" in message["content"] for message in combined)


def test_agent_context_compaction_is_opt_in() -> None:
    long_instructions = (
        "You are an agent running in a local co-scientist harness.\n"
        + "Full realistic context. " * 400
    )
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a command in a PTY. " * 100,
            "parameters": {
                "type": "object",
                "required": ["cmd"],
                "properties": {
                    "cmd": {"type": "string"},
                    "workdir": {"type": "string"},
                },
            },
        }
    ]

    default_messages = build_generation_messages(
        {
            "instructions": long_instructions,
            "input": "Reply exactly OK.",
            "tools": tools,
        }
    )
    assert long_instructions.strip() in default_messages[0]["content"]
    assert '"description": "Run a command in a PTY.' in default_messages[0]["content"]

    compact_messages = build_generation_messages(
        {
            "instructions": long_instructions,
            "input": "Reply exactly OK.",
            "tools": tools,
        },
        compact_agent_context=True,
    )
    assert "You are an agent running in a local co-scientist harness." in compact_messages[0][
        "content"
    ]
    assert "[context truncated]" in compact_messages[0]["content"]
    assert len(compact_messages[0]["content"]) < len(default_messages[0]["content"])
    assert "exec_command(cmd*: string, workdir: string)" in compact_messages[0]["content"]


def test_tool_instruction_rendering_preserves_full_schema_by_default() -> None:
    tools = [
        {
            "type": "function",
            "name": "shell",
            "description": "x" * 500,
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]
    full = render_tool_instructions(tools)
    compact = render_tool_instructions(tools, compact=True)

    assert '"description": "' + "x" * 500 in full
    assert len(compact) < len(full)
    assert "shell(cmd: string)" in compact


def test_responses_payload_includes_output_text_and_assistant_message() -> None:
    payload = build_responses_payload(
        request_payload={"model": "gemma4-31b-control", "input": "hello"},
        generated_text="world",
    )
    assert payload["output_text"] == "world"
    assert payload["output"][0]["type"] == "message"
    assert payload["output"][0]["role"] == "assistant"
    assert payload["output"][0]["content"][0]["text"] == "world"


def test_responses_streaming_emits_lifecycle_events() -> None:
    payload = build_responses_payload(
        request_payload={"model": "gemma4-31b-control", "input": "hello"},
        generated_text="streamed text",
    )
    stream = "".join(responses_sse_events(payload))
    assert "event: response.created" in stream
    assert "event: response.output_text.delta" in stream
    assert "streamed text" in stream
    assert "event: response.completed" in stream
    assert "data: [DONE]" in stream


def test_tool_call_parser_and_responses_function_call_items() -> None:
    generated = json.dumps(
        {"tool_calls": [{"name": "shell", "arguments": {"cmd": "pwd"}}]}
    )
    calls = parse_tool_calls(generated)
    assert calls == [
        {
            "call_id": calls[0]["call_id"],
            "name": "shell",
            "arguments": '{"cmd":"pwd"}',
        }
    ]

    payload = build_responses_payload(
        request_payload={
            "model": "gemma4-31b-control",
            "input": "run pwd",
            "tools": [{"type": "function", "name": "shell"}],
        },
        generated_text=generated,
    )
    assert payload["output_text"] == ""
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["name"] == "shell"
    assert payload["output"][0]["arguments"] == '{"cmd":"pwd"}'


def test_tool_call_parser_accepts_python_literal_tool_calls() -> None:
    generated = (
        "{'role': 'assistant', 'tool_calls': [{'type': 'function', "
        "'function': {'name': 'exec_command', 'arguments': {'cmd': 'pwd'}}}]}"
    )
    calls = parse_tool_calls(generated)

    assert calls == [
        {
            "call_id": calls[0]["call_id"],
            "name": "exec_command",
            "arguments": '{"cmd":"pwd","yield_time_ms":30000}',
        }
    ]


def test_tool_call_parser_drops_invalid_exec_sandbox_permission() -> None:
    generated = json.dumps(
        {
            "tool_calls": [
                {
                    "name": "exec_command",
                    "arguments": {
                        "cmd": "pwd",
                        "sandbox_permissions": "workspace-write",
                    },
                }
            ]
        }
    )
    calls = parse_tool_calls(generated)

    assert calls[0]["arguments"] == '{"cmd":"pwd","yield_time_ms":30000}'


def test_tool_call_parser_raises_short_exec_yield_time() -> None:
    generated = json.dumps(
        {
            "tool_calls": [
                {
                    "name": "exec_command",
                    "arguments": {
                        "cmd": "python3 slow.py",
                        "yield_time_ms": 100,
                    },
                }
            ]
        }
    )
    calls = parse_tool_calls(generated)

    assert calls[0]["arguments"] == '{"cmd":"python3 slow.py","yield_time_ms":30000}'
