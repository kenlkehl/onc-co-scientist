from __future__ import annotations

import json

from onc_co_scientist.caa_server import (
    CAA_MODEL_ALIASES,
    build_generation_messages,
    build_responses_payload,
    models_payload,
    parse_tool_calls,
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
    assert messages[1]["role"] == "tool"
    assert "tool result" in messages[1]["content"]


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
