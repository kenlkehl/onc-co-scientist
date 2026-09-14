"""Native Azure wire format, quota retries, and accounting without a CLI agent."""

import json
import urllib.error
from contextlib import contextmanager

import pytest

from onc_co_scientist.external import azure
from onc_co_scientist.external.config import ExternalSpec
from onc_co_scientist.external.transport import RunBroker, model_request, usage_summary


@pytest.fixture
def spec(tmp_path):
    return ExternalSpec(
        input_root=tmp_path,
        output_root=tmp_path / "out",
        llm_backend="azure",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        base_url="https://example.openai.azure.com/openai/v1",
    )


def raw_response(status="completed"):
    return {
        "id": "response-1",
        "model": "gpt-5.6-luna",
        "status": status,
        "output": [
            {"type": "reasoning", "summary": [{"text": "private reasoning"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "print(2 + 2)"}]},
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 32,
            "total_tokens": 132,
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }


def test_native_request_preserves_dialogue_and_requested_model(spec):
    body = model_request(
        spec,
        {
            "messages": [
                {"role": "system", "content": "native planning"},
                {"role": "assistant", "content": "prior code", "reasoning": "hidden"},
                {"role": "user", "content": "observation"},
            ],
            "stop": ["</execute>"],
        },
    )
    wire = azure.responses_request(body)
    assert wire["model"] == "gpt-5.6-luna"
    assert wire["reasoning"] == {"effort": "medium"}
    assert wire["max_output_tokens"] == 125000
    assert [x["content"] for x in wire["input"]] == ["native planning", "prior code", "observation"]
    assert wire["store"] is False and wire["truncation"] == "disabled"
    assert not {"tools", "temperature", "stop", "chat_template_kwargs"} & wire.keys()


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
def test_structured_protocol_only_wraps_native_steps(spec, api):
    spec = spec.model_copy(update={"azure_native_protocol": "structured", "azure_api": api})
    messages = [
        {"role": "system", "content": "native instructions"},
        {"role": "user", "content": "scientific task"},
    ]
    mapper = azure.responses_request if api == "responses" else azure.chat_request
    field = "text" if api == "responses" else "response_format"
    helper = mapper(model_request(spec, {"messages": messages}))
    assert field not in helper
    retriever = mapper(
        model_request(spec, {"messages": messages[1:], "stop": ["</execute>", "</solution>"]})
    )
    assert field not in retriever
    body = model_request(spec, {"messages": messages, "stop": ["</execute>", "</solution>"]})
    wire = mapper(body)
    assert field in wire
    assert wire.get("input", wire.get("messages"))[:-1] == messages
    assert "_biomni_native_step" not in wire
    assert wire["model"] == spec.model


@pytest.mark.parametrize(
    "action,content", [("execute", "print(2 + 2)"), ("solution", "Observed 4")]
)
def test_structured_response_decodes_only_visible_content(action, content):
    raw = raw_response()
    raw["output"][1]["content"][0]["text"] = json.dumps({"action": action, "content": content})
    result = azure.native_step_response(azure.chat_response(raw))
    assert result["choices"][0]["message"]["content"] == f"<{action}>\n{content}\n</{action}>"
    assert "private reasoning" not in json.dumps(result)
    assert result["usage"]["completion_tokens"] == 32


@pytest.mark.parametrize(
    "text",
    [
        '{"action":"execute","content":"print(1)","action":"solution"}',
        '{"action":"execute","content":"print(1)","extra":true}',
        '{"action":"execute","content":"</execute><solution>done"}',
        '{"action":"execute","content":"```python\\nprint(1)\\n```"}',
        '{"action":"execute","content":""}',
        '{"action":"observe","content":"imagined"}',
        '{"action":"execute","content":null}',
        '["execute","print(1)"]',
    ],
)
def test_invalid_structured_output_never_reaches_native_execution(
    spec, tmp_path, monkeypatch, text
):
    raw = raw_response()
    raw["output"][1]["content"][0]["text"] = text
    fake_provider(monkeypatch, [raw])
    broker = RunBroker(
        spec.model_copy(update={"azure_native_protocol": "structured"}),
        None,
        tmp_path / "run",
        "secret",
    )
    with pytest.raises(ValueError):
        broker.complete(
            {
                "messages": [{"role": "system", "content": "native"}],
                "stop": ["</execute>", "</solution>"],
            }
        )
    assert "response" not in broker.requests[0]
    assert usage_summary(broker.requests)["output_tokens"] == 32


def test_structured_truncation_preserves_usage_without_decoding(spec, tmp_path, monkeypatch):
    raw = raw_response("incomplete")
    raw["output"][1]["content"][0]["text"] = '{"action":"execute","content":"print('
    fake_provider(monkeypatch, [raw])
    broker = RunBroker(
        spec.model_copy(update={"azure_native_protocol": "structured"}),
        None,
        tmp_path / "run",
        "secret",
    )
    with pytest.raises(ValueError, match="Length-truncated"):
        broker.complete(
            {
                "messages": [{"role": "system", "content": "native"}],
                "stop": ["</execute>", "</solution>"],
            }
        )
    assert broker.fatal == "completion_length_exhausted"
    assert usage_summary(broker.requests)["output_tokens"] == 32


def test_accounting_preserves_reasoning_tokens_but_not_reasoning_text():
    result = azure.chat_response(raw_response())
    assert result["choices"][0]["message"]["content"] == "print(2 + 2)"
    assert "private reasoning" not in json.dumps(result)
    usage = usage_summary([{"response": result}])
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 32
    assert usage["reasoning_tokens"] == 20
    missing = raw_response()
    missing.pop("usage")
    assert not usage_summary([{"response": azure.chat_response(missing)}])["usage_complete"]


def fake_provider(monkeypatch, responses):
    state = {"tokens": 0, "admissions": 0, "throttles": 0, "successes": 0}

    class Provider:
        class config:
            azure_auth_retries = 3

        def environment(self):
            state["tokens"] += 1
            return {"OCS_AZURE_ACCESS_TOKEN": "secret-token-do-not-save"}

        def auth_retry_delay(self, retry):
            return 0

        @contextmanager
        def request_slot(self, *args):
            state["admissions"] += 1
            yield self

        def throttled(self, *args):
            state["throttles"] += 1

        def succeeded(self):
            state["successes"] += 1

    iterator = iter(responses)

    def http(*args, **kwargs):
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(azure, "azure_provider", lambda *args: Provider())
    monkeypatch.setattr(azure, "http_json", http)
    return state


@pytest.mark.parametrize("code", [401, 429])
def test_rejected_calls_refresh_token_and_do_not_consume_another_scientific_request(
    spec, tmp_path, monkeypatch, code
):
    state = fake_provider(
        monkeypatch,
        [
            urllib.error.HTTPError("https://example", code, "rejected", {"retry-after": "1"}, None),
            raw_response(),
        ],
    )
    broker = RunBroker(spec, None, tmp_path / "run", "broker-secret")
    result = broker.complete({"messages": [{"role": "user", "content": "test"}]})
    assert result["usage"]["prompt_tokens"] == 100
    assert len(broker.requests) == 1 and len(broker.requests[0]["transport_attempts"]) == 2
    assert state["tokens"] == state["admissions"] == 2
    audit = (tmp_path / "run/llm/000001.json").read_text()
    assert "secret-token-do-not-save" not in audit
    assert json.loads(audit)["raw_response"] == raw_response()


def test_truncated_output_is_not_executable_and_usage_is_retained(spec, tmp_path, monkeypatch):
    fake_provider(monkeypatch, [raw_response("incomplete")])
    broker = RunBroker(spec, None, tmp_path / "run", "broker-secret")
    with pytest.raises(ValueError, match="Length-truncated"):
        broker.complete({"messages": []})
    assert broker.fatal == "completion_length_exhausted"
    assert usage_summary(broker.requests)["output_tokens"] == 32


def test_ambiguous_network_failure_is_not_replayed(spec, tmp_path, monkeypatch):
    state = fake_provider(monkeypatch, [TimeoutError("response lost")])
    broker = RunBroker(spec, None, tmp_path / "run", "broker-secret")
    with pytest.raises(TimeoutError):
        broker.complete({"messages": []})
    assert state["admissions"] == 1
    assert not usage_summary(broker.requests)["usage_complete"]


def test_azure_configuration_rejects_other_credentials_and_adaptive_budget(spec):
    for updates in ({"api_key_env": "OPENAI_API_KEY"}, {"completion_policy": "adaptive"}):
        with pytest.raises(ValueError):
            ExternalSpec.model_validate({**spec.model_dump(), **updates})


def test_chat_completions_preserves_native_commands_and_usage(spec, tmp_path, monkeypatch):
    spec = spec.model_copy(update={"azure_api": "chat_completions"})
    raw = {
        "model": "gpt-5.6-luna",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "<execute>print(4)</execute>",
                    "reasoning_content": "hidden",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "completion_tokens_details": {"reasoning_tokens": 22},
        },
    }
    fake_provider(monkeypatch, [raw])
    broker = RunBroker(spec, None, tmp_path / "run", "broker-secret")
    result = broker.complete({"messages": [{"role": "user", "content": "unchanged task"}]})
    assert result["choices"][0]["message"] == {
        "role": "assistant",
        "content": "<execute>print(4)</execute>",
    }
    assert usage_summary(broker.requests)["reasoning_tokens"] == 22
    record = broker.requests[0]
    assert record["prompt_token_count"] == 123
    assert record["raw_response"] == raw
    assert record["wire_request"] == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "unchanged task"}],
        "reasoning_effort": "medium",
        "max_completion_tokens": 125000,
        "stream": False,
    }


def test_chat_completions_truncation_stays_fatal(spec, tmp_path, monkeypatch):
    spec = spec.model_copy(update={"azure_api": "chat_completions"})
    fake_provider(
        monkeypatch,
        [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "<execute>partial"},
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 10},
            }
        ],
    )
    broker = RunBroker(spec, None, tmp_path / "run", "broker-secret")
    with pytest.raises(ValueError, match="Length-truncated"):
        broker.complete({"messages": []})
    assert broker.fatal == "completion_length_exhausted"
    assert usage_summary(broker.requests)["output_tokens"] == 10
