from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.gemini_vertex import (
    GeminiVertexClient,
    GeminiVertexConfig,
    GeminiVertexProvider,
)


def client(**kwargs):
    return GeminiVertexClient(
        GeminiVertexConfig(project_id="test-project", **kwargs),
        credentials=SimpleNamespace(valid=True),
    )


def response(parts, reason="STOP"):
    return {
        "modelVersion": "gemini-3.8-flash",
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": reason}],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 20,
        },
    }


def test_native_history_roundtrip_and_accounting(monkeypatch):
    from scripts.vllm_cli_json_adapter import _native_tool_call

    c = client()
    signed = {
        "functionCall": {"name": "python", "args": {"code": "print(5)"}},
        "thoughtSignature": "opaque-signature",
    }
    seen = []

    def generate(model, body):
        seen.append(body)
        return response([signed] if len(seen) == 1 else [{"text": "done"}])

    monkeypatch.setattr(c, "generate", generate)
    messages = [
        {"role": "system", "content": "scientist"},
        {"role": "user", "content": "calculate"},
    ]
    result = c.create(model="gemini-3.8-flash", messages=messages)
    assert result.usage.completion_tokens == 25
    call, assistant = _native_tool_call(result)
    assert assistant["gemini_parts"] == [signed]
    messages += [assistant, {"role": "tool", "tool_call_id": call["id"], "content": "5"}]
    c.complete(model="gemini-3.8-flash", messages=messages)
    assert seen[1]["systemInstruction"] == {"parts": [{"text": "scientist"}]}
    assert seen[1]["contents"][-2] == {"role": "model", "parts": [signed]}
    assert seen[1]["contents"][-1]["parts"][0]["functionResponse"] == {
        "name": "python",
        "response": {"result": "5"},
    }


def test_json_schema_and_thinking_mapping(monkeypatch):
    c = client(reasoning_effort="high")
    seen = []
    monkeypatch.setattr(
        c,
        "generate",
        lambda m, b: (
            seen.append(b)
            or response([{"text": "internal", "thought": True}, {"text": '{"ok":true}'}])
        ),
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    r = c.complete(
        model="gemini-3.8-flash",
        messages=[{"role": "user", "content": "test"}],
        max_completion_tokens=1234,
        response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    )
    assert r["choices"][0]["message"]["content"] == '{"ok":true}'
    assert seen[0]["generationConfig"] == {
        "maxOutputTokens": 1234,
        "thinkingConfig": {"thinkingLevel": "HIGH"},
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"service_tier": "default"},
        {"reasoning_effort": "minimal"},
        {"extra_body": {"repetition_penalty": 1.1}},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    ],
)
def test_unsupported_controls_fail_before_network(options):
    with pytest.raises(ValueError):
        client().complete(model="gemini-3.8-flash", messages=[], **options)


@pytest.mark.parametrize(
    "raw",
    [
        {"promptFeedback": {"blockReason": "SAFETY"}},
        response([], "SAFETY"),
        {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "no usage"}]}}]},
    ],
)
def test_blocked_and_missing_usage_are_failures(monkeypatch, raw):
    c = client()
    monkeypatch.setattr(c, "generate", lambda *a: raw)
    with pytest.raises(RuntimeError):
        c.complete(model="gemini-3.8-flash", messages=[])


def test_registry_and_truncation(monkeypatch):
    from onc_co_scientist.providers.base import ChatMessage
    from onc_co_scientist.providers.registry import get_provider

    monkeypatch.setattr(GeminiVertexClient, "__init__", lambda self, config: None)
    monkeypatch.setattr(GeminiVertexClient, "generate", lambda *a: response([], "MAX_TOKENS"))
    provider = get_provider({"kind": "gemini_vertex", "model_id": "gemini-3.8-flash"})
    assert isinstance(provider, GeminiVertexProvider)
    provider._client.config = GeminiVertexConfig()
    with pytest.raises(RuntimeError, match="truncated"):
        provider.chat([ChatMessage(role="user", content="test")])


def test_endpoint_resolution_and_timeout_clone(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    c = client(location="global")
    clone = c.with_options(timeout=3)
    assert clone.credentials is c.credentials
    assert clone.project_id == "test-project"
    assert clone.location == "global"
    assert clone.config.timeout_s == 3
    assert clone.base_url.startswith("https://aiplatform.googleapis.com/")


def test_gemini_does_not_use_vllm_failover(monkeypatch):
    from scripts.vllm_cli_json_adapter import _api_target

    monkeypatch.setenv("OCS_VLLM_BASE_URL_OVERRIDE", "http://other")
    monkeypatch.setenv("OCS_VLLM_MODEL_ID_OVERRIDE", "other-model")
    assert _api_target(
        SimpleNamespace(provider="gemini-vertex", base_url="https://google"), "gemini-3.8-flash"
    ) == ("https://google", "gemini-3.8-flash")


def test_structured_runner_gemini_submissions(monkeypatch, tmp_path):
    from onc_co_scientist.harness.structured_runner import StructuredRunner

    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": "test", "max_iterations": 1}))
    monkeypatch.setattr(
        GeminiVertexClient,
        "__init__",
        lambda self, config: setattr(self, "base_url", "https://google"),
    )
    seen = []
    record = {"index": 1, "proposed_hypotheses": [], "analyses": []}

    def complete(self, **body):
        seen.append(body)
        message = {"role": "assistant", "content": "done"}
        if len(seen) == 1:
            message["tool_calls"] = [
                {
                    "id": "a",
                    "function": {
                        "name": "submit_iteration",
                        "arguments": json.dumps({"iteration": record}),
                    },
                }
            ]
        return {
            "model": "gemini-3.8-flash",
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 27},
        }

    monkeypatch.setattr(GeminiVertexClient, "complete", complete)
    runner = StructuredRunner(tmp_path, provider="gemini-vertex", model="gemini-3.8-flash")
    result = runner.run()
    assert len(result.iterations) == 1
    assert runner._tokens_used == 54
    assert seen[1]["messages"][-2]["role"] == "tool"
    assert (
        json.loads((tmp_path / "runtime_metadata.json").read_text())["provider"] == "gemini-vertex"
    )


def test_gemini_judge_cli_selection(monkeypatch):
    from onc_co_scientist.cli import JudgeBackend, _build_judge
    from onc_co_scientist.scoring.judge import GeminiVertexJudge

    monkeypatch.setattr(GeminiVertexClient, "__init__", lambda *a: None)
    judge = _build_judge(
        JudgeBackend.gemini_vertex,
        judge_cli="auto",
        judge_model=None,
        batch_size=2,
        cache_dir=None,
        stub_config_path=None,
    )
    assert isinstance(judge, GeminiVertexJudge)
    assert judge.model_id == "gemini-3.8-flash"


def test_credentials_used_only_in_transport_headers(monkeypatch):
    import urllib.request
    from contextlib import contextmanager
    from io import StringIO

    secret = "test-only-credential-sentinel"
    monkeypatch.setenv("GEMINI_API_KEY", "unused-test-api-key")
    credentials = SimpleNamespace(
        valid=True, apply=lambda headers: headers.update({"Authorization": "Bearer " + secret})
    )
    c = GeminiVertexClient(GeminiVertexConfig(project_id="test-project"), credentials=credentials)
    seen = []

    @contextmanager
    def urlopen(request, timeout):
        assert request.headers["Authorization"] == "Bearer " + secret
        seen.append(request.data.decode())
        yield StringIO(json.dumps(response([{"text": "ok"}])))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    result = c.complete(model="gemini-3.8-flash", messages=[{"role": "user", "content": "ok"}])
    serialized = json.dumps(result) + "".join(seen)
    assert secret not in serialized
    assert "unused-test-api-key" not in serialized
    assert not hasattr(c.config, "api_key")


def test_legacy_agent_uses_fresh_public_workspace(monkeypatch, tmp_path):
    import sys

    from onc_co_scientist.harness import gemini_agent

    source = tmp_path / "bundle"
    source.mkdir()
    (source / "agent_instructions.md").write_text(
        "**Dataset:** `real-id`\n**Maximum iterations (N):** 3\n"
    )
    (source / "dataset.parquet").write_text("public fixture")
    (source / "private_key.json").write_text("do not copy")
    (source / "runs").mkdir()
    (source / "runs/old-result.json").write_text("do not copy")
    monkeypatch.chdir(source)
    monkeypatch.setenv("OCS_RUN_DIR", str(source / "runs/run_001"))
    monkeypatch.setattr(sys, "argv", ["ocs-gemini-agent", "custom task"])
    seen = []

    class Runner:
        def __init__(self, workspace, **kwargs):
            self.workspace = workspace
            seen.append(kwargs)

        def run(self):
            assert not (self.workspace / "private_key.json").exists()
            assert not (self.workspace / "runs").exists()
            assert json.loads((self.workspace / "metadata.json").read_text()) == {
                "dataset_id": "real-id",
                "max_iterations": 3,
            }
            assert "custom task" in (self.workspace / "agent_instructions.md").read_text()
            (self.workspace / "transcript.json").write_text("{}")

    monkeypatch.setattr(gemini_agent, "StructuredRunner", Runner)
    gemini_agent.main()
    assert seen[0]["provider"] == "gemini-vertex"
    assert seen[0]["project_id"] is None
    assert (source / "transcript.json").read_text() == "{}"
