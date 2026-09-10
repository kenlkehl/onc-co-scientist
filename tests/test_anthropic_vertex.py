from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.anthropic_vertex import (
    AnthropicVertexClient,
    AnthropicVertexConfig,
    AnthropicVertexProvider,
)
from onc_co_scientist.providers.base import ChatMessage
from onc_co_scientist.providers.registry import get_provider


def reply(parts=None, reason="end_turn", **extra):
    return {
        "model": "claude-opus-5",
        "stop_reason": reason,
        "content": parts if parts is not None else [{"type": "text", "text": "done"}],
        "usage": {"input_tokens": 12, "output_tokens": 27, "cache_read_input_tokens": 7},
        **extra,
    }


def fake_sdk(monkeypatch, responses):
    seen = []

    @contextmanager
    def stream(**body):
        seen.append(copy.deepcopy(body))
        raw = responses(body) if callable(responses) else responses.pop(0)
        yield SimpleNamespace(
            get_final_message=lambda: SimpleNamespace(model_dump=lambda **kw: raw)
        )

    monkeypatch.setattr(
        AnthropicVertexClient,
        "_build_client",
        lambda self: SimpleNamespace(
            project_id="adc-project", messages=SimpleNamespace(stream=stream)
        ),
    )
    return seen


def test_signed_parallel_tool_roundtrip(monkeypatch):
    signed = {"type": "thinking", "thinking": "", "signature": "opaque"}
    parts = [
        signed,
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        *[
            {"type": "tool_use", "id": f"c{i}", "name": "execute_python", "input": {"code": str(i)}}
            for i in range(2)
        ],
    ]
    seen = fake_sdk(monkeypatch, [reply(parts, "tool_use"), reply()])
    c = AnthropicVertexClient(AnthropicVertexConfig(reasoning_effort="medium"))
    history = [{"role": "system", "content": "science"}, {"role": "user", "content": "go"}]
    result = c.complete(model="claude-opus-5", messages=history, temperature=0)
    assert result["usage"] == {"prompt_tokens": 19, "completion_tokens": 27}
    history += [
        result["choices"][0]["message"],
        *[{"role": "tool", "tool_call_id": f"c{i}", "content": str(i)} for i in range(2)],
    ]
    c.complete(model="claude-opus-5", messages=history)
    assert seen[1]["messages"][-2]["content"] == parts
    assert [p["tool_use_id"] for p in seen[1]["messages"][-1]["content"]] == ["c0", "c1"]
    assert seen[0]["system"] == "science"
    assert "temperature" not in seen[0]
    assert seen[0]["extra_body"] == {"output_config": {"effort": "medium"}}


def test_registry_options_and_system_roles(monkeypatch):
    seen = fake_sdk(monkeypatch, [reply()])
    p = get_provider(
        {
            "kind": "anthropic_vertex",
            "model_id": "claude-opus-5",
            "reasoning_effort": "max",
            "timeout_s": 1800,
            "region": "global",
            "project_id": "explicit",
        }
    )
    r = p.chat([ChatMessage("system", "second"), ChatMessage("user", "test")], system="first")
    assert r.text == "done"
    assert seen[0]["system"] == "first\n\nsecond"
    assert p._client.config.timeout_s == 1800
    assert p._client.project_id == "explicit"
    assert p._client.base_url.startswith("https://aiplatform.googleapis.com/")
    with pytest.raises(TypeError):
        get_provider({"kind": "anthropic_vertex", "service_tier": "default"})


@pytest.mark.parametrize(
    "raw,match",
    [
        (reply(reason="max_tokens"), "truncated"),
        (reply(reason="refusal"), "refusal"),
        (reply(usage={}), "token usage"),
        (reply(model="other-model"), "different model"),
        (reply([]), "no text"),
    ],
)
def test_failures_retain_usage_for_matrix_and_raise_for_chat(monkeypatch, raw, match):
    fake_sdk(monkeypatch, [raw, raw])
    p = AnthropicVertexProvider(AnthropicVertexConfig(model_id="claude-opus-5"))
    response = p.chat_for_retry([ChatMessage("user", "test")])
    assert match in response.raw["adapter_error"]
    assert response.raw["anthropic_response"]["usage"] == raw["usage"]
    with pytest.raises(RuntimeError, match=match):
        p.chat([ChatMessage("user", "test")])


def test_sdk_auth_timeout_and_retries(monkeypatch):
    import anthropic

    seen = []
    monkeypatch.setattr(anthropic, "AnthropicVertex", lambda **kw: seen.append(kw) or object())
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project")
    monkeypatch.setenv("CLOUD_ML_REGION", "global")
    c = AnthropicVertexClient(AnthropicVertexConfig(timeout_s=300, max_retries=1))
    assert seen == [
        {"project_id": "env-project", "region": "global", "timeout": 300, "max_retries": 1}
    ]
    with pytest.raises(ValueError, match="service tiers"):
        c.complete(model="claude-opus-5", messages=[], service_tier="default")
    with pytest.raises(ValueError, match="effort"):
        AnthropicVertexConfig(reasoning_effort="ultra")


def test_structured_runner_native_submission(monkeypatch, tmp_path):
    from onc_co_scientist.harness.structured_runner import StructuredRunner

    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": "d", "max_iterations": 1}))
    record = {"index": 1, "proposed_hypotheses": [], "analyses": []}
    seen = fake_sdk(
        monkeypatch,
        [
            reply(
                [
                    {"type": "thinking", "thinking": "", "signature": "signed"},
                    {
                        "type": "tool_use",
                        "id": "submit",
                        "name": "submit_iteration",
                        "input": {"iteration": record},
                    },
                ],
                "tool_use",
            ),
            reply(),
        ],
    )
    runner = StructuredRunner(tmp_path, provider="anthropic-vertex", model="claude-opus-5")
    # Test controller integration independently of the host's namespace support.
    monkeypatch.setattr(
        runner,
        "_sandbox",
        lambda: SimpleNamespace(
            verify=lambda: None,
            collect_summary=lambda: None,
        ),
    )
    transcript = runner.run()
    assert len(transcript.iterations) == 1
    assert runner._tokens_used == 54
    assert seen[0]["tools"][1]["name"] == "submit_iteration"
    assert seen[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    logs = [json.loads(line) for line in runner.log_path.read_text().splitlines()]
    assert logs[0]["response"]["anthropic_response"]["content"][0]["signature"] == "signed"
    assert (
        json.loads((tmp_path / "runtime_metadata.json").read_text())["provider"]
        == "anthropic-vertex"
    )


def test_structured_runner_truncation_never_executes_partial_tools(monkeypatch, tmp_path):
    from onc_co_scientist.harness.structured_runner import StructuredRunner

    (tmp_path / "metadata.json").write_text(json.dumps({"dataset_id": "d", "max_iterations": 1}))
    fake_sdk(monkeypatch, [reply(reason="max_tokens")])
    runner = StructuredRunner(tmp_path, provider="anthropic-vertex", model="claude-opus-5")
    monkeypatch.setattr(runner, "_sandbox", lambda: SimpleNamespace(verify=lambda: None))
    with pytest.raises(RuntimeError, match="truncated"):
        runner.run()
    assert runner._tokens_used == 27
    assert runner.log_path.exists()
    assert not (tmp_path / "transcript.json").exists()


def test_expected_surprising_matrix_uses_real_vertex_adapter(monkeypatch, tmp_path):
    import yaml

    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import run_experiment
    from tests.test_expected_surprising_experiment import config
    from tests.test_expected_surprising_workflow import ScriptedScientist

    _, pairs, raw, path = config(tmp_path)
    scientist = ScriptedScientist(pairs[0])

    def respond(body):
        last = body["messages"][-1]["content"][-1]["text"]
        answer = scientist.chat([ChatMessage("user", last)])
        return reply([{"type": "text", "text": answer.text}])

    seen = fake_sdk(monkeypatch, respond)
    raw["models"] = [
        {
            "id": "claude",
            "adapter": "provider",
            "model_id": "claude-opus-5",
            "reasoning_effort": "medium",
            "provider_config": {
                "kind": "anthropic_vertex",
                "model_id": "claude-opus-5",
                "reasoning_effort": "medium",
                "region": "global",
                "project_id": "test",
            },
        }
    ]
    path.write_text(yaml.safe_dump(raw))
    spec = load_experiment_spec(path)
    result = run_experiment(spec)
    assert result["n_completed"] == 6
    assert result["n_failed"] == 0
    assert all("temperature" not in body for body in seen)
    count = len(seen)
    resumed = run_experiment(spec, resume=True)
    assert resumed["n_resumed"] == 6
    assert len(seen) == count


def test_prepare_named_masked_expected_surprising_grids(tmp_path, monkeypatch):
    from pathlib import Path

    from onc_co_scientist.harness.experiment import load_experiment_spec
    from scripts.expected_surprising.prepare_claude_grid import prepare_grid
    from tests.test_expected_surprising_experiment import config

    config(tmp_path)
    monkeypatch.setattr(AnthropicVertexClient, "_build_client", lambda self: pytest.fail("network"))
    out = tmp_path / "claude-grid"
    grids = prepare_grid(tmp_path / "packages", out, project_id="test-project")
    assert [g["runs"] for g in grids] == [6, 6]
    specs = [load_experiment_spec(Path(g["config"])) for g in grids]
    assert all(s.models[0].provider_config["model_id"] == "claude-opus-5" for s in specs)
    assert all(s.models[0].provider_config["project_id"] == "test-project" for s in specs)
    assert all(len(s.tasks) == 2 for s in specs)
    assert {t.semantic_condition for t in specs[0].tasks} == {"expected", "surprising"}
    assert specs[0].schedule_seed == specs[1].schedule_seed
    with pytest.raises(FileExistsError):
        prepare_grid(tmp_path / "packages", out, project_id="test-project")
    invalid = tmp_path / "bad"
    with pytest.raises(ValueError, match="No pairs"):
        prepare_grid(tmp_path / "missing", invalid, project_id="test-project")
    assert not invalid.exists()


def test_named_masked_preparation_freezes_claude_and_tool_instructions(tmp_path):
    import sys
    from pathlib import Path

    from experiments.aim1_recovery.prepare import prepare
    from experiments.aim1_recovery.run_batch import validate_launch

    plan = prepare(
        Path(__file__).resolve().parents[1],
        tmp_path / "aim1",
        Path(sys.executable),
        tasks=("nsclc",),
        clinical_repeats=1,
        depmap_repeats=0,
        backend="anthropic-vertex",
        model="claude-opus-5",
        service_tier=None,
    )
    assert len(plan["jobs"]) == 2
    for job in plan["jobs"]:
        text = (Path(job["workspace"]) / "agent_instructions.md").read_text()
        assert "call submit_iteration" in text
        assert "Use execute_python" in text
        assert "structured_runner submit" not in text
        assert "structured_runner finalize" not in text
    args = SimpleNamespace(
        backend="anthropic-vertex",
        model="claude-opus-5",
        reasoning_effort="medium",
        service_tier=None,
    )
    validate_launch(plan, args)
    args.model = "different"
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_launch(plan, args)
