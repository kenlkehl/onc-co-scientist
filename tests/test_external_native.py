"""Native exchange semantics and compatibility with the existing scientific controller."""

import copy
import json

import pytest

from onc_co_scientist.expected_surprising.generation import sample
from onc_co_scientist.expected_surprising.prompting import translate
from onc_co_scientist.expected_surprising.schemas import ValidationPolicy
from onc_co_scientist.expected_surprising.scoring import evidence_status
from onc_co_scientist.expected_surprising.workflow import WorkflowController, finalize_workflow
from onc_co_scientist.external.config import ExternalSpec
from onc_co_scientist.external.exchanges import ExchangeGateway, NativeController
from onc_co_scientist.external.transport import model_request, usage_summary
from tests.test_expected_surprising import pair


@pytest.fixture
def gateway():
    spec = pair()
    controller = NativeController(
        spec,
        "expected",
        sample(spec, "expected", seed=123),
        ValidationPolicy.default(spec.profile, 6),
        "replicate-001",
    )
    return ExchangeGateway(controller, 6)


def exchange(g, action, payload=None):
    return g.exchange(
        {
            "request_id": str(len(g.state["receipts"]) + 1),
            "round": g.state["round"],
            "action": action,
            "payload": payload or {},
        }
    )


def proposal(g, index=0):
    h = g.controller.spec.discoveries[index].hypothesis
    return {
        **h.model_dump(exclude={"id"}),
        "anticipated_direction": h.direction,
        "initial_status": "unresolved",
    }


def assess_pending(g):
    refs = g.public_state()["claims"]
    required = {r["claim"] for r in g.pending()}
    judgments = []
    for row in refs:
        if row["ref"] not in required:
            continue
        # Scripted fixture uses evaluator objects; the native scientist never sees them.
        h = g.controller.state["hypotheses"][row["ref"]]
        available = [e for e in g.controller.state["events"] if e["hypothesis_id"] == h.id]
        from onc_co_scientist.expected_surprising.schemas import EvidenceResult

        decision = (
            evidence_status(EvidenceResult.model_validate(available[-1]["result"]))
            if available
            else "unresolved"
        )
        judgments.append(
            {"claim": row["ref"], "status": decision or "unresolved", "investigation": "active"}
        )
    if judgments:
        exchange(g, "assess", {"assessments": judgments})


def test_native_rounds_registration_analysis_validation_and_deadlines(gateway):
    g = gateway
    exchange(g, "register", {"proposals": [proposal(g)]})
    exchange(g, "analyze", {"run_analyses": ["H1"]})
    assert g.pending()
    with pytest.raises(ValueError, match="Assess outstanding"):
        exchange(g, "prepare_close")
    assess_pending(g)
    exchange(g, "validate", {"validate": "H1"})
    for _ in range(6):
        assess_pending(g)
        exchange(g, "prepare_close")
        assess_pending(g)
        exchange(g, "close")
    assert g.public_state()["finished"]
    events = [e for e in g.controller.state["events"] if e["source"] == "voluntary"]
    assert events[0]["immediate"] and events[0]["delayed"]
    assert not g.public_state()["claims"][0]["initial_expectation"]["pre_evidence"]


def test_idempotency_and_transactional_rollback(gateway):
    request = {
        "request_id": "one",
        "round": 1,
        "action": "register",
        "payload": {"proposals": [proposal(gateway)]},
    }
    first = gateway.exchange(request)
    assert gateway.exchange(request) == first
    with pytest.raises(ValueError, match="reused"):
        gateway.exchange({**request, "action": "state"})
    before = copy.deepcopy(gateway.controller.state)
    with pytest.raises(ValueError):
        exchange(gateway, "analyze", {"run_analyses": ["bad"]})
    assert gateway.state["analyses"] == 0
    assert gateway.controller.state == before


def test_cumulative_round_budget_and_no_cross_round_reuse(gateway):
    exchange(gateway, "register", {"proposals": [proposal(gateway)]})
    for _ in range(12):
        exchange(gateway, "analyze", {"run_analyses": ["H1"]})
    with pytest.raises(ValueError, match="12 canonical"):
        exchange(gateway, "analyze", {"run_analyses": ["H1"]})
    with pytest.raises(ValueError, match="Wrong reporting round"):
        gateway.exchange({"request_id": "future", "round": 2, "action": "state"})


def test_scheduled_release_and_sealed_round(gateway):
    g = gateway
    exchange(g, "register", {"proposals": [proposal(g)]})
    exchange(g, "analyze", {"run_analyses": ["H1"]})
    assess_pending(g)
    exchange(g, "prepare_close")
    assert not any(e["source"] == "automatic" for e in g.controller.state["events"])
    with pytest.raises(ValueError, match="sealed"):
        exchange(g, "register", {"proposals": [proposal(g, 1)]})
    exchange(g, "close")
    exchange(g, "prepare_close")
    assert any(e["source"] == "automatic" for e in g.controller.state["events"])
    with pytest.raises(ValueError, match="Explicitly assess"):
        exchange(g, "close")


def test_public_projection_excludes_private_scoring(gateway):
    exchange(gateway, "register", {"proposals": [proposal(gateway)]})
    result = exchange(gateway, "analyze", {"run_analyses": ["H1"]})
    text = json.dumps(result)
    for secret in (
        "delta",
        "discoveries",
        "expected_direction",
        "eligible_pool",
        "selection_strata",
    ):
        assert f'"{secret}"' not in text


def test_settings_and_missing_usage_are_explicit(tmp_path):
    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out")
    body = model_request(
        spec,
        {"messages": [], "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert body["max_tokens"] == 125000
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    usage = usage_summary(
        [
            {"response": {"usage": {"prompt_tokens": 5, "completion_tokens": 10}}},
            {"status": "failed"},
        ]
    )
    assert usage["input_tokens"] is None and usage["known_output_tokens"] == 10
    assert usage["unknown_usage_requests"] == 1 and usage["cost_usd"] is None


def test_score_parity_with_staged_controller(gateway, tmp_path):
    g = gateway
    c = WorkflowController(
        g.controller.spec,
        "expected",
        g.controller.frame,
        g.controller.service.policy,
        "replicate-001",
    )

    def apply(stage, payload, iteration):
        record, _, _ = translate(c, stage, iteration, payload)
        return c.apply(record)

    payload = {"proposals": [proposal(g)]}
    apply("explore", payload, 1)
    exchange(g, "register", payload)
    apply("analyze", {"run_analyses": ["H1"]}, 1)
    exchange(g, "analyze", {"run_analyses": ["H1"]})
    assessment = {
        "assessments": [{"claim": "H1", "status": "unresolved", "investigation": "active"}]
    }
    for i in range(1, 7):
        if i > 1:
            apply("explore", {}, i)
            apply("analyze", {}, i)
        c.service.select(i, c.state["tests"])
        apply("appraise", assessment, i)
        exchange(g, "assess", assessment)
        exchange(g, "prepare_close")
        apply("synthesize", assessment, i)
        exchange(g, "close", assessment)
    public = tmp_path / "public"
    public.mkdir()
    g.controller.frame.to_parquet(public / "dataset.parquet")
    reports = []
    for name, controller in (("staged", c), ("native", g.controller)):
        out = tmp_path / name
        out.mkdir()
        reports.append(
            finalize_workflow(
                controller, public, out, model_id="test", run_id="test", iterations=6, versions={}
            )
        )
    assert reports[0]["scores"] == reports[1]["scores"]
    assert reports[0]["responsiveness"]["B"] == reports[1]["responsiveness"]["B"]


def test_proxy_accounts_failed_and_helper_requests(tmp_path, gateway, monkeypatch):
    from onc_co_scientist.external import transport

    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out", max_requests=2)
    calls = []

    def endpoint(url, data=None, **kwargs):
        calls.append((url, data))
        if url.endswith("/tokenize"):
            return {"count": 100}
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "<solution>OK</solution>",
                        "reasoning": "Reasoning is separate.",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }

    monkeypatch.setattr(transport, "http_json", endpoint)
    broker = transport.RunBroker(spec, gateway, tmp_path / "run", "test")
    broker.complete({"messages": [{"role": "user", "content": "helper"}], "model": "cloud-model"})
    assert calls[-1][1]["model"] == spec.model
    usage = transport.usage_summary(broker.requests)
    assert usage["output_tokens"] == 30 and usage["reasoning_tokens"] == 20

    def failure(*args, **kwargs):
        raise TimeoutError("test transport timeout")

    monkeypatch.setattr(transport, "http_json", failure)
    with pytest.raises(TimeoutError):
        broker.complete({"messages": []})
    assert transport.usage_summary(broker.requests)["output_tokens"] is None
    with pytest.raises(ValueError, match="request_budget"):
        broker.complete({"messages": []})
    assert len(list((tmp_path / "run" / "llm").glob("*.json"))) == 2


def test_proxy_context_and_truncated_code_fail_closed(tmp_path, gateway, monkeypatch):
    from onc_co_scientist.external import transport

    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out")
    broker = transport.RunBroker(spec, gateway, tmp_path / "run", "test")
    monkeypatch.setattr(transport, "http_json", lambda *a, **k: {"count": 262144})
    with pytest.raises(ValueError, match="exceeds configured context"):
        broker.complete({"messages": []})
    assert broker.fatal == "context_exhausted"

    def endpoint(url, *args, **kwargs):
        if url.endswith("/tokenize"):
            return {"count": 10}
        return {
            "choices": [{"finish_reason": "length", "message": {"content": "<execute>broken"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 125000},
        }

    monkeypatch.setattr(transport, "http_json", endpoint)
    with pytest.raises(ValueError, match="truncated"):
        broker.complete({"messages": []})
    assert broker.requests[-1]["response"]["usage"]["completion_tokens"] == 125000


def test_native_replay_has_identical_public_state(gateway):
    g = gateway
    exchange(g, "register", {"proposals": [proposal(g)]})
    exchange(g, "analyze", {"run_analyses": ["H1"]})
    assess_pending(g)
    exchange(g, "prepare_close")
    exchange(g, "close")
    c = NativeController(
        g.controller.spec,
        "expected",
        g.controller.frame,
        g.controller.service.policy,
        "replicate-001",
    )
    replay = ExchangeGateway(c, 6)
    for receipt in g.state["receipts"].values():
        assert replay.exchange(receipt["request"]) == receipt["response"]
    assert replay.public_state() == g.public_state()


def test_sandbox_mounts_exclude_repository_and_private_assets(tmp_path):
    from pathlib import Path

    from onc_co_scientist.external.sandbox import sandbox_command

    spec = ExternalSpec(input_root=tmp_path / "input", output_root=tmp_path / "out")
    command = sandbox_command(spec, tmp_path / "public", tmp_path / "scratch", tmp_path / "config")
    assert "--clearenv" in command and "--unshare-pid" in command
    assert str(spec.input_root) not in command
    assert str(Path(__file__).parents[1]) not in command
    assert "/proc" in command and "--ro-bind" in command
    assert "--ro-bind / /" not in " ".join(command)


def test_namespace_checkpoint_preserves_values_and_closed_handles(tmp_path):
    cloudpickle = pytest.importorskip("cloudpickle")
    import numpy as np
    import pandas as pd

    from onc_co_scientist.external.biomni_worker import dump_namespace

    with (tmp_path / "analysis.txt").open("w") as handle:
        handle.write("retained result")
        with pytest.raises(TypeError, match="Live file"):
            dump_namespace({"handle": handle})
    namespace = {
        "handle": handle,
        "array": np.array([2, 4, 6]),
        "frame": pd.DataFrame({"x": [1, 2]}),
        "nested": {"closed": handle},
    }
    restored = cloudpickle.loads(dump_namespace(namespace))
    assert restored["handle"].closed
    assert restored["handle"] is restored["nested"]["closed"]
    assert restored["handle"].name == str(tmp_path / "analysis.txt")
    with pytest.raises(ValueError, match="closed"):
        restored["handle"].read()
    assert restored["array"].tolist() == [2, 4, 6]
    pd.testing.assert_frame_equal(restored["frame"], namespace["frame"])
    assert (tmp_path / "analysis.txt").read_text() == "retained result"


def test_comparison_preserves_missing_metrics_and_rejects_changed_data(tmp_path):
    from onc_co_scientist.external.comparison import compare_reports
    from onc_co_scientist.external.transport import write_json

    base = {
        "versions": {"scoring": "profile-3.0.0"},
        "run_id": "native",
        "pair_id": "pair",
        "version": "expected",
        "dataset_sha256": "original",
        "model": "qwen",
        "harness": "external-native",
        "resource_policy": "full",
        "clock": "reporting_round",
        "protocol_errors": [],
        "scores": {"discovery": {"exact": {"R": 0.5, "Q": None}}, "D": 0, "E": 10, "B": None},
    }
    native = tmp_path / "native.json"
    baseline = tmp_path / "baseline.json"
    write_json(native, base)
    write_json(
        baseline,
        {
            **base,
            "run_id": "baseline",
            "harness": "persistent",
            "resource_policy": "benchmark-controller",
            "clock": "workflow_iteration",
        },
    )
    result = compare_reports([native, baseline], tmp_path / "comparison")
    assert len(result["conditions"]) == 2
    assert all(c["P"]["mean"] is None and c["P"]["missing_n"] == 1 for c in result["conditions"])
    assert all(c["B"]["mean"] is None for c in result["conditions"])
    write_json(baseline, {**base, "dataset_sha256": "changed"})
    with pytest.raises(ValueError, match="Dataset bytes differ"):
        compare_reports([native, baseline], tmp_path / "bad")


def test_config_preserves_virtual_environment_interpreter(tmp_path):
    target = tmp_path / "system-python"
    target.touch()
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(target)
    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out", python=executable)
    assert spec.python == executable
    assert spec.python != target


def test_usage_handles_null_reasoning_details():
    result = usage_summary(
        [
            {
                "response": {
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "completion_tokens_details": None,
                    }
                }
            }
        ]
    )
    assert result["output_tokens"] == 2
    assert result["reasoning_tokens"] is None


def test_resume_never_replays_uncheckpointed_native_work(gateway, tmp_path):
    from types import SimpleNamespace

    from onc_co_scientist.external.runner import run_cell
    from onc_co_scientist.external.transport import write_json

    public = tmp_path / "public"
    public.mkdir()
    gateway.controller.frame.to_parquet(public / "dataset.parquet")
    write_json(public / "task.json", {})
    write_json(public / "data_dictionary.json", {})
    private = tmp_path / "private"
    private.mkdir()
    write_json(private / "workflow.json", {})
    write_json(private / "assignment.json", {"expected": {"sha256": "source"}})
    task = SimpleNamespace(
        id="opaque",
        semantic_condition="expected",
        public_workspace=public,
        private_evaluation_path=private / "pair.json",
    )
    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out", rounds=6)
    run = spec.output_root / "runs" / "opaque__biomni_native__r001"
    write_json(run / "run.json", {"status": "running", "fingerprint": "frozen"})
    write_json(run / "llm" / "000001.json", {"status": "inflight"})

    class MustNotRun:
        def run(self, *args):
            raise AssertionError("Replayed arbitrary native work")

    with pytest.raises(ValueError, match="No native checkpoint"):
        run_cell(
            spec,
            task,
            gateway.controller.spec,
            gateway.controller.service.policy,
            1,
            "frozen",
            resume=True,
            runner=MustNotRun(),
        )


@pytest.mark.parametrize("masked", [False, True])
def test_native_run_writes_scores_token_curve_and_verifies_resume(
    gateway, tmp_path, monkeypatch, masked
):
    from types import SimpleNamespace

    from onc_co_scientist.external.runner import run_cell, write_reports
    from onc_co_scientist.external.transport import RunBroker, write_json

    monkeypatch.setattr(RunBroker, "start", lambda self: "http://unused")
    monkeypatch.setattr(RunBroker, "stop", lambda self: None)
    public = tmp_path / "public"
    public.mkdir()
    gateway.controller.frame.to_parquet(public / "dataset.parquet")
    write_json(public / "task.json", {})
    write_json(public / "data_dictionary.json", {})
    private = tmp_path / "private"
    private.mkdir()
    write_json(private / "workflow.json", {})
    write_json(private / "assignment.json", {"expected": {"sha256": "source"}})
    task = SimpleNamespace(
        id="opaque",
        semantic_condition="expected",
        public_workspace=public,
        private_evaluation_path=private / "pair.json",
    )
    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out", rounds=6)

    if masked:
        from onc_co_scientist.expected_surprising.masking import MASKING_VERSION, SemanticMask
        from onc_co_scientist.synthetic.anonymize import build_column_mapping, build_value_mapping

        frame = gateway.controller.frame
        mapping = SemanticMask(
            {
                "version": MASKING_VERSION,
                "columns": build_column_mapping(
                    list(frame), [o.name for o in gateway.controller.spec.outcomes]
                ),
                "values": build_value_mapping(frame, seed=7),
            }
        )
        mapping.frame(frame).to_parquet(public / "dataset.parquet")
        write_json(private / "workflow.json", {"masking": mapping.payload})

    class ScriptedNative:
        def run(self, spec, public, scratch, config_path, broker):
            g = broker.gateway
            if masked:
                from onc_co_scientist.expected_surprising.masking import MaskedValidationService

                assert isinstance(g.controller.service, MaskedValidationService)
                assert all(c in g.controller.frame for c in mapping.columns.values())
                prompt = json.loads(config_path.read_text())["prompt"]
                assert "opaque labels" in prompt
                assert "Research signatures and markers" not in prompt
            counter = 0

            def send(action, payload=None):
                nonlocal counter
                counter += 1
                return broker.exchange(
                    {
                        "request_id": str(counter),
                        "round": g.state["round"],
                        "action": action,
                        "payload": payload or {},
                    }
                )

            send("register", {"proposals": [proposal(g)]})
            send("analyze", {"run_analyses": ["H1"]})
            assessments = {
                "assessments": [{"claim": "H1", "status": "unresolved", "investigation": "active"}]
            }
            for _ in range(6):
                send("assess", assessments)
                send("prepare_close")
                send("close", assessments)
            return 0

    result = run_cell(
        spec,
        task,
        gateway.controller.spec,
        gateway.controller.service.policy,
        1,
        "frozen",
        runner=ScriptedNative(),
    )
    assert result["status"] == "completed"
    root = spec.output_root / "runs" / result["run_id"]
    report = json.loads((root / "report.json").read_text())
    assert len(report["exploration_by_tokens"]) == 6
    assert report["clock"] == "reporting_round"
    assert report["dataset_view"] == ("masked" if masked else "named")
    assert report["scores"]["discovery"]["exact"]["Q"] is None
    write_reports(spec.output_root, [result])
    assert (spec.output_root / "summary.csv").exists()
    assert (
        run_cell(
            spec,
            task,
            gateway.controller.spec,
            gateway.controller.service.policy,
            1,
            "frozen",
            resume=True,
        )
        == result
    )
    with (root / "exchanges.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        run_cell(
            spec,
            task,
            gateway.controller.spec,
            gateway.controller.service.policy,
            1,
            "frozen",
            resume=True,
        )


def test_adaptive_cap_preserves_history_and_audits_reasoning(tmp_path, gateway, monkeypatch):
    from onc_co_scientist.external import transport

    spec = ExternalSpec(
        input_root=tmp_path, output_root=tmp_path / "out", completion_policy="adaptive"
    )
    calls = []
    prompt_count = 200000

    def endpoint(url, data=None, **kwargs):
        calls.append((url, copy.deepcopy(data)))
        if url.endswith("/tokenize"):
            return {"count": prompt_count}
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                        "reasoning": "private trace",
                        "reasoning_content": "another trace",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": 100,
                "completion_tokens_details": {"reasoning_tokens": 90},
            },
        }

    monkeypatch.setattr(transport, "http_json", endpoint)
    broker = transport.RunBroker(spec, gateway, tmp_path / "run", "test")
    original = {
        "messages": [
            {"role": "user", "content": "research question"},
            {"role": "assistant", "content": "earlier final", "reasoning": "old trace"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "old thought"},
                    {"type": "text", "text": "tool result interpretation"},
                ],
            },
        ]
    }
    result = broker.complete(original)
    sent = calls[-1][1]
    assert sent["max_tokens"] == 61888
    assert sent["messages"] == calls[0][1]["messages"]
    assert len(sent["messages"]) == 3
    assert sent["messages"][1] == {"role": "assistant", "content": "earlier final"}
    assert sent["messages"][2]["content"] == [
        {"type": "text", "text": "tool result interpretation"}
    ]
    assert original["messages"][1]["reasoning"] == "old trace"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "final answer"}
    audit = json.loads((tmp_path / "run/llm/000001.json").read_text())
    assert audit["response"]["choices"][0]["message"]["reasoning"] == "private trace"
    assert audit["effective_max_tokens"] == 61888
    assert audit["requested_max_tokens"] == 125000
    assert audit["completion_policy"] == "adaptive"
    assert result["usage"]["completion_tokens"] == 100
    prompt_count = 1000
    broker.complete(original)
    assert calls[-1][1]["max_tokens"] == 125000
    prompt_count = spec.context_length - spec.context_guard_tokens - spec.min_completion_tokens
    broker.complete(original)
    assert calls[-1][1]["max_tokens"] == spec.min_completion_tokens
    prompt_count += 1
    with pytest.raises(ValueError, match="exceeds configured context"):
        broker.complete(original)
    assert calls[-1][0].endswith("/tokenize")
    assert broker.fatal == "context_exhausted"


def test_reasoning_in_plain_content_fails_closed(tmp_path):
    from onc_co_scientist.external.transport import without_reasoning

    with pytest.raises(ValueError, match="Reasoning must be separated"):
        without_reasoning({"role": "assistant", "content": "<think>trace</think>answer"})
    with pytest.raises(ValueError, match="Reasoning must be separated"):
        without_reasoning({"content": [{"type": "text", "text": "<think>trace</think>"}]})


def test_adaptive_campaign_requires_full_length_gate(tmp_path, monkeypatch):
    from onc_co_scientist.external import runner
    from onc_co_scientist.external.transport import fingerprint, write_json

    spec = ExternalSpec(
        input_root=tmp_path, output_root=tmp_path / "out", completion_policy="adaptive"
    )
    monkeypatch.setattr(runner, "validate_inputs", lambda spec: ([], None, None))
    monkeypatch.setattr(runner, "preflight", lambda spec: None)
    monkeypatch.setattr(runner, "provenance", lambda spec, tasks: {})
    gate = {"status": "passed", "fingerprint": fingerprint({}), "rounds": 6}
    write_json(spec.output_root / "smoke_gate.json", gate)
    with pytest.raises(ValueError, match="full-length excluded pilot"):
        runner.run_experiment(spec)
    assert not (spec.output_root / "provenance.json").exists()
    gate["rounds"] = 25
    write_json(spec.output_root / "smoke_gate.json", gate)
    assert runner.run_experiment(spec) == []


def test_combined_launch_stops_if_pilot_fails(tmp_path, monkeypatch):
    import typer

    from onc_co_scientist.expected_surprising.cli import external_run
    from onc_co_scientist.external import config, runner, smoke

    spec = ExternalSpec(input_root=tmp_path, output_root=tmp_path / "out")
    monkeypatch.setattr(config, "load_spec", lambda path: spec)
    calls = []
    monkeypatch.setattr(
        smoke, "run_smoke", lambda spec, **kwargs: calls.append(kwargs) or {"status": "failed"}
    )
    monkeypatch.setattr(
        runner,
        "run_experiment",
        lambda *args, **kwargs: pytest.fail("Failed pilot launched formal runs"),
    )
    with pytest.raises(typer.Exit):
        external_run(tmp_path / "config.yaml", full_length_smoke=True)
    assert calls == [{"full_length": True}]
