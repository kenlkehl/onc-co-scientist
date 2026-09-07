"""Integration tests for scientific authority, conversation isolation and paired inference."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from onc_co_scientist.expected_surprising import experiment as integration
from onc_co_scientist.expected_surprising.coordination import StageCoordinator, response_usage
from onc_co_scientist.expected_surprising.experiment_report import _estimate, summarize_matrix
from onc_co_scientist.expected_surprising.generation import write_pair
from onc_co_scientist.expected_surprising.packaging import repackage
from onc_co_scientist.expected_surprising.summary import paired_summary
from onc_co_scientist.expected_surprising.workflow import WorkflowInfrastructureError
from onc_co_scientist.harness.experiment import (
    ResourceBudget,
    WorkflowSpec,
    default_stages,
    load_experiment_spec,
)
from onc_co_scientist.harness.orchestrator import build_run_plans, run_experiment
from onc_co_scientist.providers.base import ChatMessage, ChatResponse
from tests.test_expected_surprising import pair
from tests.test_expected_surprising_workflow import ScriptedScientist


def config(tmp_path, *, profiles=("nsclc_clinical",), iterations=6, rounds=1):
    private = []
    for profile in profiles:
        spec = pair(profile)
        private.append(spec)
        write_pair(spec, tmp_path / "original")
    repackage(tmp_path / "original", tmp_path / "packages")
    raw = {
        "experiment_id": "aim2-test",
        "output_root": "output",
        "expected_surprising": {
            "root": "packages",
            "max_tokens_per_call": 8192,
            "persistent_history_chars": None,
        },
        "iteration_policy": {"iterations": iterations},
        "models": [
            {
                "id": "fixture",
                "model_id": "scripted-workflow",
                "adapter": "provider",
                "provider_config": {"kind": "vllm_openai", "model_id": "scripted-workflow"},
            }
        ],
        "workflows": [
            {
                "id": mode,
                "mode": mode,
                **(
                    {"agents_per_stage": 2, "deliberation_rounds": rounds}
                    if mode == "deliberative"
                    else {}
                ),
            }
            for mode in ("persistent", "sequential", "deliberative")
        ],
        "budget": {"max_agent_calls": 4 * iterations * (2 * rounds + 1) + 20},
    }
    path = tmp_path / "matrix.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_experiment_spec(path), private, raw, path


class Scientist(ScriptedScientist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []

    def chat(self, messages, **kwargs):
        if kwargs.get("system"):
            messages = [ChatMessage("system", kwargs["system"]), *messages]
        self.messages.append(messages)
        return super().chat([messages[-1]], **kwargs)


def providers(monkeypatch, pair_spec, cls=Scientist):
    made = []

    def factory(config):
        provider = cls(pair_spec)
        made.append(provider)
        return provider

    monkeypatch.setattr(integration, "get_provider", factory)
    return made


def reports(result):
    return [json.loads(Path(r["scientific_report"]).read_text()) for r in result["runs"]]


def test_all_workflows_share_science_with_distinct_memory_and_calls(tmp_path, monkeypatch):
    spec, pairs, _, config_path = config(tmp_path)
    made = providers(monkeypatch, pairs[0])
    result = run_experiment(spec)
    assert result["n_completed"] == 6
    assert result["realized_agent_calls"] == 2 * (24 + 24 + 72)
    scientific = reports(result)
    for version in ("expected", "surprising"):
        group = [r for r in scientific if r["version"] == version]
        for field in ("scores", "confirmation", "state", "validation"):
            assert all(r[field] == group[0][field] for r in group)
    assert {r["replicate_id"] for r in scientific} == {"replicate-001"}
    for run, provider in zip(result["runs"], made, strict=True):
        messages = provider.messages
        if run["workflow_mode"] == "persistent":
            assert len(messages[-1]) == 48
            assert messages[-1][2].role == "assistant"
        elif run["workflow_mode"] == "sequential":
            assert all(len(m) == 2 for m in messages)
            assert "claims" in messages[-1][-1].content
        else:
            assert "Participant drafts" not in messages[0][0].content
            assert "Participant drafts" not in messages[1][0].content
            assert messages[0][-1] == messages[1][-1]
            assert "Participant drafts" in messages[2][0].content
        text = "\n".join(m.content for call in messages for m in call)
        for secret in ("selected_stratum", "focal_id", "minimum_effect", "primary_recovery"):
            assert secret not in text
    summary = json.loads((spec.output_root / "expected_surprising_summary.json").read_text())
    assert len(summary["conditions"]) == 3
    assert all(c["difference_pp"] == 0 for c in summary["workflow_contrasts"])
    assert all(c["ci95_pp"] is None for c in summary["workflow_contrasts"])
    text = (spec.output_root / "expected_surprising_report.md").read_text()
    assert "precision/positive predictive value" in text
    assert "not a run cap" in text
    with pytest.raises(ValueError, match="one model profile and workflow"):
        paired_summary(scientific, bootstrap_replicates=2)
    from onc_co_scientist.cli import app

    regenerated = CliRunner().invoke(
        app,
        [
            "expected-surprising",
            "summarize-workflows",
            "--config",
            str(config_path),
            "--bootstrap-replicates",
            "2",
        ],
    )
    assert regenerated.exit_code == 0, regenerated.output
    assert "expected_surprising_report.md" in regenerated.output


@pytest.mark.parametrize("profile", ["nsclc_clinical", "nsclc_depmap"])
def test_full_25_iterations_for_both_modalities(tmp_path, monkeypatch, profile):
    spec, pairs, _, _ = config(tmp_path, profiles=(profile,), iterations=25)
    spec.expected_surprising.persistent_history_chars = 120000
    providers(monkeypatch, pairs[0])
    result = run_experiment(spec)
    assert result["n_completed"] == 6
    assert sorted(r["agent_calls"] for r in result["runs"]) == [100, 100, 100, 100, 300, 300]
    assert all(r["successful_stages"] == 100 for r in reports(result))


def test_only_chair_can_register_or_execute(tmp_path, monkeypatch):
    spec, pairs, _, _ = config(tmp_path)
    spec.workflows = [spec.workflows[-1]]

    class ProposingPeers(Scientist):
        def chat(self, messages, **kwargs):
            if "You are the chair" in kwargs.get("system", ""):
                return ChatResponse(
                    text='{"narrative":"No proposal selected."}', model_id=self.model_id
                )
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], ProposingPeers)
    result = run_experiment(spec)
    assert result["n_completed"] == 2
    for report in reports(result):
        assert report["state"]["registrations"] == []
        assert report["state"]["executions"] == []
        assert report["final_accepted_ids"] == []


def test_chair_repair_reuses_peers_and_rolls_back_invalid_action(tmp_path, monkeypatch):
    spec, pairs, _, _ = config(tmp_path)
    spec.workflows = [spec.workflows[-1]]

    class RepairChair(Scientist):
        def chat(self, messages, **kwargs):
            self.repair = "You are the chair" in kwargs.get("system", "")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], RepairChair)
    result = run_experiment(spec)
    assert result["n_completed"] == 2
    for report in reports(result):
        assert report["coordination"]["agent_calls"] == 73
        assert len(report["recovered_stages"]) == 1
        assert len(report["state"]["registrations"]) == len(pairs[0].discoveries)


def test_resume_replays_without_new_calls_and_rejects_changed_inputs(tmp_path, monkeypatch):
    spec, pairs, _, _ = config(tmp_path)
    spec.expected_surprising.persistent_history_chars = 120000
    spec.workflows = [spec.workflows[0]]
    plans = build_run_plans(spec)
    plan = plans[0]

    class Interrupted(Scientist):
        def chat(self, messages, **kwargs):
            if self.calls == 7:
                raise KeyboardInterrupt("simulate killed process")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], Interrupted)
    with pytest.raises(KeyboardInterrupt):
        integration.run_cell(spec, plan, spec.output_root, spec.fingerprint(), resume=False)
    made = providers(monkeypatch, pairs[0])
    result = integration.run_cell(spec, plan, spec.output_root, spec.fingerprint(), resume=True)
    assert made[0].calls == 17
    assert result["agent_calls"] == 24
    assert result["resumed"]
    assert integration.run_cell(spec, plan, spec.output_root, spec.fingerprint(), resume=True)[
        "resumed"
    ]
    assert len(made) == 1
    task = plan.task.public_workspace / "instructions.md"
    task.write_text(task.read_text() + "\nChanged instructions")
    with pytest.raises(RuntimeError, match="inputs or implementation changed"):
        integration.run_cell(spec, plan, spec.output_root, spec.fingerprint(), resume=True)


def test_failed_cells_remain_in_primary_denominator(tmp_path):
    spec, _, _, _ = config(tmp_path)
    plans = build_run_plans(spec)
    results = [{**p.public_dict(), "status": "failed"} for p in plans]
    summary = summarize_matrix(spec, plans, results, bootstrap_replicates=3)
    assert all(
        c["expected_recovery"] == c["surprising_recovery"] == 0 for c in summary["conditions"]
    )
    assert all(c["failed_runs"] == c["run_n"] == 2 for c in summary["conditions"])
    with pytest.raises(ValueError, match="Every planned run"):
        summarize_matrix(spec, plans, results[:-1])


def test_openai_null_metrics_and_missing_usage_are_safe():
    raw = {"metrics": None, "usage": {"prompt_tokens": 123, "completion_tokens": 45}}
    usage = response_usage(raw, 1.5)
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 45
    assert usage["infrastructure_attempts"] is None
    assert response_usage({"metrics": None, "usage": None}, 1)["input_tokens"] is None


def test_errors_consume_calls_and_corrupt_cache_stops_replay(tmp_path):
    class Unavailable:
        model_id = "fixture"
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            raise TimeoutError("Request timed out")

    provider = Unavailable()
    arguments = (
        provider,
        WorkflowSpec(id="persistent", mode="persistent"),
        default_stages(),
        SimpleNamespace(
            max_tokens_per_call=100, max_retries_per_stage=2, persistent_history_chars=None
        ),
        ResourceBudget(max_agent_calls=1),
        tmp_path / "calls",
    )
    coordinator = StageCoordinator(*arguments)
    with pytest.raises(ValueError, match="Request timed out"):
        coordinator.respond("prompt", iteration=1, stage="explore", attempt=1)
    with pytest.raises(ValueError, match="max_agent_calls"):
        coordinator.respond("repair", iteration=1, stage="explore", attempt=2)
    assert provider.calls == 1
    assert coordinator.audit()["provider_error_calls"] == 1
    assert coordinator.audit()["usage"]["input_tokens"] is None
    resumed = StageCoordinator(*arguments)
    with pytest.raises(ValueError, match="Request timed out"):
        resumed.respond("prompt", iteration=1, stage="explore", attempt=1)
    assert provider.calls == 1
    next((tmp_path / "calls").glob("*.json")).write_text('{"invalid":true}')
    with pytest.raises(WorkflowInfrastructureError, match="Invalid cached call"):
        StageCoordinator(*arguments).respond("prompt", iteration=1, stage="explore", attempt=1)
    assert provider.calls == 1


def test_exhausted_peer_is_not_silently_dropped(tmp_path):
    class BadDraft:
        model_id = "fixture"
        calls = 0

        def chat(self, *_args, **_kwargs):
            self.calls += 1
            return ChatResponse(text="not JSON", model_id=self.model_id)

    provider = BadDraft()
    coordinator = StageCoordinator(
        provider,
        WorkflowSpec(id="deliberative", mode="deliberative", agents_per_stage=2),
        default_stages(),
        SimpleNamespace(
            max_tokens_per_call=100, max_retries_per_stage=2, persistent_history_chars=None
        ),
        ResourceBudget(max_agent_calls=30),
        tmp_path / "calls",
    )
    for attempt in (1, 2, 3):
        with pytest.raises(ValueError, match="Peer 1 exhausted"):
            coordinator.respond(f"prompt {attempt}", iteration=1, stage="explore", attempt=attempt)
    assert provider.calls == 3
    assert len(coordinator.audit()["draft_errors"]) == 3
    assert coordinator.audit()["authoritative_candidates"] == 0
    assert coordinator.audit()["committed_stages"] == []


def test_workflow_contrast_sign_and_repeat_weighting(tmp_path, monkeypatch):
    spec, _, _, _ = config(tmp_path)
    spec.replicates = 2
    plans = build_run_plans(spec)
    results = []
    for p in plans:
        # Persistent: expected 100%, surprising 0%; sequential: both 50%;
        # deliberative: expected 50%, surprising 100%.
        accepted = {
            "persistent": p.task.semantic_condition == "expected",
            "sequential": p.replicate == 1,
            "deliberative": p.task.semantic_condition == "surprising" or p.replicate == 1,
        }[p.workflow.id]
        path = tmp_path / f"{p.run_id}.json"
        path.write_text(json.dumps({"confirmation": {"primary_recovery": int(accepted)}}))
        results.append({**p.public_dict(), "status": "completed", "scientific_report": str(path)})
    monkeypatch.setattr(
        "onc_co_scientist.expected_surprising.experiment_report.paired_summary", lambda *a, **k: {}
    )
    summary = summarize_matrix(spec, plans, results, bootstrap_replicates=5)
    conditions = {c["workflow_id"]: c for c in summary["conditions"]}
    assert conditions["sequential"]["expected_recovery"] == 0.5
    assert conditions["sequential"]["paired_effect"]["base_dataset_n"] == 1
    effects = {c["workflow_id"]: c["difference_pp"] for c in summary["workflow_contrasts"]}
    assert effects == {"sequential": 100, "deliberative": 150}
    assert all(c["ci95_pp"] is None for c in summary["workflow_contrasts"])


def test_intervals_resample_base_datasets_within_modality():
    pairs = [
        {"pair_id": str(i), "modality": "clinical" if i < 2 else "depmap", "difference_pp": 20}
        for i in range(4)
    ]
    result = _estimate(pairs, bootstrap_replicates=100, seed=1)
    assert result["base_dataset_n"] == 4
    assert result["difference_pp"] == 20
    assert result["ci95_pp"] == [20, 20]


def test_bounded_persistent_memory_preserves_full_scientific_state(tmp_path, monkeypatch):
    spec, pairs, _, _ = config(tmp_path)
    spec.expected_surprising.persistent_history_chars = 120000
    made = providers(monkeypatch, pairs[0])
    result = run_experiment(spec)
    assert result["n_completed"] == 6
    scientific = reports(result)
    for version in ("expected", "surprising"):
        group = [r for r in scientific if r["version"] == version]
        for field in ("scores", "confirmation", "state", "validation"):
            assert all(r[field] == group[0][field] for r in group)
    for run, provider in zip(result["runs"], made, strict=True):
        if run["workflow_mode"] == "persistent":
            report = next(r for r in scientific if r["run_id"] == run["run_id"])
            assert report["coordination"]["memory_trims"]
            assert all(
                sum(len(m.content) for m in call[1:-1]) <= 120000 for call in provider.messages
            )
            assert "complete research ledger" in provider.messages[-1][0].content
            assert len(provider.messages[-1]) > 2


def test_provider_initialization_failure_completes_matrix_with_denominators(tmp_path, monkeypatch):
    spec, _, _, _ = config(tmp_path)

    def unavailable(_):
        raise RuntimeError("Provider setup unavailable")

    monkeypatch.setattr(integration, "get_provider", unavailable)
    result = run_experiment(spec)
    assert result["n_failed"] == 6
    assert result["realized_agent_calls"] == 0
    summary = json.loads((spec.output_root / "expected_surprising_summary.json").read_text())
    assert all(c["failed_runs"] == 2 for c in summary["conditions"])


def test_two_rounds_preserve_independent_first_drafts(tmp_path, monkeypatch):
    spec, pairs, _, _ = config(tmp_path, rounds=2)
    spec.workflows = [spec.workflows[-1]]
    made = providers(monkeypatch, pairs[0])
    result = run_experiment(spec)
    assert all(r["agent_calls"] == 120 for r in result["runs"])
    calls = made[0].messages
    assert "Participant drafts" not in calls[0][0].content
    assert "Participant drafts" not in calls[1][0].content
    assert "Participant drafts" in calls[2][0].content
    assert len(calls[2]) == 4


def test_config_defaults_pair_hashes_and_rejected_unsupported_settings(tmp_path):
    spec, _, raw, path = config(tmp_path)
    raw.pop("iteration_policy")
    raw["budget"]["max_agent_calls"] = 900
    path.write_text(yaml.safe_dump(raw))
    assert load_experiment_spec(path).iteration_policy.iterations == 25
    for field, value in (("federated", True), ("safeguards", {"independent_rerun": True})):
        changed = copy.deepcopy(raw)
        changed["workflows"][0][field] = value
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match="separate protocols"):
            load_experiment_spec(path)
    path.write_text(yaml.safe_dump(raw))
    dataset = spec.tasks[0].public_workspace / "dataset.parquet"
    with dataset.open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_experiment_spec(path)
