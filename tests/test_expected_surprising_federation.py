"""Federation boundary, shared-budget and N=1 equivalence checks."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from onc_co_scientist.expected_surprising.federation_spec import (
    FederationCell,
    FederationGrid,
    SitePartition,
)
from onc_co_scientist.expected_surprising.generation import sample
from onc_co_scientist.expected_surprising.scoring import estimate
from onc_co_scientist.expected_surprising.site_statistics import (
    combine_statistics,
    partition_frame,
    public_site_context,
    sufficient_statistics,
)
from onc_co_scientist.harness.experiment import load_experiment_spec, required_agent_calls
from onc_co_scientist.harness.orchestrator import build_run_plans, run_experiment
from onc_co_scientist.providers.base import ChatMessage, ChatResponse
from tests.test_expected_surprising import pair
from tests.test_expected_surprising_experiment import config, providers, reports
from tests.test_expected_surprising_workflow import ScriptedScientist


def test_partitions_disjoint_reproducible_and_heterogeneous():
    frame = pd.DataFrame(
        {
            "patient_id": [f"PRIVATE-{i}" for i in range(120)],
            "age": np.arange(120),
            "x": np.arange(120) % 2,
        }
    )
    random = FederationCell(sites=3)
    a, audit = partition_frame(frame, random, "pair", "repeat")
    b, again = partition_frame(frame, random, "pair", "repeat")
    assert audit == again
    assert set.union(*(set(f.index) for f in a.values())) == set(frame.index)
    assert sum(len(f) for f in a.values()) == len(frame)
    assert all(len(f) == 40 for f in a.values())
    heterogeneous = FederationCell(
        sites=3, partition=SitePartition(id="age", mode="heterogeneous", by=["age"])
    )
    parts, _ = partition_frame(frame, heterogeneous, "pair", "repeat")
    assert parts["site_1"].age.max() < parts["site_2"].age.min()
    assert parts["site_2"].age.max() < parts["site_3"].age.min()
    text = json.dumps(public_site_context(parts["site_1"], {}))
    assert "PRIVATE-" not in text and "patient_id" not in text
    assert "min" not in json.loads(text)["variables"]["age"]
    assert (
        len(
            FederationGrid(
                site_counts=[1, 3], partitions=[SitePartition(), heterogeneous.partition]
            ).cells()
        )
        == 3
    )
    with pytest.raises(ValueError):
        SitePartition(mode="heterogeneous")
    with pytest.raises(ValueError):
        FederationGrid(site_counts=[0])


@pytest.mark.parametrize("sites", [2, 4])
@pytest.mark.parametrize("version", ["expected", "surprising"])
def test_sufficient_statistics_reproduce_pooled_contrast_and_suppress_small_cells(sites, version):
    spec = pair()
    frame = sample(spec, version, seed=123)
    h = spec.discoveries[0].hypothesis
    partitions, _ = partition_frame(frame, FederationCell(sites=sites), spec.pair_id, "repeat")
    stats = [sufficient_statistics(part, h) for part in partitions.values()]
    kwargs = dict(delta=spec.outcomes[0].delta, alpha=0.05, result_id="test")
    combined = combine_statistics(stats, h, **kwargs)
    direct = estimate(frame, h, **kwargs)
    assert combined.valid and direct.valid
    assert combined.estimate == pytest.approx(direct.estimate)
    assert combined.lower == pytest.approx(direct.lower)
    assert combined.upper == pytest.approx(direct.upper)
    small = sufficient_statistics(frame.iloc[:5], h)
    assert not small["valid"] and all(
        c["mean"] is None and c["variance"] is None for c in small["cells"]
    )
    assert not combine_statistics([stats[0], small], h, **kwargs).valid


class FederatedScientist(ScriptedScientist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(([m.content for m in messages], kwargs.get("system", "")))
        prompt = messages[-1].content
        start = prompt.index("Iteration ")
        marker = prompt.index('{"schema":', start)
        payload, _ = json.JSONDecoder().raw_decode(prompt[marker:])
        first = prompt[start : prompt.index("\n", start)]
        response = super().chat(
            [ChatMessage("user", first + "\n" + json.dumps(payload, separators=(",", ":")))]
        )
        form = json.loads(response.text)
        return ChatResponse(
            text=json.dumps(form),
            model_id=self.model_id,
            raw={"usage": {"prompt_tokens": 20, "completion_tokens": 11}},
        )


def test_grid_n1_equivalence_stage_handoffs_tokens_and_shared_budget(tmp_path, monkeypatch):
    spec, pairs, raw, path = config(tmp_path, iterations=6)
    raw["federation"] = {"site_counts": [1, 2, 4]}
    raw["budget"]["max_agent_calls"] = 400
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)
    assert len(build_run_plans(grid)) == 18
    assert (
        required_agent_calls(grid, grid.tasks[0], grid.workflows[2], FederationCell(sites=2)) == 216
    )
    made = providers(monkeypatch, pairs[0], FederatedScientist)
    result = run_experiment(grid)
    assert result["n_completed"] == 18, [
        (r["run_id"], r.get("error"), r.get("call_failures", [])[:1])
        for r in result["runs"]
        if r["status"] != "completed"
    ]
    scientific = reports(result)
    for report in scientific:
        n = report["site_count"]
        per_stage = 3 if report["workflow_mode"] == "deliberative" else 1
        expected_calls = 24 * (per_stage if n == 1 else (n + 1) * per_stage)
        audit = report["coordination"]
        assert audit["agent_calls"] == expected_calls
        assert audit["usage"]["output_tokens"] > 0
        assert audit["output_token_accounting"]["missing_calls"] == 0
        assert report["validation"]["voluntary_slots_used"] <= report["policy"]["voluntary_limit"]
        assert len(report["validation"]["cache"]) <= report["policy"]["voluntary_limit"] + len(
            report["policy"]["release_iterations"]
        )
        if n > 1:
            root = Path(
                next(
                    r["scientific_report"]
                    for r in result["runs"]
                    if r["run_id"] == report["run_id"]
                )
            ).parent
            assert len(list((root / "handoffs").glob("*.json"))) == 24 * n
            assert set(audit["scope_audits"]) == {"central", *(f"site_{i + 1}" for i in range(n))}
            assert len(audit["committed_stages"]) == 24
            assert all((root / p).exists() for p in audit["participant_artifacts"])
            assert report["successful_stages"] == 24
    # Every central/site prompt contains aggregates only, never identifiers or raw row exports.
    for provider in made:
        text = json.dumps(provider.messages)
        assert "patient_id" not in text and "membership_sha256" not in text
    summary = json.loads((grid.output_root / "expected_surprising_summary.json").read_text())
    assert len(summary["conditions"]) == 9
    assert len(summary["workflow_contrasts"]) == 6
    assert all(c["run_n"] == 2 and c["output_tokens"] > 0 for c in summary["conditions"])
    # Explicit N=1 follows the same prompts, science and seed as a grid with federation omitted.
    baseline_dir = tmp_path / "baseline"
    plain = grid.model_copy(deep=True, update={"federation": None, "output_root": baseline_dir})
    providers(monkeypatch, pairs[0], FederatedScientist)
    baseline = reports(run_experiment(plain))
    for old in baseline:
        new = next(r for r in scientific if r["run_id"] == old["run_id"])
        for field in ["scores", "confirmation", "state", "validation", "final_accepted_ids"]:
            assert new[field] == old[field]


def test_central_extra_queries_cannot_bypass_budget(tmp_path):
    from types import SimpleNamespace

    from onc_co_scientist.expected_surprising.federation import FederatedCoordinator
    from onc_co_scientist.expected_surprising.schemas import ValidationPolicy
    from onc_co_scientist.expected_surprising.workflow import WorkflowController
    from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages

    spec = pair()
    policy = ValidationPolicy.default(spec.profile, 6)
    controller = WorkflowController(
        spec, "expected", sample(spec, "expected", seed=123), policy, "test"
    )
    federation = FederatedCoordinator(
        FederatedScientist(spec),
        WorkflowSpec(id="sequential", mode="sequential"),
        default_stages(),
        SimpleNamespace(
            max_tokens_per_call=100, max_retries_per_stage=0, persistent_history_chars=None
        ),
        ResourceBudget(max_agent_calls=2),
        tmp_path,
        FederationCell(sites=2),
    )
    federation.bind(controller, {"instructions": "test", "outcomes": [], "variables": {}})
    assert controller.frame.empty
    federation.active = "i001-analyze"
    federation.prepared[federation.active] = {"analyses": []}
    with pytest.raises(ValueError, match="not sent to the sites"):
        controller.analysis_estimator(
            controller.frame, spec.discoveries[0].hypothesis, delta=0.1, alpha=0.05, result_id="x"
        )


def test_constant_variance_is_invalid_like_single_site():
    h = pair().discoveries[0].hypothesis
    item = {
        "valid": True,
        "cells": [
            {"n": 30, "mean": 2.0, "variance": 0.0, "sign": 1},
            {"n": 30, "mean": 2.0, "variance": 0.0, "sign": -1},
        ],
    }
    assert not combine_statistics([item, item], h, delta=0.1, alpha=0.05, result_id="x").valid


def test_participant_call_cap_is_shared_by_sites_and_center(tmp_path):
    from types import SimpleNamespace

    from onc_co_scientist.expected_surprising.federation import FederatedCoordinator
    from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages

    class Provider:
        model_id = "test"
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            return ChatResponse(text="{}", model_id="test", raw={"usage": {"completion_tokens": 5}})

    provider = Provider()
    fed = FederatedCoordinator(
        provider,
        WorkflowSpec(id="sequential", mode="sequential"),
        default_stages(),
        SimpleNamespace(
            max_tokens_per_call=100, max_retries_per_stage=0, persistent_history_chars=None
        ),
        ResourceBudget(max_agent_calls=2),
        tmp_path,
        FederationCell(sites=2),
    )

    def call(coordinator, slot):
        return coordinator._call(
            slot,
            [ChatMessage("system", "test"), ChatMessage("user", "test")],
            session=slot,
            authoritative=True,
            iteration=1,
            stage="explore",
            kind="linear",
        )

    call(fed.central, "center")
    call(fed.sites["site_1"], "site1")
    with pytest.raises(ValueError, match="max_agent_calls"):
        call(fed.sites["site_2"], "site2")
    assert provider.calls == 2
    assert fed.shared_budget["calls"] == 2
    assert fed.sites["site_1"].records[0]["request"]["authoritative_candidate"] is False


def test_resume_mid_stage_and_central_repair_reuse_site_handoffs(tmp_path, monkeypatch):
    from onc_co_scientist.expected_surprising import experiment as integration

    _, pairs, raw, path = config(tmp_path)
    raw["federation"] = {"site_counts": [2]}
    raw["workflows"] = [raw["workflows"][0]]  # Persistent memories must replay identically.
    raw["budget"]["max_agent_calls"] = 120
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)
    plan = build_run_plans(grid)[0]

    class Repairing(FederatedScientist):
        def chat(self, messages, **kwargs):
            response = super().chat(messages, **kwargs)
            prompt = messages[-1].content
            if (
                "Site consultation is complete." in prompt
                and "Iteration 1/6, stage analyze, attempt 1." in prompt
            ):
                value = json.loads(response.text)
                value["run_analyses"] = ["H999"]  # Invalid reference forces central repair.
                response = replace(response, text=json.dumps(value))
            return response

    class Interrupted(Repairing):
        def chat(self, messages, **kwargs):
            if len(self.messages) == 5:
                raise KeyboardInterrupt("simulate interruption before central repair")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], Interrupted)
    with pytest.raises(KeyboardInterrupt):
        integration.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=False)
    made = providers(monkeypatch, pairs[0], Repairing)
    result = integration.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=True)
    assert result["status"] == "completed", result.get("error")
    assert result["agent_calls"] == 73
    assert len(made[0].messages) == 68
    audit = reports({"runs": [result]})[0]["coordination"]
    assert audit["scope_audits"]["site_1"]["agent_calls"] == 24
    assert audit["scope_audits"]["site_2"]["agent_calls"] == 24
    root = Path(result["scientific_report"]).parent
    for site in ("central", "site_1", "site_2"):
        assert (root / "calls" / "requests" / site / "i001-analyze-agent-a1.json").exists()


@pytest.mark.parametrize("mode", ["persistent", "sequential", "deliberative"])
def test_results_are_citable_when_first_shown_and_roles_have_study_context(
    tmp_path, monkeypatch, mode
):
    """Regression: a rejected analyze form cannot see or guess its future R IDs."""
    from onc_co_scientist.expected_surprising import experiment as integration

    _, pairs, raw, path = config(tmp_path)
    raw["federation"] = {"site_counts": [2]}
    raw["workflows"] = [w for w in raw["workflows"] if w["mode"] == mode]
    raw["budget"]["max_agent_calls"] = 280
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)

    class LedgerReader(FederatedScientist):
        def chat(self, messages, **kwargs):
            response = super().chat(messages, **kwargs)
            prompt = messages[-1].content
            if (
                "Site consultation is complete." in prompt
                and "Iteration 1/6, stage analyze, attempt 1." in prompt
            ):
                form = json.loads(response.text)
                form["run_analyses"] = ["H999"]  # Force rollback and central repair.
                response = replace(response, text=json.dumps(form))
            return response

    made = providers(monkeypatch, pairs[0], LedgerReader)
    result = integration.run_cell(
        grid, build_run_plans(grid)[0], grid.output_root, grid.fingerprint(), resume=False
    )
    assert result["status"] == "completed", result.get("call_failures")
    saw = set()
    for messages, system in made[0].messages:
        prompt = messages[-1]
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"schema":') :])
        assert "Research goal:" in system
        assert pairs[0].outcomes[0].name in system
        assert "explore (propose comparisons)" in system
        assert "scientist for the" not in system
        # Task instructions are visible before the schema/large claim ledger.
        assert prompt.index("Study instructions:") < prompt.index('{"schema":')
        if "federation" in payload:
            assert system.startswith("You are a scientist on one federated site team.")
            assert "only the central orchestrator" in system.lower()
            saw.add("site")
            continue
        assert system.startswith("You are the federated research orchestrator.")
        assert "Test the candidate hypotheses" not in system
        assert "You are the federated orchestrator" not in prompt
        marker = "committed local discovery summaries:\n"
        handoffs, _ = json.JSONDecoder().raw_decode(prompt.split(marker)[1])
        if "Iteration 1/6, stage analyze," in prompt:
            assert all(not h["evidence"] for h in payload["claims"])
            assert all(not h["analysis_results"] for h in handoffs)
            assert "Combined summaries computed" not in prompt
            assert "assess them in appraise" in prompt
            saw.add("analyze_retry" if "attempt 2." in prompt else "analyze")
        if "Iteration 1/6, stage appraise," in prompt:
            cards = {h["ref"]: h for h in payload["claims"]}
            assert all(h["analysis_results"] for h in handoffs)
            for handoff in handoffs:
                for claim, local in handoff["analysis_results"].items():
                    assert "id" not in local
                    assert set(local["shared_evidence"]) <= {
                        e["ref"] for e in cards[claim]["evidence"]
                    }
                    assert cards[claim]["evidence"]
                    assert all(e["ref"].startswith("R") for e in cards[claim]["evidence"])
            # Assessments using omitted evidence resolve to these displayed R IDs.
            assert payload["assessments_due"]
            saw.add("appraise")
    assert saw == {"site", "analyze", "analyze_retry", "appraise"}
    report = reports({"runs": [result]})[0]
    assert report["coordination"]["scope_audits"]["site_1"]["agent_calls"] == (
        72 if mode == "deliberative" else 24
    )


@pytest.mark.parametrize("mode", ["persistent", "sequential", "deliberative"])
@pytest.mark.parametrize("exhaust", [False, True])
def test_site_reference_repairs_and_fallback_preserve_shared_science(tmp_path, mode, exhaust):
    from types import SimpleNamespace

    from onc_co_scientist.expected_surprising.federation import FederatedCoordinator
    from onc_co_scientist.expected_surprising.prompting import build_prompt, translate
    from onc_co_scientist.expected_surprising.schemas import ValidationPolicy
    from onc_co_scientist.expected_surprising.workflow import WorkflowController
    from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages

    spec = pair()
    controller = WorkflowController(
        spec,
        "expected",
        sample(spec, "expected", seed=123),
        ValidationPolicy.default(spec.profile, 6),
        "repeat",
    )

    class BadSite(FederatedScientist):
        def chat(self, messages, **kwargs):
            response = super().chat(messages, **kwargs)
            prompt = messages[-1].content
            payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"schema":') :])
            if payload.get("federation", {}).get("site") == "site_1":
                form = json.loads(response.text)
                if "stage analyze," in prompt:
                    # Recommendations do not bind the center or reduce site data coverage.
                    form["run_analyses"] = []
                if "stage appraise," in prompt and (
                    exhaust or "Repair the site form:" not in prompt
                ):
                    form["assessments"][0]["evidence"] = ["R999"]
                response = replace(response, text=json.dumps(form))
            return response

    provider = BadSite(spec)
    federation = FederatedCoordinator(
        provider,
        WorkflowSpec(id=mode, mode=mode, agents_per_stage=2 if mode == "deliberative" else 1),
        default_stages(),
        SimpleNamespace(
            max_tokens_per_call=1000,
            max_retries_per_stage=2,
            persistent_history_chars=None,
            peer_failure_policy="chair_with_available",
        ),
        ResourceBudget(max_agent_calls=100),
        tmp_path,
        FederationCell(sites=2),
    )
    context = {"instructions": "Investigate clinical outcomes", "variables": {}, "outcomes": []}
    federation.bind(controller, context)
    for stage in ("explore", "analyze", "appraise"):
        prompt = build_prompt(controller, context, 1, 6, stage, 1, {}, None)
        response = federation.respond(prompt, iteration=1, stage=stage, attempt=1)
        record, _, _ = translate(controller, stage, 1, json.loads(response.text))
        controller.apply(record)
        federation.commit()
    assert controller.frame.empty
    assert len(controller.state["hypotheses"]) == len(spec.discoveries)
    assert len([rid for rid in controller.state["results"] if rid.startswith("analysis-")]) == len(
        spec.discoveries
    )
    assert len(controller.service.state["voluntary_keys"]) == 1
    assert (
        federation.site_discovery["site_1"]
        == federation.prepared["i001-analyze"]["local_results"]["site_1"]
    )
    assert federation.site_discovery["site_1"]  # Included despite recommending no analyses.
    assert len(federation.handoff_errors) == (3 if exhaust else 1)
    handoff = federation.prepared["i001-appraise"]["handoffs"]["site_1"]
    assert handoff["unavailable"] is exhaust
    assert "R999" not in json.dumps(handoff)
    # Only successfully selected site forms enter persistent memory.
    assert len(federation.sites["site_1"].committed_stages) == (2 if exhaust else 3)
    for coordinator in (federation.central, *federation.sites.values()):
        assert coordinator.workflow.mode == mode
        assert bool(coordinator.history) is (mode == "persistent")


def test_handoff_storage_failure_is_infrastructure_not_a_scientific_retry(tmp_path, monkeypatch):
    from onc_co_scientist.expected_surprising import federation as module
    from onc_co_scientist.expected_surprising.workflow import WorkflowInfrastructureError

    fed = object.__new__(module.FederatedCoordinator)
    fed.root = tmp_path

    def fail(*args):
        raise OSError("unavailable storage")

    monkeypatch.setattr(module, "atomic_write_json", fail)
    with pytest.raises(WorkflowInfrastructureError, match="Handoff persistence failed"):
        fed._write_handoff("i001-analyze", "site_1", {})
