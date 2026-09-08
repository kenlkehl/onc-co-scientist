"""Federation boundary, shared-budget and N=1 equivalence checks."""

import json
import re
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


def test_sufficient_statistics_reproduce_pooled_contrast_and_suppress_small_cells():
    spec = pair()
    frame = sample(spec, "expected", seed=123)
    h = spec.discoveries[0].hypothesis
    partitions, _ = partition_frame(frame, FederationCell(sites=2), spec.pair_id, "repeat")
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
        stage = re.search(r"stage (\w+)", first)[1]
        if "Before this stage, direct the sites." in prompt:
            refs = (
                [h["ref"] for h in payload["claims"]]
                if stage == "analyze" and first.startswith("Iteration 1/")
                else []
            )
            value = {
                "goal": "One shared scientific goal",
                "analyses": refs,
                "site_instructions": {},
            }
            return ChatResponse(
                text=json.dumps(value),
                model_id=self.model_id,
                raw={"usage": {"prompt_tokens": 20, "completion_tokens": 7}},
            )
        response = super().chat(
            [ChatMessage("user", first + "\n" + json.dumps(payload, separators=(",", ":")))]
        )
        # Site and central forms record precisely the currently directed comparisons.
        form = json.loads(response.text)
        if stage == "analyze":
            if "federation" in payload:
                form["run_analyses"] = payload["federation"]["approved_analyses"]
            elif "All sites completed this stage." in prompt:
                form["run_analyses"] = json.loads(prompt.rsplit(": ", 1)[1])
        return ChatResponse(
            text=json.dumps(form),
            model_id=self.model_id,
            raw={"usage": {"prompt_tokens": 20, "completion_tokens": 11}},
        )


def test_grid_n1_equivalence_stage_handoffs_tokens_and_shared_budget(tmp_path, monkeypatch):
    spec, pairs, raw, path = config(tmp_path, iterations=6)
    raw["federation"] = {"site_counts": [1, 2]}
    raw["budget"]["max_agent_calls"] = 200
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)
    assert len(build_run_plans(grid)) == 12
    assert (
        required_agent_calls(grid, grid.tasks[0], grid.workflows[2], FederationCell(sites=2)) == 192
    )
    made = providers(monkeypatch, pairs[0], FederatedScientist)
    result = run_experiment(grid)
    assert result["n_completed"] == 12, [
        (r["run_id"], r.get("error"), r.get("call_failures", [])[:1])
        for r in result["runs"]
        if r["status"] != "completed"
    ]
    scientific = reports(result)
    for report in scientific:
        n = report["site_count"]
        per_stage = 3 if report["workflow_mode"] == "deliberative" else 1
        expected_calls = 24 * (per_stage if n == 1 else n * per_stage + 2)
        audit = report["coordination"]
        assert audit["agent_calls"] == expected_calls
        assert audit["usage"]["output_tokens"] > 0
        assert audit["output_token_accounting"]["missing_calls"] == 0
        assert report["validation"]["voluntary_slots_used"] <= report["policy"]["voluntary_limit"]
        assert len(report["validation"]["cache"]) <= report["policy"]["voluntary_limit"] + len(
            report["policy"]["release_iterations"]
        )
        if n == 2:
            root = Path(
                next(
                    r["scientific_report"]
                    for r in result["runs"]
                    if r["run_id"] == report["run_id"]
                )
            ).parent
            assert len(list((root / "handoffs").glob("*.json"))) == 48
            assert set(audit["scope_audits"]) == {"central", "site_1", "site_2"}
            assert len(audit["committed_stages"]) == 24
            assert all((root / p).exists() for p in audit["participant_artifacts"])
            assert report["successful_stages"] == 24
    # Every central/site prompt contains aggregates only, never identifiers or raw row exports.
    for provider in made:
        text = json.dumps(provider.messages)
        assert "patient_id" not in text and "membership_sha256" not in text
    summary = json.loads((grid.output_root / "expected_surprising_summary.json").read_text())
    assert len(summary["conditions"]) == 6
    assert len(summary["workflow_contrasts"]) == 4
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

    from onc_co_scientist.expected_surprising.federation import Direction, FederatedCoordinator
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
    federation.prepared[federation.active] = {"direction": {"analyses": []}}
    with pytest.raises(ValueError, match="not sent to the sites"):
        controller.analysis_estimator(
            controller.frame, spec.discoveries[0].hypothesis, delta=0.1, alpha=0.05, result_id="x"
        )
    with pytest.raises(ValueError):
        Direction(analyses=[f"H{i}" for i in range(13)])


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
                "All sites completed this stage." in prompt
                and "Iteration 1/6, stage analyze, attempt 1." in prompt
            ):
                value = json.loads(response.text)
                value["run_analyses"] = []  # Central cannot omit executed site comparisons.
                response.text = json.dumps(value)
            return response

    class Interrupted(Repairing):
        def chat(self, messages, **kwargs):
            if len(self.messages) == 8:
                raise KeyboardInterrupt("simulate interruption before central repair")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], Interrupted)
    with pytest.raises(KeyboardInterrupt):
        integration.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=False)
    made = providers(monkeypatch, pairs[0], Repairing)
    result = integration.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=True)
    assert result["status"] == "completed", result.get("error")
    assert result["agent_calls"] == 97
    assert len(made[0].messages) == 89
    audit = reports({"runs": [result]})[0]["coordination"]
    assert audit["scope_audits"]["site_1"]["agent_calls"] == 24
    assert audit["scope_audits"]["site_2"]["agent_calls"] == 24
    root = Path(result["scientific_report"]).parent
    for site in ("central", "site_1", "site_2"):
        assert (root / "calls" / "requests" / site / "i001-analyze-agent-a1.json").exists()
