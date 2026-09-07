"""Acceptance fixtures for the appraisal-first workflow and score profile."""

import copy
import json
from pathlib import Path

import pytest

from onc_co_scientist.expected_surprising.evaluation import WorkflowValidationService
from onc_co_scientist.expected_surprising.events import response_summary
from onc_co_scientist.expected_surprising.generation import sample, write_pair
from onc_co_scientist.expected_surprising.packaging import PUBLIC_INSTRUCTIONS, repackage, sha256
from onc_co_scientist.expected_surprising.rollout import run, stage_transaction
from onc_co_scientist.expected_surprising.schemas import (
    Assessment,
    Condition,
    EvidenceResult,
    Hypothesis,
    ValidationPolicy,
    ValidationRequest,
    WorkflowStageRecord,
    WorkflowVersions,
)
from onc_co_scientist.expected_surprising.scoring import (
    claim_key,
    comparison_key,
    discovery_performance,
    estimate,
    evidence_status,
    exploration_coverage,
    refinement_types,
    workflow_discovery,
)
from onc_co_scientist.expected_surprising.summary import paired_summary
from onc_co_scientist.expected_surprising.workflow import (
    STAGE_PROMPT,
    WorkflowController,
    public_event,
)
from onc_co_scientist.providers.base import ChatResponse
from tests.test_expected_surprising import pair


@pytest.fixture
def packages(tmp_path):
    spec = pair()
    write_pair(spec, tmp_path / "old")
    manifest = repackage(tmp_path / "old", tmp_path / "new")
    return spec, tmp_path / "new", manifest


class ScriptedScientist:
    model_id = "scripted-workflow"

    def __init__(self, spec, behavior="follows", *, repair=False, exhaust=False):
        self.spec, self.behavior = spec, behavior
        self.repair, self.exhaust = repair, exhaust
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        prompt = messages[0].content
        payload = json.loads(prompt[prompt.index('{"schema":') :])
        compact_payload = payload if "claims" in payload else None
        if compact_payload is not None:
            known_rows = compact_payload["claims"]
            hs = [dict(id=row["ref"], **row["comparison"]) for row in known_rows]
            results = [
                {
                    "id": result["ref"],
                    "hypothesis_id": row["ref"],
                    **{k: v for k, v in result.items() if k != "ref"},
                }
                for row in known_rows
                for result in row["evidence"]
            ]
            payload = {
                "history": [{"record": {"hypotheses": hs}, "results": results}],
                "required_assessments": [
                    {"hypothesis_id": x["claim"], "result_id": x["evidence"], "reason": x["reason"]}
                    for x in compact_payload["assessments_due"]
                ],
                "current_assessments": {
                    claim_key(Hypothesis.model_validate(h)): row["current"]
                    for h, row in zip(hs, known_rows, strict=True)
                },
            }
        first = prompt.splitlines()[0].replace(",", "").replace(".", "").split()
        iteration, stage, attempt = int(first[1].split("/")[0]), first[3], int(first[5])
        record = WorkflowStageRecord(iteration=iteration, stage=stage)
        known, evidence, requested = {}, {}, set()
        for event in payload["history"]:
            if "record" not in event:
                continue
            for h in event["record"]["hypotheses"]:
                known[h["id"]] = Hypothesis.model_validate(h)
            for r in event["results"]:
                evidence[r["id"]] = r
                if r["source"] != "discovery":
                    requested.add(comparison_key(known[r["hypothesis_id"]]))
        if iteration == 1 and stage == "explore" and self.behavior != "ineligible":
            hs = [d.hypothesis.model_copy(deep=True) for d in self.spec.discoveries]
            # One contradicted directional claim supplies an exclusion-class opportunity.
            hs[1].direction *= -1
            record.hypotheses = hs
            record.anticipated_directions = {h.id: h.direction for h in hs}
            record.assessments = {h.id: "unresolved" for h in hs}
            if self.exhaust or (self.repair and attempt == 1):
                record.hypotheses[-1].exposure = "nonexistent_variable"
        if stage == "analyze" and iteration == 1:
            record.executed_ids = list(known)
        current = {key: value["status"] for key, value in payload["current_assessments"].items()}
        by_id = {}
        for item in payload["required_assessments"]:
            hid, rid = item["hypothesis_id"], item["result_id"]
            status = (
                evidence_status(
                    EvidenceResult.model_validate(
                        {
                            **{
                                k: v
                                for k, v in evidence[rid].items()
                                if k not in {"source", "cached"}
                            },
                            # Only this evaluator-aware fixture uses the private reference.
                            "delta": next(
                                o.delta for o in self.spec.outcomes if o.name == known[hid].outcome
                            ),
                        }
                    )
                )
                or "unresolved"
            )
            if self.behavior in {"accept_all", "ignore_validation", "perseveres"}:
                status = "accept"
            if self.behavior == "cautious" and item["reason"] != "response_deadline":
                status = "unresolved"
            if hid not in by_id:
                by_id[hid] = Assessment(
                    hypothesis_id=hid,
                    result_ids=[],
                    status=status,
                    investigation="active" if status == "unresolved" else "closed",
                )
            by_id[hid].result_ids.append(rid)
            current[claim_key(known[hid])] = status
        record.assessment_records = list(by_id.values())
        if stage == "appraise" and self.behavior in {"follows", "ignore_validation"}:
            for h in known.values():
                key = comparison_key(h)
                discovery = next(
                    (
                        r
                        for r in evidence.values()
                        if r["source"] == "discovery"
                        and comparison_key(known[r["hypothesis_id"]]) == key
                        and r["valid"]
                    ),
                    None,
                )
                if discovery and key not in requested:
                    record.validation_request = ValidationRequest(
                        hypothesis_id=h.id, triggering_result_ids=[discovery["id"]]
                    )
                    break
        if stage == "synthesize":
            record.accepted_ids = [
                h.id for h in known.values() if current[claim_key(h)] == "accept"
            ]
        if compact_payload is not None:
            form = {
                "proposals": [
                    dict(
                        **h.model_dump(exclude={"id"}),
                        anticipated_direction=record.anticipated_directions[h.id],
                        initial_status=record.assessments[h.id],
                    )
                    for h in record.hypotheses
                ],
                "assessments": [
                    dict(
                        claim=a.hypothesis_id,
                        evidence=a.result_ids,
                        status=a.status,
                        investigation=a.investigation,
                    )
                    for a in record.assessment_records
                ],
                "narrative": record.narrative,
            }
            if stage == "analyze":
                form["run_analyses"] = record.executed_ids
            if stage == "appraise":
                form["validate"] = (
                    record.validation_request.hypothesis_id if record.validation_request else None
                )
            return ChatResponse(text=json.dumps(form), model_id=self.model_id)
        return ChatResponse(text=record.model_dump_json(), model_id=self.model_id)


def scripted_run(packages, tmp_path, behavior, **kwargs):
    spec, root, manifest = packages
    public = root / "public" / manifest["tasks"][0]["task_id"]
    provider = ScriptedScientist(spec, behavior, **kwargs)
    report = run(
        spec,
        "expected",
        public,
        tmp_path / behavior,
        provider,
        run_id=behavior,
        replicate_id="fixture-replicate",
        iterations=6,
    )
    return report, provider


@pytest.mark.parametrize(
    "behavior",
    [
        "follows",
        "perseveres",
        "accept_all",
        "ignore_validation",
        "automatic",
        "ineligible",
        "cautious",
    ],
)
def test_scripted_behavior_profiles(packages, tmp_path, behavior):
    report, provider = scripted_run(packages, tmp_path, behavior)
    assert report["protocol_errors"] == [], report["attempt_errors"]
    assert report["successful_stages"] == provider.calls == 24
    assert report["versions"] == WorkflowVersions().model_dump()
    assert report["policy"]["release_iterations"] == [2, 4]
    assert report["validation"]["alpha"] == 0.05 / 12
    if behavior == "ineligible":
        assert report["scores"]["D"] == report["scores"]["E"] == 0
        assert report["scores"]["B"] is None
        assert report["behavior"]["validation_choice"]["scheduled_unavailable_n"] == 2
    else:
        assert report["scores"]["E"] == 100
        assert report["responsiveness"]["total_events"] >= 2
        assert report["scores"]["B"] is None  # no ambiguous evidence in these fixtures
        if behavior in {"automatic", "cautious"}:
            assert report["validation"]["voluntary_slots_used"] == 0
            assert [e["iteration"] for e in report["responsiveness"]["events"]] == [2, 4]
        if behavior == "cautious":
            events = report["responsiveness"]["events"]
            assert all(e["immediate"]["status"] == "unresolved" for e in events)
            assert all(e["score"] == 1 for e in events)
        if behavior in {"perseveres", "ignore_validation", "accept_all"}:
            assert report["scores"]["discovery"]["unsupported_acceptance_n"] >= 1
        if behavior in {"follows", "automatic", "cautious"}:
            assert report["responsiveness"]["accuracy"] == 1
    transcript = (tmp_path / behavior / "transcript.jsonl").read_text()
    for entry in map(json.loads, transcript.splitlines()):
        if entry["kind"] == "model":
            prompt = entry["prompt"]
            assert "eligible_pool" not in prompt
            assert "desired_stratum" not in prompt
            assert "Accept if" not in prompt and "accept if" not in prompt
            assert "expected_direction" not in prompt and "category" not in prompt
            assert '"delta":' not in prompt
            assert "minimum effect" not in prompt


def test_recovered_and_exhausted_contracts(packages, tmp_path):
    repaired, provider = scripted_run(packages, tmp_path, "follows", repair=True)
    assert repaired["protocol_errors"] == []
    assert provider.calls == 25
    assert len(repaired["attempt_errors"]) == 1
    assert repaired["scores"]["discovery"]["exact"]["first_attempt_D"] == 0
    exhausted, _ = scripted_run(packages, tmp_path, "automatic", exhaust=True)
    assert exhausted["scores"]["D"] == 0
    assert exhausted["confirmation"]["primary_recovery"] == 0
    assert exhausted["state"]["registrations"] == []


def test_repackaging_hashes_and_public_contract(packages, tmp_path):
    _, root, manifest = packages
    for t in manifest["tasks"]:
        old, new = tmp_path / "old/public" / t["source_task_id"], root / "public" / t["task_id"]
        assert (
            sha256(old / "dataset.parquet") == sha256(new / "dataset.parquet") == t["source_sha256"]
        )
        assert "Accept if" in (old / "instructions.md").read_text()
        assert (new / "instructions.md").read_text() == PUBLIC_INSTRUCTIONS
        assert t["policy"]["release_iterations"] == [5, 12, 20]
        for name in ("instructions.md", "task.json", "stage_schema.json"):
            text = (new / name).read_text()
            assert not any(
                term in text
                for term in ("eligible_pool", "category", "focal", "Accept if", "delta")
            )
    with pytest.raises(FileExistsError):
        repackage(tmp_path / "old", root)
    source = tmp_path / "old/public" / manifest["tasks"][0]["source_task_id"] / "dataset.parquet"
    source.write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        repackage(tmp_path / "old", tmp_path / "corrupt-export")
    assert not (tmp_path / "corrupt-export").exists()


def test_schedule_cache_recoding_seed_and_rollback(monkeypatch):
    spec = pair()
    policy = ValidationPolicy.default(spec.profile, 6)
    service = WorkflowValidationService(spec, "expected", "replicate", policy)
    h = spec.discoveries[0].hypothesis
    frame = sample(spec, "expected", seed=71)
    r = estimate(frame, h, delta=0.1, alpha=0.05, result_id="analysis-a")
    tests = {comparison_key(h): {"hypothesis": h.model_dump(), "result": r.model_dump()}}
    draws = []

    def fake_sample(*args, seed):
        draws.append(seed)
        return frame

    monkeypatch.setattr("onc_co_scientist.expected_surprising.evaluation.sample", fake_sample)
    opportunity = copy.deepcopy(service.select(1, tests))
    assert draws == []
    assert opportunity["release_iteration"] == 2
    old = copy.deepcopy(service.state)
    with pytest.raises(ValueError), stage_transaction([service.state]):
        service.deliver(h, 1, "voluntary")
        raise ValueError("late stage contract error")
    assert service.state == old
    voluntary, fresh = service.deliver(h, 1, "voluntary")
    assert fresh and draws[0] == draws[1]
    assert not service.due(2)
    opposite = h.model_copy(update={"id": "opposite", "direction": -h.direction})
    inverted, fresh = service.deliver(opposite, 2, "voluntary")
    assert not fresh and inverted.lower == -voluntary.upper and inverted.upper == -voluntary.lower
    recoded = h.model_copy(
        update={
            "id": "recoded",
            "exposed": h.comparator,
            "comparator": h.exposed,
            "direction": -h.direction,
        }
    )
    same, fresh = service.deliver(recoded, 2, "voluntary")
    assert not fresh and same.estimate == voluntary.estimate
    assert service.export()["voluntary_slots_used"] == service.export()["distinct_comparisons"] == 1
    assert len(draws) == 2
    auto = WorkflowValidationService(spec, "expected", "replicate", policy)
    auto.select(1, tests)
    automatic, fresh = auto.deliver(h, 2, "automatic")
    assert automatic == voluntary
    partner = WorkflowValidationService(spec, "surprising", "replicate", policy)
    assert partner.derive_seed("independent-validation", comparison_key(h)) == draws[0]
    assert partner.seed == service.seed
    altered = spec.model_copy(deep=True)
    altered.discoveries.reverse()
    altered.discoveries[0].category = "expected"
    changed = WorkflowValidationService(altered, "expected", "replicate", policy)
    assert changed.select(1, tests) == opportunity


def test_registration_exposure_refinements_and_temporal_contract():
    spec = pair()
    c = WorkflowController(
        spec,
        "expected",
        sample(spec, "expected", seed=2),
        ValidationPolicy.default(spec.profile, 6),
        "test",
    )
    h = spec.discoveries[0].hypothesis
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="explore",
            hypotheses=[h],
            anticipated_directions={h.id: 1},
            assessments={h.id: "unresolved"},
        )
    )
    assert c.state["expectations"][h.id]["pre_evidence"]
    c.apply(WorkflowStageRecord(iteration=1, stage="analyze", executed_ids=[h.id]))
    rid = next(iter(c.state["results"]))
    opposite = h.model_copy(update={"id": "inverse", "direction": -h.direction})
    record = WorkflowStageRecord(
        iteration=1,
        stage="appraise",
        hypotheses=[opposite],
        parent_ids={opposite.id: h.id},
        motivating_result_ids={opposite.id: [rid]},
        anticipated_directions={opposite.id: 1},
        assessments={opposite.id: "unresolved"},
        assessment_records=[
            Assessment(
                hypothesis_id=h.id, result_ids=[rid], status="accept", investigation="closed"
            )
        ],
    )
    event = c.apply(record)
    assert not c.state["expectations"][opposite.id]["pre_evidence"]
    assert c.state["registrations"][-1]["refinement_types"] == ["direction_change"]
    assert event["reused_evidence"][0]["estimate"] == -c.state["results"][rid]["result"].estimate
    assert len(c.state["tests"]) == 1
    with (
        pytest.raises(ValueError, match="available"),
        stage_transaction([c.state, c.service.state]),
    ):
        c.apply(
            WorkflowStageRecord(
                iteration=1,
                stage="synthesize",
                assessment_records=[
                    Assessment(
                        hypothesis_id=h.id,
                        result_ids=["future"],
                        status="accept",
                        investigation="closed",
                    )
                ],
            )
        )
    with pytest.raises(ValueError, match="Initial direction"), stage_transaction([c.state]):
        c.apply(
            WorkflowStageRecord(iteration=1, stage="synthesize", anticipated_directions={h.id: -1})
        )
    refinement = h.model_copy(
        update={"subgroup": [Condition(variable="research_signature_a", op="ge", value=0)]}
    )
    assert refinement_types(h, refinement) == ["subgroup_condition_addition"]


def test_score_arithmetic_coverage_and_untested_claims():
    score = discovery_performance({"expected": 0.75, "neutral": 0.4, "surprising": 0.65}, 0.9)
    assert score["R"] == pytest.approx(0.6)
    assert score["D"] == pytest.approx(72)
    assert (
        discovery_performance(dict.fromkeys(("expected", "neutral", "surprising"), 0), None)["D"]
        == 0
    )
    spec = pair()
    claims = [d.hypothesis for d in spec.discoveries]
    records = [
        dict(
            hypothesis_id=h.id,
            id=h.id,
            estimate=1,
            lower=0.5,
            upper=1.5,
            delta=0.1,
            alpha=0.05 / 7,
            cell_n=[100, 100],
            valid=True,
            diagnostic="ok",
        )
        for h in claims
    ]
    extra = claims[0].model_copy(
        update={
            "id": "extra",
            "exposure": "sex_female",
            "subgroup": [Condition(variable="research_signature_c", op="ge", value=0)],
        }
    )
    records.append({**records[0], "id": extra.id, "hypothesis_id": extra.id})
    tested = {comparison_key(h) for h in claims[1:] + [extra]}
    scores = workflow_discovery(claims + [extra], spec.discoveries, tested, {"results": records})
    assert scores["exact"]["Q"] == 6 / 7  # confirmed extra counts, untested focal does not
    assert scores["exact"]["R_c"]["expected"] == 2 / 3
    assert (
        workflow_discovery(claims, spec.discoveries, tested, {"results": records}, failed=True)[
            "exact"
        ]["D"]
        == 0
    )
    tests = [
        {"hypothesis": h.model_dump(), "iteration": 1, "result": r}
        for h, r in zip(claims, records, strict=False)
    ]
    early = exploration_coverage(tests, spec.discoveries, 6)
    assert early["exact"]["E"] == 100
    late = exploration_coverage([{**t, "iteration": 3} for t in tests], spec.discoveries, 6)
    assert late["exact"]["E"] == pytest.approx(100 * 4 / 6)
    assert exploration_coverage(tests * 2, spec.discoveries, 6) == early
    assert (
        exploration_coverage(
            [{**t, "result": {**t["result"], "valid": False}} for t in tests], spec.discoveries, 6
        )["exact"]["E"]
        == 0
    )


def test_response_missing_late_invalid_and_interrupted():
    spec = pair()
    h = spec.discoveries[0].hypothesis
    events = []
    for i, (lo, hi, status) in enumerate(
        ((0.2, 0.4, "accept"), (-0.2, 0, "reject"), (0, 0.2, "unresolved"))
    ):
        result = EvidenceResult(
            id=f"v{i}",
            hypothesis_id=h.id,
            estimate=(lo + hi) / 2,
            lower=lo,
            upper=hi,
            delta=0.1,
            alpha=0.01,
            cell_n=[100, 100],
            valid=True,
            diagnostic="ok",
        )
        events.append(
            dict(
                hypothesis=h.model_dump(),
                result=result.model_dump(),
                iteration=2,
                due_iteration=4,
                source="automatic",
                prior="unresolved",
                anticipated_direction=1,
                claim_direction=h.direction,
                immediate={"status": "unresolved"},
                delayed={"status": status},
            )
        )
    stages = [{"iteration": 6, "stage": "explore"}]
    assert response_summary(events, stages, 6, spec.discoveries)["B"] == 100
    assert response_summary(events[:1], stages, 6, spec.discoveries)["B"] is None
    events[0]["delayed"] = None
    s = response_summary(events, stages, 6, spec.discoveries)
    assert s["B"] == pytest.approx(200 / 3)
    assert s["missing_due_n"] == 1
    events[0]["due_iteration"] = 7
    events[1]["result"]["valid"] = False
    s = response_summary(events, [{"iteration": 3}], 6, spec.discoveries)
    assert s["B"] is None and s["eligible_n"] == 0
    assert s["exclusions"] == {"late": 1, "invalid": 1, "interrupted": 1}


def test_hierarchical_scores_and_missing_response_classes(packages, tmp_path):
    report, _ = scripted_run(packages, tmp_path, "automatic")
    reports = []
    for pair_id, value, repeats in (("a", 100, 3), ("b", 0, 1)):
        for version in ("expected", "surprising"):
            for i in range(repeats):
                r = copy.deepcopy(report)
                r.update(
                    pair_id=pair_id,
                    version=version,
                    run_id=f"{pair_id}-{version}-{i}",
                    replicate_id=str(i),
                )
                r["scores"]["D"] = value
                reports.append(r)
    summary = paired_summary(reports, bootstrap_replicates=20)
    assert summary["D"] == 50
    assert summary["B"] is None
    assert summary["response_components"]["ambiguous"]["events"] == 0
    assert summary["response_components"]["supported"]["eligible_base_datasets"] == 2
    reports[0]["policy"]["selection_seed"] = 1
    with pytest.raises(ValueError, match="identical budgets and policies"):
        paired_summary(reports)


def test_no_public_numerical_decision_coaching():
    for prompt in (PUBLIC_INSTRUCTIONS, STAGE_PROMPT):
        assert not any(
            text in prompt.lower()
            for text in ("accept if", "reject if", "lower bound exceeds", "minimum effect", "delta")
        )
        assert "scientific judgment" in prompt
    assert Path("src/onc_co_scientist/expected_surprising/rollout.py").exists()


def test_public_evidence_omits_private_cutoffs_for_all_delivery_routes():
    result = {
        "id": "validation-a",
        "hypothesis_id": "h1",
        "estimate": 0.04,
        "lower": 0.01,
        "upper": 0.07,
        "delta": 0.1,
        "alpha": 0.05 / 12,
        "cell_n": [100, 100],
        "valid": True,
        "diagnostic": "welch_independent_cells",
    }
    for source in ("discovery", "voluntary", "automatic"):
        for cached in (True, False):
            event = {
                "results": [{**result, "source": source, "cached": cached}],
                "reused_evidence": [result],
            }
            visible = public_event(event)
            assert '"delta":' not in json.dumps(visible)
            assert event["results"][0]["delta"] == result["delta"]
            for key in ("results", "reused_evidence"):
                for field in ("estimate", "lower", "upper", "alpha", "cell_n"):
                    assert visible[key][0][field] == result[field]


def test_thirteen_sample_bound_and_one_new_request_each_iteration(monkeypatch):
    spec = pair()
    policy = ValidationPolicy.default(spec.profile, 25)
    service = WorkflowValidationService(spec, "expected", "budget-test", policy)
    frame = sample(spec, "expected", seed=9)
    monkeypatch.setattr(
        "onc_co_scientist.expected_surprising.evaluation.sample", lambda *args, **kwargs: frame
    )
    candidates, tests = [], {}
    for i in range(25):
        h = spec.discoveries[0].hypothesis.model_copy(
            update={
                "id": f"h{i}",
                "subgroup": [
                    Condition(variable="research_signature_a", op="ge", value=-1 + i / 50)
                ],
            }
        )
        candidates.append(h)
        r = estimate(frame, h, delta=0.1, alpha=0.05, result_id=f"a{i}")
        assert r.valid
        tests[comparison_key(h)] = {"hypothesis": h.model_dump(), "result": r.model_dump()}
    for iteration in range(1, 26):
        service.select(iteration, tests)
        selected = {o["comparison_key"] for o in service.state["opportunities"]}
        if iteration <= 10:
            h = next(
                h
                for h in candidates
                if comparison_key(h) not in selected | service.state["cache"].keys()
            )
            service.deliver(h, iteration, "voluntary")
            other = next(
                h
                for h in candidates
                if comparison_key(h) not in selected | service.state["cache"].keys()
            )
            with pytest.raises(ValueError, match="budget exhausted"):
                service.deliver(other, iteration, "voluntary")
        for o in service.due(iteration):
            service.deliver(Hypothesis.model_validate(o["hypothesis"]), iteration, "automatic")
    assert service.export()["distinct_comparisons"] == 13
    assert service.export()["voluntary_slots_used"] == 10
    assert all(e["result"].alpha == 0.05 / 13 for e in service.state["cache"].values())
    with pytest.raises(ValueError, match="budget exhausted"):
        service.deliver(candidates[-1], 25, "voluntary")
    cached = next(iter(service.state["cache"].values()))["hypothesis"]
    service.deliver(cached, 25, "voluntary")
    assert service.export()["distinct_comparisons"] == 13


def test_renaming_reanalysis_and_first_acceptance_history():
    spec = pair()
    c = WorkflowController(
        spec,
        "expected",
        sample(spec, "expected", seed=2),
        ValidationPolicy.default(spec.profile, 6),
        "rename",
    )
    h = spec.discoveries[0].hypothesis
    renamed = h.model_copy(update={"id": "renamed"})
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="explore",
            hypotheses=[h, renamed],
            anticipated_directions={h.id: 1, renamed.id: 1},
            assessments={h.id: "unresolved", renamed.id: "unresolved"},
        )
    )
    c.apply(WorkflowStageRecord(iteration=1, stage="analyze", executed_ids=[h.id, renamed.id]))
    assert len(c.required(1, "appraise")) == 1
    rid = next(iter(c.state["results"]))
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="appraise",
            assessment_records=[
                Assessment(
                    hypothesis_id=h.id, result_ids=[rid], status="accept", investigation="active"
                )
            ],
            validation_request=ValidationRequest(hypothesis_id=h.id, triggering_result_ids=[rid]),
        )
    )
    assert len(c.state["tests"]) == 1 and len(c.state["events"]) == 2
    first = next(iter(c.state["first_acceptances"].values()))
    assert first["receipt_before"] is None and first["requests_before"] == []
    vrid = next(r for r in c.state["results"] if r.startswith("validation-"))
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="synthesize",
            assessment_records=[
                Assessment(
                    hypothesis_id=h.id, result_ids=[vrid], status="accept", investigation="closed"
                )
            ],
            accepted_ids=[renamed.id],
        )
    )
    assert len(c.state["first_acceptances"]) == 1
    c.apply(
        WorkflowStageRecord(
            iteration=2,
            stage="explore",
            assessment_records=[
                Assessment(
                    hypothesis_id=renamed.id,
                    result_ids=[vrid],
                    status="reject",
                    investigation="closed",
                )
            ],
        )
    )
    c.apply(WorkflowStageRecord(iteration=2, stage="analyze", executed_ids=[renamed.id]))
    c.apply(
        WorkflowStageRecord(
            iteration=2,
            stage="appraise",
            assessment_records=[
                Assessment(
                    hypothesis_id=renamed.id,
                    result_ids=[rid],
                    status="accept",
                    investigation="closed",
                )
            ],
            validation_request=ValidationRequest(
                hypothesis_id=renamed.id, triggering_result_ids=[rid]
            ),
        )
    )
    assert len(c.state["events"]) == 2
    assert c.service.export()["distinct_comparisons"] == 1
    assert c.service.export()["voluntary_slots_used"] == 1
    assert len(c.state["first_acceptances"]) == 1
    assert next(iter(c.state["first_acceptances"].values())) == first


def test_aggregate_balanced_score_can_be_available_with_unavailable_run_scores(packages, tmp_path):
    report, _ = scripted_run(packages, tmp_path, "automatic")
    reports = []
    for version, cls in (("expected", "supported"), ("surprising", "excluded")):
        for replicate, other in (("0", "ambiguous"), ("1", cls)):
            r = copy.deepcopy(report)
            r.update(version=version, run_id=f"{version}-{replicate}", replicate_id=replicate)
            r["responsiveness"]["components"] = {
                k: {
                    "accuracy": 1.0 if k == other else None,
                    "numerator": int(k == other),
                    "denominator": int(k == other),
                }
                for k in ("supported", "excluded", "ambiguous")
            }
            reports.append(r)
    result = paired_summary(reports, bootstrap_replicates=10)
    assert all(r["scores"]["B"] is None for r in reports)
    assert result["B"] == 100
    assert result["response_components"]["ambiguous"]["eligible_runs"] == 2
    assert result["score_uncertainty"]["B"]["ci95"] is None


def test_opposite_and_recoded_evidence_diagnostics():
    from onc_co_scientist.expected_surprising.events import behavioral_summary
    from onc_co_scientist.expected_surprising.scoring import oriented

    spec = pair()
    h = spec.discoveries[0].hypothesis
    for recode in (False, True):
        c = WorkflowController(
            spec,
            "expected",
            sample(spec, "expected", seed=2),
            ValidationPolicy.default(spec.profile, 6),
            "diagnostics",
        )
        changed = h.model_copy(
            update={
                "id": "changed",
                "direction": -h.direction,
                **({"exposed": h.comparator, "comparator": h.exposed} if recode else {}),
            }
        )
        c.apply(
            WorkflowStageRecord(
                iteration=1,
                stage="explore",
                hypotheses=[h],
                anticipated_directions={h.id: h.direction},
                assessments={h.id: "unresolved"},
            )
        )
        c.apply(WorkflowStageRecord(iteration=1, stage="analyze", executed_ids=[h.id]))
        rid = next(iter(c.state["results"]))
        c.apply(
            WorkflowStageRecord(
                iteration=1,
                stage="appraise",
                hypotheses=[changed],
                anticipated_directions={changed.id: changed.direction},
                assessments={changed.id: "unresolved"},
                assessment_records=[
                    Assessment(
                        hypothesis_id=h.id,
                        result_ids=[rid],
                        status="unresolved",
                        investigation="active",
                    )
                ],
            )
        )
        c.apply(
            WorkflowStageRecord(
                iteration=2,
                stage="appraise",
                validation_request=ValidationRequest(
                    hypothesis_id=changed.id, triggering_result_ids=[rid]
                ),
            )
        )
        c.state["completed_stages"].append({"iteration": 6, "stage": "synthesize"})
        summary = behavioral_summary(
            c.state, c.service.export(), spec.discoveries, 6, spec.focal_id
        )
        validation = next(e for e in summary["followups"] if e["source"] == "voluntary")
        assert validation["evidence_class"] == validation["exploratory_evidence_class"]
        assert not validation["exploratory_validation_disagreement"]
        assert not summary["milestones"][0]["surprising_evidence_encountered"]
        if recode:
            original = c.state["results"][rid]["result"]
            assert oriented(original, h, changed).cell_n == original.cell_n[::-1]


@pytest.mark.parametrize(
    "profile,budget,schedule",
    [("nsclc_clinical", 25, [5, 12, 20]), ("nsclc_depmap", 25, [3, 6, 8])],
)
def test_full_budget_provider_independent_packaged_policy(tmp_path, profile, budget, schedule):
    spec = pair(profile=profile)
    write_pair(spec, tmp_path / "source")
    for path in (tmp_path / "source/public").glob("*/task.json"):
        assert json.loads(path.read_text())["iterations"] == budget
    # Explicit replay retains the old budget; repackaging must upgrade it without
    # editing the historical task or regenerating its numerical dataset.
    write_pair(spec, tmp_path / "historical", historical_replay=True)
    old_budget = 10 if profile.endswith("depmap") else 25
    for path in (tmp_path / "historical/public").glob("*/task.json"):
        assert json.loads(path.read_text())["iterations"] == old_budget
    historical_manifest = repackage(tmp_path / "historical", tmp_path / "upgraded")
    for item in historical_manifest["tasks"]:
        old = tmp_path / "historical/public" / item["source_task_id"]
        new = tmp_path / "upgraded/public" / item["task_id"]
        assert json.loads((old / "task.json").read_text())["iterations"] == old_budget
        assert json.loads((new / "task.json").read_text())["iterations"] == budget
        assert sha256(old / "dataset.parquet") == sha256(new / "dataset.parquet")
    manifest = repackage(tmp_path / "source", tmp_path / "new")
    policy_path = tmp_path / "new/private" / spec.pair_id / "workflow.json"
    policy = json.loads(policy_path.read_text())
    policy["policy"]["selection_seed"] = 99
    policy_path.write_text(json.dumps(policy))
    provider = ScriptedScientist(spec, "automatic")
    report = run(
        spec,
        "expected",
        tmp_path / "new/public" / manifest["tasks"][0]["task_id"],
        tmp_path / "run",
        provider,
        run_id="full-budget",
        replicate_id="r0",
    )
    assert report["protocol_errors"] == [], report["attempt_errors"]
    assert report["successful_stages"] == provider.calls == 4 * budget
    assert report["policy"]["selection_seed"] == 99
    assert report["policy"]["release_iterations"] == schedule
    assert [e["iteration"] for e in report["responsiveness"]["events"]] == schedule
    assert all(e["eligible"] for e in report["responsiveness"]["events"])
    assert report["validation"]["max_comparisons"] == 13
