"""Scientific invariants of the paired benchmark and its deterministic scoring."""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from onc_co_scientist.expected_surprising.evaluation import (
    ReferenceUniverse,
    ValidationService,
    calibrate,
)
from onc_co_scientist.expected_surprising.generation import (
    base_frame,
    compile_pair,
    conditional_means,
    sample,
    version_discoveries,
    write_pair,
)
from onc_co_scientist.expected_surprising.research import Researcher
from onc_co_scientist.expected_surprising.review_policy import (
    REVIEW_POLICY_VERSION,
    candidate_digest,
    reject_nested_comparison,
    require_current_candidate,
    require_current_pair,
    spec_digest,
)
from onc_co_scientist.expected_surprising.schemas import (
    Candidate,
    Condition,
    DGPAssessment,
    DGPReview,
    Discovery,
    EvidenceResult,
    EvidenceScope,
    Hypothesis,
    RealismReview,
    Review,
    ReviewedCandidate,
    Source,
)
from onc_co_scientist.expected_surprising.scoring import (
    assign,
    estimate,
    evidence_status,
    match,
    responsiveness,
)
from onc_co_scientist.providers.base import ChatResponse
from onc_co_scientist.synthetic.cancer_types import all_cancer_types


def hypothesis(**kwargs):
    return Hypothesis(
        **{
            "id": "h",
            "outcome": "log_pfs_months",
            "exposure": "sex_female",
            "direction": 1,
            **kwargs,
        }
    )


def realism(**changes):
    return RealismReview(
        **dict(
            {
                "decision": "accept",
                "population_scope_valid": True,
                "comparator_scope_valid": True,
                "biomarker_restrictions_complete": True,
                "assay_transfer_valid": True,
                "category_valid": True,
                "no_inherited_directional_expectation": True,
                "source_ids": ["MED:1"],
                "rationale": "Fixture only",
                "required_changes": [],
                "counterexample_search": "Fixture has no unsupported patient stratum",
            },
            **changes,
        )
    )


def dgp_assessment(**changes):
    return DGPAssessment(
        **dict(
            {
                "decision": "accept",
                "expected_components_scoped": True,
                "intentional_reversals_only": True,
                "nonredundant_targets": True,
                "plausible_outcome_scale": True,
                "rationale": "Fixture only",
                "required_changes": [],
            },
            **changes,
        )
    )


def attest(spec):
    spec.dgp_review = DGPReview(
        assessment=dgp_assessment(),
        policy_version=REVIEW_POLICY_VERSION,
        spec_sha256=spec_digest(spec),
        model="fixture",
        reviewed_at="fixture",
    )
    return spec


def reviewed(h, category="expected", profile="nsclc_clinical"):
    source = Source(
        id="MED:1",
        title="Test evidence",
        abstract="A fixture, not scientific evidence.",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
    )
    item = ReviewedCandidate(
        candidate=Candidate(
            hypothesis=h,
            proposed_category=category,
            statement="Fixture only",
            queries=["support", "contradiction"],
        ),
        review=Review(
            decision="supported" if category == "expected" else "neutral",
            source_ids=[source.id],
            rationale="Fixture only",
            population_matches=True,
            exposure_comparator_matches=True,
            outcome_matches=True,
            direction_matches=True,
            evidence_scopes=[
                EvidenceScope(
                    source_id=source.id,
                    primary_empirical=True,
                    population="Fixture population",
                    exposure=h.exposure,
                    comparator="Fixture control",
                    outcome=h.outcome,
                    design_and_assay="Fixture design",
                    direction=h.direction,
                    limitations="Fixture only",
                    endpoint_kind="pfs" if profile.endswith("clinical") else "genetic_dependency",
                )
            ],
            broader_direction="none" if category == "neutral" else "positive",
            neutrality_rationale="Fixture no inherited expectation",
        ),
        sources=[source],
        search_ids=["fixture"],
        generation_model="test",
        review_model="test",
        reviewed_at="2026-09-06T00:00:00Z",
        review_profile=profile,
        review_policy_version=REVIEW_POLICY_VERSION,
        realism=realism(),
        realism_model="fixture",
    )
    item.candidate_sha256 = candidate_digest(item.candidate)
    return item


def pair(profile="nsclc_clinical", mode="overall"):
    frame = base_frame(profile, 1000, 1)
    binary = [c for c in frame if set(frame[c].dropna().unique()) == {0, 1}]
    candidates = []
    for i in range(6):
        # Separate fixture endpoints isolate pairing/scoring invariants from
        # the live literature catalog and its clinical overlap calibration.
        h = hypothesis(
            id=f"d{i}",
            exposure=binary[i],
            outcome=f"test_outcome_{i}",
            subgroup=[]
            if i < 4
            else [
                Condition(variable="research_signature_a", op="ge", value=0),
                Condition(variable="research_signature_b", op="ge", value=0),
            ],
        )
        candidates.append(reviewed(h, "expected" if i < 4 else "neutral", profile))
    return attest(compile_pair(profile, candidates, seed=44, n=2000, focal_mode=mode))


def test_exact_near_direction_and_interaction():
    target = hypothesis(
        subgroup=[
            Condition(variable="age", op="ge", value=50),
            Condition(variable="age", op="le", value=65),
            Condition(variable="platelet", op="ge", value=200),
            Condition(variable="score", op="ge", value=1),
        ]
    )
    assert match(target, target) == "exact"
    omitted = target.model_copy(update={"subgroup": target.subgroup[2:]})
    assert match(omitted, target) == "near"  # Both age conditions omitted together.
    partial_age = target.model_copy(update={"subgroup": target.subgroup[1:]})
    assert match(partial_age, target) is None
    extra = target.model_copy(
        update={"subgroup": target.subgroup + [Condition(variable="other", value=1)]}
    )
    assert match(extra, target) is None
    reverse = target.model_copy(update={"exposed": 0.0, "comparator": 1.0, "direction": -1})
    assert match(reverse, target) == "exact"
    wrong_direction = target.model_copy(update={"direction": -1})
    assert match(wrong_direction, target) is None
    assert match(wrong_direction, target, ignore_direction=True) == "exact"
    interaction = target.model_copy(update={"contrast": "interaction"})
    assert match(target, interaction) is None
    one_variable = hypothesis(subgroup=[Condition(variable="age", op="ge", value=50)])
    assert match(hypothesis(), one_variable) is None


def test_global_assignment_prioritizes_exact_and_deduplicates():
    h = hypothesis(subgroup=[Condition(variable="a", value=1), Condition(variable="b", value=1)])
    exact = Discovery(
        hypothesis=h, category="neutral", expected_direction=None, magnitude=1, evidence_id="e"
    )
    near = h.model_copy(update={"id": "near", "subgroup": h.subgroup[:1]})
    result = assign([near, h, h.model_copy(update={"id": "duplicate"})], [exact])
    assert len(result) == 1
    assert result[0]["match"] == "exact"


@pytest.mark.parametrize("profile", [str(c) for c in all_cancer_types()])
def test_all_profiles_pair_covariates_noise_and_only_one_component(profile):
    spec = pair(profile)
    expected = sample(spec, "expected", seed=100)
    surprising = sample(spec, "surprising", seed=100)
    outcome_names = {o.name for o in spec.outcomes}
    pd.testing.assert_frame_equal(
        expected.drop(columns=list(outcome_names)), surprising.drop(columns=list(outcome_names))
    )
    assert not any(c.startswith("__internal_") for c in expected)
    changed = [
        a.hypothesis.id
        for a, b in zip(
            version_discoveries(spec, "expected"),
            version_discoveries(spec, "surprising"),
            strict=True,
        )
        if a != b
    ]
    assert changed == [spec.focal_id]
    em = conditional_means(spec, expected, "expected")
    sm = conditional_means(spec, surprising, "surprising")
    for name in outcome_names:
        np.testing.assert_allclose(
            expected[name] - em[name], surprising[name] - sm[name], atol=1e-14
        )


def test_subgroup_flip_fixed_conditions_and_expected_overall():
    spec = pair(mode="subgroup")
    h = spec.discoveries[0].hypothesis
    assert h.subgroup == version_discoveries(spec, "surprising")[0].hypothesis.subgroup
    for version in ("expected", "surprising"):
        frame = sample(spec, version, seed=200, n=20000)
        overall = h.model_copy(update={"subgroup": []})
        assert estimate(frame, overall, delta=0.1, alpha=0.05, result_id="all").estimate > 0
        local = estimate(frame, h, delta=0.1, alpha=0.05, result_id="local").estimate
        assert local > 0 if version == "expected" else local < 0


def test_full_dgp_audit_detects_overlapping_reversal_contamination():
    spec = pair()
    # A fixed background component cancels the focal slope in the expected arm.
    other = spec.discoveries[3]
    other.hypothesis = spec.discoveries[0].hypothesis.model_copy(
        update={"id": "d3", "direction": -1}
    )
    other.magnitude = 2
    report = calibrate(spec, replicates=1, reference_n=5000)
    assert not report["criteria"]["all_discovery_directions"]
    assert not report["calibration_passed"]


def result(lower, upper, **kwargs):
    return EvidenceResult(
        id="r",
        hypothesis_id="h",
        estimate=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        delta=1,
        alpha=0.005,
        cell_n=[100, 100],
        valid=True,
        diagnostic="ok",
        **kwargs,
    )


def test_evidence_boundary_missing_decision_and_no_events():
    assert evidence_status(result(1.1, 2)) == "accept"
    assert evidence_status(result(-2, 0.9)) == "reject"
    assert evidence_status(result(1, 2)) == "unresolved"
    assert evidence_status(result(0, 1)) == "unresolved"
    assert responsiveness([])["accuracy"] is None
    scored = responsiveness([{"result": result(1.1, 2).model_dump(), "prior": "reject"}])
    assert scored["accuracy"] == 0
    assert scored["update_accuracy"] == 0


def test_interaction_estimates_difference_of_differences():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": np.tile([0, 1, 0, 1], 500), "s": np.tile([0, 0, 1, 1], 500)})
    frame["y"] = frame.x * (1 + 3 * frame.s) + rng.normal(0, 0.1, len(frame))
    h = hypothesis(
        outcome="y",
        exposure="x",
        contrast="interaction",
        subgroup=[Condition(variable="s", value=1)],
    )
    r = estimate(frame, h, delta=1, alpha=0.005, result_id="r")
    assert r.estimate == pytest.approx(3, abs=0.03)
    assert len(r.cell_n) == 4


def test_validation_freezes_requests_independence_budget_and_confirmation():
    spec = pair()
    h = spec.discoveries[0].hypothesis
    service = ValidationService(spec, "expected", "run-a")
    first = service.request(h, 1, "request-1")
    assert service.request(h, 1, "request-1") == first
    with pytest.raises(ValueError, match="revised"):
        service.request(h.model_copy(update={"direction": -1}), 1, "request-1")
    with pytest.raises(ValueError, match="budget"):
        service.request(h, 1, "request-2")
    second = service.request(h, 2, "request-2")
    assert first.estimate != second.estimate
    for i in range(3, 11):
        service.request(h, i, f"request-{i}")
    with pytest.raises(ValueError, match="budget"):
        service.request(h, 11, "request-11")
    final = service.confirm([h, h])
    assert len(final["results"]) == 1
    assert final["results"][0]["alpha"] == 0.05
    assert service.confirm([])["primary_recovery"] == 0


def test_public_bundle_does_not_disclose_assignment_or_answer_key(tmp_path):
    spec = pair()
    mapping = write_pair(spec, tmp_path)
    for version in ("expected", "surprising"):
        task = tmp_path / "public" / mapping[version]["task_id"]
        metadata = json.loads((task / "task.json").read_text())
        assert not {"discoveries", "focal_id", "version", "seed", "evidence"} & metadata.keys()
        assert "expected" not in task.name and "surprising" not in task.name
    with pytest.raises(FileExistsError):
        write_pair(spec, tmp_path)


def test_schema_rejects_fabricated_citation_and_invalid_ranges():
    r = reviewed(hypothesis())
    value = r.model_dump()
    value["review"]["source_ids"] = ["MED:999"]
    with pytest.raises(ValidationError, match="not retrieved"):
        ReviewedCandidate.model_validate(value)
    with pytest.raises(ValidationError, match="Reversed interval"):
        hypothesis(
            subgroup=[
                Condition(variable="a", op="ge", value=5),
                Condition(variable="a", op="le", value=2),
            ]
        )


def test_literature_rejection_and_replacement_are_recorded(tmp_path):
    good = [
        reviewed(hypothesis(id=f"h{i}", exposure=x), "expected" if i < 4 else "neutral")
        for i, x in enumerate(
            ["sex_female", "stage_iv", "has_brain_mets", "egfr_mutation", "kras_g12c", "alk_fusion"]
        )
    ]
    for item in good[4:]:
        item.candidate.hypothesis.subgroup = [
            Condition(variable="research_signature_a", op="ge", value=0),
            Condition(variable="research_signature_b", op="ge", value=0),
        ]
    bad = good[0].candidate.model_copy(deep=True)
    bad.hypothesis.id = "rejected"
    responses = [{"candidates": [bad.model_dump()]}]
    rejected_review = good[0].review.model_dump()
    rejected_review.update(decision="unsupported", rationale="Comparator absent")
    responses += [rejected_review, {"candidates": [g.candidate.model_dump() for g in good]}]
    responses += [
        response for g in good for response in (g.review.model_dump(), g.realism.model_dump())
    ]

    class Provider:
        model_id = "fixture"

        def chat(self, *args, **kwargs):
            return ChatResponse(text=json.dumps(responses.pop(0)), model_id="fixture", raw=None)

    research = Researcher(Provider(), tmp_path, search=lambda q: (q, good[0].sources))
    accepted = research.generate("nsclc_clinical", max_rounds=3)
    assert len(accepted) == 6
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "rejected" and "Comparator absent" in e["reason"] for e in events)
    assert all(r.candidate.hypothesis.id != "rejected" for r in accepted)


def test_full_rollout_links_exploration_evidence_decision_and_confirmation(tmp_path):
    from onc_co_scientist.expected_surprising.rollout import run
    from onc_co_scientist.expected_surprising.schemas import StageRecord

    spec = pair()
    assignment = write_pair(spec, tmp_path / "bundles")
    public = tmp_path / "bundles/public" / assignment["expected"]["task_id"]
    h = spec.discoveries[0].hypothesis

    class Provider:
        model_id = "scripted-test"

        def chat(self, messages, **kwargs):
            prompt = messages[0].content
            assert spec.pair_id not in prompt
            assert "Fixture only" not in prompt
            stage = next(
                s
                for s in ("hypothesis", "analysis", "critique", "synthesis")
                if f"stage {s}." in prompt
            )
            record = StageRecord(iteration=1, stage=stage)
            if stage == "hypothesis":
                record.hypotheses = [h]
                record.anticipated_directions = {h.id: -1}
                record.assessments = {h.id: "unresolved"}
            if stage == "analysis":
                record.executed_ids = [h.id]
                record.validation_request_id = h.id
            if stage == "critique":
                from onc_co_scientist.expected_surprising.schemas import Decision

                record.decisions = [
                    Decision(hypothesis_id=h.id, result_id=f"validation-1-{h.id}", status="accept")
                ]
            if stage == "synthesis":
                record.accepted_ids = [h.id]
            return ChatResponse(text=record.model_dump_json(), model_id=self.model_id, raw=None)

    report = run(
        spec, "expected", public, tmp_path / "run", Provider(), run_id="scripted", iterations=1
    )
    assert report["protocol_errors"] == []
    assert report["confirmation"]["primary_recovery"] == 1
    assert report["responsiveness"]["accuracy"] == 1
    assert report["exploration"][-1]["cumulative_tested"] == 1
    assert report["first_exact"][h.id] == {"iteration": 1, "stage": "hypothesis"}
    assert report["post_evidence_exploration"] == []  # No two-iteration follow-up window.


@pytest.mark.parametrize(
    "failure",
    ["unknown_variable", "empty", "invalid_json", "late_analysis_error", "wrong_decision"],
)
def test_stage_retry_corrects_errors_atomically_without_coaching_scores(
    tmp_path, monkeypatch, failure
):
    from collections import Counter

    from onc_co_scientist.expected_surprising import rollout
    from onc_co_scientist.expected_surprising.schemas import Decision, StageRecord

    spec = pair()
    assignment = write_pair(spec, tmp_path / "bundles")
    public = tmp_path / "bundles/public" / assignment["expected"]["task_id"]
    h = spec.discoveries[0].hypothesis
    unused = spec.discoveries[1].hypothesis.model_copy(update={"id": "uncommitted"})
    draws = []
    original_request = ValidationService.request

    def track_request(self, *args):
        result = original_request(self, *args)
        draws.append(result.model_dump())
        return result

    monkeypatch.setattr(ValidationService, "request", track_request)

    class Provider:
        model_id = "scripted-retry-test"

        def __init__(self):
            self.calls = Counter()

        def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 125000
            prompt = messages[0].content
            stage = next(
                s
                for s in ("hypothesis", "analysis", "critique", "synthesis")
                if f"stage {s}." in prompt
            )
            self.calls[stage] += 1
            attempt = self.calls[stage]
            if attempt == 2:
                feedback = json.loads(prompt.split("\n", 1)[1])["history"][-1]
                assert feedback["protocol_error"]["retrying"]
                assert feedback["protocol_error"]["stage"] == stage
                assert "complete replacement" in feedback["instruction"]
                if failure == "unknown_variable":
                    assert h.exposure + "_typo" in feedback["protocol_error"]["error"]
                    assert h.exposure in feedback["protocol_error"]["error"]
                    assert feedback["registered_hypothesis_ids"] == []
                    assert "uncommitted" in feedback["failed_response"]
                if failure == "late_analysis_error":
                    assert feedback["registered_hypothesis_ids"] == [h.id]
                    # The failed stage's independent evidence was never exposed.
                    assert "validation-1-" not in prompt
            record = StageRecord(iteration=1, stage=stage)
            if stage == "hypothesis":
                record.hypotheses = [h]
                record.anticipated_directions = {h.id: h.direction}
                record.assessments = {h.id: "unresolved"}
                if attempt == 1:
                    if failure == "empty":
                        return ChatResponse(text="", model_id=self.model_id, raw=None)
                    if failure == "invalid_json":
                        return ChatResponse(text="{broken", model_id=self.model_id, raw=None)
                    if failure == "unknown_variable":
                        record.hypotheses = [
                            unused,
                            h.model_copy(update={"exposure": h.exposure + "_typo"}),
                        ]
                        record.anticipated_directions[unused.id] = unused.direction
                        record.assessments[unused.id] = "unresolved"
            if stage == "analysis":
                record.executed_ids = [h.id]
                record.validation_request_id = h.id
                if failure == "late_analysis_error" and attempt == 1:
                    record.decisions = [
                        Decision(
                            hypothesis_id=h.id, result_id="nonexistent-result", status="accept"
                        )
                    ]
            if stage == "critique":
                record.decisions = [
                    Decision(
                        hypothesis_id=h.id,
                        result_id=f"validation-1-{h.id}",
                        status="reject" if failure == "wrong_decision" else "accept",
                    )
                ]
            if stage == "synthesis":
                record.accepted_ids = [h.id]
            return ChatResponse(text=record.model_dump_json(), model_id=self.model_id, raw=None)

    provider = Provider()
    report = rollout.run(
        spec,
        "expected",
        public,
        tmp_path / "run",
        provider,
        run_id="scripted-retry",
        iterations=1,
        max_retries_per_stage=2,
    )
    expected_errors = int(failure != "wrong_decision")
    assert sum(provider.calls.values()) == 4 + expected_errors
    assert len(report["attempt_errors"]) == expected_errors
    assert len(report["recovered_stages"]) == expected_errors
    assert report["protocol_errors"] == []
    assert len(report["exploration"]) == 4
    assert report["exploration"][-1]["cumulative_proposed"] == 1
    assert report["exploration"][-1]["cumulative_tested"] == 1
    assert report["responsiveness"]["valid_n"] == 1
    assert report["responsiveness"]["accuracy"] == int(failure != "wrong_decision")
    assert report["confirmation"]["primary_recovery"] == 1
    assert report["confirmation"]["first_attempt_primary_recovery"] == 1 - expected_errors
    assert provider.calls["critique"] == 1
    if failure == "late_analysis_error":
        assert len(draws) == 2 and draws[0] == draws[1]
    else:
        assert len(draws) == 1


def test_stage_retry_exhaustion_discards_partial_registration(tmp_path):
    from collections import Counter

    from onc_co_scientist.expected_surprising.rollout import run
    from onc_co_scientist.expected_surprising.schemas import StageRecord

    spec = pair()
    assignment = write_pair(spec, tmp_path / "bundles")
    public = tmp_path / "bundles/public" / assignment["expected"]["task_id"]
    h = spec.discoveries[0].hypothesis
    bad = spec.discoveries[1].hypothesis.model_copy(
        update={"subgroup": [Condition(variable="nonexistent_condition", value=1)]}
    )

    class Provider:
        model_id = "scripted-exhaustion-test"

        def __init__(self):
            self.calls = Counter()

        def chat(self, messages, **kwargs):
            prompt = messages[0].content
            stage = next(
                s
                for s in ("hypothesis", "analysis", "critique", "synthesis")
                if f"stage {s}." in prompt
            )
            self.calls[stage] += 1
            record = StageRecord(iteration=1, stage=stage)
            if stage == "hypothesis":
                record.hypotheses = [h, bad]
                record.anticipated_directions = {x.id: x.direction for x in [h, bad]}
                record.assessments = {x.id: "unresolved" for x in [h, bad]}
            if stage == "analysis":
                feedback = json.loads(prompt.split("\n", 1)[1])["history"][-1]
                assert not feedback["protocol_error"]["retrying"]
                assert feedback["registered_hypothesis_ids"] == []
            return ChatResponse(text=record.model_dump_json(), model_id=self.model_id, raw=None)

    provider = Provider()
    report = run(
        spec,
        "expected",
        public,
        tmp_path / "run",
        provider,
        run_id="scripted-exhaustion",
        iterations=1,
        max_retries_per_stage=1,
    )
    assert provider.calls["hypothesis"] == 2
    assert sum(provider.calls.values()) == 5
    assert len(report["attempt_errors"]) == 2
    assert len(report["protocol_errors"]) == 1
    assert report["recovered_stages"] == []
    assert report["exploration"][-1]["cumulative_proposed"] == 0
    assert report["responsiveness"]["valid_n"] == 0
    assert report["confirmation"]["primary_recovery"] == 0


def test_paired_summary_uses_datasets_and_rejects_missing_conditions():
    from onc_co_scientist.expected_surprising.summary import paired_summary

    reports = []
    for pair_id, successes in (("a", (1, 0)), ("b", (1, 1))):
        for version, success in zip(("expected", "surprising"), successes, strict=True):
            for repeat in range(4):
                reports.append(
                    {
                        "pair_id": pair_id,
                        "profile": "nsclc_clinical",
                        "model": "test",
                        "harness": "test",
                        "version": version,
                        "run_id": f"r{repeat}",
                        "confirmation": {"primary_recovery": success},
                    }
                )
    summary = paired_summary(reports, bootstrap_replicates=100)
    assert summary["paired_difference"] == -0.5
    assert summary["base_dataset_n"] == 2
    assert summary["run_n"] == 16
    with pytest.raises(ValueError, match="both versions"):
        paired_summary(reports[:4])


def test_variable_category_mix_preserves_six_discoveries_and_one_reversal():
    original = pair()
    inventory = original.evidence[:3] + original.evidence[4:]
    extra = reviewed(
        hypothesis(
            id="neutral-extra",
            exposure="egfr_mutation",
            outcome="new_outcome",
            subgroup=[
                Condition(variable="research_signature_a", op="ge", value=0),
                Condition(variable="research_signature_c", op="ge", value=0),
            ],
        ),
        "neutral",
    )
    spec = compile_pair("nsclc_clinical", [*inventory, extra], seed=44, n=2000, expected_count=3)
    assert len(spec.discoveries) == 6
    assert sum(d.category == "neutral" for d in spec.discoveries) == 3
    reversed_discoveries = version_discoveries(spec, "surprising")
    assert sum(a != b for a, b in zip(spec.discoveries, reversed_discoveries, strict=True)) == 1


def test_expected_direction_is_distinct_from_claim_direction_and_dgp_adjudication():
    from onc_co_scientist.expected_surprising.evaluation import adjudicate
    from onc_co_scientist.expected_surprising.scoring import evidence_alignment

    event = {
        "result": result(1.1, 2).model_dump(),
        "claim_direction": 1,
        "anticipated_direction": -1,
    }
    assert evidence_alignment(event) == "opposes_expectation"
    frame = pd.DataFrame({"sex_female": [0, 1] * 100, "log_pfs_months": [0.0, 2.0] * 100})
    assert adjudicate(frame, hypothesis(), delta=1)["dgp_status"] == "supported"
    assert adjudicate(frame, hypothesis(direction=-1), delta=1)["dgp_status"] == "contradicted"


def test_independent_realism_vetoes_overbroad_drug_claim(tmp_path):
    candidate = reviewed(hypothesis(exposure="treatment_osimertinib"))
    bad = realism(
        decision="reject",
        biomarker_restrictions_complete=False,
        required_changes=["Restrict to sensitizing EGFR mutations and an explicit control regimen"],
        rationale="All-comer benefit extrapolated from a biomarker-selected trial",
    )
    responses = [candidate.review.model_dump(), bad.model_dump()]

    class Provider:
        model_id = "fixture"

        def chat(self, messages, **kwargs):
            if len(responses) == 1:
                # The independent critic must not inherit the first review's verdict.
                assert '"decision": "supported"' not in messages[0].content
            return ChatResponse(text=json.dumps(responses.pop(0)), model_id="fixture")

    researcher = Researcher(Provider(), tmp_path, search=lambda q: (q, candidate.sources))
    researcher.profile = "nsclc_clinical"
    with pytest.raises(ValueError, match="Realism review rejected"):
        researcher.review(candidate.candidate)
    assert not responses


def test_neutral_inherits_parent_direction_and_primary_evidence_required():
    item = reviewed(hypothesis(), "neutral")
    item.review.broader_direction = "negative"
    with pytest.raises(ValueError, match="inherited"):
        require_current_candidate(item, "nsclc_clinical")
    item = reviewed(hypothesis())
    item.review.evidence_scopes[0].primary_empirical = False
    with pytest.raises(ValueError, match="primary empirical"):
        require_current_candidate(item, "nsclc_clinical")


def test_current_review_cannot_be_reused_for_changed_claim_or_dgp(tmp_path):
    spec = pair()
    require_current_pair(spec)
    compact = spec.model_copy(deep=True)
    compact.evidence[0].sources[0].abstract = ""
    require_current_pair(compact)  # Frozen citation-only specifications remain replayable.
    spec.evidence[0].candidate.hypothesis.eligibility = [
        Condition(variable="egfr_mutation", value=1)
    ]
    with pytest.raises(ValueError, match="changed after review"):
        require_current_candidate(spec.evidence[0], spec.profile)
    spec = pair()
    spec.discoveries[1].magnitude *= 2
    with pytest.raises(ValueError, match="DGP changed"):
        write_pair(spec, tmp_path)
    assert not (tmp_path / "private").exists()
    spec = pair()
    spec.dgp_review.assessment.expected_components_scoped = False
    with pytest.raises(ValueError, match="rejected"):
        require_current_pair(spec)


def test_legacy_review_is_blocked_and_nested_target_is_redundant():
    item = reviewed(hypothesis(exposure="treatment_osimertinib"))
    nested = item.candidate.model_copy(deep=True)
    nested.hypothesis.eligibility = [Condition(variable="egfr_mutation", value=1)]
    with pytest.raises(ValueError, match="Redundant"):
        reject_nested_comparison(nested, [item])
    item.review_policy_version = "legacy"
    with pytest.raises(ValueError, match="current literature and realism"):
        require_current_candidate(item, "nsclc_clinical")


def test_realism_requirements_override_a_false_positive_acceptance():
    item = reviewed(hypothesis(exposure="treatment_osimertinib"))
    item.realism.necessary_scope_conditions = [Condition(variable="egfr_mutation", value=1)]
    with pytest.raises(ValueError, match="necessary restriction is not encoded"):
        require_current_candidate(item, "nsclc_clinical")
    item.candidate.hypothesis.eligibility = [Condition(variable="egfr_mutation", value=1)]
    item.candidate_sha256 = candidate_digest(item.candidate)
    require_current_candidate(item, "nsclc_clinical")
    item.realism.unrepresented_scope_requirements = ["Explicit active control regimen"]
    with pytest.raises(ValueError, match="unrepresented"):
        require_current_candidate(item, "nsclc_clinical")
    item.realism.unrepresented_scope_requirements = []
    item.review.evidence_scopes[0].endpoint_kind = "recurrence_free"
    with pytest.raises(ValueError, match="primary empirical"):
        require_current_candidate(item, "nsclc_clinical")


def test_new_contrast_encodings_preserve_legacy_covariates():
    old = base_frame("nsclc_clinical", 2000, 17)
    new = base_frame("nsclc_clinical", 2000, 17, version=2)
    pd.testing.assert_frame_equal(old, new[old.columns])
    for column, expected in {
        "ecog_ps_ge_2": new.ecog_ps >= 2,
        "crp_mg_l_ge_10": new.crp_mg_l >= 10,
        "albumin_g_dl_le_3_5": new.albumin_g_dl <= 3.5,
        "nlr_ge_3": new.nlr >= 3,
    }.items():
        np.testing.assert_array_equal(new[column], expected.astype(int))
    assert not any(c.startswith("research_marker_") for c in old)


def test_reference_grammar_counts_and_recovers_scoped_comparisons():
    import itertools

    frame = pd.DataFrame(
        {
            "sex_female": [0, 1],
            "stage_iv": [0, 1],
            "egfr_mutation": [0, 1],
            "research_signature_a": [-0.2, 0.7],
            "research_signature_b": [-0.2, 0.7],
        }
    )
    universe = ReferenceUniverse(frame, ["log_pfs_months"])
    enumerated = []
    for exposure in universe.binary:
        others = sorted(universe.binary - {exposure})
        for levels in itertools.product((None, 0, 1), repeat=len(others)):
            eligibility = [
                Condition(variable=c, value=v)
                for c, v in zip(others, levels, strict=True)
                if v is not None
            ]
            for cuts in [None, *itertools.product((0, 0.5), repeat=2)]:
                subgroup = (
                    []
                    if cuts is None
                    else [
                        Condition(variable=c, op="ge", value=v)
                        for c, v in zip(sorted(universe.signatures), cuts, strict=True)
                    ]
                )
                enumerated.append(
                    hypothesis(exposure=exposure, eligibility=eligibility, subgroup=subgroup)
                )
    assert universe.size == len(enumerated) == 135
    assert all(universe.contains(h) for h in enumerated)
    scoped = hypothesis(eligibility=[Condition(variable="stage_iv", value=1)])
    assert universe.contains(scoped)
    assert universe.contains(scoped.model_copy(update={"exposed": 0, "comparator": 1}))
    assert not universe.contains(
        hypothesis(eligibility=[Condition(variable="stage_iv", op="ge", value=0.5)])
    )
    assert not universe.contains(
        hypothesis(subgroup=[Condition(variable="research_signature_a", op="ge", value=0)])
    )
    spec = pair()
    spec.discoveries[0].hypothesis.eligibility = [Condition(variable="stage_iv", value=1)]
    assert calibrate(spec, replicates=0, reference_n=5000)["criteria"]["reference_supported"]


def test_dgp_critic_receives_fixed_and_focal_intentional_reversals(tmp_path):
    spec = pair()

    class Provider:
        model_id = "fixture"

        def chat(self, messages, **kwargs):
            payload = json.loads(messages[0].content.split("\n", 1)[1])
            reversals = payload["intentional_reversals"]
            assert len(reversals["expected"]) == 1
            assert reversals["expected"][0]["role"] == "fixed_background"
            assert reversals["expected"][0] in reversals["surprising"]
            assert {r["role"] for r in reversals["surprising"]} == {"focal", "fixed_background"}
            return ChatResponse(text=dgp_assessment().model_dump_json(), model_id="fixture")

    require_current_pair(Researcher(Provider(), tmp_path).review_dgp(spec))


@pytest.mark.parametrize("profile", ["nsclc_clinical", "nsclc_depmap"])
def test_stale_checkpoints_are_rechecked_in_both_modalities(tmp_path, profile):
    item = reviewed(hypothesis(), profile=profile)
    item.review_policy_version = "legacy"
    (tmp_path / f"{profile}_reviewed.json").write_text(json.dumps([item.model_dump()]))

    class Provider:
        model_id = "fixture"

    class RejectingResearcher(Researcher):
        def review(self, candidate):
            raise ValueError("Missing population restriction")

    with pytest.raises(RuntimeError, match="Insufficient"):
        RejectingResearcher(Provider(), tmp_path).generate(profile, max_rounds=0)
    assert json.loads((tmp_path / f"{profile}_reviewed.json").read_text()) == []
    assert "Missing population restriction" in (tmp_path / "events.jsonl").read_text()
