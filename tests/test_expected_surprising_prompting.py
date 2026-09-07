"""The simpler interface must preserve scientific decisions and evidence accounting."""

import copy
import json

import pytest
from pydantic import ValidationError

from onc_co_scientist.expected_surprising.generation import sample
from onc_co_scientist.expected_surprising.prompting import (
    FORMS,
    build_prompt,
    ledger,
    references,
    translate,
)
from onc_co_scientist.expected_surprising.rollout import stage_transaction
from onc_co_scientist.expected_surprising.schemas import ValidationPolicy
from onc_co_scientist.expected_surprising.workflow import WorkflowController
from tests.test_expected_surprising import pair


@pytest.fixture
def controller():
    spec = pair()
    return WorkflowController(
        spec,
        "expected",
        sample(spec, "expected", seed=123),
        ValidationPolicy.default(spec.profile, 6),
        "simple-interface",
    )


def proposal(controller, index=0, **updates):
    h = controller.spec.discoveries[index].hypothesis
    return {
        **h.model_dump(exclude={"id"}),
        "anticipated_direction": h.direction,
        "initial_status": "unresolved",
        **updates,
    }


def apply(controller, stage, data, iteration=1):
    with stage_transaction([controller.state, controller.service.state]):
        record, form, audit = translate(controller, stage, iteration, data)
        event = controller.apply(record)
    return record, form, audit, event


def seed_analysis(controller):
    apply(controller, "explore", {"proposals": [proposal(controller)]})
    apply(controller, "analyze", {"run_analyses": ["H1"]})


def test_forms_remove_redundancy_and_forbidden_actions():
    for stage, form in FORMS.items():
        fields = form.model_json_schema()["properties"]
        assert (
            not {"iteration", "stage", "accepted_ids", "executed_ids", "decisions"} & fields.keys()
        )
        assert ("run_analyses" in fields) == (stage == "analyze")
        assert ("validate" in fields) == (stage == "appraise")
    with pytest.raises(ValidationError):
        FORMS["synthesize"].model_validate({"run_analyses": ["H1"]})


def test_controller_attaches_evidence_and_acceptance_without_validation(controller):
    seed_analysis(controller)
    record, _, audit, _ = apply(
        controller,
        "appraise",
        {"assessments": [{"claim": "H1", "status": "accept", "investigation": "closed"}]},
    )
    assert record.assessment_records[0].result_ids == list(controller.state["results"])
    assert audit["assessment_links"][0]["mode"] == "displayed_claim_evidence"
    assert not controller.service.state["requests"]
    record, *_ = apply(controller, "synthesize", {})
    assert record.accepted_ids == ["H1"]
    record, *_ = apply(
        controller,
        "synthesize",
        {"assessments": [{"claim": "H1", "status": "reject", "investigation": "closed"}]},
        2,
    )
    assert record.accepted_ids == []


def test_validation_request_and_deadline_auto_links(controller):
    seed_analysis(controller)
    record, *_ = apply(
        controller,
        "appraise",
        {
            "assessments": [{"claim": "H1", "status": "unresolved", "investigation": "active"}],
            "validate": "H1",
        },
    )
    assert len(record.validation_request.triggering_result_ids) == 1
    assert len(controller.service.state["cache"]) == 1
    for iteration in (1, 3):
        record, *_ = apply(
            controller,
            "synthesize",
            {"assessments": [{"claim": "H1", "status": "accept", "investigation": "closed"}]},
            iteration,
        )
        assert len(record.assessment_records[0].result_ids) == 2
    validation = next(e for e in controller.state["events"] if e["source"] != "discovery")
    assert validation["immediate"]["status"] == validation["delayed"]["status"] == "accept"


def test_reversal_shows_oriented_evidence_and_preserves_initial_exposure(controller):
    seed_analysis(controller)
    before = ledger(controller)[0]["evidence"][0]["estimate"]
    p = proposal(controller, direction=-controller.state["hypotheses"]["H1"].direction, parent="H1")
    apply(controller, "explore", {"proposals": [p]}, 2)
    rows = ledger(controller)
    assert rows[1]["evidence"][0]["estimate"] == -before
    assert rows[0]["evidence"][0]["ref"] == rows[1]["evidence"][0]["ref"] == "R1"
    assert not rows[1]["initial_expectation"]["pre_evidence"]
    assert controller.state["registrations"][1]["motivating_result_ids"] == list(
        controller.state["results"]
    )


def test_failed_attempt_does_not_allocate_references_or_mutate_state(controller):
    seed_analysis(controller)
    before = copy.deepcopy(controller.state)
    refs = references(controller)
    with pytest.raises(ValueError, match="Required assessment missing"):
        apply(controller, "appraise", {"proposals": [proposal(controller, 1)]})
    assert controller.state == before
    assert references(controller) == refs
    with pytest.raises(ValueError, match="Unknown evidence"):
        apply(
            controller,
            "appraise",
            {
                "assessments": [
                    dict(claim="H1", status="accept", investigation="closed", evidence=["R999"])
                ]
            },
        )
    assert controller.state == before


def test_required_new_evidence_cannot_be_omitted_explicitly(controller):
    seed_analysis(controller)
    apply(
        controller,
        "appraise",
        {
            "assessments": [dict(claim="H1", status="accept", investigation="active")],
            "validate": "H1",
        },
    )
    with pytest.raises(ValueError, match="Required assessment missing"):
        apply(
            controller,
            "synthesize",
            {
                "assessments": [
                    dict(claim="H1", status="accept", investigation="active", evidence=["R1"])
                ]
            },
        )


def test_prompt_projects_public_state_not_history_or_private_rules(controller):
    seed_analysis(controller)
    prompt = build_prompt(
        controller,
        {"instructions": "Scientific judgment"},
        1,
        6,
        "appraise",
        2,
        {"notes": "Investigate a reversal", "narratives": {"explore": "prior reasoning"}},
        {"error": "Use an existing claim"},
    )
    payload = json.loads(prompt[prompt.index('{"schema":') :])
    assert "history" not in payload
    assert payload["research_notes"] == "Investigate a reversal"
    assert payload["assessments_due"] == [dict(claim="H1", evidence="R1", reason="new_result")]
    assert not any(
        x in prompt
        for x in (
            "analysis-",
            "validation-",
            "delta",
            "target_id",
            "desired_stratum",
            "supported",
            "accepted_ids",
        )
    )
    assert payload["claims"][0]["evidence"][0]["valid"]


def test_legacy_and_compact_interfaces_preserve_scripted_scientific_outcomes(tmp_path):
    from onc_co_scientist.expected_surprising.generation import write_pair
    from onc_co_scientist.expected_surprising.packaging import repackage
    from tests.test_expected_surprising_workflow import scripted_run

    spec = pair()
    write_pair(spec, tmp_path / "old")
    manifest = repackage(tmp_path / "old", tmp_path / "new")
    packages = spec, tmp_path / "new", manifest
    compact, _ = scripted_run(packages, tmp_path / "compact", "follows")
    legacy_versions = dict(
        compact["versions"],
        workflow="appraisal-3.1.0",
        prompt="exploration-3.1.0",
        schema_version="stage-3.0.0",
    )
    for path in (tmp_path / "new" / "public").glob("*/task.json"):
        data = json.loads(path.read_text())
        data["versions"] = legacy_versions
        path.write_text(json.dumps(data))
    legacy, _ = scripted_run(packages, tmp_path / "legacy", "follows")
    assert not compact["protocol_errors"] and not legacy["protocol_errors"]
    # Only the controller-assigned public hypothesis labels differ.
    compact_scores = copy.deepcopy(compact["scores"])
    legacy_scores = copy.deepcopy(legacy["scores"])
    for scores in (compact_scores, legacy_scores):
        for match in scores["discovery"]["confirmed_matches"]:
            match.pop("hypothesis_id")
    assert compact_scores == legacy_scores
    assert compact["responsiveness"]["components"] == legacy["responsiveness"]["components"]
    assert len(compact["validation"]["requests"]) == len(legacy["validation"]["requests"])


def test_notebook_and_repair_context_commit_only_successful_responses(tmp_path):
    from onc_co_scientist.expected_surprising.generation import write_pair
    from onc_co_scientist.expected_surprising.packaging import repackage
    from onc_co_scientist.expected_surprising.rollout import run
    from onc_co_scientist.providers.base import ChatResponse
    from tests.test_expected_surprising_workflow import ScriptedScientist

    spec = pair()
    write_pair(spec, tmp_path / "old")
    manifest = repackage(tmp_path / "old", tmp_path / "new")

    class Scientist(ScriptedScientist):
        prompts = []

        def chat(self, messages, **kwargs):
            self.prompts.append(messages[0].content)
            result = super().chat(messages, **kwargs)
            form = json.loads(result.text)
            if self.calls == 1:
                form["research_notes"] = "FAILED NOTE MUST NOT PERSIST"
            if self.calls == 2:
                form["research_notes"] = "Remember the longer-term research plan"
            return ChatResponse(text=json.dumps(form), model_id=self.model_id)

    scientist = Scientist(spec, repair=True)
    report = run(
        spec,
        "expected",
        tmp_path / "new/public" / manifest["tasks"][0]["task_id"],
        tmp_path / "run",
        scientist,
        run_id="memory",
        iterations=6,
    )
    assert not report["protocol_errors"]
    assert all("FAILED NOTE MUST NOT PERSIST" not in p for p in scientist.prompts)
    contexts = [json.loads(p[p.index('{"schema":') :]) for p in scientist.prompts]
    assert "repair" in contexts[1]
    assert "repair" not in contexts[2]
    assert all(
        c["research_notes"] == "Remember the longer-term research plan" for c in contexts[2:]
    )


@pytest.mark.parametrize(
    "kind,constructor",
    [
        ("vllm_openai", "VLLMProvider"),
        ("gemini_vertex", "GeminiVertexProvider"),
        ("anthropic_vertex", "AnthropicVertexProvider"),
        ("codex_cli", "CodexCLIProvider"),
    ],
)
def test_compact_workflow_is_selected_by_package_for_every_provider(
    kind, constructor, tmp_path, monkeypatch
):
    from onc_co_scientist.expected_surprising.generation import write_pair
    from onc_co_scientist.expected_surprising.packaging import repackage
    from onc_co_scientist.expected_surprising.rollout import run
    from onc_co_scientist.providers import registry
    from tests.test_expected_surprising_workflow import ScriptedScientist

    spec = pair()
    write_pair(spec, tmp_path / "old")
    manifest = repackage(tmp_path / "old", tmp_path / "new")

    class Transport(ScriptedScientist):
        def chat(self, messages, **kwargs):
            prompt = messages[0].content
            context = json.loads(prompt[prompt.index('{"schema":') :])
            assert "claims" in context and "history" not in context
            assert "accepted_ids" not in context["schema"]["properties"]
            return super().chat(messages, **kwargs)

    transport = Transport(spec)
    # Mock network construction only; use the real registry and common workflow.
    monkeypatch.setattr(registry, constructor, lambda config: transport)
    provider = registry.get_provider({"kind": kind, "model_id": "transport-test"})
    report = run(
        spec,
        "expected",
        tmp_path / "new/public" / manifest["tasks"][0]["task_id"],
        tmp_path / "run",
        provider,
        run_id=kind,
        iterations=6,
    )
    assert report["successful_stages"] == 24
    assert report["versions"]["prompt"] == "ledger-1.0.0"
    assert not report["protocol_errors"]
