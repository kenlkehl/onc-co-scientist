"""Regression checks for explicit peer fallback and non-scientific bookkeeping."""

from types import SimpleNamespace

import pytest

from onc_co_scientist.expected_surprising.coordination import StageCoordinator
from onc_co_scientist.expected_surprising.generation import sample
from onc_co_scientist.expected_surprising.rollout import stage_transaction
from onc_co_scientist.expected_surprising.schemas import (
    Assessment,
    ValidationPolicy,
    WorkflowStageRecord,
)
from onc_co_scientist.expected_surprising.scoring import claim_key
from onc_co_scientist.expected_surprising.workflow import (
    WorkflowController,
    WorkflowInfrastructureError,
    run_workflow,
)
from onc_co_scientist.harness.experiment import ResourceBudget, WorkflowSpec, default_stages
from onc_co_scientist.providers.base import ChatResponse
from tests.test_expected_surprising import pair
from tests.test_expected_surprising_workflow import ScriptedScientist, packages  # noqa: F401


def controller():
    spec = pair()
    c = WorkflowController(
        spec,
        "expected",
        sample(spec, "expected", seed=2),
        ValidationPolicy.default(spec.profile, 6),
        "bookkeeping",
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
    return c, h


def test_investigation_edit_does_not_invent_an_evidence_response():
    c, h = controller()
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="explore",
            assessment_records=[
                Assessment(
                    hypothesis_id=h.id, status="unresolved", investigation="deferred", result_ids=[]
                )
            ],
        )
    )
    assert c.state["current"][claim_key(h)] == {"status": "unresolved", "investigation": "deferred"}
    assert c.state["assessments"] == []
    assert c.state["bookkeeping"][-1]["kind"] == "investigation_only"
    with pytest.raises(ValueError, match="requires available"), stage_transaction([c.state]):
        c.apply(
            WorkflowStageRecord(
                iteration=1,
                stage="explore",
                assessment_records=[
                    Assessment(
                        hypothesis_id=h.id,
                        status="accept",
                        investigation="closed",
                        result_ids=[],
                    )
                ],
            )
        )


def test_bookkeeping_cannot_satisfy_a_required_evidence_response():
    c, h = controller()
    c.apply(WorkflowStageRecord(iteration=1, stage="analyze", executed_ids=[h.id]))
    with pytest.raises(ValueError, match="Required assessment missing"):
        c.apply(
            WorkflowStageRecord(
                iteration=1,
                stage="appraise",
                assessment_records=[
                    Assessment(
                        hypothesis_id=h.id,
                        status="unresolved",
                        investigation="active",
                        result_ids=[],
                    )
                ],
            )
        )


def test_duplicate_proposal_preserves_belief_and_records_reconciliation():
    c, h = controller()
    alias = h.model_copy(update={"id": "alias"})
    c.apply(
        WorkflowStageRecord(
            iteration=1,
            stage="explore",
            hypotheses=[alias],
            anticipated_directions={alias.id: 1},
            assessments={alias.id: "accept"},
        )
    )
    assert c.state["current"][claim_key(h)]["status"] == "unresolved"
    assert c.state["expectations"][alias.id]["assessment"] == "unresolved"
    assert c.state["first_acceptances"] == {}
    assert c.state["bookkeeping"][-1]["requested_status"] == "accept"


@pytest.mark.parametrize("bad_peers", [{1}, {1, 2}])
def test_chair_fallback_is_explicit_cached_and_still_requires_chair(bad_peers, tmp_path):
    class Provider:
        model_id = "fixture"

        def chat(self, messages, **kwargs):
            # Count independent calls; the chair receives explicit missing-peer notices.
            if "independent peer" in kwargs["system"]:
                self.peer_calls += 1
                peer = 1 if self.peer_calls <= 3 else 2
                if peer in bad_peers:
                    raise RuntimeError("transport failure")
            else:
                self.chair_systems.append(kwargs["system"])
            return ChatResponse(
                text='{"narrative":"available evidence only"}', model_id=self.model_id
            )

        peer_calls = 0
        chair_systems = []

    provider = Provider()
    source = SimpleNamespace(
        max_tokens_per_call=100,
        max_retries_per_stage=2,
        persistent_history_chars=None,
        peer_failure_policy="chair_with_available",
    )
    args = (
        provider,
        WorkflowSpec(id="d", mode="deliberative", agents_per_stage=2),
        default_stages(),
        source,
        ResourceBudget(max_agent_calls=30),
        tmp_path / "calls",
    )
    c = StageCoordinator(*args)
    c.respond("prompt", iteration=1, stage="explore", attempt=1)
    n = provider.peer_calls
    c.reject()
    c.respond("repair", iteration=1, stage="explore", attempt=2)
    assert provider.peer_calls == n  # Failed drafts do not incur another hidden retry budget.
    c.commit()
    audit = c.audit()
    assert len(audit["peer_fallbacks"]) == len(bad_peers)
    assert audit["degraded_stages"] == ["i001-explore"]
    assert audit["authoritative_candidates"] == 2
    assert all('"unavailable": true' in p for p in provider.chair_systems)
    # A durable replay reproduces the same chair input and fallback metadata.
    replay = StageCoordinator(*args)
    replay.respond("prompt", iteration=1, stage="explore", attempt=1)
    assert replay.audit()["peer_fallbacks"] == audit["peer_fallbacks"]
    (tmp_path / "calls/i001-explore-peer1-r1-a1.json").write_text("{}")
    with pytest.raises(WorkflowInfrastructureError):
        StageCoordinator(*args).respond("prompt", iteration=1, stage="explore", attempt=1)


def test_failed_late_stage_retains_discoveries_only_under_explicit_policy(packages, tmp_path):  # noqa: F811
    spec, root, manifest = packages
    public = root / "public" / manifest["tasks"][0]["task_id"]

    class LateFailure(ScriptedScientist):
        def chat(self, messages, **kwargs):
            if messages[-1].content.startswith("Iteration 6/6, stage synthesize"):
                raise RuntimeError("last stage transport failure")
            return super().chat(messages, **kwargs)

    results = []
    for policy in ("zero_run", "retain_scientific_scores"):
        results.append(
            run_workflow(
                spec,
                "expected",
                public,
                tmp_path / policy,
                LateFailure(spec, "automatic"),
                run_id=policy,
                replicate_id="same-scientific-seed",
                iterations=6,
                stage_failure_policy=policy,
            )
        )
    old, new = results
    assert old["protocol_errors"] and new["protocol_errors"]
    assert old["state"] == new["state"]
    assert old["scores"]["D"] == 0
    assert new["scores"]["D"] > 0
    assert new["scores"]["D"] == old["scores"]["discovery"]["exact"]["diagnostic_D"]
    assert new["scores"]["discovery"]["exact"]["failure_penalized_D"] == 0
    assert new["completion_rate"] < 1
