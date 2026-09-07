"""Small agent-facing forms; translate them to the unchanged controller audit records.

Only public state is projected here. References are deterministic views of committed
state, so retries cannot allocate IDs or alter evidence/validation accounting.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from .schemas import (
    Assessment,
    Condition,
    Hypothesis,
    Status,
    StrictModel,
    ValidationRequest,
    WorkflowStageRecord,
)
from .scoring import claim_key, comparison_key, oriented


class Proposal(StrictModel):
    outcome: str
    exposure: str
    exposed: str | float = 1.0
    comparator: str | float = 0.0
    contrast: Literal["mean_difference", "interaction"] = "mean_difference"
    eligibility: list[Condition] = Field(default_factory=list)
    subgroup: list[Condition] = Field(default_factory=list)
    direction: Literal[-1, 1]
    anticipated_direction: Literal[-1, 0, 1]
    initial_status: Status
    parent: str | None = None
    motivating_evidence: list[str] | None = None


class Judgment(StrictModel):
    claim: str
    status: Status
    investigation: Literal["active", "deferred", "closed"]
    evidence: list[str] | None = None


class ResearchForm(StrictModel):
    proposals: list[Proposal] = Field(default_factory=list)
    assessments: list[Judgment] = Field(default_factory=list)
    narrative: str = ""
    research_notes: str | None = None


class ExploreForm(ResearchForm):
    pass


class AnalyzeForm(ResearchForm):
    run_analyses: list[str] = Field(default_factory=list, max_length=12)


class AppraiseForm(ResearchForm):
    validation_claim: str | None = Field(default=None, alias="validate")


class SynthesizeForm(ResearchForm):
    pass


FORMS = dict(
    explore=ExploreForm, analyze=AnalyzeForm, appraise=AppraiseForm, synthesize=SynthesizeForm
)

COMMON = """You are investigating the supplied research dataset. Return one JSON object using
only this stage's response form. The service performs analyses; you choose the science.
Use your judgment about effect size, uncertainty, and conclusions. Acceptance does not require
independent validation first. Explain your reasoning in narrative.

The ledger contains every registered claim, its current assessment, and all its available direct
evidence. References H1, H2, ... identify claims; R1, R2, ... identify evidence. Copy these short
references when needed. New proposals receive references automatically after this stage. A changed
comparison or reversed claim is a new proposal, optionally linked to an existing parent.
Proposal direction is the claim (+1 positive, -1 negative); anticipated_direction is your
expectation
of the raw exposed-minus-comparator contrast before seeing evidence (-1, 0 uncertain, +1).
initial_status is your assessment at registration. These initial judgments are recorded once.

For each claim in assessments_due, return one assessment with claim,
status (accept/reject/unresolved),
and investigation (active/deferred/closed). You may also reassess other claims. Omitted claims keep
their current assessment. Assess the original claim even when proposing a refinement or reversal.
By omitting evidence in an assessment, you assess all evidence displayed on that claim's card;
the controller attaches those references. To specify a subset or related evidence, provide evidence
with short R references; include the evidence listed in assessments_due for that claim.
For a refinement, parent identifies the earlier claim; omitted motivating_evidence attaches the
parent's displayed evidence. These automatic links record evidence supplied for your assessment
or refinement, not proof that you discussed or understood every result.

The controller maintains the accepted set from your assessments. Do not reconstruct it. Optional
research_notes replaces your persistent notebook; omit it to keep the existing notes. Use it for
plans or reasoning worth retaining. Latest stage narratives are also retained. All earlier numerical
evidence remains in the ledger. Do not repeat completed actions or old responses.
Omit unused fields.
"""

STAGE_INSTRUCTIONS = {
    "explore": "Propose scientifically useful new comparisons or refinements for later analysis.",
    "analyze": "Select up to 12 existing claim references in run_analyses. "
    "This requests analyses now; "
    "previously tested comparisons reuse their data and give no new sample.",
    "appraise": "Assess the newly delivered discovery evidence. Optionally set validate to one "
    "eligible claim reference for independent validation; omit it for no request. "
    "The controller attaches the discovery result. Validation arrives after this stage.",
    "synthesize": "Assess new validation and any response deadlines listed below. Explain what you "
    "conclude and what to investigate next. Accepted claims update automatically.",
}


def schemas():
    return {stage: form.model_json_schema() for stage, form in FORMS.items()}


def references(controller):
    s = controller.state
    return (
        {f"H{i}": hid for i, hid in enumerate(s["hypotheses"], 1)},
        {f"R{i}": rid for i, rid in enumerate(s["results"], 1)},
    )


def direct_results(controller, h):
    return [
        rid
        for rid, saved in controller.state["results"].items()
        if comparison_key(saved["hypothesis"]) == comparison_key(h)
    ]


def ledger(controller):
    s = controller.state
    hs, rs = references(controller)
    reverse_r = {rid: ref for ref, rid in rs.items()}
    rows = []
    for ref, hid in hs.items():
        h = s["hypotheses"][hid]
        evidence = []
        for rid in direct_results(controller, h):
            saved = s["results"][rid]
            result = oriented(saved["result"], saved["hypothesis"], h)
            evidence.append(
                dict(
                    ref=reverse_r[rid],
                    source="discovery" if rid.startswith("analysis-") else "validation",
                    **{
                        k: getattr(result, k)
                        for k in (
                            "estimate",
                            "lower",
                            "upper",
                            "alpha",
                            "cell_n",
                            "valid",
                            "diagnostic",
                        )
                    },
                )
            )
        current = s["current"][claim_key(h)]
        rows.append(
            dict(
                ref=ref,
                comparison=h.model_dump(exclude={"id"}),
                initial_expectation={
                    k: s["expectations"][hid][k]
                    for k in ("anticipated_direction", "assessment", "pre_evidence")
                },
                current={k: current[k] for k in ("status", "investigation")},
                evidence=evidence,
            )
        )
    return rows


def build_prompt(controller, task_context, iteration, iterations, stage, attempt, memory, feedback):
    hs, rs = references(controller)
    hr, rr = {v: k for k, v in hs.items()}, {v: k for k, v in rs.items()}
    due = [
        dict(claim=hr[x["hypothesis_id"]], evidence=rr[x["result_id"]], reason=x["reason"])
        for x in controller.required(iteration, stage)
    ]
    s = controller.state
    payload = dict(
        schema=FORMS[stage].model_json_schema(),
        task=task_context,
        claims=ledger(controller),
        assessments_due=due,
        validation=dict(
            remaining_new_requests=controller.service.policy.voluntary_limit
            - len(controller.service.state["voluntary_keys"]),
            eligible_claims=[
                ref for ref, hid in hs.items() if comparison_key(s["hypotheses"][hid]) in s["tests"]
            ],
            already_validated=[
                ref
                for ref, hid in hs.items()
                if comparison_key(s["hypotheses"][hid]) in controller.service.state["cache"]
            ],
        ),
        research_notes=memory.get("notes", ""),
        latest_stage_narratives=memory.get("narratives", {}),
    )
    if feedback:
        payload["repair"] = feedback
    return (
        f"Iteration {iteration}/{iterations}, stage {stage}, attempt {attempt}.\n"
        + COMMON
        + "\nYour action now: "
        + STAGE_INSTRUCTIONS[stage]
        + "\n"
        + json.dumps(payload, separators=(",", ":"))
    )


def translate(controller, stage, iteration, data):
    """Validate a small form, then attach only evidence available before this stage."""
    form = FORMS[stage].model_validate(data)
    hs, rs = references(controller)
    s = controller.state
    audit = dict(assessment_links=[], refinement_links=[])

    def claim(ref):
        if ref not in hs:
            raise ValueError(
                f"Unknown claim {ref!r}. Choose a reference from the current ledger: {list(hs)}"
            )
        return hs[ref]

    def evidence(refs):
        unknown = [ref for ref in refs if ref not in rs]
        if unknown:
            raise ValueError(
                f"Unknown evidence {unknown}. Use the R references in the current ledger."
            )
        if len(set(refs)) != len(refs):
            raise ValueError("List each evidence reference only once.")
        return [rs[ref] for ref in refs]

    record = WorkflowStageRecord(iteration=iteration, stage=stage, narrative=form.narrative)
    for i, p in enumerate(form.proposals, len(hs) + 1):
        hid = f"H{i}"
        h = Hypothesis(
            id=hid,
            **p.model_dump(
                exclude={"anticipated_direction", "initial_status", "parent", "motivating_evidence"}
            ),
        )
        record.hypotheses.append(h)
        record.anticipated_directions[hid] = p.anticipated_direction
        record.assessments[hid] = p.initial_status
        if p.parent is not None:
            parent = claim(p.parent)
            record.parent_ids[hid] = parent
            record.motivating_result_ids[hid] = (
                direct_results(controller, s["hypotheses"][parent])
                if p.motivating_evidence is None
                else evidence(p.motivating_evidence)
            )
            audit["refinement_links"].append(
                dict(
                    hypothesis_id=hid,
                    mode="displayed_parent_evidence"
                    if p.motivating_evidence is None
                    else "agent_selected",
                )
            )
        elif p.motivating_evidence is not None:
            record.motivating_result_ids[hid] = evidence(p.motivating_evidence)
    for a in form.assessments:
        hid = claim(a.claim)
        ids = (
            direct_results(controller, s["hypotheses"][hid])
            if a.evidence is None
            else evidence(a.evidence)
        )
        record.assessment_records.append(
            Assessment(
                hypothesis_id=hid, result_ids=ids, status=a.status, investigation=a.investigation
            )
        )
        audit["assessment_links"].append(
            dict(
                hypothesis_id=hid,
                mode="displayed_claim_evidence" if a.evidence is None else "agent_selected",
            )
        )
    if stage == "analyze":
        record.executed_ids = [claim(ref) for ref in form.run_analyses]
    if stage == "appraise" and form.validation_claim is not None:
        hid = claim(form.validation_claim)
        key = comparison_key(s["hypotheses"][hid])
        if key not in s["tests"]:
            raise ValueError(
                f"{form.validation_claim} has no valid discovery analysis. "
                "Choose an eligible claim."
            )
        record.validation_request = ValidationRequest(
            hypothesis_id=hid, triggering_result_ids=[s["tests"][key]["result"]["id"]]
        )
    if stage == "synthesize":
        current = {k: v["status"] for k, v in s["current"].items()}
        all_h = {**s["hypotheses"], **{h.id: h for h in record.hypotheses}}
        for h in record.hypotheses:
            current.setdefault(claim_key(h), record.assessments[h.id])
        for a in record.assessment_records:
            current[claim_key(all_h[a.hypothesis_id])] = a.status
        representatives = {}
        for hid, h in all_h.items():
            if current[claim_key(h)] == "accept":
                representatives.setdefault(claim_key(h), hid)
        record.accepted_ids = list(representatives.values())
    return record, form, audit


def repair_message(controller, error):
    """Use the same short references in feedback as in the supplied ledger."""
    hs, rs = references(controller)
    for ref, internal in sorted({**hs, **rs}.items(), key=lambda x: -len(x[1])):
        error = error.replace(internal, ref)
    return error
