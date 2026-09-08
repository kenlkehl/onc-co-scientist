"""Versioned appraisal-first controller; scientific choices never trigger repairs."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from ..providers.base import ChatMessage
from .evaluation import WorkflowValidationService
from .generation import version_discoveries
from .packaging import sha256
from .research import json_response, now
from .review_policy import require_current_pair
from .rollout import stage_transaction, unknown_variable
from .schemas import (
    AssessmentRecord,
    EvidenceResult,
    Hypothesis,
    InitialExpectation,
    ValidationPolicy,
    WorkflowStageRecord,
    WorkflowVersions,
)
from .scoring import (
    assign,
    claim_key,
    comparison_key,
    estimate,
    exploration_coverage,
    family,
    oriented,
    recovery,
    refinement_types,
    workflow_discovery,
)

STAGES = ("explore", "analyze", "appraise", "synthesize")
STAGE_PROMPT = """Return one complete WorkflowStageRecord JSON object for this stage.
Explore registers comparisons. Analyze selects previously registered IDs in executed_ids (at most
12). Appraise records assessments of all newly delivered discovery results and may set
validation_request after appraisal. Synthesize assesses new validation and every due response
checkpoint and lists the complete accepted_ids set, consistent with current assessments.
Every stage may register new hypotheses for later analysis. New IDs require anticipated_directions
(-1,0,1) and initial assessments dictionaries; these dictionaries are ONLY for initial registration.
Use assessment_records for subsequent decisions: hypothesis_id, result_ids considered, status
(accept/reject/unresolved), investigation (active/deferred/closed). Initial assessments and
anticipated directions are immutable. Refinements use a new ID, parent_ids and motivating_result_ids
that cite already available evidence. Opposite claims may reuse oriented evidence from the same
comparison. Such registrations are marked as already exposed to direct evidence.
Assess every item in required_assessments, citing its result_id, in assessment_records.
Optional decisions must agree with assessment_records. Only synthesize sets accepted_ids, with
exactly one ID per currently accepted canonical claim. To withdraw a claim, assess it accordingly.
For validation_request specify hypothesis_id and triggering_result_ids for a valid discovery
analysis of that comparison. Other stages set validation_request to null. Returned estimates and
intervals are already signed to their indicated hypothesis.
Use your scientific judgment about effect sizes, uncertainty, and the conclusions supported by
the supplied data. Explain your reasoning in the narrative and decide how to investigate further.
All evidence citations must already be available before this stage. Failed stages are rolled back;
correct contract feedback with a complete replacement. There are no other analysis or search tools.
"""


def public_event(event):
    """Project numerical evidence for the agent while retaining private scoring records."""
    fields = {
        "id",
        "hypothesis_id",
        "estimate",
        "lower",
        "upper",
        "alpha",
        "cell_n",
        "valid",
        "diagnostic",
        "source",
        "cached",
    }
    return {
        **event,
        **{
            key: [{k: v for k, v in result.items() if k in fields} for result in event[key]]
            for key in ("results", "reused_evidence")
            if key in event
        },
    }


class WorkflowController:
    def __init__(self, spec, version, frame, policy, replicate_id):
        self.spec, self.version, self.frame = spec, version, frame
        self.analysis_estimator = estimate
        self.outcomes = {o.name: o for o in spec.outcomes}
        self.service = WorkflowValidationService(spec, version, replicate_id, policy)
        self.state = {
            "hypotheses": {},
            "expectations": {},
            "registrations": [],
            "assessments": [],
            "current": {},
            "results": {},
            "tests": {},
            "executions": [],
            "events": [],
            "first_acceptances": {},
            "accepted_ids": [],
            "exploration": [],
            "completed_stages": [],
        }

    def required(self, iteration, stage):
        required = []
        for event in self.state["events"]:
            immediate = event["iteration"] == iteration and stage == (
                "appraise" if event["source"] == "discovery" else "synthesize"
            )
            due = (
                event["source"] != "discovery"
                and event["due_iteration"] == iteration
                and stage == "synthesize"
            )
            if immediate or due:
                required.append(
                    {
                        "hypothesis_id": event["hypothesis"]["id"],
                        "result_id": event["result"]["id"],
                        "reason": "response_deadline" if due else "new_result",
                    }
                )
        # Invalid discovery analyses are appraised too, but have no reference score.
        if stage == "appraise":
            for execution in self.state["executions"]:
                if execution["iteration"] == iteration:
                    item = {
                        "hypothesis_id": execution["hypothesis"]["id"],
                        "result_id": execution["result"]["id"],
                        "reason": "new_result",
                    }
                    key = claim_key(Hypothesis.model_validate(execution["hypothesis"]))
                    if not any(
                        claim_key(self.state["hypotheses"][r["hypothesis_id"]]) == key
                        and r["result_id"] == item["result_id"]
                        for r in required
                    ):
                        required.append(item)
        return required

    def _available(self, ids, available):
        if len(ids) != len(set(ids)) or not set(ids) <= available:
            raise ValueError("Cite distinct result IDs available before this stage")

    def _first_acceptance(self, h, status, iteration, stage, sequence):
        key, comparison = claim_key(h), comparison_key(h)
        if status != "accept" or key in self.state["first_acceptances"]:
            return
        requests = [
            r.copy() for r in self.service.state["requests"] if r["comparison_key"] == comparison
        ]
        receipt = self.service.state["cache"].get(comparison)
        self.state["first_acceptances"][key] = {
            "claim_key": key,
            "comparison_key": comparison,
            "hypothesis_id": h.id,
            "iteration": iteration,
            "stage": stage,
            "sequence": sequence,
            "tested_at_acceptance": comparison in self.state["tests"],
            "requests_before": requests,
            "receipt_before": None
            if receipt is None
            else {
                "iteration": receipt["iteration"],
                "route": receipt["route"],
                "sequence": receipt["sequence"],
                "result_id": receipt["result"].id,
            },
        }

    def _register(self, record, sequence, available):
        s = self.state
        new_ids = {h.id for h in record.hypotheses} - s["hypotheses"].keys()
        if len({h.id for h in record.hypotheses}) != len(record.hypotheses):
            raise ValueError("Duplicate hypothesis IDs in registration")
        if set(record.anticipated_directions) != new_ids or set(record.assessments) != new_ids:
            raise ValueError(
                "Initial direction and assessment dictionaries must name each new ID only"
            )
        if (
            not set(record.parent_ids) <= new_ids
            or not set(record.motivating_result_ids) <= new_ids
        ):
            raise ValueError("Parent and motivating result links are immutable registration fields")
        for h in record.hypotheses:
            if h.outcome not in self.outcomes:
                raise unknown_variable(h.outcome, "outcome", self.outcomes)
            if h.exposure not in self.frame:
                raise unknown_variable(h.exposure, "exposure", self.frame.columns)
            for condition in h.subgroup + h.eligibility:
                if condition.variable not in self.frame:
                    raise unknown_variable(condition.variable, "condition", self.frame.columns)
            if h.id in s["hypotheses"]:
                if h != s["hypotheses"][h.id]:
                    raise ValueError("Registered hypotheses cannot change; use a new ID")
                continue
            parent_id = record.parent_ids.get(h.id)
            motivations = record.motivating_result_ids.get(h.id, [])
            self._available(motivations, available)
            if parent_id and parent_id not in s["hypotheses"]:
                raise ValueError("Unknown parent hypothesis")
            if parent_id and not motivations:
                raise ValueError("Refinements require available motivating result IDs")
            direct = [
                rid
                for rid, value in s["results"].items()
                if comparison_key(value["hypothesis"]) == comparison_key(h)
            ]
            related = [
                rid
                for rid, value in s["results"].items()
                if rid not in direct
                and (family(value["hypothesis"]) == family(h) or rid in motivations)
            ]
            expectation = InitialExpectation(
                hypothesis_id=h.id,
                anticipated_direction=record.anticipated_directions[h.id],
                assessment=record.assessments[h.id],
                iteration=record.iteration,
                stage=record.stage,
                sequence=sequence,
                direct_result_ids=tuple(direct),
                related_result_ids=tuple(related),
                pre_evidence=not direct,
            )
            s["expectations"][h.id] = expectation.model_dump()
            key = claim_key(h)
            if key in s["current"] and record.assessments[h.id] != s["current"][key]["status"]:
                raise ValueError("A renamed canonical claim must retain its current assessment")
            s["hypotheses"][h.id] = h
            s["current"].setdefault(
                key, {"status": record.assessments[h.id], "investigation": "active"}
            )
            s["registrations"].append(
                {
                    "hypothesis": h.model_dump(),
                    "iteration": record.iteration,
                    "stage": record.stage,
                    "sequence": sequence,
                    "parent_id": parent_id,
                    "motivating_result_ids": motivations,
                    "refinement_types": refinement_types(s["hypotheses"][parent_id], h)
                    if parent_id
                    else [],
                    "expectation": expectation.model_dump(),
                }
            )
            self._first_acceptance(
                h, record.assessments[h.id], record.iteration, record.stage, sequence
            )

    def _assess(self, record, sequence, available):
        s = self.state
        seen = {}
        for a in record.assessment_records:
            if a.hypothesis_id not in s["hypotheses"]:
                raise ValueError("Unknown assessed hypothesis")
            self._available(a.result_ids, available)
            if not a.result_ids:
                raise ValueError("Post-registration assessments require available result IDs")
            h = s["hypotheses"][a.hypothesis_id]
            key = claim_key(h)
            if key in seen:
                raise ValueError("Assess a canonical claim once per stage")
            seen[key] = a
            entry = AssessmentRecord(
                **a.model_dump(), iteration=record.iteration, stage=record.stage, sequence=sequence
            ).model_dump()
            s["assessments"].append(entry)
            s["current"][key] = entry
            self._first_acceptance(h, a.status, record.iteration, record.stage, sequence)
        for required in self.required(record.iteration, record.stage):
            # The original claim must be explicitly reassessed; a replacement direction is separate.
            found = next(
                (
                    a
                    for a in record.assessment_records
                    if a.hypothesis_id == required["hypothesis_id"]
                    and required["result_id"] in a.result_ids
                ),
                None,
            )
            if found is None:
                raise ValueError(f"Required assessment missing: {required}")
        for decision in record.decisions:
            if decision.hypothesis_id not in s["hypotheses"]:
                raise ValueError("Unknown decision hypothesis")
            h = s["hypotheses"][decision.hypothesis_id]
            a = seen.get(claim_key(h))
            if a is None or decision.status != a.status or decision.result_id not in a.result_ids:
                raise ValueError("Result decisions must agree with the stage assessment")
            result = s["results"][decision.result_id]
            if comparison_key(result["hypothesis"]) != comparison_key(h):
                raise ValueError("Decision result belongs to another comparison")
        for event in s["events"]:
            a = next(
                (
                    a
                    for a in s["assessments"]
                    if a["sequence"] == sequence
                    and a["hypothesis_id"] == event["hypothesis"]["id"]
                    and event["result"]["id"] in a["result_ids"]
                ),
                None,
            )
            if a is None:
                continue
            if event["iteration"] == record.iteration and record.stage == (
                "appraise" if event["source"] == "discovery" else "synthesize"
            ):
                event["immediate"] = a
            if (
                event["source"] != "discovery"
                and event["due_iteration"] == record.iteration
                and (record.stage == "synthesize")
            ):
                event["delayed"] = a

    def _deliver(self, h, result, iteration, sequence, source, fresh):
        s = self.state
        s["results"].setdefault(result.id, {"hypothesis": h, "result": result})
        if fresh:
            current = s["current"][claim_key(h)]
            s["events"].append(
                {
                    "hypothesis": h.model_dump(),
                    "hypothesis_id": h.id,
                    "comparison_key": comparison_key(h),
                    "result": result.model_dump(),
                    "iteration": iteration,
                    "sequence": sequence,
                    "source": source,
                    "due_iteration": iteration + self.service.policy.response_window,
                    "prior": current["status"],
                    "prior_investigation": current["investigation"],
                    "voluntary_slots_remaining": self.service.policy.voluntary_limit
                    - len(self.service.state["voluntary_keys"]),
                    "anticipated_direction": s["expectations"][h.id]["anticipated_direction"],
                    "claim_direction": h.direction,
                    "expectation": s["expectations"][h.id],
                    "immediate": None,
                    "delayed": None,
                }
            )
        return {**result.model_dump(), "source": source, "cached": not fresh}

    def apply(self, record):
        s = self.state
        iteration, stage = record.iteration, record.stage
        sequence = len(s["completed_stages"]) + 1
        available = set(s["results"])
        previously_registered = set(s["hypotheses"])
        self._register(record, sequence, available)
        self._assess(record, sequence, available)
        if record.executed_ids and stage != "analyze":
            raise ValueError("Execute analyses only in analyze")
        if len(record.executed_ids) > 12 or len(set(record.executed_ids)) != len(
            record.executed_ids
        ):
            raise ValueError("Analysis budget exceeded or duplicate IDs")
        if not set(record.executed_ids) <= previously_registered:
            raise ValueError("Analysis requires previously registered IDs")
        if record.validation_request and stage != "appraise":
            raise ValueError("Request validation only in appraise")
        if record.accepted_ids and stage != "synthesize":
            raise ValueError("Set accepted_ids only in synthesize")
        results = []
        for hid in record.executed_ids:
            h = s["hypotheses"][hid]
            key = comparison_key(h)
            cached = s["tests"].get(key)
            if cached:
                result = oriented(
                    EvidenceResult.model_validate(cached["result"]),
                    Hypothesis.model_validate(cached["hypothesis"]),
                    h,
                )
            else:
                result = self.analysis_estimator(
                    self.frame,
                    h,
                    delta=self.outcomes[h.outcome].delta,
                    alpha=0.05,
                    result_id="analysis-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                )
            execution = {
                "hypothesis": h.model_dump(),
                "result": result.model_dump(),
                "comparison_key": key,
                "iteration": iteration,
                "stage": stage,
                "sequence": sequence,
                "repeated": cached is not None,
            }
            s["executions"].append(execution)
            if result.valid:
                s["tests"].setdefault(key, execution)
            results.append(
                self._deliver(
                    h,
                    result,
                    iteration,
                    sequence,
                    "discovery",
                    fresh=cached is None and result.valid,
                )
            )
        if record.validation_request:
            req = record.validation_request
            if req.hypothesis_id not in previously_registered:
                raise ValueError("Validation requires a previously registered hypothesis")
            h = s["hypotheses"][req.hypothesis_id]
            self._available(req.triggering_result_ids, available)
            key = comparison_key(h)
            if key not in s["tests"] or not any(
                rid == s["tests"][key]["result"]["id"] for rid in req.triggering_result_ids
            ):
                raise ValueError(
                    "Validation must cite a valid discovery analysis of this comparison"
                )
            result, fresh = self.service.deliver(
                h,
                iteration,
                "voluntary",
                triggering_result_ids=req.triggering_result_ids,
                prior=s["current"][claim_key(h)].copy(),
                sequence=sequence,
            )
            results.append(self._deliver(h, result, iteration, sequence, "voluntary", fresh))
        if stage == "appraise":
            for opportunity in self.service.due(iteration):
                h = s["hypotheses"][opportunity["hypothesis_id"]]
                result, fresh = self.service.deliver(h, iteration, "automatic", sequence=sequence)
                results.append(self._deliver(h, result, iteration, sequence, "automatic", fresh))
        if stage == "synthesize":
            if not set(record.accepted_ids) <= s["hypotheses"].keys():
                raise ValueError("Unknown accepted ID")
            keys = [claim_key(s["hypotheses"][hid]) for hid in record.accepted_ids]
            required = {key for key, a in s["current"].items() if a["status"] == "accept"}
            if len(set(keys)) != len(keys) or set(keys) != required:
                raise ValueError(
                    "accepted_ids must contain one ID for every currently accepted claim"
                )
            s["accepted_ids"] = record.accepted_ids.copy()
        discoveries = version_discoveries(self.spec, self.version)
        tested = [Hypothesis.model_validate(t["hypothesis"]) for t in s["tests"].values()]
        s["exploration"].append(
            {
                "iteration": iteration,
                "stage": stage,
                "new_hypotheses": len(
                    {claim_key(h) for h in record.hypotheses}
                    - {
                        claim_key(Hypothesis.model_validate(r["hypothesis"]))
                        for r in s["registrations"]
                        if r["sequence"] < sequence
                    }
                ),
                "cumulative_proposed": len({claim_key(h) for h in s["hypotheses"].values()}),
                "cumulative_tested": len(s["tests"]),
                "families": len({family(h) for h in s["hypotheses"].values()}),
                "tested_families": len({family(h) for h in tested}),
                "coverage": assign(tested, discoveries, ignore_direction=True),
                "hypothesis_ids": [h.id for h in record.hypotheses],
                "executed_ids": record.executed_ids,
            }
        )
        s["completed_stages"].append({"iteration": iteration, "stage": stage})
        # Explicitly expose previously available numerical results oriented to new claims.
        reused = []
        for h in record.hypotheses:
            for rid in s["expectations"][h.id]["direct_result_ids"]:
                saved = s["results"][rid]
                reused.append(oriented(saved["result"], saved["hypothesis"], h).model_dump())
        return {"record": record.model_dump(), "results": results, "reused_evidence": reused}


class WorkflowInfrastructureError(RuntimeError):
    """Stop rather than treating a persistence/provenance error as an agent mistake."""


def run_workflow(
    spec,
    version,
    public,
    out,
    provider,
    *,
    run_id,
    iterations=None,
    max_tokens_per_call=125000,
    max_retries_per_stage=2,
    policy=None,
    replicate_id=None,
    stage_executor=None,
):
    from .events import behavioral_summary, response_summary

    require_current_pair(spec)
    task = json.loads((public / "task.json").read_text())
    versions = WorkflowVersions.model_validate(task["versions"]).model_dump()
    iterations = task["iterations"] if iterations is None else iterations
    if not 1 <= iterations <= task["iterations"]:
        raise ValueError("Iteration count is outside the task budget")
    if max_tokens_per_call < 1 or max_retries_per_stage < 0:
        raise ValueError("Positive tokens and nonnegative retries required")
    packaged_policy_path = public.parent.parent / "private" / spec.pair_id / "workflow.json"
    if policy is None and iterations == task["iterations"]:
        packaged = json.loads(packaged_policy_path.read_text())
        if packaged["versions"] != versions:
            raise ValueError("Private workflow policy and public task versions differ")
        policy = packaged["policy"]
    policy = (
        ValidationPolicy.default(spec.profile, iterations)
        if policy is None
        else ValidationPolicy.model_validate(policy).for_budget(iterations)
    )
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    frame = pd.read_parquet(public / "dataset.parquet")
    controller = WorkflowController(spec, version, frame, policy, replicate_id or run_id)
    compact = versions["prompt"] == "ledger-1.0.0"
    from .prompting import build_prompt, references, repair_message, translate

    memory, repair_feedback = {}, None
    task_context = {
        "instructions": (public / "instructions.md").read_text(),
        "outcomes": task["outcomes"],
        "observations": len(frame),
        "variables": {
            name: {k: v for k, v in stats.items() if v != ""}
            for name, stats in json.loads(
                frame.describe(include="all").fillna("").to_json()
            ).items()
            if name not in {"patient_id", "cell_line_id"}
        },
    }
    if stage_executor is not None and hasattr(stage_executor, "bind"):
        stage_executor.bind(controller, task_context)
    history = [
        {
            "task": task,
            "instructions": (public / "instructions.md").read_text(),
            "summary": json.loads(frame.describe(include="all").fillna("").to_json()),
        }
    ]
    errors, exhausted, recovered = [], [], []
    sequence = 0

    def log(kind, payload):
        nonlocal sequence
        sequence += 1
        with (out / "transcript.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"sequence": sequence, "timestamp": now(), "kind": kind, **payload})
                + "\n"
            )

    log(
        "run",
        {
            "versions": versions,
            "policy": policy.model_dump(),
            "run_id": run_id,
            "replicate_id": controller.service.replicate_id,
        },
    )
    for iteration in range(1, iterations + 1):
        for stage in STAGES:
            if stage == "appraise":
                opportunity = controller.service.select(iteration, controller.state["tests"])
                if opportunity:
                    log("private_selection", opportunity)
            required = controller.required(iteration, stage)
            repair_feedback = None
            for attempt in range(1, max_retries_per_stage + 2):
                response_text = ""
                if compact:
                    prompt = build_prompt(
                        controller,
                        task_context,
                        iteration,
                        iterations,
                        stage,
                        attempt,
                        memory,
                        repair_feedback,
                    )
                else:
                    prompt = (
                        f"Iteration {iteration}/{iterations}, stage {stage}, attempt {attempt}.\n"
                    )
                    prompt += (
                        STAGE_PROMPT
                        + "\n"
                        + json.dumps(
                            {
                                "schema": WorkflowStageRecord.model_json_schema(),
                                "history": history,
                                "required_assessments": required,
                                "current_assessments": controller.state["current"],
                            }
                        )
                    )
                try:
                    with stage_transaction([controller.state, controller.service.state]):
                        if stage_executor is None:
                            response = provider.chat(
                                [ChatMessage(role="user", content=prompt)],
                                temperature=0,
                                max_tokens=max_tokens_per_call,
                            )
                        else:
                            response = stage_executor.respond(
                                prompt, iteration=iteration, stage=stage, attempt=attempt
                            )
                        response_text = response.text
                        log(
                            "model",
                            {
                                "iteration": iteration,
                                "stage": stage,
                                "attempt": attempt,
                                "model": response.model_id,
                                "prompt": prompt,
                                "response": response_text,
                            },
                        )
                        if not response_text.strip():
                            raise ValueError("Empty response; return a complete JSON stage record")
                        if compact:
                            record, form, interface_audit = translate(
                                controller, stage, iteration, json_response(response_text)
                            )
                        else:
                            record = WorkflowStageRecord.model_validate(
                                json_response(response_text)
                            )
                        if record.stage != stage or record.iteration != iteration:
                            raise ValueError(f"Return iteration={iteration}, stage={stage}")
                        event = controller.apply(record)
                    if stage_executor is not None:
                        stage_executor.commit()
                    if compact:
                        if form.research_notes is not None:
                            memory["notes"] = form.research_notes
                        memory.setdefault("narratives", {})[stage] = form.narrative
                        log(
                            "interface_translation",
                            {
                                "iteration": iteration,
                                "stage": stage,
                                "attempt": attempt,
                                "reference_map": references(controller),
                                **interface_audit,
                            },
                        )
                    history.append(public_event(event))
                    log("stage", {**event, "attempt": attempt})
                    if attempt > 1:
                        recovered.append(
                            {"iteration": iteration, "stage": stage, "attempts": attempt}
                        )
                        log("stage_recovered", recovered[-1])
                    break
                except WorkflowInfrastructureError:
                    raise
                except Exception as exc:
                    if stage_executor is not None:
                        stage_executor.reject()
                    retrying = attempt <= max_retries_per_stage
                    error = {
                        "iteration": iteration,
                        "stage": stage,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "retrying": retrying,
                    }
                    errors.append(error)
                    log("attempt_error", error)
                    if not retrying:
                        exhausted.append(error)
                        log("protocol_error", error)
                    feedback = {
                        "protocol_error": error,
                        "registered_hypothesis_ids": sorted(controller.state["hypotheses"]),
                        "failed_response": response_text[:24000],
                        "instruction": "Failed attempt rolled back. Return a complete replacement."
                        if retrying
                        else "Stage exhausted; continue with committed state.",
                    }
                    repair_feedback = {
                        "error": repair_message(controller, error["error"])
                        if compact
                        else error["error"],
                        "instruction": (
                            "Nothing in the failed attempt was committed. Return a corrected "
                            "complete form for this stage. The ledger is unchanged."
                        ),
                    }
                    history.append(feedback)
                    log("error_feedback", feedback)
            # Scheduled release is controller work even if appraisal exhausts its retries.
            if stage == "appraise" and controller.service.due(iteration):
                deliveries = []
                for o in controller.service.due(iteration):
                    h = controller.state["hypotheses"][o["hypothesis_id"]]
                    result, fresh = controller.service.deliver(
                        h,
                        iteration,
                        "automatic",
                        sequence=len(controller.state["completed_stages"]) + 1,
                    )
                    deliveries.append(
                        controller._deliver(
                            h,
                            result,
                            iteration,
                            len(controller.state["completed_stages"]) + 1,
                            "automatic",
                            fresh,
                        )
                    )
                event = {"controller_delivery": True, "results": deliveries}
                history.append(public_event(event))
                log("controller_delivery", event)
    s = controller.state
    # Assessments are authoritative, including when the final synthesis has a contract error.
    unique = {}
    for h in s["hypotheses"].values():
        if s["current"][claim_key(h)]["status"] == "accept":
            unique.setdefault(claim_key(h), h)
    accepted = list(unique.values())
    confirmation = controller.service.confirm(accepted)
    discoveries = version_discoveries(spec, version)
    discovery = workflow_discovery(
        accepted,
        discoveries,
        s["tests"],
        confirmation,
        failed=bool(exhausted),
        repaired=bool(errors),
    )
    confirmation["confirmed_recovery"] = {
        "matches": discovery["confirmed_matches"],
        "categories": discovery["exact"]["R_c"],
    }
    primary = int(
        any(
            m["discovery_id"] == spec.focal_id and m["match"] == "exact"
            for m in discovery["confirmed_matches"]
        )
    )
    confirmation.update(
        primary_recovery=0 if exhausted else primary,
        first_attempt_primary_recovery=0 if errors else primary,
        confirmed_matches=discovery["confirmed_matches"],
    )
    coverage = exploration_coverage(list(s["tests"].values()), discoveries, iterations)
    responses = response_summary(s["events"], s["completed_stages"], iterations, discoveries)
    behavior = behavioral_summary(
        s, controller.service.export(), discoveries, iterations, spec.focal_id
    )
    confirmed_ids = {m["discovery_id"] for m in discovery["confirmed_matches"]}
    for milestone in behavior["milestones"]:
        milestone["independently_confirmed"] = milestone["discovery_id"] in confirmed_ids
    report = {
        "run_id": run_id,
        "replicate_id": controller.service.replicate_id,
        "pair_id": spec.pair_id,
        "profile": spec.profile,
        "version": version,
        "harness": "appraisal_stage",
        "model": provider.model_id,
        "versions": versions,
        "iterations": iterations,
        "max_tokens_per_call": max_tokens_per_call,
        "policy": policy.model_dump(),
        "dataset_sha256": sha256(public / "dataset.parquet"),
        "public_package_sha256": {
            p.name: sha256(p) for p in sorted(public.iterdir()) if p.is_file()
        },
        "retry_policy": {
            "version": "1.0.0",
            "max_retries_per_stage": max_retries_per_stage,
            "atomic_stage": True,
            "primary_failure_rule": "unrecovered_stage_error",
        },
        "attempt_errors": errors,
        "protocol_errors": exhausted,
        "recovered_stages": recovered,
        "successful_stages": len(s["completed_stages"]),
        "expected_stages": 4 * iterations,
        "completion_rate": len(s["completed_stages"]) / (4 * iterations),
        "termination": "budget_completed",
        "discovery": recovery(accepted, discoveries),
        "confirmation": confirmation,
        "exploration": s["exploration"],
        "scores": {
            "D": discovery["exact"]["D"],
            "E": coverage["exact"]["E"],
            "B": responses["B"],
            "discovery": discovery,
            "coverage": coverage,
        },
        "responsiveness": responses,
        "validation": controller.service.export(),
        "behavior": behavior,
        "post_evidence_exploration": behavior["followups"],
        "state": {
            key: s[key]
            for key in (
                "registrations",
                "assessments",
                "events",
                "executions",
                "first_acceptances",
                "completed_stages",
                "current",
            )
        },
        "final_accepted_ids": [h.id for h in accepted],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
