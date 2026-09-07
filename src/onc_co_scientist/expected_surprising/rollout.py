"""Stage-based deployment with trusted analyses and private independent validation.

The provider receives public data summaries and result records only. All
analysis execution and scoring take place in Python, outside the LLM.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

from ..providers.base import ChatMessage, LLMProvider
from .evaluation import ValidationService
from .generation import version_discoveries
from .research import json_response, now
from .review_policy import require_current_pair
from .schemas import Hypothesis, PairSpec, StageRecord
from .scoring import (
    assign,
    canonical,
    estimate,
    evidence_alignment,
    family,
    recovery,
    responsiveness,
)


@contextmanager
def stage_transaction(containers):
    """Commit a whole stage or restore registration, evidence and analysis budgets."""
    snapshots = deepcopy(containers)
    try:
        yield
    except Exception:
        for current, saved in zip(containers, snapshots, strict=True):
            current.clear()
            if isinstance(current, list):
                current.extend(saved)
            else:
                current.update(saved)
        raise


def unknown_variable(name, role, available):
    suggestions = get_close_matches(name, sorted(available), n=3, cutoff=0.5)
    return ValueError(
        f"Unknown {role} variable {name!r}. "
        f"{'Closest public names' if suggestions else 'Allowed public names'}: "
        f"{suggestions or sorted(available)}. Copy a name from the public data schema."
    )


def run(
    spec: PairSpec,
    version: str,
    public: Path,
    out: Path,
    provider: LLMProvider,
    *,
    run_id: str,
    iterations: int | None = None,
    max_tokens_per_call: int = 125000,
    max_retries_per_stage: int = 2,
) -> dict:
    require_current_pair(spec)
    if max_retries_per_stage < 0 or max_tokens_per_call < 1:
        raise ValueError("Nonnegative retries and a positive completion allowance are required")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    task = json.loads((public / "task.json").read_text())
    frame = pd.read_parquet(public / "dataset.parquet")
    expected_iterations = task["iterations"]
    iterations = iterations or expected_iterations
    if not 1 <= iterations <= expected_iterations:
        raise ValueError("Iteration count is outside the task budget")
    outcomes = {o.name: o for o in spec.outcomes}
    service = ValidationService(spec, version, run_id)
    instructions = (public / "instructions.md").read_text()
    summary = frame.describe(include="all").fillna("").to_json()
    history = [{"task": task, "instructions": instructions, "summary": summary}]
    hypotheses: dict[str, Hypothesis] = {}
    assessments, parents, result_index, first_exact, expectations = {}, {}, {}, {}, {}
    proposed, tested, families = set(), set(), set()
    tested_hypotheses, exploration, evidence_events = [], [], []
    accepted_ids, protocol_errors = [], []
    attempt_errors, recovered_stages = [], []
    transactional_state = [
        hypotheses,
        assessments,
        parents,
        result_index,
        first_exact,
        expectations,
        proposed,
        tested,
        families,
        tested_hypotheses,
        exploration,
        evidence_events,
        accepted_ids,
        service.results,
        service.iterations,
    ]
    stage_schema = StageRecord.model_json_schema()
    stages = ("hypothesis", "analysis", "critique", "synthesis")
    sequence = 0

    def log(kind, payload):
        nonlocal sequence
        sequence += 1
        with (out / "transcript.jsonl").open("a") as f:
            f.write(
                json.dumps({"sequence": sequence, "timestamp": now(), "kind": kind, **payload})
                + "\n"
            )

    for iteration in range(1, iterations + 1):
        for stage in stages:
            for attempt in range(1, max_retries_per_stage + 2):
                response_text = None
                prompt = (
                    f"You are conducting iteration {iteration}/{iterations}, stage {stage}. "
                    f"Attempt {attempt}/{max_retries_per_stage + 1}. "
                    "If the latest history entry reports an error for this stage, correct it and "
                    "return the complete StageRecord for this stage. Failed attempts are rolled "
                    "back; "
                    "only previously successful stages have registered hypotheses or results. "
                    "Return JSON matching StageRecord. During hypothesis generation register "
                    "explicit "
                    "hypotheses with unique immutable IDs, anticipated_directions (-1,0,+1), and "
                    "assessments (accept,reject,unresolved). Use new IDs and parent_ids for "
                    "refinements. "
                    "In analysis select registered IDs in executed_ids; a trusted service will "
                    "execute "
                    "the specified mean differences or interactions. In the analysis stage you may "
                    "select one previously registered hypothesis using validation_request_id for "
                    "independent evidence; set this field to null in every other stage. "
                    "There are ten requests total, one per iteration. Critique must record a "
                    "decision for each validation result from the preceding analysis stage, linked "
                    "to its exact result ID. Also record decisions for the discovery-sample "
                    "analyses. In synthesis "
                    "specify all currently accepted_ids for the final report "
                    "and your narrative. Every stage may propose new hypotheses for later "
                    "analysis. Anticipated direction records your expectation; hypothesis "
                    "direction defines the "
                    "signed claim actually tested. Every result's estimate, lower and upper are "
                    "ALREADY SIGNED to that claim: estimate = hypothesis.direction * raw contrast. "
                    "The raw contrast is exposed minus comparator (or the specified interaction). "
                    "Do not multiply the reported estimate or interval by direction again when "
                    "deciding whether to accept the claim. For direction=-1, a raw difference of "
                    "-0.4 produces a reported estimate of +0.4 and supports the negative-direction "
                    "claim. A reported estimate of -0.4 opposes that claim. For BOTH directions, "
                    "accept if the REPORTED lower bound exceeds positive delta; reject if the "
                    "REPORTED upper bound is below positive delta; otherwise remain unresolved. "
                    "A rejected directional claim can motivate a new opposite-direction "
                    "hypothesis. "
                    "Outcome thresholds and comparison definitions are "
                    "in the task. Do not repeat IDs with changed hypotheses. You can perform at "
                    "most 12 analyses per iteration. No other tool calls are available in this "
                    "reference "
                    "harness.\n" + json.dumps({"schema": stage_schema, "history": history})
                )
                try:
                    with stage_transaction(transactional_state):
                        response = provider.chat(
                            [ChatMessage(role="user", content=prompt)],
                            temperature=0,
                            max_tokens=max_tokens_per_call,
                        )
                        log(
                            "model",
                            {
                                "iteration": iteration,
                                "stage": stage,
                                "attempt": attempt,
                                "model": response.model_id,
                                "prompt": prompt,
                                "response": response.text,
                            },
                        )
                        response_text = response.text
                        if not response_text.strip():
                            raise ValueError(
                                "Empty response: no final JSON was returned. Finish reasoning and "
                                "return a complete StageRecord JSON object for this stage."
                            )
                        record = StageRecord.model_validate(json_response(response_text))
                        if record.iteration != iteration or record.stage != stage:
                            raise ValueError(f"Return iteration={iteration}, stage={stage!r}")
                        previously_registered = set(hypotheses)
                        new_n = 0
                        new_ids = []
                        for h in record.hypotheses:
                            if h.outcome not in outcomes:
                                raise unknown_variable(h.outcome, "outcome", outcomes)
                            if h.exposure not in frame:
                                raise unknown_variable(h.exposure, "exposure", frame.columns)
                            for condition in h.subgroup + h.eligibility:
                                if condition.variable not in frame:
                                    raise unknown_variable(
                                        condition.variable, "condition", frame.columns
                                    )
                            if h.id in hypotheses and canonical(hypotheses[h.id]) != canonical(h):
                                raise ValueError(
                                    f"Registered hypothesis {h.id!r} cannot change. "
                                    "Use a new ID for a refinement and link it with parent_ids."
                                )
                            if h.id not in hypotheses:
                                if (
                                    h.id not in record.anticipated_directions
                                    or h.id not in record.assessments
                                ):
                                    raise ValueError(
                                        f"New hypothesis {h.id!r} needs an anticipated_direction "
                                        "and assessment in the corresponding dictionaries"
                                    )
                                if (
                                    h.id in record.parent_ids
                                    and record.parent_ids[h.id] not in hypotheses
                                ):
                                    raise ValueError(
                                        f"Unknown parent hypothesis {record.parent_ids[h.id]!r}"
                                    )
                                parents[h.id] = record.parent_ids.get(h.id)
                                expectations[h.id] = record.anticipated_directions[h.id]
                            hypotheses[h.id] = h
                            key = canonical(h)
                            new_n += key not in proposed
                            if key not in proposed:
                                new_ids.append(h.id)
                            proposed.add(key)
                            families.add(family(h))
                        if len(record.executed_ids) > 12 or len(set(record.executed_ids)) != len(
                            record.executed_ids
                        ):
                            raise ValueError("Analysis budget exceeded or duplicate analysis ID")
                        if record.executed_ids and stage != "analysis":
                            raise ValueError("Execute analyses in the analysis stage")
                        unknown_ids = set(record.executed_ids) - previously_registered
                        if unknown_ids:
                            raise ValueError(
                                f"Analysis IDs {sorted(unknown_ids)} are not previously "
                                "registered. "
                                "Select IDs registered before this stage."
                            )
                        if record.validation_request_id and (
                            stage != "analysis"
                            or record.validation_request_id not in previously_registered
                        ):
                            raise ValueError(
                                "Validation must select a previously registered hypothesis "
                                "at analysis"
                            )
                        results = []
                        for hid in record.executed_ids:
                            h = hypotheses[hid]
                            r = estimate(
                                frame,
                                h,
                                delta=outcomes[h.outcome].delta,
                                alpha=0.05,
                                result_id=f"analysis-{iteration}-{hid}",
                            )
                            results.append(r.model_dump())
                            result_index[r.id] = r.hypothesis_id
                            if r.valid:
                                tested.add(canonical(h))
                                tested_hypotheses.append(h)
                        if record.validation_request_id:
                            hid = record.validation_request_id
                            r = service.request(
                                hypotheses[hid], iteration, f"validation-{iteration}-{hid}"
                            )
                            evidence_events.append(
                                {
                                    "iteration": iteration,
                                    "hypothesis_id": hid,
                                    "prior": assessments.get(hid, "unresolved"),
                                    "anticipated_direction": expectations[hid],
                                    "claim_direction": hypotheses[hid].direction,
                                    "result": r.model_dump(),
                                }
                            )
                            results.append(r.model_dump())
                            result_index[r.id] = r.hypothesis_id
                        for decision in record.decisions:
                            if result_index.get(decision.result_id) != decision.hypothesis_id:
                                raise ValueError(
                                    f"Decision ({decision.hypothesis_id!r}, "
                                    f"{decision.result_id!r}) does not identify an executed "
                                    "result. Copy both IDs from its record."
                                )
                            matching = [
                                e
                                for e in evidence_events
                                if e["result"]["id"] == decision.result_id
                                and e["hypothesis_id"] == decision.hypothesis_id
                            ]
                            if decision.result_id.startswith("validation-"):
                                if not matching or "decision" in matching[0]:
                                    raise ValueError(
                                        "Unknown validation result or repeated decision"
                                    )
                                # The first critique immediately after a result is scored.
                                if stage != "critique" or matching[0]["iteration"] != iteration:
                                    raise ValueError(
                                        "Validation decisions must be immediate critiques"
                                    )
                                matching[0]["decision"] = decision.status
                            assessments[decision.hypothesis_id] = decision.status
                        for hid, assessment in record.assessments.items():
                            if hid not in previously_registered:
                                assessments[hid] = assessment
                        if stage == "synthesis":
                            if any(hid not in hypotheses for hid in record.accepted_ids):
                                raise ValueError("Unknown accepted hypothesis")
                            accepted_ids[:] = record.accepted_ids
                        coverage = assign(
                            tested_hypotheses,
                            version_discoveries(spec, version),
                            ignore_direction=True,
                        )
                        proposed_recovery = recovery(
                            list(hypotheses.values()), version_discoveries(spec, version)
                        )
                        for matched in proposed_recovery["matches"]:
                            if matched["match"] == "exact":
                                first_exact.setdefault(
                                    matched["discovery_id"],
                                    {"iteration": iteration, "stage": stage},
                                )
                        exploration.append(
                            {
                                "iteration": iteration,
                                "stage": stage,
                                "new_hypotheses": new_n,
                                "cumulative_proposed": len(proposed),
                                "cumulative_tested": len(tested),
                                "families": len(families),
                                "coverage": coverage,
                                "proposed_recovery": proposed_recovery,
                                "accepted_recovery": recovery(
                                    [hypotheses[h] for h in accepted_ids],
                                    version_discoveries(spec, version),
                                ),
                                "hypothesis_ids": [h.id for h in record.hypotheses],
                                "new_hypothesis_ids": new_ids,
                                "executed_ids": record.executed_ids,
                            }
                        )
                        public_event = {"record": record.model_dump(), "results": results}
                        history.append(public_event)
                        log("stage", {**public_event, "attempt": attempt})
                    if attempt > 1:
                        recovered = {"iteration": iteration, "stage": stage, "attempts": attempt}
                        recovered_stages.append(recovered)
                        log("stage_recovered", recovered)
                    break
                except Exception as exc:
                    retrying = attempt <= max_retries_per_stage
                    error = {
                        "iteration": iteration,
                        "stage": stage,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "retrying": retrying,
                    }
                    attempt_errors.append(error)
                    log("attempt_error", error)
                    feedback = {
                        "protocol_error": error,
                        "registered_hypothesis_ids": sorted(hypotheses),
                        "failed_response": (response_text or "")[:24000],
                        "failed_response_truncated": len(response_text or "") > 24000,
                        "instruction": (
                            "Correct the error and return a complete replacement for the same "
                            "stage. No changes, analyses or validation requests from the failed "
                            "attempt were committed."
                            if retrying
                            else "Stage retry budget exhausted; proceed to the next stage."
                        ),
                    }
                    history.append(feedback)
                    log("error_feedback", feedback)
                    if not retrying:
                        protocol_errors.append(error)
                        log("protocol_error", error)
    accepted = [hypotheses[hid] for hid in sorted(set(accepted_ids))]
    confirmation = service.confirm(accepted)
    confirmation["first_attempt_primary_recovery"] = (
        0 if attempt_errors else confirmation["primary_recovery"]
    )
    if protocol_errors:
        confirmation["primary_recovery"] = 0
    followups = []
    for e in evidence_events:
        if e["iteration"] > iterations - 2:
            continue
        future = [x for x in exploration if e["iteration"] < x["iteration"] <= e["iteration"] + 2]
        new_ids = {hid for x in future for hid in x["new_hypothesis_ids"]}
        followups.append(
            {
                "result_id": e["result"]["id"],
                "anticipated_direction": e["anticipated_direction"],
                "evidence_alignment": evidence_alignment(e),
                "new_hypotheses": sum(x["new_hypotheses"] for x in future),
                "linked_refinements": sum(
                    parents.get(hid) == e["hypothesis_id"] for hid in new_ids
                ),
                "tested_hypotheses": len({hid for x in future for hid in x["executed_ids"]}),
            }
        )
    report = {
        "run_id": run_id,
        "pair_id": spec.pair_id,
        "profile": spec.profile,
        "version": version,
        "harness": "reference_stage",
        "model": provider.model_id,
        "iterations": iterations,
        "max_tokens_per_call": max_tokens_per_call,
        "retry_policy": {
            "version": "1.0.0",
            "max_retries_per_stage": max_retries_per_stage,
            "atomic_stage": True,
            "primary_failure_rule": "unrecovered_stage_error",
        },
        "attempt_errors": attempt_errors,
        "recovered_stages": recovered_stages,
        "protocol_errors": protocol_errors,
        "discovery": recovery(accepted, version_discoveries(spec, version)),
        "confirmation": confirmation,
        "exploration": exploration,
        "responsiveness": responsiveness(evidence_events),
        "post_evidence_exploration": followups,
        "first_exact": first_exact,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
