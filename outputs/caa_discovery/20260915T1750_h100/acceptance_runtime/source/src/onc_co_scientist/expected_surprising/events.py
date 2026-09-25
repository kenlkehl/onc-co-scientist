"""Private event scoring and descriptive investigation/validation-choice summaries."""

from __future__ import annotations

from collections import defaultdict

from .schemas import EvidenceResult, Hypothesis, ScoreComponent
from .scoring import (
    EVIDENCE_CLASSES,
    assign,
    claim_key,
    comparison_key,
    evidence_alignment,
    evidence_class,
    evidence_status,
    family,
    match,
    oriented,
)


def _component(events):
    n = len(events)
    correct = sum(e["score"] for e in events)
    return ScoreComponent(
        numerator=correct, denominator=n, accuracy=correct / n if n else None
    ).model_dump()


def balanced_components(events):
    components = {
        c: _component([e for e in events if e["evidence_class"] == c]) for c in EVIDENCE_CLASSES
    }
    accuracies = [v["accuracy"] for v in components.values()]
    return {
        "components": components,
        "B": None if None in accuracies else 100 * sum(accuracies) / 3,
    }


def response_summary(events, completed_stages, iterations, discoveries):
    reached = max((s["iteration"] for s in completed_stages), default=0)
    scored = []
    for event in events:
        if event["source"] == "discovery":
            continue
        result = EvidenceResult.model_validate(event["result"])
        reference = evidence_status(result)
        reason = (
            "invalid"
            if reference is None
            else "late"
            if event["due_iteration"] > iterations
            else "interrupted"
            if event["due_iteration"] > reached
            else None
        )
        matched = assign(
            [Hypothesis.model_validate(event["hypothesis"])], discoveries, ignore_direction=True
        )
        delayed = event.get("delayed")
        immediate = event.get("immediate")
        eligible = reason is None
        scored.append(
            {
                **event,
                "evidence_class": evidence_class(result),
                "reference": reference,
                "eligible": eligible,
                "exclusion": reason,
                "missing_due": eligible and delayed is None,
                "score": int(delayed is not None and delayed["status"] == reference)
                if eligible
                else None,
                "immediate_score": int(immediate is not None and immediate["status"] == reference)
                if reference is not None
                else None,
                "requires_update": reference != event["prior"],
                "category": matched[0]["category"] if matched else None,
                "target_match": matched[0] if matched else None,
                "evidence_alignment": evidence_alignment(event),
            }
        )
    eligible = [e for e in scored if e["eligible"]]
    result = balanced_components(eligible)
    result.update(
        events=scored,
        eligible_n=len(eligible),
        valid_n=sum(e["reference"] is not None for e in scored),
        accuracy=sum(e["score"] for e in eligible) / len(eligible) if eligible else None,
        total_events=len(scored),
        missing_due_n=sum(e["missing_due"] for e in scored),
        exclusions={
            k: sum(e["exclusion"] == k for e in scored) for k in ("invalid", "late", "interrupted")
        },
        reporting_level="run",
    )
    result["by_source"] = {
        source: balanced_components([e for e in eligible if e["source"] == source])
        for source in ("voluntary", "automatic")
    }
    result["by_category"] = {
        (c or "unmatched"): balanced_components([e for e in eligible if e["category"] == c])
        for c in ("expected", "neutral", "surprising", None)
    }
    result["by_expectation"] = {
        a: balanced_components([e for e in eligible if e["evidence_alignment"] == a])
        for a in sorted({e["evidence_alignment"] for e in eligible})
    }
    return result


def behavioral_summary(state, validation, discoveries, iterations, focal_id):
    hypotheses = state["hypotheses"]
    tests = list(state["tests"].values())
    events = state["events"]
    registrations = state["registrations"]
    reached = max((s["iteration"] for s in state["completed_stages"]), default=0)
    followups = []
    for event in events:
        r = EvidenceResult.model_validate(event["result"])
        h = Hypothesis.model_validate(event["hypothesis"])
        start, end = event["iteration"], event["iteration"] + 2
        complete = end <= min(iterations, reached)
        future_tests = [t for t in tests if start < t["iteration"] <= end]
        future_registrations = [r for r in registrations if start < r["iteration"] <= end]
        linked = [
            r
            for r in future_registrations
            if event["result"]["id"] in r["motivating_result_ids"]
            and r["parent_id"] is not None
            and r["refinement_types"]
        ]
        tested_keys = {t["comparison_key"] for t in future_tests}
        old_families = {
            family(Hypothesis.model_validate(t["hypothesis"]))
            for t in tests
            if t["iteration"] <= start
        }
        future_families = {family(Hypothesis.model_validate(t["hypothesis"])) for t in future_tests}
        same_claim = [
            a
            for a in state["assessments"]
            if start < a["iteration"] <= end
            and claim_key(hypotheses[a["hypothesis_id"]]) == claim_key(h)
        ]
        opposite = [
            r
            for r in future_registrations
            if comparison_key(Hypothesis.model_validate(r["hypothesis"])) == comparison_key(h)
            and claim_key(Hypothesis.model_validate(r["hypothesis"])) != claim_key(h)
        ]
        matched = assign([h], discoveries, ignore_direction=True)
        cls = evidence_class(r)
        reference_tests = [t for t in tests if t["comparison_key"] == comparison_key(h)]
        discovery_class = (
            evidence_class(
                oriented(
                    EvidenceResult.model_validate(reference_tests[0]["result"]),
                    Hypothesis.model_validate(reference_tests[0]["hypothesis"]),
                    h,
                )
            )
            if reference_tests
            else None
        )
        later_validation = [
            e for e in events if e["source"] != "discovery" and start < e["iteration"] <= end
        ]
        overlaps = [
            e["result"]["id"]
            for e in events
            if e is not event and max(start, e["iteration"]) < min(end, e["iteration"] + 2)
        ]
        followups.append(
            {
                "result_id": r.id,
                "hypothesis_id": h.id,
                "source": event["source"],
                "iteration": start,
                "complete": complete,
                "exclusion": None if complete else "late" if end > iterations else "interrupted",
                "category": matched[0]["category"] if matched else None,
                "evidence_class": cls,
                "exploratory_evidence_class": discovery_class,
                "exploratory_validation_disagreement": event["source"] != "discovery"
                and cls is not None
                and discovery_class is not None
                and cls != discovery_class,
                "evidence_alignment": evidence_alignment(event),
                "anticipated_direction": event["anticipated_direction"],
                "pre_evidence_expectation": event["expectation"]["pre_evidence"],
                "normalized_effect": r.estimate / r.delta if r.valid else None,
                "normalized_interval": [r.lower / r.delta, r.upper / r.delta] if r.valid else None,
                "cell_n": r.cell_n,
                "minimum_cell_n": min(r.cell_n) if r.cell_n else None,
                "subgroup_complexity": len({c.variable for c in h.subgroup}),
                "eligibility_complexity": len({c.variable for c in h.eligibility}),
                "remaining_iterations": iterations - start,
                "voluntary_slots_remaining": event.get("voluntary_slots_remaining"),
                "prior_status": event["prior"],
                "prior_investigation": event["prior_investigation"],
                "followup_status": same_claim[-1]["status"] if same_claim else event["prior"],
                "followup_investigation": same_claim[-1]["investigation"]
                if same_claim
                else event["prior_investigation"],
                "comparison_validation": {
                    route: any(
                        e["source"] == route and e["comparison_key"] == comparison_key(h)
                        for e in later_validation
                    )
                    for route in ("voluntary", "automatic")
                },
                "new_tested_comparisons": len(tested_keys),
                "new_comparison_keys": sorted(tested_keys),
                "linked_tested_refinements": len(
                    {
                        comparison_key(Hypothesis.model_validate(r["hypothesis"]))
                        for r in linked
                        if comparison_key(Hypothesis.model_validate(r["hypothesis"])) in tested_keys
                    }
                ),
                "linked_refinements": linked,
                "opposite_direction_registrations": [r["hypothesis"]["id"] for r in opposite],
                "opposite_direction_acceptances": list(
                    {
                        a["hypothesis_id"]
                        for a in state["assessments"]
                        if start < a["iteration"] <= end
                        and a["status"] == "accept"
                        and comparison_key(hypotheses[a["hypothesis_id"]]) == comparison_key(h)
                        and claim_key(hypotheses[a["hypothesis_id"]]) != claim_key(h)
                    }
                ),
                "further_validation": {
                    route: len(
                        {e["comparison_key"] for e in later_validation if e["source"] == route}
                    )
                    for route in ("voluntary", "automatic")
                },
                "repeated_analyses": sum(
                    e["repeated"] for e in state["executions"] if start < e["iteration"] <= end
                ),
                "assessment_and_investigation_transitions": same_claim,
                "new_tested_families": len(future_families - old_families),
                "target_coverage": assign(
                    [Hypothesis.model_validate(t["hypothesis"]) for t in future_tests],
                    discoveries,
                    ignore_direction=True,
                ),
                "overlapping_event_ids": overlaps,
            }
        )
    opportunities = validation["opportunities"]
    available = [o for o in opportunities if o["comparison_key"] is not None]
    first = [dict(a) for a in state["first_acceptances"].values()]
    for a in first:
        h = hypotheses[a["hypothesis_id"]]
        matched = assign([h], discoveries, ignore_direction=True)
        a["category"] = matched[0]["category"] if matched else None
        a["unmatched"] = not matched
        a["history_partition"] = (
            "before_independent_evidence"
            if a["receipt_before"] is None
            else "after_" + a["receipt_before"]["route"]
        )
        a["requests_after"] = [
            r
            for r in validation["requests"]
            if r["comparison_key"] == a["comparison_key"]
            and r["sequence"] >= a["sequence"]
            and r not in a["requests_before"]
        ]
        a["requested_before_received_later"] = bool(
            a["requests_before"] and a["receipt_before"] is None
        )
        a["voluntary_before_acceptance"] = bool(
            a["requests_before"]
            and a["receipt_before"]
            and a["receipt_before"]["route"] == "voluntary"
        )
    tested_first = [a for a in first if a["tested_at_acceptance"]]
    milestones = []
    for d in discoveries:
        target = d.hypothesis
        comparison_tests = [
            t
            for t in tests
            if match(Hypothesis.model_validate(t["hypothesis"]), target, ignore_direction=True)
            == "exact"
        ]
        correct = [
            r
            for r in registrations
            if match(Hypothesis.model_validate(r["hypothesis"]), target) == "exact"
        ]
        acceptances = [a for a in first if match(hypotheses[a["hypothesis_id"]], target) == "exact"]
        target_events = [
            e
            for e in events
            if match(Hypothesis.model_validate(e["hypothesis"]), target, ignore_direction=True)
            == "exact"
        ]
        validation_events = [e for e in target_events if e["source"] != "discovery"]
        supportive = [
            e
            for e in validation_events
            if match(Hypothesis.model_validate(e["hypothesis"]), target) == "exact"
            and evidence_status(EvidenceResult.model_validate(e["result"])) == "accept"
            and e["due_iteration"] <= min(iterations, reached)
        ]
        unexpected = [
            e
            for e in target_events
            if d.expected_direction
            and evidence_alignment(
                {
                    **e,
                    "anticipated_direction": d.expected_direction,
                    "claim_direction": target.direction,
                    "result": oriented(
                        EvidenceResult.model_validate(e["result"]),
                        Hypothesis.model_validate(e["hypothesis"]),
                        target,
                    ).model_dump(),
                }
            )
            == "opposes_expectation"
        ]
        milestones.append(
            {
                "discovery_id": target.id,
                "category": d.category,
                "focal": target.id == focal_id,
                "first_test": min((t["iteration"] for t in comparison_tests), default=None),
                "first_correct_direction": min((r["iteration"] for r in correct), default=None),
                "first_acceptance": min((a["iteration"] for a in acceptances), default=None),
                "surprising_evidence_encountered": bool(unexpected),
                "validation_requested": any(
                    r["comparison_key"] == comparison_key(target) for r in validation["requests"]
                ),
                "validation_received": [
                    {"iteration": e["iteration"], "source": e["source"]} for e in validation_events
                ],
                "supportive_validation_events": len(supportive),
                "accepted_after_supportive_validation": sum(
                    e.get("delayed") is not None and e["delayed"]["status"] == "accept"
                    for e in supportive
                ),
            }
        )
    # Descriptive conditional cells retain covariates in the underlying rows.
    cells = defaultdict(list)
    for f in followups:
        if f["complete"] and f["evidence_class"]:
            key = (
                f["source"],
                f["category"],
                f["evidence_class"],
                f["evidence_alignment"],
                f["subgroup_complexity"],
                "20-99"
                if f["minimum_cell_n"] < 100
                else "100-999"
                if f["minimum_cell_n"] < 1000
                else "1000+",
                "0-2"
                if f["remaining_iterations"] <= 2
                else "3-5"
                if f["remaining_iterations"] <= 5
                else "6+",
            )
            cells[key].append(f)
    conditional = [
        {
            "source": k[0],
            "category": k[1],
            "evidence_class": k[2],
            "expectation_alignment": k[3],
            "subgroup_complexity": k[4],
            "group_size_band": k[5],
            "remaining_budget_band": k[6],
            "event_n": len(v),
            "active_before_rate": sum(f["prior_investigation"] == "active" for f in v) / len(v),
            "active_after_rate": sum(f["followup_investigation"] == "active" for f in v) / len(v),
            "accepted_before_rate": sum(f["prior_status"] == "accept" for f in v) / len(v),
            "accepted_after_rate": sum(f["followup_status"] == "accept" for f in v) / len(v),
            "subsequent_voluntary_validation_rate": sum(
                f["comparison_validation"]["voluntary"] for f in v
            )
            / len(v),
            "mean_new_comparisons": sum(f["new_tested_comparisons"] for f in v) / len(v),
            "mean_linked_tested_refinements": sum(f["linked_tested_refinements"] for f in v)
            / len(v),
        }
        for k, v in cells.items()
    ]
    return {
        "followups": followups,
        "conditional_investigation": conditional,
        "overlap_warning": "Windows overlap; descriptive cells are not independent replicates.",
        "milestones": milestones,
        "validation_choice": {
            "scheduled_available_n": len(available),
            "scheduled_unavailable_n": len(opportunities) - len(available),
            "scheduled_requested_n": sum(o["obtained"] == "voluntary" for o in available),
            "scheduled_request_rate": sum(o["obtained"] == "voluntary" for o in available)
            / len(available)
            if available
            else None,
            "eligible_pool_sizes": [len(o["eligible_pool"]) for o in opportunities],
            "voluntary_slots_used": validation["voluntary_slots_used"],
            "voluntary_slots_available": validation["policy"]["voluntary_limit"]
            - validation["voluntary_slots_used"],
            "first_acceptances": first,
            "tested_first_acceptance_n": len(tested_first),
            "validated_before_acceptance_n": sum(
                a["voluntary_before_acceptance"] for a in tested_first
            ),
            "validated_before_acceptance_rate": sum(
                a["voluntary_before_acceptance"] for a in tested_first
            )
            / len(tested_first)
            if tested_first
            else None,
            "first_acceptance_partitions": {
                p: sum(a["history_partition"] == p for a in tested_first)
                for p in ("before_independent_evidence", "after_voluntary", "after_automatic")
            },
            "unmatched_accepted_n": sum(a["unmatched"] for a in first),
            "untested_first_acceptance_n": len(first) - len(tested_first),
        },
    }
