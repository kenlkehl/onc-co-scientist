"""Fail-closed gates for reviewed scientific claims and their compiled DGPs."""

from __future__ import annotations

import hashlib
import json

from .schemas import Candidate, PairSpec, ReviewedCandidate

REVIEW_POLICY_VERSION = "2.1.0"


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def candidate_digest(candidate: Candidate) -> str:
    return digest(candidate.model_dump())


def spec_digest(spec: PairSpec) -> str:
    # Review metadata and the administrative lock status do not alter the DGP.
    payload = spec.model_dump(exclude={"dgp_review", "status"})
    for item in payload["evidence"]:
        for source in item["sources"]:
            source.pop("abstract", None)
            source.pop("full_text_excerpt", None)
    return digest(payload)


def require_current_candidate(item: ReviewedCandidate, profile: str):
    item = ReviewedCandidate.model_validate(item.model_dump())
    if item.review_policy_version != REVIEW_POLICY_VERSION or item.review_profile != profile:
        raise ValueError("Candidate needs current literature and realism review for this profile")
    if item.candidate_sha256 != candidate_digest(item.candidate):
        raise ValueError("Candidate changed after review")
    review, realism = item.review, item.realism
    if realism is None or realism.decision != "accept" or realism.required_changes:
        raise ValueError("Candidate did not pass independent realism review")
    if realism.unrepresented_scope_requirements or not realism.counterexample_search.strip():
        raise ValueError(
            "Realism review has unrepresented scope requirements or no counterexample audit"
        )
    encoded = item.candidate.hypothesis.eligibility + item.candidate.hypothesis.subgroup
    for requirement in realism.necessary_scope_conditions:
        if not any(condition_implies(c, requirement) for c in encoded):
            raise ValueError(
                f"Clinically necessary restriction is not encoded: {requirement.model_dump()}"
            )
    if not all(
        (
            realism.population_scope_valid,
            realism.comparator_scope_valid,
            realism.biomarker_restrictions_complete,
            realism.assay_transfer_valid,
            realism.category_valid,
        )
    ):
        raise ValueError("Realism review identified an invalid scope, assay, or category")
    ids = {s.id for s in item.sources}
    if not set(realism.source_ids) <= ids or any(
        s.source_id not in ids for s in review.evidence_scopes
    ):
        raise ValueError("Realism or evidence scope cites a source that was not retrieved")
    if item.candidate.proposed_category == "expected":
        matching = [
            s
            for s in review.evidence_scopes
            if s.primary_empirical
            and s.source_id in review.source_ids
            and s.direction == item.candidate.hypothesis.direction
            and s.endpoint_kind == ("pfs" if profile.endswith("clinical") else "genetic_dependency")
        ]
        if not matching:
            raise ValueError(
                "Expected candidate needs an explicit primary empirical evidence scope"
            )
    elif (
        review.broader_direction != "none"
        or not review.neutrality_rationale.strip()
        or not realism.no_inherited_directional_expectation
    ):
        raise ValueError("Neutrality requires absence of an inherited directional expectation")


def condition_implies(encoded, required):
    if encoded.variable != required.variable:
        return False
    if required.op == "eq":
        return encoded.op == "eq" and encoded.value == required.value
    if isinstance(encoded.value, str) or isinstance(required.value, str):
        return False
    if required.op == "ge":
        return encoded.op in {"eq", "ge"} and encoded.value >= required.value
    return encoded.op in {"eq", "le"} and encoded.value <= required.value


def reject_nested_comparison(candidate: Candidate, accepted: list[ReviewedCandidate]):
    h = candidate.hypothesis
    conditions = {(c.variable, c.op, c.value) for c in h.eligibility}
    for previous in accepted:
        p = previous.candidate.hypothesis
        if (h.exposure, h.outcome, h.exposed, h.comparator) != (
            p.exposure,
            p.outcome,
            p.exposed,
            p.comparator,
        ):
            continue
        other = {(c.variable, c.op, c.value) for c in p.eligibility}
        if conditions <= other or other <= conditions:
            raise ValueError(
                "Redundant overall/nested comparison for the same exposure and outcome"
            )


def require_current_pair(spec: PairSpec):
    for item in spec.evidence:
        require_current_candidate(item, spec.profile)
    review = spec.dgp_review
    if review is None or review.policy_version != REVIEW_POLICY_VERSION:
        raise ValueError(
            "Pair needs current full-DGP realism review before materialization or evaluation"
        )
    if review.spec_sha256 != spec_digest(spec):
        raise ValueError("DGP changed after realism review")
    a = review.assessment
    if (
        a.decision != "accept"
        or a.required_changes
        or not all(
            (
                a.expected_components_scoped,
                a.intentional_reversals_only,
                a.nonredundant_targets,
                a.plausible_outcome_scale,
            )
        )
    ):
        raise ValueError("Full-DGP realism review rejected the pair")
