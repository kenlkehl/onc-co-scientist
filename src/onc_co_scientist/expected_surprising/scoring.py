"""Deterministic discovery matching and prespecified evidence calculations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import t

from .generation import mask
from .schemas import Discovery, EvidenceResult, Hypothesis


def conditions_key(conditions) -> tuple:
    return tuple(sorted((c.variable, c.op, scalar_key(c.value)) for c in conditions))


def scalar_key(value) -> str:
    return json.dumps(float(value) if isinstance(value, (float, int)) else value)


def canonical(h: Hypothesis, *, ignore_direction=False) -> tuple:
    a, b = sorted((scalar_key(h.exposed), scalar_key(h.comparator)))
    direction = h.direction * (1 if scalar_key(h.exposed) == a else -1)
    return (
        h.outcome,
        h.exposure,
        a,
        b,
        h.contrast,
        conditions_key(h.eligibility),
        conditions_key(h.subgroup),
        0 if ignore_direction else direction,
    )


def family(h: Hypothesis) -> tuple:
    key = canonical(h, ignore_direction=True)
    return key[:6] + (tuple(sorted({c.variable for c in h.subgroup})),)


def match(h: Hypothesis, target: Hypothesis, *, ignore_direction=False) -> str | None:
    hk, tk = (
        canonical(h, ignore_direction=ignore_direction),
        canonical(target, ignore_direction=ignore_direction),
    )
    if hk == tk:
        return "exact"
    if hk[:6] != tk[:6] or hk[-1] != tk[-1]:
        return None
    hv, tv = {c.variable for c in h.subgroup}, {c.variable for c in target.subgroup}
    if len(tv) >= 2 and len(tv - hv) == 1 and not hv - tv:
        retained = [c for c in target.subgroup if c.variable in hv]
        if conditions_key(h.subgroup) == conditions_key(retained):
            return "near"
    return None


def assign(
    hypotheses: list[Hypothesis], discoveries: list[Discovery], *, ignore_direction=False
) -> list[dict]:
    unique = {}
    for h in hypotheses:
        unique.setdefault(canonical(h, ignore_direction=ignore_direction), h)
    claims = sorted(unique.values(), key=lambda h: h.id)
    targets = sorted(discoveries, key=lambda d: d.hypothesis.id)
    if not claims or not targets:
        return []
    weights = np.zeros((len(claims), len(targets)), dtype=int)
    for i, h in enumerate(claims):
        for j, d in enumerate(targets):
            kind = match(h, d.hypothesis, ignore_direction=ignore_direction)
            weights[i, j] = len(targets) + 1 if kind == "exact" else int(kind == "near")
    rows, cols = linear_sum_assignment(weights, maximize=True)
    result = []
    for i, j in zip(rows, cols, strict=True):
        if not weights[i, j]:
            continue
        h, d = claims[i], targets[j]
        omitted = {c.variable for c in d.hypothesis.subgroup} - {c.variable for c in h.subgroup}
        result.append(
            {
                "hypothesis_id": h.id,
                "discovery_id": d.hypothesis.id,
                "category": d.category,
                "match": "exact" if weights[i, j] > 1 else "near",
                "omitted_variables": sorted(omitted),
                "omits_paradigm_variable": bool(omitted & set(d.paradigm_bearing_variables)),
            }
        )
    return result


def cells(frame: pd.DataFrame, h: Hypothesis) -> tuple[list[np.ndarray], list[int]]:
    if h.exposure not in frame or h.outcome not in frame:
        raise ValueError("Unknown exposure or outcome")
    eligible = mask(frame, h.eligibility)
    subgroup = mask(frame, h.subgroup)
    # Missing subgroup values are excluded from both S and its complement.
    for c in h.subgroup:
        eligible &= frame[c.variable].notna().to_numpy()
    groups, signs = [], []
    for region, weight in ((subgroup, 1), (~subgroup, -1)):
        if weight == -1 and h.contrast != "interaction":
            break
        for level, sign in ((h.exposed, 1), (h.comparator, -1)):
            selected = eligible & region & (frame[h.exposure] == level).to_numpy()
            values = frame.loc[selected, h.outcome].to_numpy(dtype=float)
            groups.append(values[np.isfinite(values)])
            signs.append(weight * sign * h.direction)
    return groups, signs


def estimate(
    frame: pd.DataFrame,
    h: Hypothesis,
    *,
    delta: float,
    alpha: float,
    result_id: str,
    min_cell_n: int = 20,
) -> EvidenceResult:
    if not 0 < alpha < 1 or delta <= 0:
        raise ValueError("Alpha must be in (0,1) and delta positive")
    kwargs = dict(id=result_id, hypothesis_id=h.id, delta=delta, alpha=alpha)
    try:
        groups, signs = cells(frame, h)
    except (ValueError, KeyError, TypeError) as exc:
        return EvidenceResult(**kwargs, cell_n=[], valid=False, diagnostic=str(exc))
    sizes = [len(g) for g in groups]
    if min(sizes) < min_cell_n:
        return EvidenceResult(**kwargs, cell_n=sizes, valid=False, diagnostic="insufficient_cell_n")
    variances = np.array([np.var(g, ddof=1) / len(g) for g in groups])
    variance = variances.sum()
    if variance <= 0 or not np.isfinite(variance):
        return EvidenceResult(**kwargs, cell_n=sizes, valid=False, diagnostic="invalid_variance")
    dof = variance**2 / sum(v**2 / (n - 1) for v, n in zip(variances, sizes, strict=True))
    value = sum(sign * float(g.mean()) for g, sign in zip(groups, signs, strict=True))
    half = float(t.ppf(1 - alpha / 2, dof) * np.sqrt(variance))
    return EvidenceResult(
        **kwargs,
        estimate=value,
        lower=value - half,
        upper=value + half,
        cell_n=sizes,
        valid=True,
        diagnostic="welch_independent_cells",
    )


def evidence_status(result: EvidenceResult) -> str | None:
    if not result.valid:
        return None
    if result.lower > result.delta:
        return "accept"
    if result.upper < result.delta:
        return "reject"
    return "unresolved"


def responsiveness(events: list[dict]) -> dict:
    scored = []
    invalid = 0
    for event in events:
        result = EvidenceResult.model_validate(event["result"])
        expected = evidence_status(result)
        if expected is None:
            invalid += 1
            continue
        scored.append(
            {
                "result_id": result.id,
                "expected": expected,
                "score": int(event.get("decision") == expected),
                "requires_update": event.get("prior") != expected,
                "anticipated_direction": event.get("anticipated_direction"),
                "evidence_alignment": evidence_alignment(event),
            }
        )
    updates = [e for e in scored if e["requires_update"]]
    maintained = [e for e in scored if not e["requires_update"]]

    def rate(records):
        return sum(r["score"] for r in records) / len(records) if records else None

    return {
        "valid_n": len(scored),
        "invalid_n": invalid,
        "accuracy": rate(scored),
        "update_accuracy": rate(updates),
        "maintenance_accuracy": rate(maintained),
        "events": scored,
    }


def evidence_alignment(event: dict) -> str:
    anticipated = event.get("anticipated_direction")
    if not anticipated:
        return "no_directional_expectation"
    result = EvidenceResult.model_validate(event["result"])
    if not result.valid:
        return "invalid_analysis"
    sign = anticipated * event["claim_direction"]
    lower, upper = sorted((result.lower * sign, result.upper * sign))
    if lower > 0:
        return "agrees_with_expectation"
    if upper < 0:
        return "opposes_expectation"
    return "direction_unresolved"


def recovery(hypotheses: list[Hypothesis], discoveries: list[Discovery]) -> dict:
    assigned = assign(hypotheses, discoveries)
    categories = {}
    for category in ("expected", "neutral", "surprising"):
        n = sum(d.category == category for d in discoveries)
        categories[category] = {
            "available": n,
            **{
                kind: sum(m["category"] == category and m["match"] == kind for m in assigned)
                for kind in ("exact", "near")
            },
        }
    return {"categories": categories, "matches": assigned}


CATEGORIES = ("expected", "neutral", "surprising")
EVIDENCE_CLASSES = ("supported", "excluded", "ambiguous")


def comparison_key(h: Hypothesis) -> str:
    return json.dumps(canonical(h, ignore_direction=True), separators=(",", ":"))


def claim_key(h: Hypothesis) -> str:
    return json.dumps(canonical(h), separators=(",", ":"))


def oriented(result: EvidenceResult, original: Hypothesis, h: Hypothesis) -> EvidenceResult:
    """Reuse one numerical comparison under either direction or exposure recoding."""
    if comparison_key(original) != comparison_key(h):
        raise ValueError("Cannot reorient a different comparison")
    sign = canonical(original)[-1] * canonical(h)[-1]
    changes = {"hypothesis_id": h.id}
    if scalar_key(original.exposed) != scalar_key(h.exposed):
        changes["cell_n"] = [
            n for i in range(0, len(result.cell_n), 2) for n in reversed(result.cell_n[i : i + 2])
        ]
    if result.valid:
        lo, hi = sorted((sign * result.lower, sign * result.upper))
        changes.update(estimate=sign * result.estimate, lower=lo, upper=hi)
    return result.model_copy(update=changes)


def evidence_class(result: EvidenceResult) -> str | None:
    return {"accept": "supported", "reject": "excluded", "unresolved": "ambiguous"}.get(
        evidence_status(result)
    )


def refinement_types(parent: Hypothesis, child: Hypothesis) -> list[str]:
    """Derive structural changes, independent of prose and arbitrary identifiers."""
    pk, ck = canonical(parent), canonical(child)
    changes = []
    if pk[-1] != ck[-1]:
        changes.append("direction_change")
    for field in ("eligibility", "subgroup"):
        before, after = getattr(parent, field), getattr(child, field)
        bv, av = {c.variable for c in before}, {c.variable for c in after}
        if av - bv:
            changes.append(f"{field}_condition_addition")
        if bv - av:
            changes.append(f"{field}_condition_removal")
        common = av & bv
        if conditions_key([c for c in before if c.variable in common]) != conditions_key(
            [c for c in after if c.variable in common]
        ):
            changes.append("cutoff_change")
    if pk[:5] != ck[:5]:
        changes.append("comparison_change")
    return sorted(set(changes))


def discovery_performance(recovery_rates: dict, q: float | None) -> dict:
    r = sum(recovery_rates[c] for c in CATEGORIES) / 3
    return {
        "R_c": recovery_rates,
        "R": r,
        "Q": q,
        "D": 0.0 if q is None or r + q == 0 else 100 * 2 * r * q / (r + q),
    }


def workflow_discovery(
    accepted, discoveries, tested_keys, confirmation, *, failed=False, repaired=False
):
    claims = list({claim_key(h): h for h in accepted}.values())
    results = {
        r["hypothesis_id"]: EvidenceResult.model_validate(r) for r in confirmation["results"]
    }
    supported = [
        h for h in claims if h.id in results and evidence_status(results[h.id]) == "accept"
    ]
    confirmed = [h for h in supported if comparison_key(h) in tested_keys]
    assigned = assign(confirmed, discoveries)
    q = len(confirmed) / len(claims) if claims else None
    scores = {}
    for label, kinds in (("exact", {"exact"}), ("exact_or_near", {"exact", "near"})):
        rates = {
            c: sum(m["category"] == c and m["match"] in kinds for m in assigned)
            / sum(d.category == c for d in discoveries)
            for c in CATEGORIES
        }
        scores[label] = discovery_performance(rates, q)
        scores[label]["diagnostic_D"] = scores[label]["D"]
        if failed:
            scores[label]["D"] = 0.0
        scores[label]["first_attempt_D"] = 0.0 if repaired else scores[label]["D"]
    scores.update(
        accepted_n=len(claims),
        tested_accepted_n=sum(comparison_key(h) in tested_keys for h in claims),
        confirmed_tested_n=len(confirmed),
        independently_supported_n=len(supported),
        confirmed_matches=assigned,
        unsupported_acceptance_n=sum(
            h.id in results and evidence_status(results[h.id]) == "reject" for h in claims
        ),
        unconfirmed_n=len(claims) - len(supported),
    )
    return scores


def exploration_coverage(tests, discoveries, iterations):
    curves = {kind: {c: [] for c in CATEGORIES} for kind in ("exact", "exact_or_near")}
    for iteration in range(1, iterations + 1):
        hypotheses = [
            Hypothesis.model_validate(e["hypothesis"])
            for e in tests
            if e["iteration"] <= iteration and e["result"]["valid"]
        ]
        matches = assign(hypotheses, discoveries, ignore_direction=True)
        for kind, categories in curves.items():
            for c in CATEGORIES:
                categories[c].append(
                    sum(
                        m["category"] == c and (kind == "exact_or_near" or m["match"] == "exact")
                        for m in matches
                    )
                    / sum(d.category == c for d in discoveries)
                )
    return {
        kind: {
            "E": 100 * sum(sum(v) for v in categories.values()) / (3 * iterations),
            "curves": categories,
            "final_coverage": {c: v[-1] for c, v in categories.items()},
            "configured_iterations": iterations,
        }
        for kind, categories in curves.items()
    }
