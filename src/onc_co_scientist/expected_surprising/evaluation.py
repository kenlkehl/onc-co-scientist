"""Private validation service, paired calibration, and final confirmation."""

from __future__ import annotations

import hashlib
import itertools
from math import comb

import numpy as np
from scipy.stats import norm

from .generation import base_frame, conditional_means, sample, version_discoveries
from .schemas import Condition, Hypothesis, PairSpec
from .scoring import assign, canonical, cells, estimate, evidence_status, recovery

REFERENCE_POLICY_VERSION = "2.0.0"
MAX_REFERENCE_ELIGIBILITY = 4


class ReferenceUniverse:
    """Public v2 comparison grammar, counted without constructing millions of objects.

    Binary exposures; up to four other binary eligibility variables, each 0 or 1;
    and zero, two or three research signatures >= 0 or >= 0.5. Every comparison
    is a two-sided mean-difference test. The count includes all combinations,
    even sparse or structurally empty ones, giving a conservative correction.
    """

    def __init__(self, frame, outcomes):
        self.outcomes = set(outcomes)
        self.signatures = {c for c in frame if c.startswith("research_signature_")}
        self.binary = {
            c
            for c in frame
            if c not in self.signatures | self.outcomes
            and set(frame[c].dropna().unique()) == {0, 1}
        }
        b, s = len(self.binary), len(self.signatures)
        eligibility_n = sum(
            comb(b - 1, k) * 2**k for k in range(min(MAX_REFERENCE_ELIGIBILITY, b - 1) + 1)
        )
        subgroup_n = 1 + sum(comb(s, k) * 2**k for k in (2, 3) if k <= s)
        self.size = b * len(self.outcomes) * eligibility_n * subgroup_n

    def contains(self, h: Hypothesis) -> bool:
        return (
            h.exposure in self.binary
            and h.outcome in self.outcomes
            and {h.exposed, h.comparator} == {0, 1}
            and h.contrast == "mean_difference"
            and len(h.eligibility) <= MAX_REFERENCE_ELIGIBILITY
            and len({c.variable for c in h.eligibility}) == len(h.eligibility)
            and all(
                c.variable in self.binary - {h.exposure} and c.op == "eq" and c.value in (0, 1)
                for c in h.eligibility
            )
            and len(h.subgroup) in (0, 2, 3)
            and len({c.variable for c in h.subgroup}) == len(h.subgroup)
            and all(
                c.variable in self.signatures and c.op == "ge" and c.value in (0, 0.5)
                for c in h.subgroup
            )
        )


class ValidationService:
    """Run-scoped private service. Repeated IDs return the original result."""

    def __init__(self, spec: PairSpec, version: str, run_id: str):
        self.spec, self.version = spec, version
        self.seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "little")
        self.results = {}
        self.iterations = set()

    def request(self, h: Hypothesis, iteration: int, request_id: str):
        if request_id in self.results:
            old_h, old_iteration, result = self.results[request_id]
            if canonical(old_h) != canonical(h) or old_iteration != iteration:
                raise ValueError(
                    "A validation request cannot be revised after evidence is returned"
                )
            return result
        if iteration in self.iterations or len(self.results) >= 10:
            raise ValueError("Validation budget exhausted for this iteration or run")
        outcome = next((o for o in self.spec.outcomes if o.name == h.outcome), None)
        if outcome is None:
            raise ValueError("Unknown outcome")
        index = len(self.results)
        seed = int(
            np.random.SeedSequence([self.seed, self.spec.seed, 302, index]).generate_state(1)[0]
        )
        frame = sample(self.spec, self.version, seed=seed)
        result = estimate(frame, h, delta=outcome.delta, alpha=0.05 / 10, result_id=request_id)
        self.results[request_id] = (h.model_copy(deep=True), iteration, result)
        self.iterations.add(iteration)
        return result

    def confirm(self, hypotheses: list[Hypothesis]) -> dict:
        claims = list({canonical(h): h for h in hypotheses}.values())
        discoveries = version_discoveries(self.spec, self.version)
        if not claims:
            return {"primary_recovery": 0, "results": [], "confirmed_matches": [], "additional": []}
        seed = int(np.random.SeedSequence([self.seed, self.spec.seed, 915]).generate_state(1)[0])
        frame = sample(self.spec, self.version, seed=seed)
        outcomes = {o.name: o for o in self.spec.outcomes}
        results = [
            estimate(
                frame,
                h,
                delta=outcomes[h.outcome].delta,
                alpha=0.05 / len(claims),
                result_id=f"final-{i}",
            )
            for i, h in enumerate(claims)
        ]
        confirmed = [
            h for h, r in zip(claims, results, strict=True) if evidence_status(r) == "accept"
        ]
        assigned = assign(confirmed, discoveries)
        all_assignments = assign(claims, discoveries)
        target_claim_ids = {m["hypothesis_id"] for m in all_assignments}
        # Additional claims are independently confirmed or unconfirmed. No
        # unsupported inference that an unmatched association is false.
        additional = [
            {
                "hypothesis_id": h.id,
                "status": "confirmed" if evidence_status(r) == "accept" else "unconfirmed",
            }
            for h, r in zip(claims, results, strict=True)
            if h.id not in target_claim_ids
        ]
        if additional:
            reference = base_frame(
                self.spec.profile, 100000, seed + 1, version=self.spec.generation_version
            )
            for name, values in conditional_means(self.spec, reference, self.version).items():
                reference[name] = values
            by_id = {h.id: h for h in claims}
            for item in additional:
                h = by_id[item["hypothesis_id"]]
                item.update(
                    adjudicate(reference, h, outcomes[h.outcome].delta, alpha=0.05 / len(claims))
                )
        return {
            "primary_recovery": int(
                any(
                    m["discovery_id"] == self.spec.focal_id and m["match"] == "exact"
                    for m in assigned
                )
            ),
            "results": [r.model_dump() for r in results],
            "confirmed_matches": assigned,
            "confirmed_recovery": recovery(confirmed, discoveries),
            "additional": additional,
        }


def adjudicate(mean_frame, h: Hypothesis, delta: float, *, alpha=0.005) -> dict:
    """Evaluate an extra claim using full-DGP conditional means and MC uncertainty."""
    try:
        groups, signs = cells(mean_frame, h)
    except (ValueError, KeyError, TypeError):
        return {"dgp_status": "unadjudicated"}
    if min(map(len, groups)) < 20:
        return {"dgp_status": "unadjudicated"}
    effect = sum(sign * float(g.mean()) for g, sign in zip(groups, signs, strict=True))
    se = np.sqrt(sum(float(g.var(ddof=1)) / len(g) for g in groups))
    half = float(norm.ppf(1 - alpha / 2) * se)
    status = (
        "supported"
        if effect - half > delta
        else ("contradicted" if effect + half < delta else "unresolved")
    )
    return {
        "dgp_status": status,
        "dgp_signed_effect": effect,
        "mc_interval": [effect - half, effect + half],
        "mc_reference_n": len(mean_frame),
    }


def reference_grid(frame, outcomes: list[str]) -> list[Hypothesis]:
    """Fixed public search universe; no use of target IDs or private predicates."""
    binary = [c for c in frame if set(frame[c].dropna().unique()) == {0, 1}]
    treatments = [c for c in binary if c.startswith("treatment_")]
    signatures = [c for c in frame if c.startswith("research_signature_")]
    groups = [[]]
    for size in (2, 3):
        for variables in itertools.combinations(signatures, size):
            for cuts in itertools.product((0.0, 0.5), repeat=size):
                groups.append(
                    [
                        Condition(variable=v, op="ge", value=cut)
                        for v, cut in zip(variables, cuts, strict=True)
                    ]
                )
    grid = []
    for exposure, outcome, subgroup in itertools.product(binary, outcomes, groups):
        for treatment in [None] + [c for c in treatments if c != exposure]:
            eligibility = [] if treatment is None else [Condition(variable=treatment, value=1)]
            grid.append(
                Hypothesis(
                    id=f"reference-{len(grid)}",
                    exposure=exposure,
                    outcome=outcome,
                    direction=1,
                    eligibility=eligibility,
                    subgroup=subgroup,
                )
            )
    return grid


def calibrate(spec: PairSpec, *, replicates: int = 1000, reference_n: int = 100_000) -> dict:
    """Full-DGP audit plus repeated recovery of the focal contrast in a fixed grid.

    The grid tests every prespecified comparison with Bonferroni correction;
    focal recovery is therefore exactly computable from the focal test and
    grid size without evaluating every non-focal test on every replicate.
    This is a reference procedure, separate from agent performance.
    """
    if replicates < 0 or reference_n < 100:
        raise ValueError("Nonnegative replicate count and at least 100 reference rows required")
    reference = base_frame(
        spec.profile, reference_n, spec.seed + 8_000_000, version=spec.generation_version
    )
    if spec.generation_version == 1:
        grid = reference_grid(reference, [o.name for o in spec.outcomes])
        grid_keys = {canonical(h, ignore_direction=True) for h in grid}
        grid_size = len(grid)
        contains = lambda h: canonical(h, ignore_direction=True) in grid_keys  # noqa: E731
        reference_policy = "1.0.0"
    else:
        universe = ReferenceUniverse(reference, [o.name for o in spec.outcomes])
        grid_size, contains = universe.size, universe.contains
        reference_policy = REFERENCE_POLICY_VERSION
    comparisons, audit, rates = {}, {}, {}
    outcomes = {o.name: o for o in spec.outcomes}
    for version in ("expected", "surprising"):
        mean_frame = reference.copy()
        for outcome, values in conditional_means(spec, reference, version).items():
            mean_frame[outcome] = values
        checks = []
        for d in version_discoveries(spec, version):
            groups, signs = cells(mean_frame, d.hypothesis)
            effect = sum(sign * float(g.mean()) for g, sign in zip(groups, signs, strict=True))
            o = outcomes[d.hypothesis.outcome]
            variances = [float(g.var(ddof=1)) + o.sigma**2 for g in groups]
            se = np.sqrt(
                sum(
                    v / (len(g) * spec.n / reference_n)
                    for v, g in zip(variances, groups, strict=True)
                )
            )
            check = {
                "id": d.hypothesis.id,
                "category": d.category,
                "signed_effect": effect,
                "se_at_task_n": float(se),
                "snr": float(effect / se),
                "cell_n_reference": [len(g) for g in groups],
                "direction_valid": effect > o.delta,
            }
            checks.append(check)
            if d.hypothesis.id == spec.focal_id:
                comparisons[version] = check
                focal = d.hypothesis
        audit[version] = checks
        if spec.focal_mode == "subgroup":
            overall = focal.model_copy(
                update={
                    "subgroup": [],
                    "direction": next(
                        d.expected_direction
                        for d in spec.discoveries
                        if d.hypothesis.id == spec.focal_id
                    ),
                }
            )
            groups, signs = cells(mean_frame, overall)
            audit[version + "_overall_expected_effect"] = sum(
                sign * float(g.mean()) for g, sign in zip(groups, signs, strict=True)
            )
        recoveries = []
        for i in range(replicates):
            frame = sample(spec, version, seed=spec.seed + 9_000_000 + i)
            result = estimate(
                frame,
                focal,
                delta=outcomes[focal.outcome].delta,
                alpha=0.05 / grid_size,
                result_id=f"reference-{i}",
            )
            recoveries.append(contains(focal) and evidence_status(result) == "accept")
        rates[version] = float(np.mean(recoveries)) if recoveries else None

    def mismatch(field):
        a, b = abs(comparisons["expected"][field]), abs(comparisons["surprising"][field])
        return abs(a - b) / max(a, b) if max(a, b) else float("inf")

    criteria = {
        "reference_supported": contains(focal),
        "all_discovery_directions": all(
            c["direction_valid"] for v in ("expected", "surprising") for c in audit[v]
        ),
        "effect_match": mismatch("signed_effect") <= 0.10,
        "snr_match": mismatch("snr") <= 0.10,
        "reference_recovery": replicates > 0 and min(rates.values()) >= 0.80,
        "recovery_gap": replicates > 0 and abs(rates["expected"] - rates["surprising"]) <= 0.05,
        "adequate_development_replicates": replicates >= 1000,
        "overall_ordering": spec.focal_mode != "subgroup"
        or all(audit[v + "_overall_expected_effect"] > 0 for v in ("expected", "surprising")),
    }
    return {
        "pair_id": spec.pair_id,
        "replicates": replicates,
        "reference_n": reference_n,
        "reference_policy_version": reference_policy,
        "reference_grid_size": grid_size,
        "reference_recovery": rates,
        "effect_relative_difference": mismatch("signed_effect"),
        "snr_relative_difference": mismatch("snr"),
        "audit": audit,
        "criteria": criteria,
        "calibration_passed": all(criteria.values()),
        "status": "development",
        "expert_review_complete": False,
    }
