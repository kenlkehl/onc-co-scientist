"""Select an interpretable focal/background arrangement before generating a pair."""

from __future__ import annotations

import itertools

from .evaluation import calibrate
from .generation import compile_pair


def select_design(profile, reviewed, *, seed, n, focal_mode, reference_n=50000, expected_count=4):
    expected = [r for r in reviewed if r.candidate.proposed_category == "expected"]
    neutral = [r for r in reviewed if r.candidate.proposed_category == "neutral"]
    attempts, options = [], []
    required = (
        "all_discovery_directions",
        "effect_match",
        "snr_match",
        "overall_ordering",
        "reference_supported",
    )
    for focal, surprising in itertools.permutations(range(min(len(expected), 6)), 2):
        others = [r for i, r in enumerate(expected) if i not in {focal, surprising}][
            : expected_count - 2
        ]
        candidates = [expected[focal], *others, expected[surprising], *neutral]
        spec = compile_pair(
            profile,
            candidates,
            seed=seed,
            n=n,
            focal_mode=focal_mode,
            expected_count=expected_count,
        )
        report = calibrate(spec, replicates=0, reference_n=reference_n)
        passed = all(report["criteria"][k] for k in required)
        focal_checks = [
            c
            for version in ("expected", "surprising")
            for c in report["audit"][version]
            if c["id"] == spec.focal_id
        ]
        cell_support = all(min(c["cell_n_reference"]) * n / reference_n >= 30 for c in focal_checks)
        passed = passed and cell_support
        attempts.append(
            {
                "focal_id": spec.focal_id,
                "background_surprising_id": expected[surprising].candidate.hypothesis.id,
                "criteria": report["criteria"],
                "focal_cell_support": cell_support,
                "effect_relative_difference": report["effect_relative_difference"],
                "snr_relative_difference": report["snr_relative_difference"],
            }
        )
        if passed:
            options.append(
                (
                    max(report["effect_relative_difference"], report["snr_relative_difference"]),
                    spec,
                    report,
                )
            )
    if not options:
        raise ValueError(
            f"No valid focal/background arrangement for {profile}. "
            "Revise the candidate inventory before generating datasets. "
            f"Attempted {len(attempts)} arrangements."
        )
    _, spec, report = min(options, key=lambda x: x[0])
    return spec, {"attempts": attempts, "selected": report}
