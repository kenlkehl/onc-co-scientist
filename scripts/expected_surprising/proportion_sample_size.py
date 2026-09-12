#!/usr/bin/env python3
"""Plan independent binary runs against a fixed proportion benchmark.

Uses the normal approximation for a two-sided one-sample proportion test:
    n = (z_(1-alpha/2) sqrt(p0(1-p0)) + z_power sqrt(p1(1-p1)))**2 / (p1-p0)**2

This reproduces the planning calculations discussed for focal recovery. It is
not an exact binomial power calculation and uses no continuity correction.
The benchmark is treated as known without sampling error. These sample sizes
do not apply directly to continuous run scores or correlated findings/runs.

Examples (run from the repository root; Python standard library only):
    python3 scripts/expected_surprising/proportion_sample_size.py
    python3 scripts/expected_surprising/proportion_sample_size.py --benchmark .6
    python3 scripts/expected_surprising/proportion_sample_size.py --benchmark .5 --json

All probabilities are fractions: --change .05 means five percentage points.
"""

import argparse
import json
import math
from statistics import NormalDist


def required_runs(benchmark: float, alternative: float, power: float, alpha: float) -> float:
    """Return the unrounded normal-approximation sample size."""
    for name, value in (("benchmark", benchmark), ("alternative", alternative), ("alpha", alpha)):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    if not math.isfinite(power) or not 0.5 < power < 1:
        raise ValueError("power must be finite and strictly between 0.5 and 1")
    if benchmark == alternative:
        raise ValueError("alternative must differ from benchmark")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    return (
        (
            z_alpha * math.sqrt(benchmark * (1 - benchmark))
            + z_power * math.sqrt(alternative * (1 - alternative))
        )
        / abs(alternative - benchmark)
    ) ** 2


def calculate(
    benchmark: float = 0.70, change: float = 0.05, power: float = 0.80, alpha: float = 0.05
) -> dict:
    """Evaluate decreases and increases, retaining both exact formula outputs."""
    if not math.isfinite(change) or change <= 0:
        raise ValueError("change must be finite and positive")
    alternatives = []
    for direction, alternative in (
        ("decrease", benchmark - change), ("increase", benchmark + change)
    ):
        unrounded = required_runs(benchmark, alternative, power, alpha)
        alternatives.append(
            {
                "direction": direction,
                "alternative": alternative,
                "unrounded_runs": unrounded,
                "required_runs": math.ceil(unrounded),
            }
        )
    return {
        "method": "Two-sided one-sample proportion test; normal approximation",
        "assumptions": "Independent binary runs; fixed known benchmark; no continuity correction",
        "benchmark": benchmark,
        "absolute_change": change,
        "power": power,
        "alpha": alpha,
        "alternatives": alternatives,
        "runs_to_cover_either_direction": max(row["required_runs"] for row in alternatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--benchmark", type=float, default=0.70)
    parser.add_argument("--change", type=float, default=0.05, help="Absolute change (default: .05)")
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args()
    try:
        result = calculate(args.benchmark, args.change, args.power, args.alpha)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    print(result["method"])
    print(result["assumptions"])
    print(f"Benchmark: {args.benchmark:.1%}; power: {args.power:.1%}; alpha: {args.alpha:g}")
    for row in result["alternatives"]:
        print(
            f"{row['direction'].capitalize()} to {row['alternative']:.1%}: "
            f"{row['required_runs']:,} runs (unrounded: {row['unrounded_runs']:.3f})"
        )
    print(f"Plan for either direction: {result['runs_to_cover_either_direction']:,} total runs")
    print("Approximate power planning, not a confidence-interval margin-of-error calculation.")


if __name__ == "__main__":
    main()
