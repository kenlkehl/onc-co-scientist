"""Summarize a closed NSCLC batch and its archived Luna reference scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def wilson(k: int, n: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    # Keep exact endpoints at zero/all successes despite floating point rounding.
    return (0.0 if k == 0 else max(0.0, center - half), 1.0 if k == n else min(1.0, center + half))


def generate(root: Path, baseline: Path, out: Path) -> None:
    current = pd.read_csv(root / "final_report" / "run_scores.csv")
    previous = pd.read_csv(baseline)
    previous = previous[previous.task.eq("nsclc")].copy()
    assert len(current) == 40 and set(current.task) == {"nsclc"}
    assert current.groupby("variant").size().to_dict() == {"anonymized": 20, "named": 20}
    assert current.submitted_iterations.eq(25).all() and not current.terminal_failure.any()
    assert set(current.model_id) == {"gpt-5.6-sol"}
    assert len(previous) == 40 and set(previous.model_id) == {"gpt-5.6-luna"}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, frame in [("Luna 5.6 medium", previous), ("Sol 5.6 medium", current)]:
        for variant in ["named", "anonymized"]:
            part = frame[frame.variant.eq(variant)]
            n, k = len(part), int(part.primary_recovered.sum())
            lo, hi = wilson(k, n)
            rows.append(
                dict(
                    model=model,
                    condition="Masked" if variant == "anonymized" else "Named",
                    n=n,
                    recovered=k,
                    rate=k / n,
                    wilson_95_low=lo,
                    wilson_95_high=hi,
                    strict=int(part.strict_recovered.sum()),
                    confirmed=int(part.confirmed_recovered.sum()),
                    interaction_confirmed=int(part.interaction_confirmed_recovered.sum()),
                    median_unique_claims=float(part.unique_structured_claims.median()),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "model_comparison.csv", index=False)
    inspections = [
        json.loads(x) for x in (root / "interim_inspections.jsonl").read_text().splitlines()
    ]
    comparison = dict(
        scope="Descriptive comparison of model sessions on one fixed synthetic NSCLC cohort",
        formal_runs=40,
        excluded_setup_runs=2,
        total_formal_iterations=1000,
        baseline_file=baseline.as_posix(),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        groups=rows,
        interim_inspections=len(inspections),
        intervals="95% Wilson intervals for session recovery, not biological replication",
    )
    (out / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for offset, model, color in [
        (-0.16, "Luna 5.6 medium", "#8A919B"),
        (0.16, "Sol 5.6 medium", "#2463A8"),
    ]:
        part = table[table.model.eq(model)]
        x = [offset, 1 + offset]
        rates = part.rate.to_numpy() * 100
        ax.bar(x, rates, width=0.28, color=color, label=model)
        ax.errorbar(
            x,
            rates,
            yerr=[
                rates - part.wilson_95_low.to_numpy() * 100,
                part.wilson_95_high.to_numpy() * 100 - rates,
            ],
            fmt="none",
            color=color,
            capsize=4,
        )
        for xi, (_, row) in zip(x, part.iterrows(), strict=True):
            ax.text(
                xi,
                row.wilson_95_high * 100 + 2,
                f"{row.recovered}/{row.n}",
                ha="center",
                fontsize=10,
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Named", "Masked"],
        ylim=(0, 100),
        ylabel="Discovery recovery (%)",
        title="NSCLC: unchanged tasks, 20 runs per condition",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(alpha=0.16)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.5,
        0.015,
        "Whiskers: 95% Wilson intervals. Repeated sessions on one fixed cohort.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"nsclc_recovery.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# Sol 5.6 medium: NSCLC discovery recovery",
        "",
        "Completed all 40 planned formal sessions: 20 named and 20 masked, each with "
        "25 validated iteration records (1,000 total). No terminal failures, replacement "
        "sessions, or technical resumes occurred. Two successful setup sessions are excluded.",
        "",
        "| Model | Condition | Recovery | Rate | 95% Wilson interval | Strict | Confirmed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['condition']} | {row['recovered']}/{row['n']} | "
            f"{row['rate']:.0%} | {row['wilson_95_low']:.1%}–{row['wilson_95_high']:.1%} | "
            f"{row['strict']} | {row['confirmed']} |"
        )
    lines += [
        "",
        "![NSCLC recovery comparison](nsclc_recovery.png)",
        "",
        "The Sol rates are higher than the archived Luna NSCLC rates in both conditions. "
        "The observed named rate exceeds the masked rate. These are small samples of fresh "
        "model sessions on one fixed synthetic cohort; they do not establish a general model "
        "ranking or biological discovery rate. The intervals describe session variability. "
        "No significance threshold or optional stopping decision was applied to these comparisons.",
        "",
        "## What stayed fixed",
        "",
        "The source is the archived synthetic NSCLC cohort in "
        "`example_data_clinical_all_claude/ds001/nsclc`: 50,000 patients, split into the same "
        "40,000 discovery and 10,000 held-out rows with split seed 20260904. The outcome is "
        "progression-free survival in months. Naming conditions contain the same patient values; "
        "masked predictor names and categorical labels are opaque.",
        "",
        "Dataset, instructions, description, and schema bytes match the prior v2 NSCLC tasks. "
        "Only model labels and job identifiers changed in public inputs. The 25-iteration "
        "budget, required screening/multivariable/refinement/robustness actions, dispatch "
        "prompt, hidden evaluator, and deterministic recovery rules were unchanged.",
        "",
        "Primary recovery requires the complete planted subgroup, correct outcome, exposure "
        "and direction, and at least 90% subgroup precision and recall. Strict recovery requires "
        "equivalent boundaries. Held-out confirmation is secondary and requires finite linked "
        "discovery evidence plus a correctly signed held-out result under the existing "
        "per-distinct-claim allocation `alpha = 0.05/[j(j+1)]`. Both subgroup treatment "
        "effects and explicit interactions can qualify for primary recovery.",
        "",
        "## Execution and validation",
        "",
        "Every formal replicate was dispatched to a fresh `gpt-5.6-sol` agent with "
        "`reasoning_effort=medium` and `fork_turns=none`. The coordinator knew the earlier "
        "answer key; workers received only their own task workspace and no inherited context "
        "or recovery feedback. Dispatch acknowledgments and transcript model labels agree. "
        "Per-response tier, token, and compute telemetry are unavailable; priority service "
        "is advertised by the tool. Equal iteration counts do not establish equal compute "
        "or 25 independent model reasoning turns.",
        "",
        "The launch plan called for scoring after the batch closed. At the user's request, "
        f"{len(inspections)} interim inspections of validated completed runs were made and logged. "
        "Inputs, scoring, sample size, dispatch order, and stopping rules were not changed. "
        "The same final 40 attempts were retained regardless of interim outcomes.",
        "",
        "Input and evaluator hashes, transcript reconstruction, required actions, and retained "
        "script/output receipts were validated. Checkpoints were packed only while every worker "
        "was idle. See `verification.json` and `reproducibility.json` for final integrity and "
        "archive-restoration checks. The full archive retains inputs, private evaluator, both "
        "setup sessions, every research script/output/transcript, and interim reports. "
        "Its public checksum receipt is `archive_reference.json`.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python experiments/aim1_recovery/archive.py restore \\",
        "  --archive aim1_sol_nsclc_20260905_checkpoint.tar.gz --out restored_sol",
        "python experiments/aim1_recovery/score.py \\",
        "  --plan restored_sol/experiment/plan.json --out restored_sol/rescored",
        "python experiments/aim1_recovery/report_nsclc_comparison.py \\",
        "  --root restored_sol/experiment \\",
        "  --baseline experiments/aim1_recovery/results/"
        "luna_20260905_priority_v2_restored/run_scores.csv \\",
        "  --out restored_sol/comparison",
        "```",
        "",
        "Use `run_scores.csv` for per-run outcomes, `group_scores.csv` for unchanged scorer "
        "aggregates, and `model_comparison.csv` for the NSCLC-only model comparison. Generic "
        "scorer output also contains empty DepMap aggregate rows; no DepMap sessions were run.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root, args.baseline, args.out)
