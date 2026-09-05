#!/usr/bin/env python3
"""Validate receipts, score heldout recovery, export traces, and regenerate figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from onc_co_scientist.harness.structured_runner import finalize_workspace
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring.structured_batch import score_transcript, write_structured_report
from onc_co_scientist.synthetic.schemas import DatasetManifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(plan_path: Path, out: Path, allow_incomplete: bool = False) -> dict:
    plan = json.loads(plan_path.read_text())
    jobs = plan["jobs"]
    unfinished = [
        j["job_id"] for j in jobs if not (Path(j["workspace"]) / "transcript.json").exists()
    ]
    if unfinished and not allow_incomplete:
        raise ValueError(
            f"{len(unfinished)} of {len(jobs)} jobs are unfinished. Refusing a final figure."
        )
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    scores = []
    rows = []
    for job in jobs:
        ws = Path(job["workspace"])
        if not (ws / "transcript.json").exists():
            continue
        if (
            sha(ws / "dataset.parquet") != job["data_sha256"]
            or sha(ws / "agent_instructions.md") != job["instructions_sha256"]
        ):
            raise ValueError(f"Task input changed: {job['job_id']}")
        saved = Transcript.model_validate_json((ws / "transcript.json").read_text())
        assembled = finalize_workspace(
            ws, model_id=saved.model_id, harness_id=saved.harness_id, write_output=False
        )
        if saved.model_dump() != assembled.model_dump():
            raise ValueError("Transcript differs from submitted records")
        ev = Path(job["evaluator"])
        if job["task"] not in cache:
            manifest = DatasetManifest.model_validate_json((ev / "manifest.json").read_text())
            frame = pd.read_parquet(ev / "evaluation.parquet")
            mapping = json.loads((ev / "column_mapping.json").read_text())
            source = next(s for s in plan["sources"] if s["task"] == job["task"])
            if sha(ev / "evaluation.parquet") != source["evaluation_sha256"]:
                raise ValueError("Evaluation data changed")
            if sha(ev / "manifest.json") != source["manifest_sha256"]:
                raise ValueError("Answer key changed")
            cache[job["task"]] = (manifest, frame, mapping)
        manifest, frame, mapping = cache[job["task"]]
        scored = score_transcript(
            manifest, assembled, frame, column_mapping=mapping, evidence_design="heldout_20_percent"
        )
        scored.update({k: job[k] for k in ["job_id", "family", "task", "variant", "replicate"]})
        scores.append(scored)
        # Sensitivity changes overlap only, retaining the same evidence and alpha allocation.
        at95 = any(
            c.get("heldout_supported")
            and c.get("score", {}).get("match_recovered")
            and c["score"]["functional"]["precision"] >= 0.95
            and c["score"]["functional"]["recall"] >= 0.95
            for c in scored["claim_scores"]
        )
        exact_membership = any(
            c.get("heldout_supported")
            and c.get("score", {}).get("match_recovered")
            and c["score"]["functional"]["precision"] == 1
            and c["score"]["functional"]["recall"] == 1
            for c in scored["claim_scores"]
        )
        row = {
            k: scored[k]
            for k in [
                "job_id",
                "family",
                "task",
                "variant",
                "replicate",
                "model_id",
                "harness_id",
                "max_iterations",
                "submitted_iterations",
                "structured_claims",
                "unique_structured_claims",
                "primary_recovered",
                "primary_iteration",
                "strict_recovered",
                "strict_iteration",
            ]
        }
        row["recovery_at_95"] = at95
        row["recovery_at_exact_membership"] = exact_membership
        row["capped_iterations"] = scored["primary_iteration"] or scored["max_iterations"]
        rows.append(row)
        exported = out / "transcripts" / job["job_id"]
        exported.mkdir(parents=True, exist_ok=True)
        for filename in [
            "transcript.json",
            "analysis_summary.txt",
            "submission_events.jsonl",
            "runtime_metadata.json",
        ]:
            if (ws / filename).exists():
                shutil.copyfile(ws / filename, exported / filename)
        if not (exported / "iterations").exists():
            shutil.copytree(ws / "iterations", exported / "iterations")
        # Preserve executable analyses and their output without duplicating the
        # input parquet, which is reproducible from the recorded source and split.
        with tarfile.open(exported / "research_artifacts.tar.gz", "w:gz") as archive:
            for artifact in sorted(ws.iterdir()):
                if artifact.name not in {"dataset.parquet", "__pycache__"}:
                    archive.add(artifact, arcname=artifact.name)
    write_structured_report(scores, out)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No completed runs to score")
    frame.to_csv(out / "run_scores.csv", index=False)
    summaries = []
    for family in ["clinical", "depmap"]:
        for task in ["all"] + sorted(frame.loc[frame.family.eq(family), "task"].unique().tolist()):
            for variant in ["named", "anonymized"]:
                part = frame[frame.family.eq(family) & frame.variant.eq(variant)]
                if task != "all":
                    part = part[part.task.eq(task)]
                intended = sum(
                    j["family"] == family
                    and j["variant"] == variant
                    and (task == "all" or j["task"] == task)
                    for j in jobs
                )
                summaries.append(
                    {
                        "family": family,
                        "task": task,
                        "variant": variant,
                        "n": len(part),
                        "intended_n": intended,
                        "primary_n": int(part.primary_recovered.sum()),
                        "primary_rate": float(part.primary_recovered.mean()) if len(part) else None,
                        "strict_n": int(part.strict_recovered.sum()),
                        "strict_rate": float(part.strict_recovered.mean()) if len(part) else None,
                        "rate_at_95": float(part.recovery_at_95.mean()) if len(part) else None,
                        "rmst_iterations": float(part.capped_iterations.mean())
                        if len(part)
                        else None,
                    }
                )
    summary = {
        "complete": not unfinished,
        "n_expected": len(jobs),
        "n_completed": len(frame),
        "unfinished": unfinished,
        "protocol": plan["protocol"],
        "sources": plan["sources"],
        "actual_models": sorted(frame.model_id.unique().tolist()),
        "actual_harnesses": sorted(frame.harness_id.unique().tolist()),
        "groups": summaries,
        "interpretation": (
            "Fixed archived cohorts; fresh independent model sessions. Recovery differences "
            "describe this new protocol and model, not a controlled comparison with legacy "
            "results."
        ),
    }
    for filename in ["plan.json", "protocol.json", "implementation_at_launch.json"]:
        source_path = plan_path.parent / filename
        if source_path.exists():
            shutil.copyfile(source_path, out / filename)
    state_path = plan_path.parent / "coordinator_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        summary["technical_interruptions"] = state.get("technical_interruptions", [])
        shutil.copyfile(state_path, out / state_path.name)
        resumed = {j for event in summary["technical_interruptions"] for j in event["jobs"]}
        sensitivity = []
        for (family, variant), group in frame.groupby(["family", "variant"]):
            uninterrupted = group[~group.job_id.isin(resumed)]
            sensitivity.append(
                {
                    "family": family,
                    "variant": variant,
                    "excluded_resumed_n": len(group) - len(uninterrupted),
                    "n": len(uninterrupted),
                    "primary_n": int(uninterrupted.primary_recovered.sum()),
                    "strict_n": int(uninterrupted.strict_recovered.sum()),
                }
            )
        summary["technical_resume_sensitivity"] = sensitivity
    supplemental = plan_path.parent / "supplemental_analysis_plan.json"
    if supplemental.exists():
        shutil.copyfile(supplemental, out / supplemental.name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summaries).to_csv(out / "group_scores.csv", index=False)
    plot(frame, out, complete=not unfinished)
    lines = [
        "# Aim 1 structured recovery pilot",
        "",
        f"Completed {len(frame)}/{len(jobs)} planned runs.",
        "",
        f"Models recorded in transcripts: {', '.join(summary['actual_models'])}. "
        f"Harnesses: {', '.join(summary['actual_harnesses'])}.",
        "",
        "For Work runs, the protocol requests Luna 5.6, medium reasoning, "
        "priority/Fast mode as advertised by the model tool. Each replicate starts "
        "with no inherited conversation. Per-response tier and token telemetry "
        "are unavailable for Work subagents. Endpoint runs record the supplied "
        "model and returned service metadata in runtime_metadata.json.",
        "",
        "## Recovery",
        "",
        "| Family | Condition | N | Primary | Strict | RMST iterations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for g in summaries:
        if g["task"] == "all":
            lines.append(
                f"| {g['family']} | {g['variant']} | {g['n']} | "
                f"{g['primary_n']}/{g['n']} | {g['strict_n']}/{g['n']} | "
                f"{g['rmst_iterations']} |"
            )
    if plan["protocol"].get("excluded_setup_batch"):
        lines += [
            "",
            "Excluded setup batch: " + json.dumps(plan["protocol"]["excluded_setup_batch"]),
        ]
    if summary.get("technical_interruptions"):
        lines += [
            "",
            "Technical interruptions are retained in coordinator_state.json and summary.json. "
            "Affected jobs continued from their original saved work and submitted records, "
            "without recovery feedback. Some required a replacement agent context when the "
            "original runtime session became unavailable.",
        ]
        lines += [
            "",
            "Sensitivity excluding jobs that required replacement agent contexts "
            "(the primary figure retains all planned runs):",
            "",
            "| Family | Condition | Excluded | Remaining N | Primary | Strict |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for g in summary["technical_resume_sensitivity"]:
            lines.append(
                f"| {g['family']} | {g['variant']} | {g['excluded_resumed_n']} | "
                f"{g['n']} | {g['primary_n']}/{g['n']} | {g['strict_n']}/{g['n']} |"
            )
    lines += [
        "",
        "Primary recovery requires the complete defining structure, subgroup precision and "
        "recall ≥0.90, correct outcome/exposure/contrast/direction, and independent support "
        "on the held-out 20% of rows. Strict recovery additionally requires equivalent "
        "boundaries. The jth distinct submitted claim receives alpha=0.05/[j(j+1)], "
        "preventing repeated held-out tests and future look-ahead in discovery time. Claims "
        "must have a linked executed discovery analysis recorded before credit. Novelty does "
        "not enter recovery scoring.",
        "",
        "The original iteration caps and replicate counts are retained. Early stopping is "
        "allowed by the original task instructions. RMST is the mean of min(discovery "
        "iteration, task cap), with non-recovery assigned the cap; compare ratios only with "
        "the task-specific caps in mind.",
        "",
        "This rerun changes the model, output protocol, numeric tolerance, confirmatory "
        "analysis, held-out data split, and examples. Original examples that mentioned "
        "planted DepMap variables were replaced with neutral placeholders. Thresholds were "
        "selected during development after inspecting archived examples and frozen before "
        "fresh experiments; this is not an independently preregistered validation.",
        "",
        "Work sessions share a filesystem; task isolation was implemented through separate "
        "copies and explicit instructions, not an OS access boundary. Archived cohorts and "
        "public column/value semantics were retained, including category labels in masked "
        "data. Counts describe stochastic repeats on these fixed cohorts, not independent "
        "biological datasets.",
        "",
        "Files: run_scores.csv, group_scores.csv, structured_scores.json, transcripts/, "
        "aim1_recovery.png/.pdf/.svg, and discovery_curves.png/.pdf.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def plot(frame: pd.DataFrame, out: Path, *, complete: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    colors = {"named": "#193B63", "anonymized": "#3BA5A7"}
    labels = {"named": "Named", "anonymized": "Masked"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.65), sharey=True)
    for ax, family, title in zip(
        axes, ["clinical", "depmap"], ["Clinical cohorts", "DepMap cohort"], strict=True
    ):
        part = frame[frame.family.eq(family)]
        for offset, variant in zip([-0.18, 0.18], ["named", "anonymized"], strict=True):
            group = part[part.variant.eq(variant)]
            for x, col in enumerate(["primary_recovered", "recovery_at_95", "strict_recovered"]):
                rate = group[col].mean() if len(group) else 0
                ax.bar(
                    x + offset,
                    rate,
                    width=0.32,
                    color=colors[variant],
                    label=labels[variant] if x == 0 else None,
                )
                ax.text(
                    x + offset,
                    rate + 0.025,
                    f"{int(group[col].sum())}/{len(group)}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_title(title, weight="bold", pad=12)
        ax.set_xticks([0, 1, 2], ["Primary\n90% overlap", "95% overlap", "Strict\nrule recovery"])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Runs recovering the planted finding")
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(
        "Deterministic hypothesis recovery" + ("" if complete else " — INCOMPLETE"),
        fontsize=13,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"aim1_recovery.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, family, cap in zip(axes, ["clinical", "depmap"], [25, 10], strict=True):
        for variant in ["named", "anonymized"]:
            part = frame[frame.family.eq(family) & frame.variant.eq(variant)]
            x = np.arange(cap + 1)
            y = [
                float(((part.primary_iteration <= i) & part.primary_recovered).sum()) / len(part)
                if len(part)
                else 0
                for i in x
            ]
            ax.step(x, y, where="post", label=labels[variant], color=colors[variant], linewidth=2)
        ax.set_title("Clinical" if family == "clinical" else "DepMap", weight="bold")
        ax.set_xlabel("Submitted research iteration")
        ax.set_xlim(0, cap)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Cumulative primary recovery")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"discovery_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    a = p.parse_args()
    s = generate(a.plan, a.out, a.allow_incomplete)
    print(
        json.dumps({"complete": s["complete"], "n_completed": s["n_completed"], "out": str(a.out)})
    )


if __name__ == "__main__":
    main()
