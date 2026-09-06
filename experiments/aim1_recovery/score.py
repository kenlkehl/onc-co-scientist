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
from onc_co_scientist.scoring.structured_batch import (
    LEGACY_SCORER_VERSION,
    SCORER_VERSION,
    score_transcript,
    write_structured_report,
)
from onc_co_scientist.synthetic.schemas import DatasetManifest


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_transcript_records(saved: Transcript, assembled: Transcript) -> bool:
    """Compare retained values without treating a failed analysis's NaN as a change.

    Keep NaN distinct from null and finite values. The recovery scorer separately
    rejects nonfinite evidence; this comparison only checks record consistency.
    """
    return json.dumps(saved.model_dump(mode="json"), sort_keys=True, allow_nan=True) == json.dumps(
        assembled.model_dump(mode="json"), sort_keys=True, allow_nan=True
    )


def failed_run_score(job: dict, protocol: dict, failure: dict) -> dict:
    """Retain a documented terminal failure without scoring its artifacts."""
    if (
        failure.get("retain_in_denominator") is not True
        or failure.get("recovery_credit") is not False
    ):
        raise ValueError("Terminal failures must remain in the denominator with no recovery credit")
    if not failure.get("reason") or failure.get("job_id") != job["job_id"]:
        raise ValueError("Terminal failure needs a matching job ID and reason")
    result = {
        "scorer_version": protocol.get("scorer_version", LEGACY_SCORER_VERSION),
        "primary_scoring_backend": "deterministic",
        "dataset_id": job["dataset_id"],
        "model_id": protocol["model_id"],
        "harness_id": protocol.get("harness_id", f"{protocol['backend']}-structured-v2"),
        "evidence_design": "terminal failure; no claims evaluated",
        "max_iterations": job["max_iterations"],
        "submitted_iterations": len(list((Path(job["workspace"]) / "iterations").glob("*.json"))),
        "structured_claims": 0,
        "unique_structured_claims": 0,
        "unstructured_claims": 0,
        "per_association": [],
        "claim_scores": [],
        "terminal_failure": failure,
    }
    for prefix in ("primary", "strict", "confirmed", "interaction_confirmed"):
        result[prefix + "_recovered"] = False
        result[prefix + "_iteration"] = None
    return result


def generate(plan_path: Path, out: Path, allow_incomplete: bool = False) -> dict:
    plan = json.loads(plan_path.read_text())
    version = plan["protocol"].get("scorer_version", LEGACY_SCORER_VERSION)
    v2 = version == SCORER_VERSION
    jobs = plan["jobs"]
    failure_path = plan_path.parent / "terminal_failures.json"
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    if not isinstance(failures, dict) or set(failures) - {j["job_id"] for j in jobs}:
        raise ValueError("Terminal failure ledger contains unknown jobs or has invalid format")
    unfinished = [
        j["job_id"]
        for j in jobs
        if j["job_id"] not in failures and not (Path(j["workspace"]) / "transcript.json").exists()
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
        if job["job_id"] not in failures and not (ws / "transcript.json").exists():
            continue
        if (
            sha(ws / "dataset.parquet") != job["data_sha256"]
            or sha(ws / "agent_instructions.md") != job["instructions_sha256"]
        ):
            raise ValueError(f"Task input changed: {job['job_id']}")
        for filename, expected in job.get("public_input_sha256", {}).items():
            if sha(ws / filename) != expected:
                raise ValueError(f"Task input changed: {job['job_id']}/{filename}")
        if job["job_id"] in failures:
            scored = failed_run_score(job, plan["protocol"], failures[job["job_id"]])
        else:
            saved = Transcript.model_validate_json((ws / "transcript.json").read_text())
            assembled = finalize_workspace(
                ws, model_id=saved.model_id, harness_id=saved.harness_id, write_output=False
            )
            if not same_transcript_records(saved, assembled):
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
                manifest,
                assembled,
                frame,
                column_mapping=mapping,
                evidence_design="heldout_20_percent",
                scorer_version=version,
            )
        scored.update({k: job[k] for k in ["job_id", "family", "task", "variant", "replicate"]})
        scores.append(scored)
        # Sensitivity changes overlap only, retaining the same evidence and alpha allocation.
        at95 = any(
            (v2 or c.get("heldout_supported"))
            and c.get("score", {}).get("match_recovered")
            and c["score"]["functional"]["precision"] >= 0.95
            and c["score"]["functional"]["recall"] >= 0.95
            for c in scored["claim_scores"]
        )
        exact_membership = any(
            (v2 or c.get("heldout_supported"))
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
        row["terminal_failure"] = job["job_id"] in failures
        row["recovery_at_95"] = at95
        if v2:
            for field in (
                "confirmed_recovered",
                "confirmed_iteration",
                "interaction_confirmed_recovered",
                "interaction_confirmed_iteration",
            ):
                row[field] = scored[field]
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
        if (ws / "iterations").exists() and not (exported / "iterations").exists():
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
                if v2:
                    summaries[-1].update(
                        confirmed_n=int(part.confirmed_recovered.sum()),
                        interaction_confirmed_n=int(part.interaction_confirmed_recovered.sum()),
                    )
    summary = {
        "complete": not unfinished,
        "n_expected": len(jobs),
        "n_completed": len(frame) - len(failures),
        "n_terminal": len(frame),
        "terminal_failures": failures,
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
    for filename in [
        "plan.json",
        "protocol.json",
        "implementation_at_launch.json",
        "implementation_for_scoring.json",
        "environment.json",
        "verification.json",
        "terminal_failures.json",
        "runtime_reset_recovery.json",
        "restoration_20260905.json",
    ]:
        source_path = plan_path.parent / filename
        if source_path.exists():
            shutil.copyfile(source_path, out / filename)
    state_path = plan_path.parent / "coordinator_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        summary["technical_interruptions"] = state.get("technical_interruptions", [])
        shutil.copyfile(state_path, out / state_path.name)
        resumed = {
            j
            for event in summary["technical_interruptions"]
            for j in event.get("jobs", event.get("affected_jobs", []))
        }
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
    validation_notes = plan_path.parent / "validation_notes.json"
    if validation_notes.exists():
        summary["validation_notes"] = json.loads(validation_notes.read_text())
        shutil.copyfile(validation_notes, out / validation_notes.name)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summaries).to_csv(out / "group_scores.csv", index=False)
    plot(frame, out, complete=not unfinished)
    if v2:
        lines = [
            "# Aim 1 structured recovery v2",
            "",
            f"Finalized {len(frame)}/{len(jobs)} planned attempts: "
            f"{len(frame) - len(failures)} completed and "
            f"{len(failures)} terminal failures. "
            "Failures remain in every primary denominator with zero recovery credit.",
            "",
            "Primary recovery measures complete subgroup identity, correct "
            "outcome/exposure/direction "
            "and ≥90% subgroup precision and recall. Clinical treatment effects"
            " within the subgroup "
            "and treatment interactions both qualify. Strict recovery requires "
            "equivalent boundaries. "
            "Statistical support does not gate either identity endpoint.",
            "",
            "Confirmation is secondary: finite linked discovery analysis plus "
            "held-out, correctly signed "
            "evidence for the candidate's declared contrast, with "
            "alpha=0.05/[j(j+1)] per distinct claim. "
            "Interaction confirmation counts only explicitly submitted treatment interactions. "
            "Novelty judging is optional and was not used by this report.",
            "",
            "| Family | Condition | N | Identity | Strict identity | Confirmed "
            "| Interaction confirmed |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for g in summaries:
            if g["task"] == "all":
                lines.append(
                    f"| {g['family']} | {g['variant']} | {g['n']} | {g['primary_n']} | "
                    f"{g['strict_n']} | {g['confirmed_n']} | {g['interaction_confirmed_n']} |"
                )
        lines += [
            "",
            "Protocol and limitations",
            "",
            (
                "Research follows the archived Claude task brief with a 25-iteration ceiling "
                "and agent-selected early stopping. No phase schedule, four-action quota, or "
                "per-iteration script/output hashes are required. Structured findings and "
                "ordered iteration JSON receipts are retained; these do not prove execution "
                "timing or scientific originality. Actual iterations and tokens/compute are "
                "not matched. Research sessions receive no recovery feedback."
                if plan["protocol"].get("prompt_style") == "claude-legacy-loose-v1"
                else "All research runs must complete 25 clinical or 10 DepMap iterations, "
                "including screening, multivariable exploration, refinement, and robustness. "
                "Records retain script/output hashes; these checks do not establish equal "
                "tokens/compute or certify scientific originality. Neutral examples are "
                "identical across naming conditions. Research sessions receive no recovery "
                "feedback. This is a revised protocol on fixed archived cohorts, not an "
                "isolated test of model ability or scoring alone."
            ),
            "",
            f"Requested model: {plan['protocol'].get('model_id')}; "
            f"reasoning: {plan['protocol'].get('reasoning_effort')}; "
            f"service: {plan['protocol'].get('service_tier_requested')}. "
            f"Actual models: {summary['actual_models']}. "
            "See launch evidence and runtime metadata for capability/telemetry verification.",
        ]
        if summary.get("validation_notes"):
            lines += ["", "Restoration and protocol limitations", ""]
            lines += [str(value) for value in summary["validation_notes"].values()]
        if summary.get("technical_resume_sensitivity"):
            lines += [
                "",
                "Sensitivity excluding documented or potentially reset-affected jobs",
                "",
                "| Family | Condition | Excluded | Remaining N | Identity | Strict |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for group in summary["technical_resume_sensitivity"]:
                lines.append(
                    f"| {group['family']} | {group['variant']} | {group['excluded_resumed_n']} | "
                    f"{group['n']} | {group['primary_n']} | {group['strict_n']} |"
                )
        (out / "report.md").write_text("\n".join(lines) + "\n")
        return summary
    lines = [
        "# Aim 1 structured recovery pilot",
        "",
        f"Finalized {len(frame)}/{len(jobs)} planned attempts: "
        f"{len(frame) - len(failures)} completed and "
        f"{len(failures)} terminal failures. "
        "Failures remain in every primary denominator with zero recovery credit.",
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
        "The retained launch hashes are historical. The scoring implementation commit "
        "and hashes are recorded separately in implementation_for_scoring.json; software "
        "versions are in environment.json. The frozen scientific criteria are in "
        "protocol.json.",
        "",
        "Validation notes, when present, are retained in validation_notes.json. Failed "
        "analyses recorded as NaN remain unchanged in the original records and cannot "
        "satisfy the finite-evidence requirement. Record comparison treats matching NaNs "
        "as identical while distinguishing them from null and finite values.",
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
        axes, ["clinical", "depmap"], ["A  Clinical cohorts", "B  DepMap cohort"], strict=True
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
