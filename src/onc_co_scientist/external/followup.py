"""Finish a named campaign, run its masked twin, and publish scored reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .comparison import compare_reports
from .config import load_spec
from .runner import run_experiment
from .smoke import run_smoke
from .transport import write_json


def terminal_runs(root):
    plan = json.loads((root / "plan.json").read_text())
    expected = {f"{p['task_id']}__biomni_native__r{p['replicate']:03d}" for p in plan}
    results = {}
    for run_id in expected:
        path = root / "runs" / run_id / "run.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["status"] in {"completed", "failed"}:
                results[run_id] = result
    return len(results) == len(expected), results


def scalar_metrics(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_metrics(item, f"{prefix}.{key}" if prefix else key)
    elif value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
        yield prefix, value


def publish_report(named, masked, out):
    """Use canonical final reports, retaining failures, nulls, and all scientific detail."""
    from ..expected_surprising.packaging import sha256

    roots = [named, masked]
    for root in roots:
        complete, results = terminal_runs(root)
        if not complete or len(results) != 20:
            raise ValueError(f"Campaign is not terminal with 20 runs: {root}")
        for run_id, result in results.items():
            run_root = root / "runs" / run_id
            if sha256(run_root / "report.json") != result["report_sha256"]:
                raise ValueError(f"Scientific report hash changed: {run_id}")
            for name, digest in result.get("artifact_sha256", {}).items():
                if sha256(run_root / name) != digest:
                    raise ValueError(f"Audited artifact changed: {run_id}/{name}")
    comparison = compare_reports(roots, out)
    details, grouped = [], defaultdict(list)
    for view, root in zip(("named", "masked"), roots, strict=True):
        for path in sorted(root.glob("runs/*/report.json")):
            report = json.loads(path.read_text())
            status = json.loads((path.parent / "run.json").read_text())
            sections = {
                k: report[k]
                for k in (
                    "scores",
                    "discovery",
                    "confirmation",
                    "exploration",
                    "responsiveness",
                    "validation",
                    "behavior",
                    "post_evidence_exploration",
                    "usage",
                    "exploration_by_tokens",
                )
                if k in report
            }
            detail = {
                "view": view,
                "version": report["version"],
                "run_id": report["run_id"],
                "status": status["status"],
                "error": status.get("error"),
                "completed_rounds": report["completed_rounds"],
                "report_sha256": status["report_sha256"],
                "metrics": sections,
            }
            details.append(detail)
            grouped[(view, report["version"])].append(detail)
    summaries = []
    for (view, version), runs in sorted(grouped.items()):
        metrics = defaultdict(list)
        for run in runs:
            for key, value in scalar_metrics(run["metrics"]):
                metrics[key].append(value)
        summaries.append(
            {
                "view": view,
                "version": version,
                "n": len(runs),
                "statuses": dict(Counter(r["status"] for r in runs)),
                "metrics": {
                    key: {
                        "mean": mean(known) if known else None,
                        "available_n": len(known),
                        "missing_n": len(runs) - len(known),
                    }
                    for key, values in sorted(metrics.items())
                    for known in [[v for v in values if v is not None]]
                },
            }
        )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "conditions": summaries,
        "comparison": comparison,
        "runs": details,
        "interpretation": "One fixed paired dataset, ten repeats per condition. "
        "Named and masked use the same rows and numerical validation seeds. "
        "Failures remain included under zero_run scoring; null metrics are not zero. "
        "Scalar summaries are descriptive means; per-run JSON retains curves, evidence "
        "classes, denominators and diagnostics. Local dollar cost is unavailable. "
        "No custom-harness outputs were supplied in the GCS data directory.",
    }
    write_json(out / "biomni_report.json", payload)
    lines = [
        "# Biomni: named and masked expected/surprising NSCLC",
        "",
        payload["interpretation"],
        "",
        "## Primary metrics",
        "",
        "R and P are fractions; F1*, E and B are on 0–100 scales. "
        "Parentheses give available-run counts.",
        "",
        "| View | Version | Completed | Failed | R | P | F1* | E | B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in comparison["conditions"]:
        cells = [
            condition["dataset_view"],
            condition["version"],
            str(condition["runs"] - condition["failures"]),
            str(condition["failures"]),
        ]
        for key in ("R", "P", "F1_star", "E", "B"):
            item = condition[key]
            cells.append(
                "NA" if item["mean"] is None else f"{item['mean']:.4f} ({item['available_n']})"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Failures", ""])
    for run in details:
        if run["status"] == "failed":
            lines.append(
                f"- {run['view']} / {run['version']} / {run['run_id']}: "
                f"{run['error']}; {run['completed_rounds']}/25 rounds."
            )
    for condition in summaries:
        lines.extend(
            [
                "",
                f"## All scalar metrics: {condition['view']} / {condition['version']}",
                "",
                "| Metric | Mean | Available / total |",
                "|---|---:|---:|",
            ]
        )
        for key, item in condition["metrics"].items():
            value = "NA" if item["mean"] is None else f"{item['mean']:.6g}"
            lines.append(f"| {key} | {value} | {item['available_n']} / {condition['n']} |")
    (out / "biomni_report.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--named-root", type=Path, required=True)
    parser.add_argument("--named-pid", type=int, required=True)
    parser.add_argument("--named-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--bucket-prefix", required=True)
    args = parser.parse_args()
    spec = load_spec(args.config)
    status_path = args.report_out / "followup_status.json"

    def status(phase, **extra):
        write_json(
            status_path, {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), **extra}
        )

    try:
        while True:
            finished, results = terminal_runs(args.named_root)
            status("waiting_for_named", terminal_runs=len(results), total=20)
            try:
                os.kill(args.named_pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if finished and not alive:
                break
            if not alive and not finished:
                raise RuntimeError(
                    "Named campaign exited with unfinished runs; no automatic replay"
                )
            time.sleep(30)
        status("updating_original_checkout")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(
            ["git", "-C", str(args.named_repo), "merge", "--ff-only", commit], check=True
        )
        status("masked_full_length_pilots")
        gate = run_smoke(spec, full_length=True)
        if gate["status"] != "passed":
            raise RuntimeError("Masked full-length pilot gate failed; formal campaign not started")
        status("masked_formal_campaign")
        run_experiment(spec, resume=True)
        status("scoring")
        publish_report(args.named_root, spec.output_root, args.report_out)
        status("uploading")
        for name in (
            "biomni_report.md",
            "biomni_report.json",
            "comparison.csv",
            "comparison.json",
            "comparison.md",
        ):
            for attempt in range(3):
                result = subprocess.run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        str(args.report_out / name),
                        args.bucket_prefix.rstrip("/") + "/" + name,
                    ]
                )
                if result.returncode == 0:
                    break
                if attempt == 2:
                    result.check_returncode()
                time.sleep(30)
        status(
            "completed",
            report=str(args.report_out / "biomni_report.md"),
            uploaded_to=args.bucket_prefix,
        )
        print((args.report_out / "biomni_report.md").read_text(), flush=True)
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
