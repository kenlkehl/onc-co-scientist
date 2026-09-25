"""Compare the shared score profile without pooling distinct execution protocols."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .transport import write_json


def compare_reports(roots, out):
    paths = set()
    for root in map(Path, roots):
        if root.is_file():
            paths.add(root.resolve())
        else:
            paths.update(p.resolve() for p in root.glob("runs/*/report.json"))
    if not paths:
        raise ValueError("No run reports found")
    rows, groups = [], defaultdict(list)
    hashes = defaultdict(set)
    source_hashes = defaultdict(set)
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if report.get("versions", {}).get("scoring") != "profile-3.0.0":
            raise ValueError(f"Incompatible scoring profile: {path}")
        scores = report["scores"]
        discovery = scores["discovery"]["exact"]
        usage = report.get("usage", report.get("coordination", {}).get("usage", {}))
        row = {
            "run_id": report["run_id"],
            "pair_id": report["pair_id"],
            "version": report["version"],
            "dataset_sha256": report["dataset_sha256"],
            "model": report.get("model_profile", report["model"]),
            "workflow": report.get("workflow_id", report["harness"]),
            "resources": report.get("resource_policy", "benchmark-controller"),
            "clock": report.get("clock", "workflow_iteration"),
            "completion_policy": report.get("completion_policy", "unspecified"),
            "dataset_view": report.get("dataset_view", "named"),
            "source_dataset_sha256": report.get("source_dataset_sha256", report["dataset_sha256"]),
            "failed": bool(report["protocol_errors"]),
            "R": discovery["R"],
            "P": discovery["Q"],
            "F1_star": scores["D"],
            "E": scores["E"],
            "B": scores["B"],
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        hashes[(row["pair_id"], row["version"], row["dataset_view"])].add(row["dataset_sha256"])
        source_hashes[(row["pair_id"], row["version"])].add(row["source_dataset_sha256"])
        rows.append(row)
        groups[
            tuple(
                row[k]
                for k in (
                    "pair_id",
                    "version",
                    "model",
                    "workflow",
                    "resources",
                    "clock",
                    "completion_policy",
                    "dataset_view",
                )
            )
        ].append(row)
    if any(len(values) > 1 for values in hashes.values()):
        raise ValueError(
            "Dataset bytes differ for the same pair/version; compare as distinct studies"
        )
    if any(len(values) > 1 for values in source_hashes.values()):
        raise ValueError("Source dataset bytes differ between named/masked conditions")
    summaries = []
    keys = (
        "pair_id",
        "version",
        "model",
        "workflow",
        "resources",
        "clock",
        "completion_policy",
        "dataset_view",
    )
    for key, members in sorted(groups.items()):
        summary = {
            **dict(zip(keys, key, strict=True)),
            "runs": len(members),
            "failures": sum(row["failed"] for row in members),
        }
        for metric in ("R", "P", "F1_star", "E", "B", "input_tokens", "output_tokens"):
            values = [row[metric] for row in members if row[metric] is not None]
            summary[metric] = {
                "mean": mean(values) if values else None,
                "available_n": len(values),
                "missing_n": len(members) - len(values),
            }
        summaries.append(summary)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "runs": rows,
        "conditions": summaries,
        "interpretation": "Descriptive fixed-dataset repeats. Resource access and clock differ; "
        "no claim of matched compute or across-dataset uncertainty.",
    }
    write_json(out / "comparison.json", result)
    with (out / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# External co-scientist comparison",
        "",
        result["interpretation"],
        "",
        "| Model | Workflow | Version | Resources | Clock | Completion policy | "
        "Dataset view | Runs | Failures | "
        "R | P | F1* | E | B |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        values = [
            str(row[k])
            for k in (
                "model",
                "workflow",
                "version",
                "resources",
                "clock",
                "completion_policy",
                "dataset_view",
                "runs",
                "failures",
            )
        ]
        values.extend(
            "NA" if row[k]["mean"] is None else f"{row[k]['mean']:.3f} ({row[k]['available_n']})"
            for k in ("R", "P", "F1_star", "E", "B")
        )
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "Metric cells show the mean and available-run count. Missing evidence is not zero.",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    return result
