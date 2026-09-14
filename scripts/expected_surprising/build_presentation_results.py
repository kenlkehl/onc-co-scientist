"""Rebuild the selected clinical comparison from durable per-run reports.

Read-only with respect to experiments; no providers, evaluators, or monitors run.
Figures require matplotlib; the combined PDF also uses reportlab and pypdf.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = (
    "expected-unmasked", "expected-masked", "surprising-unmasked", "surprising-masked"
)
METRICS = {
    "precision": "Precision (%)",
    "recall": "Recall (%)",
    "f1": "Scientific F1*",
    "exploration": "Exploration E",
    "responsiveness": "Evidence responsiveness B",
    "focal_recovery": "Focal recovery (%)",
}
COMPONENT_WEIGHTS = {"f1": .35, "focal_recovery": .25, "exploration": .20,
                     "responsiveness": .20}
CONDITION_METRICS = {**METRICS, "condition_score": "Weighted condition score C"}
CROSS_METRICS = {"summary_score": "Grounded discovery and adaptation S",
                 "label_surprise_interaction": "Label-by-surprise interaction I (pp)"}
SCORE_SETTINGS = {
    "component_weights": COMPONENT_WEIGHTS,
    "condition_weights": {c: .25 for c in CONDITIONS},
    "summary": "geometric mean of four condition-level weighted scores",
    "interaction": "(focal_EU - focal_SU) - (focal_EM - focal_SM), percentage points",
    "summary_ci": "whole-run bootstrap, independent draws within each condition",
    "interaction_ci": "Wilson-based MOVER for independent binomial proportions",
    "interaction_ci_reference": "https://doi.org/10.1016/j.csda.2008.09.033",
    "missing_policy": "required missing component or condition makes score unavailable",
    "status": "provisional descriptive index; weights specified after these experiments",
}
CLASSES = ("supported", "excluded", "ambiguous")
GROUP = ("llm", "harness", "workflow", "reasoning_effort")
NAMES = {
    "gpt-6-astra": "Astra", "gpt-5.6-sol": "Sol", "gpt-5.6-terra": "Terra",
    "gpt-5.6-luna": "Luna", "claude-opus-5": "Claude Opus 5",
    "Inferact/Qwen3.8-27B-NVFP4": "Qwen 3.8 27B",
    "RedHatAI/Gemma-4-31B-IT-FP8-Dynamic": "Gemma 4 31B",
}
WORKFLOW_COLORS = {
    "persistent": "#2575ad", "sequential": "#249c89", "deliberative": "#ad6aab",
    "Biomni": "#d79929",
}
CI_SETTINGS = {
    "confidence": .95, "bootstrap_resamples": 10000, "seed": 20260912,
    "continuous_method": "percentile bootstrap of whole runs within each fixed dataset",
    "focal_method": "Wilson score interval (single fixed dataset)",
    "minimum_valid_bootstrap_fraction": .95,
    "scope": "Repeated-run uncertainty conditional on this dataset; not across-dataset inference",
}


def read(path):
    return json.loads(Path(path).read_text())


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def atomic_text(path, value):
    """Readers see either the previous output or the fully written replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        f.write(value)
        temp = f.name
    os.replace(temp, path)


def write_csv(path, rows):
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


class Sources:
    def __init__(self, repo):
        self.repo = repo.resolve()
        self.provenance = {}

    def path(self, value):
        path = Path(value)
        if not path.is_absolute():
            return self.repo / path
        # Saved manifests can move with a checkout. Rebase known repository paths.
        if "onc-co-scientist" in path.parts:
            index = path.parts.index("onc-co-scientist")
            return self.repo.joinpath(*path.parts[index + 1:])
        return path

    def read(self, value):
        path = self.path(value)
        raw = path.read_bytes()
        self.provenance[str(path)] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        return json.loads(raw)


def select_jobs(sources, config):
    """Choose by recorded identity/ownership, never completion or score."""
    plans = sources.read(config["unmasked_plan"])
    replacements = {
        r["run_id"] for r in sources.read(config["codex_selection"])["rows"] if r["replace"]
    }
    local = set(sources.read(config["vllm_selection"]))
    planned = {p["run_id"] for p in plans}
    if not replacements <= planned or not local <= planned or replacements & local:
        raise ValueError("Replacement selections do not match the unmasked plan")
    jobs = []
    for p in plans:
        owner = (sources.path(config["vllm_replacement_root"]) if p["run_id"] in local
                 else sources.path(config["codex_replacement_root"])
                 if p["run_id"] in replacements
                 else sources.path(config["unmasked_plan"]).parent)
        jobs.append((p, "unmasked", owner))
    masked = sources.read(config["masked_plan"])
    owners = sources.read(config["masked_owners"])
    if set(owners) != {p["run_id"] for p in masked}:
        raise ValueError("Masked ownership must cover exactly the planned runs")
    jobs.extend((p, "masked", sources.path(owners[p["run_id"]])) for p in masked)
    claude_ids = defaultdict(set)
    for path in config.get("claude_selections", []):
        for view, ids in sources.read(path).items():
            claude_ids["unmasked" if view == "named" else view].update(ids)
    for view, root in config["claude_roots"].items():
        owner = sources.path(root)
        plan = sources.read(owner / "plan.json")
        if not claude_ids[view] <= {p["run_id"] for p in plan}:
            raise ValueError("Claude selection contains unplanned runs")
        for p in plan:
            # Preserve new launches on rerun, but omit unused original plan slots.
            if (not config.get("claude_selections") or p["run_id"] in claude_ids[view]
                    or (owner / "runs" / p["run_id"]).exists()):
                jobs.append((p, view, owner))
    ids = [(view, p["run_id"]) for p, view, _ in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate selected run identities")
    return jobs


def scientific_metrics(report):
    exact = report["scores"]["discovery"]["exact"]
    recall, precision = exact["R"], exact["Q"]
    # diagnostic_D retains the scientific trace even under an old zero_run policy.
    f1 = exact["diagnostic_D"]
    calculated = 100 * 2 * recall * precision / (recall + precision) if (
        precision is not None and recall + precision > 0
    ) else 0.0
    if not math.isclose(f1, calculated, abs_tol=1e-7):
        raise ValueError("Stored diagnostic F1 disagrees with recall and precision")
    focal_id = next(m["discovery_id"] for m in report["behavior"]["milestones"] if m["focal"])
    focal_recovered = any(
        m["discovery_id"] == focal_id and m["match"] == "exact"
        for m in report["scores"]["discovery"]["confirmed_matches"]
    )
    result = {
        "precision": 100 * precision if precision is not None else None,
        "recall": 100 * recall, "f1": f1,
        "f1_saved": report["scores"]["D"],
        "exploration": report["scores"]["E"],
        "focal_recovery": 100 * focal_recovered,
    }
    for cls in CLASSES:
        result["response_" + cls] = report["responsiveness"]["components"][cls]["accuracy"]
    result["responsiveness"] = report["scores"]["B"]
    for key in METRICS:
        value = result[key]
        if value is not None and (not math.isfinite(value) or not -1e-8 <= value <= 100 + 1e-8):
            raise ValueError(f"Invalid {key}: {value}")
    return result


def inspect_job(job, aliases=None):
    aliases = aliases or {}
    plan, view, owner = job
    folder = owner / "runs" / plan["run_id"]
    model = plan["model_id"]
    row = dict(
        llm=model, harness="Codex" if model.startswith("gpt-") else "Custom runner",
        workflow=plan["workflow_id"], reasoning_effort="medium", view=view,
        version=plan["semantic_condition"], run_id=plan["run_id"],
        replicate=plan["replicate"], iterations=plan["iterations"],
        source=str(folder), pair_id=None, report_sha256=None, report_available=False,
        source_generation=owner.name, versions=None, stage_failure_policy=None,
        reported_model=None, output_tokens=None, input_tokens=None, missing_token_calls=None,
        unknown_internal_attempts=None, usage_source=None, report_error=None,
    )
    # A directory is created when execution starts; no process-liveness inference.
    row["status"] = "unfinished" if folder.exists() else "queued"
    result_path = folder / "run.json"
    if result_path.exists():
        result = read(result_path)
        row["status"] = result["status"]
        if result["run_id"] != row["run_id"]:
            raise ValueError(f"Run identity mismatch: {result_path}")
        if row["status"] in ("completed", "failed"):
            report_path = folder / "report.json"
            try:
                raw = report_path.read_bytes()
            except FileNotFoundError:
                row["report_error"] = "Terminal run has no scientific report"
            else:
                digest = hashlib.sha256(raw).hexdigest()
                expected = result.get("scientific_report_sha256")
                if not expected or digest != expected:
                    raise ValueError(f"Scientific report checksum mismatch: {report_path}")
                report = json.loads(raw)
                reported_model = report["model"]
                if (report["version"] != row["version"]
                        or aliases.get(reported_model, reported_model) != model):
                    raise ValueError(f"Report metadata mismatch: {report_path}")
                row.update(scientific_metrics(report))
                row.update(
                    pair_id=report["pair_id"], report_sha256=digest, report_available=True,
                    reported_model=reported_model,
                    versions=json.dumps(report["versions"], sort_keys=True),
                    stage_failure_policy=result.get("stage_failure_policy", "zero_run"),
                )
                usage = report.get("coordination", {}).get("output_token_accounting", {})
                row.update(
                    output_tokens=usage.get("known_output_tokens"),
                    input_tokens=result.get("usage", {}).get("input_tokens"),
                    missing_token_calls=usage.get("missing_calls"),
                    unknown_internal_attempts=usage.get("unaccounted_infrastructure_attempts"),
                    usage_source="coordination.output_token_accounting",
                )
    # Fixed columns even for pending or missing-report runs.
    for key in (*METRICS, "f1_saved", *("response_" + c for c in CLASSES)):
        row.setdefault(key, None)
    row["attempted"] = row["status"] != "queued"
    return row


def biomni_rows(sources, config):
    payload = sources.read(config["biomni_report"])
    lookup = {(r["dataset_view"], r["run_id"]): r for r in payload["comparison"]["runs"]}
    rows = []
    for run in payload["runs"]:
        meta = lookup[run["view"], run["run_id"]]
        usage = run["metrics"]["usage"]
        row = dict(
            llm=config["biomni_model"], harness="Biomni", workflow="Biomni",
            reasoning_effort=config["biomni_reasoning"],
            view="unmasked" if run["view"] == "named" else run["view"],
            version=run["version"], run_id=run["run_id"],
            replicate=int(run["run_id"].rsplit("r", 1)[-1]), iterations=25,
            source=str(sources.path(config["biomni_report"])), pair_id=meta["pair_id"],
            report_sha256=run["report_sha256"], report_available=True,
            source_generation="imported Biomni " + payload["generated_at"],
            versions=None, stage_failure_policy="zero_run",
            reported_model=meta["model"], output_tokens=usage.get("known_output_tokens"),
            input_tokens=usage.get("known_input_tokens"),
            missing_token_calls=usage.get("unknown_usage_requests"),
            unknown_internal_attempts=None, usage_source="Biomni metrics.usage",
            report_error=None, status=run["status"], attempted=True,
        )
        row.update(scientific_metrics(run["metrics"]))
        rows.append(row)
    if len({(r["view"], r["run_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate Biomni runs")
    return rows, payload["generated_at"]


def key(row):
    return tuple(row[k] for k in GROUP)



def bootstrap_group_identity(group):
    """Keep historical bootstrap draws stable when only a presentation label changes."""
    if group[1] == "Biomni" and group[2] == "Biomni":
        return (*group[:2], "Native adaptive", group[3])
    return group


def group_order(group):
    models = list(NAMES)
    model, harness, workflow, _ = group
    workflows = ["persistent", "sequential", "deliberative", "Biomni"]
    return (models.index(model) if model in models else 99, harness,
            workflows.index(workflow) if workflow in workflows else 99, group)


def hierarchical_mean(rows, field):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row[field])
    return mean([mean(values) for values in pairs.values()])


def confidence_interval(rows, metric, *, seed=20260912, resamples=10000):
    """Marginal run-level CIs. B recomputes the class-balanced statistic in each draw."""
    import numpy as np

    result = dict(ci95_low=None, ci95_high=None, ci_method="run bootstrap",
                  ci_status="insufficient eligible runs", ci_valid_resamples=0)
    if not rows:
        return result
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    if metric == "focal_recovery" and len(by_pair) == 1:
        values = [r[metric] for r in rows if r[metric] is not None]
        if not values:
            return result
        if not all(v in (0, 100) for v in values):
            raise ValueError("Focal recovery must be binary for Wilson intervals")
        n, p = len(values), statistics.mean(values) / 100
        z = statistics.NormalDist().inv_cdf(.975)
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return dict(ci95_low=max(0, 100 * (center - radius)),
                    ci95_high=min(100, 100 * (center + radius)), ci_method="Wilson score",
                    ci_status="available", ci_valid_resamples=0)
    fields = ["response_" + c for c in CLASSES] if metric == "responsiveness" else [metric]
    rng = np.random.default_rng(seed)
    pair_draws = []
    available_counts = np.zeros(len(fields), dtype=int)
    for pair in sorted(by_pair):
        # Stable ordering makes reruns independent of manifest scheduling order.
        items = sorted(by_pair[pair], key=lambda r: str(r.get("run_id", "")))
        data = np.array([[r[f] if r[f] is not None else np.nan for f in fields]
                         for r in items], dtype=float)
        n = len(data)
        available_counts += np.isfinite(data).sum(axis=0)
        if n < 2:
            return result
        sample = data[rng.integers(0, n, size=(resamples, n))]
        counts = np.isfinite(sample).sum(axis=1)
        pair_draws.append(np.divide(np.nansum(sample, axis=1), counts,
                                   out=np.full((resamples, len(fields)), np.nan), where=counts > 0))
    if (available_counts < 2).any():
        return result
    stacked = np.array(pair_draws)
    counts = np.isfinite(stacked).sum(axis=0)
    component_draws = np.divide(np.nansum(stacked, axis=0), counts,
                               out=np.full((resamples, len(fields)), np.nan), where=counts > 0)
    valid = np.isfinite(component_draws).all(axis=1)
    result["ci_valid_resamples"] = int(valid.sum())
    if valid.mean() < CI_SETTINGS["minimum_valid_bootstrap_fraction"]:
        result["ci_status"] = "too many bootstrap draws lack eligible evidence"
        return result
    draws = component_draws[valid].mean(axis=1) * (100 if metric == "responsiveness" else 1)
    low, high = np.quantile(draws, [.025, .975])
    result.update(ci95_low=float(low), ci95_high=float(high), ci_status="available")
    return result


def aggregate(rows, cohort):
    groups = sorted({key(r) for r in rows}, key=group_order)
    table, cells = [], []
    for metric, label in METRICS.items():
        for group in groups:
            items = [r for r in rows if key(r) == group]
            completed = [r for r in items if r["status"] == "completed"]
            tokens = [r["output_tokens"] for r in completed if r["output_tokens"] is not None]
            entry = dict(metric=label, **dict(zip(GROUP, group, strict=True)),
                         runs_attempted=sum(r["attempted"] for r in items),
                         runs_successfully_completed=len(completed),
                         runs_ended_with_errors=sum(r["status"] == "failed" for r in items),
                         median_output_tokens_per_completed_run=(
                             statistics.median(tokens) if tokens else None))
            for condition in CONDITIONS:
                version, view = condition.split("-")
                selected = [r for r in items if r["view"] == view and r["version"] == version]
                statuses = Counter(r["status"] for r in selected)
                eligible = [r for r in selected if r["report_available"] and (
                    r["status"] == "completed" or (cohort == "finished" and r["status"] == "failed")
                )]
                value = hierarchical_mean(eligible, metric)
                # B balances evidence classes AFTER averaging each class across runs.
                components = {c: hierarchical_mean(eligible, "response_" + c) for c in CLASSES}
                if metric == "responsiveness":
                    value = (100 * statistics.mean(components.values())
                             if all(v is not None for v in components.values()) else None)
                seed_key = json.dumps([*bootstrap_group_identity(group), condition, metric], separators=(",", ":"))
                seed = CI_SETTINGS["seed"] + int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
                ci = confidence_interval(eligible, metric, seed=seed,
                                         resamples=CI_SETTINGS["bootstrap_resamples"])
                entry[condition] = value
                entry[condition + "_ci95_low"] = ci["ci95_low"]
                entry[condition + "_ci95_high"] = ci["ci95_high"]
                cells.append(dict(
                    metric=metric, **dict(zip(GROUP, group, strict=True)), condition=condition,
                    value=value, **ci, runs_planned=len(selected),
                    runs_attempted=sum(r["attempted"] for r in selected),
                    runs_completed=statuses["completed"], runs_failed=statuses["failed"],
                    runs_unfinished=statuses["unfinished"], runs_queued=statuses["queued"],
                    reports_included=len(eligible),
                    metric_available_runs=sum(r[metric] is not None for r in eligible),
                    **{"response_" + c + "_available_runs": sum(
                        r["response_" + c] is not None for r in eligible) for c in CLASSES},
                ))
            table.append(entry)
    return table, cells, groups



def condition_score(values):
    """Combine aggregate metrics, not complete-case per-run composites."""
    if any(values.get(k) is None for k in COMPONENT_WEIGHTS):
        return None
    return sum(w * values[k] for k, w in COMPONENT_WEIGHTS.items())


def summary_score(values):
    if len(values) != 4 or any(v is None for v in values):
        return None
    if any(not 0 <= v <= 100 for v in values):
        raise ValueError("Condition scores must be between 0 and 100")
    return 100 * math.prod(v / 100 for v in values) ** .25


def interaction_score(values):
    """CONDITIONS order is EU, EM, SU, SM; larger is a greater unmasked penalty."""
    if len(values) != 4 or any(v is None for v in values):
        return None
    return sum(sign * v for sign, v in zip((1, -1, -1, 1), values, strict=True))


def bootstrap_condition_score(rows, seed, resamples):
    """Resample all components together, including B's three evidence classes."""
    import numpy as np

    fields = ["f1", "focal_recovery", "exploration"] + ["response_" + c for c in CLASSES]
    items = sorted(rows, key=lambda r: str(r.get("run_id", "")))
    if not items:
        return np.full(resamples, np.nan), False
    data = np.array([[r[f] if r[f] is not None else np.nan for f in fields]
                     for r in items], dtype=float)
    rng = np.random.default_rng(seed)
    sample = data[rng.integers(0, len(data), size=(resamples, len(data)))]
    counts = np.isfinite(sample).sum(axis=1)
    means = np.divide(np.nansum(sample, axis=1), counts,
                      out=np.full((resamples, len(fields)), np.nan), where=counts > 0)
    draws = (COMPONENT_WEIGHTS["f1"] * means[:, 0]
             + COMPONENT_WEIGHTS["focal_recovery"] * means[:, 1]
             + COMPONENT_WEIGHTS["exploration"] * means[:, 2]
             + COMPONENT_WEIGHTS["responsiveness"] * 100 * means[:, 3:].mean(axis=1))
    return draws, bool((np.isfinite(data).sum(axis=0) >= 2).all())


def bootstrap_score_interval(draws, sufficient):
    import numpy as np

    valid = np.isfinite(draws)
    result = dict(ci95_low=None, ci95_high=None, ci_method="joint whole-run bootstrap",
                  ci_status="insufficient eligible runs", ci_valid_resamples=int(valid.sum()))
    if not sufficient:
        return result
    if valid.mean() < CI_SETTINGS["minimum_valid_bootstrap_fraction"]:
        result["ci_status"] = "too many bootstrap draws lack eligible evidence"
        return result
    low, high = np.quantile(draws[valid], [.025, .975])
    result.update(ci95_low=float(low), ci95_high=float(high), ci_status="available")
    return result


def interaction_interval(focal_cells):
    """Independent-proportion MOVER with Wilson limits (Zou et al., 2009)."""
    result = dict(ci95_low=None, ci95_high=None, ci_method="Wilson-based MOVER",
                  ci_status="missing focal condition", ci_valid_resamples=0)
    values = [c["value"] for c in focal_cells]
    value = interaction_score(values)
    if value is None or any(c["ci95_low"] is None or c["ci95_high"] is None
                            for c in focal_cells):
        return result
    lower_variance = upper_variance = 0
    for sign, cell in zip((1, -1, -1, 1), focal_cells, strict=True):
        below = cell["value"] - cell["ci95_low"]
        above = cell["ci95_high"] - cell["value"]
        lower_variance += (below if sign > 0 else above) ** 2
        upper_variance += (above if sign > 0 else below) ** 2
    result.update(ci95_low=max(-200, value - math.sqrt(lower_variance)),
                  ci95_high=min(200, value + math.sqrt(upper_variance)), ci_status="available")
    return result


def aggregate_cross_condition(rows, cohort, table, cells, groups):
    """Append auditable C rows and return the separate across-condition score table."""
    import numpy as np

    eligible = [r for r in rows if r["report_available"] and (r["status"] == "completed"
                or (cohort == "finished" and r["status"] == "failed"))]
    # These intervals condition on one fixed clinical dataset pair. Never silently
    # apply the independent-binomial calculation to a multi-dataset hierarchy.
    if len({r["pair_id"] for r in eligible}) > 1:
        raise ValueError("Cross-condition scores currently require one fixed dataset pair")
    lookup = {(key(r), r["condition"], r["metric"]): r for r in cells}
    cross = []
    for group in groups:
        template = next(r for r in table if key(r) == group)
        common = {k: v for k, v in template.items() if k != "metric"
                  and not any(k.startswith(c) for c in CONDITIONS)}
        condition_entry = dict(metric=CONDITION_METRICS["condition_score"], **common)
        condition_values, condition_draws, enough, focal = [], [], [], []
        for condition in CONDITIONS:
            originals = {m: lookup[group, condition, m] for m in COMPONENT_WEIGHTS}
            value = condition_score({m: c["value"] for m, c in originals.items()})
            selected = [r for r in eligible if key(r) == group
                        and r["version"] + "-" + r["view"] == condition]
            seed_key = json.dumps([*bootstrap_group_identity(group), condition, "condition_score"], separators=(",", ":"))
            seed = CI_SETTINGS["seed"] + int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
            draws, sufficient = bootstrap_condition_score(
                selected, seed, CI_SETTINGS["bootstrap_resamples"])
            ci = bootstrap_score_interval(draws, sufficient)
            cell = dict(originals["f1"], metric="condition_score", value=value, **ci)
            # C has aggregate, class-specific denominators, not a per-run score count.
            cell["metric_available_runs"] = None
            cells.append(cell)
            condition_entry[condition] = value
            condition_entry[condition + "_ci95_low"] = ci["ci95_low"]
            condition_entry[condition + "_ci95_high"] = ci["ci95_high"]
            condition_values.append(value)
            condition_draws.append(draws)
            enough.append(sufficient)
            focal.append(originals["focal_recovery"])
        table.append(condition_entry)
        summary_draws = 100 * np.prod(np.array(condition_draws) / 100, axis=0) ** .25
        entries = [
            ("summary_score", summary_score(condition_values),
             bootstrap_score_interval(summary_draws, all(enough))),
            ("label_surprise_interaction", interaction_score([c["value"] for c in focal]),
             interaction_interval(focal)),
        ]
        for metric, value, ci in entries:
            cross.append(dict(metric=metric, **common, value=value, **ci))
    return sorted(cross, key=lambda r: (list(CROSS_METRICS).index(r["metric"]), group_order(key(r))))


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return str(value)


def render_tables(out, table, notes, *, stem="results", title="Clinical experiment results"):
    columns = list(table[0])
    headings = [c.replace("_", " ").capitalize() for c in columns]
    lines = ["# " + title, "", notes, "",
             "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(display(row[c]) for c in columns) + " |" for row in table]
    atomic_text(out / f"{stem}.md", "\n".join(lines) + "\n")
    body = "".join("<tr>" + "".join(f"<td>{html.escape(display(row[c]))}</td>" for c in columns)
                   + "</tr>" for row in table)
    header = "".join(f"<th>{html.escape(c)}</th>" for c in headings)
    atomic_text(out / f"{stem}.html", "<!doctype html><meta charset='utf-8'>"
                "<title>Clinical results</title>"
                "<style>body{font:14px system-ui;margin:32px;color:#172334}"
                "table{border-collapse:collapse}"
                "th,td{padding:9px 12px;border-bottom:1px solid #dae1e9;text-align:right}"
                "th{position:sticky;top:0;background:#eaf0f7}td:nth-child(-n+5){text-align:left}"
                "tr:nth-child(even){background:#f7f9fc}p{max-width:1100px;line-height:1.6}</style>"
                f"<h1>{html.escape(title)}</h1><p>{html.escape(notes)}</p>"
                f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>")



def persistent_comparison(group):
    """Include Biomni as a separate labeled comparator; never pool it with custom runs."""
    return group[2] == "persistent" or (group[1] == "Biomni" and group[2] == "Biomni")


def comparison_order(group):
    return (not persistent_comparison(group), group_order(group))


def export_persistent_tables(out):
    """Focused companion tables reuse the saved estimates and their actual metadata."""
    for source, stem, title in (
        ("results.csv", "persistent_results", "Persistent workflows + Biomni: condition metrics"),
        ("cross_condition_scores.csv", "persistent_cross_condition_scores",
         "Persistent workflows + Biomni: summary and focal interaction"),
    ):
        with (out / source).open() as handle:
            selected = [r for r in csv.DictReader(handle) if persistent_comparison(key(r))]
        metric_order = list(dict.fromkeys(r["metric"] for r in selected))
        selected.sort(key=lambda r: (metric_order.index(r["metric"]), group_order(key(r))))
        write_csv(out / (stem + ".csv"), selected)
        # Keep CSV precision; round display numbers using the regular table renderer.
        for row in selected:
            for field in row:
                if field in ("metric", *GROUP, "ci_method", "ci_status"):
                    continue
                if not row[field]:
                    row[field] = None
                else:
                    row[field] = float(row[field])
                    if field.startswith("runs_") or field == "ci_valid_resamples":
                        row[field] = int(row[field])
        render_tables(out, selected,
                      "Persistent workflows plus Qwen 3.8 27B / Biomni as a separate "
                      "comparator. Biomni uses xhigh reasoning; other included runs use medium. "
                      "Estimates, 95% CIs, execution counts, and token medians retain their original "
                      "definitions. No runs are pooled across harnesses or workflows.",
                      stem=stem, title=title)

def render_figures(out, cells, groups, stamp, cohort, cross):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    persistent = [g for g in groups if persistent_comparison(g)]
    render_metric_figures(out / "figures/persistent_only", cells, persistent, stamp, cohort,
                          persistent_only=True, cross=cross)
    render_metric_figures(out / "figures", cells, groups, stamp, cohort, cross=cross)
    render_model_harness_figures(out, cells, groups, stamp, cohort, cross)


def draw_ci(ax, x, cell, *, cap=3):
    """Draw endpoints directly: percentile CIs need not contain the point estimate."""
    low, high = cell["ci95_low"], cell["ci95_high"]
    if low is None or high is None:
        return
    center = (low + high) / 2
    ax.errorbar(x, center, yerr=[[center - low], [high - center]], fmt="none",
                ecolor="#202b39", elinewidth=1, capsize=cap, capthick=1, zorder=5)


def render_metric_figures(figures, cells, groups, stamp, cohort, *, persistent_only=False, cross):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    colors = dict(WORKFLOW_COLORS)
    if persistent_only:
        colors["Biomni"] = colors["persistent"]
    labels = [NAMES.get(g[0], g[0]) + "\n" + (
        "Biomni" if g[1] == "Biomni" else
        ("Codex / " if g[1] == "Codex" else "Custom / ") + g[2][:4]) for g in groups]
    if persistent_only:
        # Large type remains legible when the 20-inch canvas is reduced for a grant.
        # Workflow is stated in the title, so model labels need only name the harness.
        labels = [NAMES.get(g[0], g[0]) + "\n" +
                  ("Custom" if g[1] == "Custom runner" else g[1]) for g in groups]
    figures.mkdir(parents=True, exist_ok=True)
    with PdfPages(figures / "all_metrics.pdf") as pages:
        for metric, title in CONDITION_METRICS.items():
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
                lookup = {key(r): r for r in cells if r["metric"] == metric
                          and r["condition"] == condition}
                x = np.arange(len(groups))
                for i, group in enumerate(groups):
                    cell = lookup[group]
                    value = cell["value"]
                    if value is None:
                        ax.text(i, 4, "NA", ha="center", fontsize=20 if persistent_only else 7, color="#777777")
                    else:
                        ax.bar(i, value, width=.76, color=colors[group[2]], zorder=3)
                        draw_ci(ax, i, cell)
                        label_y = max(value, cell["ci95_high"] or value) + 1.5
                        ax.text(i, label_y, f"{value:.0f}", ha="center", fontsize=20 if persistent_only else 7)
                ax.set(ylim=(0, 110), xticks=x, xticklabels=labels, ylabel=title,
                       title=condition.replace("-", " · ").title(), xlim=(-.8, len(groups) - .2))
                ax.tick_params(axis="x", labelrotation=45 if persistent_only else 60,
                               labelsize=19 if persistent_only else 8, length=0)
                if persistent_only:
                    ax.set_ylabel("Percent" if metric in ("precision", "recall", "focal_recovery")
                                  else "Score", fontsize=22, labelpad=10)
                    ax.set_title(condition.replace("-", " · ").title(), fontsize=26, pad=16)
                    ax.tick_params(axis="y", labelsize=20)
                for label in ax.get_xticklabels():
                    label.set_ha("right")
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.grid(axis="y", color="#e5e9ef", zorder=0)
                ax.spines[["top", "right", "bottom"]].set_visible(False)
            fig.suptitle(title + (" · Persistent + Biomni" if persistent_only else ""),
                         x=.05, ha="left", fontsize=32 if persistent_only else 25, weight="bold")
            if not persistent_only:
                fig.legend(handles=[Patch(color=c, label=w) for w, c in colors.items()
                                    if any(g[2] == w for g in groups)],
                           loc="upper right", bbox_to_anchor=(.97, .985), ncol=4, frameon=False)
            cohort_label = ("Successfully completed runs" if cohort == "completed"
                            else "Ended runs, including runs with errors")
            if persistent_only:
                fig.text(.05, .025, f"Snapshot {stamp[:16]} UTC | {cohort_label}\n"
                         "Persistent: medium reasoning | Biomni: xhigh reasoning\n"
                         "95% CIs, fixed dataset: bootstrap; Wilson for focal recovery. NA / missing CIs: insufficient evidence.",
                         fontsize=18, linespacing=1.35, color="#526074")
                fig.subplots_adjust(left=.08, right=.985, bottom=.27, top=.85,
                                    hspace=1.05, wspace=.22)
            else:
                fig.text(.05, .025, f"Snapshot {stamp[:16]} UTC · {cohort_label} · "
                         "95% run-level CIs on a fixed dataset pair · NA = unavailable\n"
                         "Bootstrap CIs; Wilson for focal recovery. Missing CIs: insufficient runs/evidence. "
                         "Native Biomni uses xhigh reasoning; other runs use medium.",
                         fontsize=10, color="#526074")
                fig.subplots_adjust(left=.055, right=.985, bottom=.22, top=.91, hspace=.95, wspace=.16)
            for extension in ("png", "svg", "pdf"):
                fig.savefig(figures / f"{metric}.{extension}", dpi=180, facecolor="white")
            pages.savefig(fig)
            plt.close(fig)

        for metric, title in CROSS_METRICS.items():
            fig, ax = plt.subplots(figsize=(20, 10))
            draw_cross_panel(ax, cross, groups, metric, title, labels,
                             annotation_size=20 if persistent_only else 9, colors=colors)
            ax.set_title("")
            ax.tick_params(axis="x", labelrotation=0 if persistent_only else 45,
                           labelsize=20 if persistent_only else 10, pad=10)
            if persistent_only:
                ax.tick_params(axis="y", labelsize=20)
                ax.yaxis.label.set_size(22)
            for label in ax.get_xticklabels():
                label.set_ha("center" if persistent_only else "right")
            fig.suptitle(title + (" - Persistent + Biomni" if persistent_only else ""),
                         x=.055, ha="left", fontsize=32 if persistent_only else 25, weight="bold")
            if persistent_only:
                description = ("Equal-weight geometric mean across four conditions; higher is better (0-100)."
                               if metric == "summary_score" else
                               "I = (Focal EU - Focal SU) - (Focal EM - Focal SM)\n"
                               "Positive = larger surprise penalty with recognizable labels.")
                fig.text(.055, .87, description, fontsize=22, linespacing=1.3, color="#526074")
                fig.text(.055, .025, f"Snapshot {stamp[:16]} UTC | {cohort_label}\n"
                         "Persistent: medium | Biomni: xhigh | Fixed dataset pair\n"
                         "95% CIs: run bootstrap (S); Wilson-based MOVER (I). NA / missing CIs: insufficient data.",
                         fontsize=18, linespacing=1.35, color="#526074")
                fig.subplots_adjust(left=.09, right=.985, bottom=.23, top=.78)
            else:
                fig.text(.055, .91, cross_description(metric), fontsize=13, color="#526074")
                fig.text(.055, .025, f"Snapshot {stamp[:16]} UTC | {cohort_label} | "
                         "95% CIs conditional on the fixed dataset pair.\n"
                         "S: joint whole-run bootstrap; I: Wilson-based MOVER. "
                         "NA = unavailable; absent intervals indicate insufficient runs/evidence.",
                         fontsize=11, color="#526074")
                fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.85)
            for extension in ("png", "svg", "pdf"):
                fig.savefig(figures / f"{metric}.{extension}", dpi=180, facecolor="white")
            pages.savefig(fig)
            plt.close(fig)


def cross_description(metric):
    if metric == "summary_score":
        return "Equal-weight geometric mean across four conditions; higher is better (0-100)."
    return "(Focal EU - SU) - (Focal EM - SM); positive = larger surprise penalty with recognizable labels."


def draw_cross_panel(ax, cross, groups, metric, title, labels, *, annotation_size=9, colors=None):
    import numpy as np

    colors = WORKFLOW_COLORS if colors is None else colors
    lookup = {key(r): r for r in cross if r["metric"] == metric}
    interaction = metric == "label_surprise_interaction"
    extent = 30
    for i, group in enumerate(groups):
        cell = lookup[group]
        value = cell["value"]
        if value is None:
            ax.text(i, 3, "NA", ha="center", fontsize=annotation_size)
            continue
        ax.bar(i, value, width=.7, color=colors[group[2]], zorder=3)
        draw_ci(ax, i, cell)
        extent = max(extent, abs(value), abs(cell["ci95_low"] or 0),
                     abs(cell["ci95_high"] or 0))
        label_y = max(value, cell["ci95_high"] or value) + 2
        ax.text(i, label_y, f"{value:+.1f}" if interaction else f"{value:.1f}",
                ha="center", fontsize=annotation_size)
    ax.set(title=title, ylabel="Percentage points" if interaction else "Score (0-100)",
           xticks=np.arange(len(groups)), xticklabels=labels, xlim=(-.7, len(groups) - .3))
    if interaction:
        limit = min(215, max(40, math.ceil((extent + 10) / 10) * 10))
        ax.set_ylim(-limit, limit)
        ax.axhline(0, color="#526074", linewidth=1.1, zorder=2)
    else:
        ax.set(ylim=(0, 110), yticks=[0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e5e9ef", zorder=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def render_model_harness_figures(out, cells, groups, stamp, cohort, cross):
    """Within-model comparisons: conditions on x, adjacent workflow bars, metric panels."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    destination = out / "figures/by_model_harness"
    destination.mkdir(parents=True, exist_ok=True)
    combinations = list(dict.fromkeys((g[0], g[1], g[3]) for g in groups))
    labels = [c.replace("-", "\n").title() for c in CONDITIONS]
    with PdfPages(destination / "all_model_harnesses.pdf") as pages:
        for model, harness, effort in combinations:
            selected = [g for g in groups if (g[0], g[1], g[3]) == (model, harness, effort)]
            workflows = [g[2] for g in selected]
            lookup = {(r["metric"], r["condition"], r["workflow"]): r for r in cells
                      if (r["llm"], r["harness"], r["reasoning_effort"])
                      == (model, harness, effort)}
            fig, axes = plt.subplots(3, 3, figsize=(20, 11.25))
            width = min(.23, .8 / len(workflows))
            for ax, (metric, title) in zip(list(axes.flat)[:7], CONDITION_METRICS.items(), strict=True):
                for j, workflow in enumerate(workflows):
                    positions = np.arange(len(CONDITIONS)) + (j - (len(workflows) - 1) / 2) * width
                    for x, condition in zip(positions, CONDITIONS, strict=True):
                        cell = lookup[metric, condition, workflow]
                        value = cell["value"]
                        if value is None:
                            ax.text(x, 3, "NA", ha="center", fontsize=8,
                                    color=WORKFLOW_COLORS[workflow])
                        else:
                            ax.bar(x, value, width=width * .9,
                                   color=WORKFLOW_COLORS[workflow], zorder=3)
                            draw_ci(ax, x, cell, cap=2)
                            label_y = max(value, cell["ci95_high"] or value) + 1.7
                            ax.text(x, label_y, f"{value:.0f}", ha="center", fontsize=8)
                ax.set(title=title, ylim=(0, 110), xlim=(-.55, 3.55),
                       xticks=np.arange(len(CONDITIONS)), xticklabels=labels,
                       yticks=[0, 25, 50, 75, 100])
                ax.tick_params(axis="x", length=0, pad=8)
                ax.grid(axis="y", color="#e5e9ef", zorder=0)
                ax.spines[["top", "right", "bottom"]].set_visible(False)
            for ax, (metric, title) in zip(list(axes.flat)[7:], CROSS_METRICS.items(), strict=True):
                draw_cross_panel(ax, cross, selected, metric, title,
                                 [g[2] for g in selected])
                ax.set_title("Summary S (higher is better)" if metric == "summary_score"
                             else "Interaction I (positive = unmasked penalty)", fontsize=11)
            name = NAMES.get(model, model)
            fig.suptitle(f"{name} · {harness}", x=.055, ha="left", fontsize=25, weight="bold")
            fig.text(.055, .925, f"Workflow and condition comparison · {effort} reasoning",
                     fontsize=12, color="#526074")
            fig.legend(handles=[Patch(color=WORKFLOW_COLORS[w], label=w) for w in workflows],
                       loc="upper right", bbox_to_anchor=(.97, .985),
                       ncol=len(workflows), frameon=False)
            cohort_label = ("Successfully completed runs" if cohort == "completed"
                            else "Ended runs, including runs with errors")
            fig.text(.055, .025, f"Snapshot {stamp[:16]} UTC · {cohort_label} · "
                     "S: equal condition weights; I: signed percentage points\n"
                     "95% CIs: bootstrap; Wilson for focal recovery; Wilson-based MOVER for I. Fixed dataset pair. "
                     "NA = unavailable; missing CIs indicate insufficient runs/evidence.",
                     fontsize=10, color="#526074")
            fig.subplots_adjust(left=.055, right=.98, bottom=.14, top=.85,
                                hspace=.48, wspace=.22)
            stem = re.sub(r"[^a-z0-9]+", "_", f"{name}_{harness}_{effort}".lower()).strip("_")
            for extension in ("png", "svg", "pdf"):
                fig.savefig(destination / f"{stem}.{extension}", dpi=180, facecolor="white")
            pages.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--sources", type=Path,
                        default=Path(__file__).with_name("presentation_sources.json"))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/presentation_results")
    parser.add_argument("--cohort", choices=("finished", "completed"), default="finished")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true",
                        help="Render all three figure sets from the existing output snapshot")
    parser.add_argument("--report-only", action="store_true",
                        help="Combine saved tables and figure PDFs without refreshing experiments")
    parser.add_argument("--cost-assumptions", type=Path,
                        default=Path(__file__).with_name("presentation_costs.json"))
    parser.add_argument("--refresh-costs", action="store_true",
                        help="Reread API token audits instead of reusing unchanged terminal runs")
    args = parser.parse_args()
    if args.report_only:
        if args.figures_only or args.no_figures:
            parser.error("--report-only cannot be combined with figure options")
        build_combined_pdf(args.out.resolve(), args.repo.resolve(), args.sources.resolve(),
                           args.cost_assumptions, args.refresh_costs)
        return
    if args.figures_only:
        if args.no_figures:
            parser.error("--figures-only cannot be combined with --no-figures")
        out = args.out.resolve()
        provenance = read(out / "provenance.json")
        with (out / "condition_metrics.csv").open() as handle:
            cells = list(csv.DictReader(handle))
        for cell in cells:
            for field in ("value", "ci95_low", "ci95_high"):
                if field not in cell:
                    parser.error("Saved results lack CIs; run a full refresh first")
                cell[field] = float(cell[field]) if cell[field] else None
        groups = sorted({key(r) for r in cells}, key=group_order)
        with (out / "cross_condition_scores.csv").open() as handle:
            cross = list(csv.DictReader(handle))
        for row in cross:
            for field in ("value", "ci95_low", "ci95_high"):
                row[field] = float(row[field]) if row[field] else None
        render_figures(out, cells, groups, provenance["snapshot_started_at"], provenance["cohort"], cross)
        provenance["figure_generation"] = True
        atomic_text(out / "provenance.json", json.dumps(provenance, indent=2) + "\n")
        methods = Path(__file__).with_name("presentation_results.md").read_text()
        readme = out / "README.md"
        prefix = readme.read_text().split("## Rerun", 1)[0] if readme.exists() else ""
        atomic_text(readme, prefix + methods)
        build_combined_pdf(out, args.repo.resolve(), args.sources.resolve(),
                           args.cost_assumptions, args.refresh_costs)
        print(f"Rendered all three figure sets from saved snapshot in {out / 'figures'}")
        return
    started = datetime.now(UTC).isoformat()
    sources = Sources(args.repo)
    config = sources.read(args.sources.resolve())
    for evidence in config.get("model_alias_evidence", []):
        sources.read(evidence)
    jobs = select_jobs(sources, config)
    print(f"Reading {len(jobs)} selected local runs and the Biomni export...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(
            partial(inspect_job, aliases=config.get("model_aliases")), jobs
        ), 1):
            rows.append(row)
            if i % 100 == 0:
                print(f"Read {i}/{len(jobs)} local run slots", flush=True)
    biomni, biomni_stamp = biomni_rows(sources, config)
    rows.extend(biomni)
    rows.sort(key=lambda r: (group_order(key(r)), r["view"], r["version"], r["replicate"]))
    if any(r["view"] not in ("unmasked", "masked") or
           r["version"] not in ("expected", "surprising") for r in rows):
        raise ValueError("Unexpected condition")
    # Validate dictionary schemas before emitting any deliverable.
    columns = list(rows[0])
    rows = [{c: r.get(c) for c in columns} for r in rows]
    table, cells, groups = aggregate(rows, args.cohort)
    cross = aggregate_cross_condition(rows, args.cohort, table, cells, groups)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "results.csv", table)
    write_csv(out / "cross_condition_scores.csv", cross)
    write_csv(out / "condition_metrics.csv", cells)
    write_csv(out / "run_metrics.csv", rows)
    counts = dict(Counter(r["status"] for r in rows))
    notes = (f"Snapshot started {started}. Cohort: {args.cohort}. "
             f"Selected runs: {len(rows)}; statuses: {counts}. "
             "Attempted = selected run execution directory exists or terminal result is recorded; "
             "completed = status completed (all stages successful). "
             "Counts and median output tokens "
             "span all four conditions. Performance includes failed terminal runs in the default "
             "finished cohort; queued and unfinished runs are excluded. "
             "Missing values are unavailable, "
             "never zero. F1* uses diagnostic scientific scores consistently across harnesses. "
             f"Biomni is a local export generated {biomni_stamp}. "
             "95% CIs reflect repeated-run uncertainty on this fixed dataset pair: "
             "whole-run bootstrap for scores, Wilson for focal recovery. "
             "See README.md for methods and rerunning.")
    render_tables(out, table, notes)
    render_tables(out, cross, notes + " S uses equal condition weights; I is the signed focal "
                  "interaction in percentage points. See README.md for formulas and CI methods.",
                  stem="cross_condition_scores", title="Summary and separate focal interaction")
    with (out / "results.md").open("a") as handle:
        handle.write("\n" + (out / "cross_condition_scores.md").read_text())
    with (out / "results.html").open("a") as handle:
        handle.write("<p><a href='cross_condition_scores.html'>Summary and separate focal interaction table</a></p>")
    issues = [{"source": r["source"], "error": r["report_error"]}
              for r in rows if r["report_error"]]
    provenance = dict(snapshot_started_at=started,
                      snapshot_finished_at=datetime.now(UTC).isoformat(),
                      cohort=args.cohort, counts=counts, runs=len(rows), groups=len(groups),
                      biomni_generated_at=biomni_stamp, source_files=sources.provenance,
                      issues=issues, figure_generation=not args.no_figures,
                      confidence_intervals=CI_SETTINGS, derived_scores=SCORE_SETTINGS)
    atomic_text(out / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    if args.no_figures:
        export_persistent_tables(out)
        from presentation_costs import build as build_costs
        build_costs(out, args.cost_assumptions, args.refresh_costs)
    if not args.no_figures:
        render_figures(out, cells, groups, started, args.cohort, cross)
        build_combined_pdf(out, args.repo.resolve(), args.sources.resolve(),
                           args.cost_assumptions, args.refresh_costs)
    methods = Path(__file__).with_name("presentation_results.md").read_text()
    atomic_text(out / "README.md", "# Generated clinical comparison\n\n" + notes + "\n\n" + methods)
    print(json.dumps({"output": str(out), "counts": counts, "groups": len(groups),
                      "table_rows": len(table), "issues": issues}, indent=2))


def build_combined_pdf(out, repo, sources, cost_assumptions=None, refresh_costs=False):
    """Use installed PDF libraries, or the desktop's bundled PDF runtime."""
    import importlib.util

    export_persistent_tables(out)
    from presentation_costs import DEFAULT_CONFIG, build as build_costs
    build_costs(out, cost_assumptions or DEFAULT_CONFIG, refresh_costs)
    python = Path(sys.executable)
    if not all(importlib.util.find_spec(name) for name in ("reportlab", "pypdf")):
        runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
        python = Path(os.environ.get("OCS_PDF_PYTHON", runtime / "python/bin/python3"))
        if not python.is_file():
            raise RuntimeError("Install reportlab and pypdf, or set OCS_PDF_PYTHON to a PDF runtime")
    subprocess.run([str(python), str(Path(__file__).with_name("presentation_pdf.py")),
                    "--out", str(out), "--repo", str(repo), "--sources", str(sources)], check=True)
    readme = out / "README.md"
    if readme.exists():
        prefix = readme.read_text().split("## Rerun", 1)[0]
        atomic_text(readme, prefix + Path(__file__).with_name("presentation_results.md").read_text())


if __name__ == "__main__":
    main()
