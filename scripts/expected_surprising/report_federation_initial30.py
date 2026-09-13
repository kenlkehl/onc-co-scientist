"""Score the held federation pilot with the existing presentation definitions.

Read-only for scientific artifacts. No providers or experiment launchers are imported.
Run from the repository root with python -m scripts.expected_surprising.report_federation_initial30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from scripts.expected_surprising import build_presentation_results as scoring
from scripts.expected_surprising import presentation_costs

DEFAULT_ROOT = Path("data/expected_surprising_federation/20260912_clinical10pct_named_masked")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_selected(root, selection, grid):
    rows = []
    identities = [(r["condition"], r["run_id"]) for r in selection]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate selected identity")
    for selected in selection:
        identity = selected["condition"], selected["run_id"]
        plan = grid[identity]
        row = scoring.inspect_job(
            (plan, "unmasked" if identity[0] == "named" else "masked", root / identity[0])
        )
        if not row["report_available"] or row["status"] not in ("completed", "failed"):
            raise ValueError(f"Selected run lacks terminal verified report: {identity}")
        report = scoring.read(Path(row["source"]) / "report.json")
        result = scoring.read(Path(row["source"]) / "run.json")
        if report["site_count"] != plan["site_count"]:
            raise ValueError("Site count mismatch")
        coord = report["coordination"]
        # The top-level run usage may be null even when coordinator input receipts exist.
        row.update(
            site_count=plan["site_count"],
            partition_id=plan["partition_id"],
            input_tokens=coord.get("usage", {}).get("input_tokens"),
            input_tokens_missing_calls=coord.get("usage_missing_calls"),
            iterations_completed=result.get("iterations_completed"),
            terminal_iteration=result.get("terminal_iteration"),
            stop_reason=result.get("stop_reason"),
            accepted_claims=report["scores"]["discovery"]["accepted_n"],
            confirmed_tested_claims=report["scores"]["discovery"]["confirmed_tested_n"],
            agent_calls=coord.get("agent_calls"),
            protocol_errors=len(report.get("protocol_errors", [])),
            attempt_errors=len(report.get("attempt_errors", [])),
            run_sha256=sha(Path(row["source"]) / "run.json"),
        )
        for cls in scoring.CLASSES:
            component = report["responsiveness"]["components"][cls]
            row["response_" + cls + "_correct"] = component["numerator"]
            row["response_" + cls + "_eligible"] = component["denominator"]
        rows.append(row)
    return rows


def audit_row(row):
    # Same token reader as the earlier cost report; no pricing or model calls.
    usage = presentation_costs.audit_usage(row, float("inf"))
    root = Path(row["source"]) / "provider_audit"
    calls = [p for p in root.glob("call-*") if p.is_dir()]
    missing = sum(not (p / "metrics.json").exists() for p in calls)
    return dict(
        run_id=row["run_id"],
        view=row["view"],
        llm=row["llm"],
        workflow=row["workflow"],
        site_count=row["site_count"],
        status=row["status"],
        audit_known_input_tokens=usage["input_tokens"],
        audit_known_output_tokens=usage["output_tokens"],
        audit_cached_input_tokens=usage["cached_input_tokens"],
        audit_cache_write_input_tokens=usage["cache_write_input_tokens"],
        audit_usage_records=usage["usage_records"],
        audit_missing_usage_records=usage["missing_usage_records"],
        audit_calls_without_metrics=missing,
        audit_scope="current provider_audit call metrics; excludes recovery archives",
        audit_full_attempt_cost_known=False,
    )


def aggregate_sites(rows, cohort):
    tables, cells, cross = [], [], []
    for sites in sorted({r["site_count"] for r in rows}):
        subset = [r for r in rows if r["site_count"] == sites]
        table, cell, groups = scoring.aggregate(subset, cohort)
        across = scoring.aggregate_cross_condition(subset, cohort, table, cell, groups)
        for records in (table, cell, across):
            for r in records:
                r["site_count"] = sites
                r["partition_id"] = "random"
        for r in across:
            selected = [
                x
                for x in subset
                if scoring.key(x) == scoring.key(r)
                and (cohort == "finished" or x["status"] == "completed")
            ]
            conditions = {x["version"] + "-" + x["view"] for x in selected}
            missing = [c for c in scoring.CONDITIONS if c not in conditions]
            r["missing_conditions"] = "; ".join(missing)
            r["value_status"] = (
                "available"
                if r["value"] is not None
                else "missing conditions"
                if missing
                else "missing evidence class/component"
            )
        tables.extend(table)
        cells.extend(cell)
        cross.extend(across)
    return tables, cells, cross


def token_summary(rows, audits):
    lookup = {r["run_id"]: r for r in audits}
    result = []
    for model in sorted({r["llm"] for r in rows}):
        selected = [r for r in rows if r["llm"] == model]
        completed = [r for r in selected if r["status"] == "completed"]
        outputs = [r["output_tokens"] for r in completed if r["output_tokens"] is not None]
        inputs = [r["input_tokens"] for r in completed if r["input_tokens"] is not None]
        result.append(
            dict(
                model=model,
                runs=len(selected),
                succeeded=len(completed),
                failed=sum(r["status"] == "failed" for r in selected),
                known_output_tokens=sum(r["output_tokens"] or 0 for r in selected),
                known_input_tokens=sum(r["input_tokens"] or 0 for r in selected),
                input_totals_available_runs=sum(r["input_tokens"] is not None for r in selected),
                completed_input_totals_available_runs=len(inputs),
                median_output_tokens_per_completed_run=statistics.median(outputs)
                if outputs
                else None,
                median_input_tokens_per_completed_run=statistics.median(inputs) if inputs else None,
                missing_token_calls=sum(r["missing_token_calls"] or 0 for r in selected),
                unknown_internal_attempts=sum(
                    r["unknown_internal_attempts"] or 0 for r in selected
                ),
                audit_known_input_tokens=sum(
                    lookup[r["run_id"]]["audit_known_input_tokens"] for r in selected
                )
                if audits
                else None,
                audit_known_output_tokens=sum(
                    lookup[r["run_id"]]["audit_known_output_tokens"] for r in selected
                )
                if audits
                else None,
                audit_calls_without_metrics=sum(
                    lookup[r["run_id"]]["audit_calls_without_metrics"] for r in selected
                )
                if audits
                else None,
            )
        )
    return result


def markdown_table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(scoring.display(v) for v in row) + " |" for row in rows],
        ]
    )


def render_report(out, rows, cells, cross, tokens, stamp):
    lookup = {
        (r["llm"], r["workflow"], r["site_count"], r["condition"], r["metric"]): r["value"]
        for r in cells
    }
    display_rows = []
    for r in sorted(
        rows, key=lambda r: (r["llm"], r["workflow"], r["site_count"], r["version"], r["view"])
    ):
        cond = r["version"] + "-" + r["view"]
        display_rows.append(
            [
                scoring.NAMES[r["llm"]],
                r["workflow"],
                r["site_count"],
                cond,
                r["status"],
                *[r[m] for m in scoring.METRICS],
                lookup[r["llm"], r["workflow"], r["site_count"], cond, "condition_score"],
                r["output_tokens"],
            ]
        )
    summary = [r for r in cross if r["metric"] == "summary_score"]
    lines = [
        "# Federation pilot metrics — initial 30 runs",
        "",
        f"Generated {stamp}.",
        "",
        "**24 succeeded, 6 failed. All 30 terminal scientific reports are included. "
        "The remaining 690 runs are held. No experiment or model calls were launched.**",
        "",
        "## Main findings",
        "",
        f"Focal finding recovered in **{sum(r['focal_recovery'] == 100 for r in rows)}/30** runs. "
        "This is a descriptive count across the selected pilot, not a balanced model comparison.",
        "",
        "The pilot is uneven across models, workflows, site counts and conditions. "
        "Every observed model/workflow/site/condition cell contains one run. "
        "Most groups therefore lack a four-condition summary S, and continuous-score "
        "confidence intervals are unavailable. Missing values (—) are not zero.",
        "",
        "## Summary S by model, workflow and site count",
        "",
        markdown_table(
            ["Model", "Workflow", "Sites", "S", "Availability", "Missing conditions"],
            [
                [
                    scoring.NAMES[r["llm"]],
                    r["workflow"],
                    r["site_count"],
                    r["value"],
                    r["value_status"],
                    r["missing_conditions"],
                ]
                for r in summary
            ],
        ),
        "",
        "## Per-run scientific metrics",
        "",
        "All scores below are 0–100. P = precision; R = category-balanced recall; "
        "F1* = scientific F1; E = exploration; B = evidence responsiveness; "
        "Focal = exact independently confirmed focal recovery; C = weighted condition score. "
        "With one run per observed cell, condition means equal these run values.",
        "",
        markdown_table(
            [
                "Model",
                "Workflow",
                "Sites",
                "Condition",
                "Status",
                "P",
                "R",
                "F1*",
                "E",
                "B",
                "Focal",
                "C",
                "Known output tokens",
            ],
            display_rows,
        ),
        "",
        "## Token use",
        "",
        "Same headline definition as the earlier report: median known output tokens per "
        "successfully completed run. Totals below include failed runs. Coordinator receipts "
        "include site agents, peers, chairs and returned retries; replayed journals are counted "
        "once. Input and output totals are separate; reasoning already included in output "
        "is not added again. Unknown usage makes these observed lower bounds, "
        "not exact billing totals.",
        "",
        markdown_table(
            [
                "Model",
                "Succeeded / runs",
                "Known input",
                "Input totals available runs",
                "Known output",
                "Median output / successful run",
                "Missing usage calls",
                "Unaccounted internal attempts",
            ],
            [
                [
                    scoring.NAMES[r["model"]],
                    f"{r['succeeded']} / {r['runs']}",
                    r["known_input_tokens"],
                    r["input_totals_available_runs"],
                    r["known_output_tokens"],
                    r["median_output_tokens_per_completed_run"],
                    r["missing_token_calls"],
                    r["unknown_internal_attempts"],
                ]
                for r in tokens
            ],
        ),
        "",
        "The separate provider audit totals in token_summary.csv and provider_usage.csv count "
        "current provider-call metrics, which can include requests outside the retained scientific "
        "journal. They are not added to coordinator totals. Recovery archives, interrupted calls "
        "without receipts and excluded connection checks are outside that audit subtotal.",
        "Coordinator input totals are missing for an entire run when its input accounting is "
        "incomplete. The input column above therefore sums only runs with available totals; "
        "the coverage column shows how many. Provider receipts below recover substantially more "
        "known input usage, but still cannot establish exact total cost.",
        "",
        markdown_table(
            ["Model", "Provider known input", "Provider known output", "Calls without metrics"],
            [
                [
                    scoring.NAMES[r["model"]],
                    r["audit_known_input_tokens"],
                    r["audit_known_output_tokens"],
                    r["audit_calls_without_metrics"],
                ]
                for r in tokens
            ],
        ),
        "",
        "## Definitions and comparability",
        "",
        "- Reuses `scientific_metrics`, `aggregate`, `aggregate_cross_condition`, and confidence "
        "interval functions from the earlier presentation pipeline. No evaluator or model rerun.",
        "- P is independently confirmed tested final claims / distinct accepted final claims. "
        "R gives equal weight to expected, neutral and surprising planted-finding categories.",
        "- F1* is the saved diagnostic harmonic mean of P and R, on a 0–100 scale. Undefined "
        "P stays missing while F1 follows the prior zero convention. Saved penalized F1 remains "
        "available separately in run_metrics.csv.",
        "- E averages exact-comparison coverage across categories and all 25 iterations. "
        "B averages per-run accuracies within each evidence class, then equally averages "
        "supported, excluded and ambiguous classes. Any wholly missing class makes B unavailable.",
        "- C = 0.35 F1* + 0.25 focal recovery + 0.20 E + 0.20 B. "
        "S is the geometric mean of C across expected/unmasked, expected/masked, "
        "surprising/unmasked and surprising/masked. No reweighting around missing cells.",
        "- The separate focal interaction I = (EU − SU) − (EM − SM) is also exported. "
        "S remains the earlier provisional descriptive index, not a validated endpoint.",
        "- The earlier 10,000-draw whole-run bootstrap and Wilson focal intervals are retained. "
        "One observation per cell cannot support a repeated-run bootstrap interval. "
        "Wilson intervals with n=1 are correspondingly wide.",
        "- The main cohort includes failed terminal traces. completed_only/ gives the "
        "successful-only sensitivity analysis without replacing the main results.",
        "- Frozen reports include the known prompt/evidence harness defects. Five Luna failures "
        "were associated with reference/claim errors; the Sol failure exhausted transport retries "
        "during round 20 appraisal. These are not clean estimates of intrinsic model capability.",
        "",
        "## Files",
        "",
        "- [Condition tables](results.html) · [CSV](results.csv)",
        "- [Summary S and interaction I](cross_condition_scores.html)",
        "- [Per-run metrics and denominators](run_metrics.csv)",
        "- [Long-form condition metrics and CI availability](condition_metrics.csv)",
        "- [Token summary](token_summary.csv) · [Provider audit totals](provider_usage.csv)",
        "- [Completed-only sensitivity](completed_only/results.html)",
        "- [Score figure](metrics.png) · [Vector figure](metrics.pdf)",
        "- [Provenance](provenance.json)",
        "",
        "Rerun from the repository root: "
        "`python -m scripts.expected_surprising.report_federation_initial30`.",
        "Use `--skip-provider-audit` for the coordinator-only token view; "
        "it omits provider audit subtotals.",
    ]
    scoring.atomic_text(out / "REPORT.md", "\n".join(lines) + "\n")


def plot_metrics(out, rows):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ocs-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = sorted(
        rows, key=lambda r: (r["llm"], r["workflow"], r["site_count"], r["version"], r["view"])
    )
    metrics = list(scoring.METRICS)
    data = np.array([[r[m] if r[m] is not None else np.nan for m in metrics] for r in ordered])
    fig, ax = plt.subplots(figsize=(11, 15))
    palette = plt.get_cmap("YlGnBu").copy()
    palette.set_bad("#eeeeee")
    im = ax.imshow(data, cmap=palette, vmin=0, vmax=100, aspect="auto")
    labels = [
        f"{scoring.NAMES[r['llm']]} | {r['workflow']} | {r['site_count']} sites | "
        f"{r['version'][:3]} / {'unmasked' if r['view'] == 'unmasked' else 'masked'}"
        + (" †" if r["status"] == "failed" else "")
        for r in ordered
    ]
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.set_xticks(
        range(len(metrics)),
        [
            "Precision",
            "Recall",
            "Scientific F1*",
            "Exploration E",
            "Responsiveness B",
            "Focal recovery",
        ],
        rotation=30,
        ha="right",
    )
    for y in range(len(rows)):
        for x in range(len(metrics)):
            value = data[y, x]
            ax.text(
                x,
                y,
                "—" if np.isnan(value) else f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 65 else "black",
            )
    ax.set_title("Federation pilot: all 30 terminal scientific traces", fontsize=15, pad=18)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, label="Score (0–100)")
    fig.text(
        0.02,
        0.015,
        "† Ended with errors. One run per observed cell; uneven coverage. "
        "Missing is gray, not zero. No model ranking implied.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "metrics.png", dpi=160)
    fig.savefig(out / "metrics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-provider-audit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "analysis/initial_30_metrics"
    out.mkdir(parents=True, exist_ok=True)
    selection_path = root / "control/review_hold/initial_30_results.json"
    selection = scoring.read(selection_path)["runs"]
    grid_path = root / "grid.json"
    grid = {(r["condition"], r["run_id"]): r for r in scoring.read(grid_path)}
    started = datetime.now(UTC).isoformat()
    rows = inspect_selected(root, selection, grid)
    assert len(rows) == 30
    duplicates = Counter(
        (r["llm"], r["workflow"], r["site_count"], r["view"], r["version"]) for r in rows
    )
    assert max(duplicates.values()) == 1, "Update report wording for replicated cells"
    print("Verified and scored 30 terminal reports", flush=True)
    audits = []
    if not args.skip_provider_audit:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for audit in pool.map(audit_row, rows):
                audits.append(audit)
                print(f"Read provider usage {len(audits)}/30", flush=True)
    tables, cells, cross = aggregate_sites(rows, "finished")
    for cohort, folder in [("finished", out), ("completed", out / "completed_only")]:
        folder.mkdir(exist_ok=True)
        table, cell, across = (
            (tables, cells, cross) if cohort == "finished" else aggregate_sites(rows, cohort)
        )
        scoring.write_csv(folder / "results.csv", table)
        scoring.write_csv(folder / "condition_metrics.csv", cell)
        scoring.write_csv(folder / "cross_condition_scores.csv", across)
        notes = (
            f"Initial 30 federation runs. Cohort: {cohort}. Sites kept separate. "
            "Missing is unavailable; one run per observed cell."
        )
        scoring.render_tables(folder, table, notes, title="Federation condition metrics")
        scoring.render_tables(
            folder,
            across,
            notes,
            stem="cross_condition_scores",
            title="Federation S and focal interaction",
        )
    scoring.write_csv(out / "run_metrics.csv", rows)
    if audits:
        scoring.write_csv(out / "provider_usage.csv", audits)
    tokens = token_summary(rows, audits)
    scoring.write_csv(out / "token_summary.csv", tokens)
    render_report(out, rows, cells, cross, tokens, started)
    plot_metrics(out, rows)
    provenance = dict(
        started_at=started,
        completed_at=datetime.now(UTC).isoformat(),
        source_root=str(root),
        selected_runs=30,
        statuses=dict(Counter(r["status"] for r in rows)),
        source_files={
            str(p): sha(p)
            for p in [
                selection_path,
                grid_path,
                Path(__file__),
                Path(scoring.__file__),
                Path(presentation_costs.__file__),
            ]
        },
        reports={
            r["source"]: dict(report_sha256=r["report_sha256"], run_sha256=r["run_sha256"])
            for r in rows
        },
        versions=sorted({r["versions"] for r in rows}),
        score_settings=scoring.SCORE_SETTINGS,
        ci_settings=scoring.CI_SETTINGS,
        provider_audit_included=bool(audits),
        launched_experiments=False,
    )
    scoring.atomic_text(out / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(out),
                tokens=tokens,
                summary_scores=[r for r in cross if r["metric"] == "summary_score"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
