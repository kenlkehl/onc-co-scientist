"""Paired, workflow-specific inference and a readable Aim 2 report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..harness.durable_io import atomic_write_json, atomic_write_text
from .packaging import sha256
from .schemas import PairSpec
from .summary import paired_summary


def _estimate(pairs, *, bootstrap_replicates, seed):
    values = [p["difference_pp"] for p in pairs]
    strata = defaultdict(list)
    for p in pairs:
        strata[p["modality"]].append(p["difference_pp"])
    rng = np.random.default_rng(seed)
    enough = bool(strata) and all(len(v) >= 2 for v in strata.values())
    samples = (
        [
            np.mean([x for v in strata.values() for x in rng.choice(v, len(v))])
            for _ in range(bootstrap_replicates)
        ]
        if enough
        else []
    )
    return {
        "difference_pp": float(np.mean(values)) if values else None,
        "ci95_pp": np.quantile(samples, [0.025, 0.975]).tolist() if samples else None,
        "base_dataset_n": len(pairs),
        "bootstrap_unit": (
            "base dataset, stratified by modality; paired versions and workflows retained"
        ),
        "pairs": pairs,
    }


def summarize_matrix(spec, plans, results, *, bootstrap_replicates=2000, seed=0):
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap count must be positive")
    by_id = {r["run_id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {p.run_id for p in plans}:
        raise ValueError(
            "Every planned run must have exactly one result; missing runs are not failures"
        )
    groups = defaultdict(list)
    for plan in plans:
        result = by_id[plan.run_id]
        if result["status"] not in {"completed", "failed"}:
            raise ValueError("Cannot summarize unfinished cells")
        pair = PairSpec.model_validate_json(plan.task.private_evaluation_path.read_text())
        path = result.get("scientific_report")
        if (
            path
            and result.get("scientific_report_sha256")
            and sha256(Path(path)) != result["scientific_report_sha256"]
        ):
            raise ValueError("Scientific report checksum changed")
        report = json.loads(Path(path).read_text()) if path else None
        if result["status"] == "completed" and report is None:
            raise ValueError("Completed expected/surprising run is missing its scientific report")
        groups[(plan.model.id, plan.workflow.id)].append(
            {
                "pair_id": pair.pair_id,
                "profile": pair.profile,
                "version": plan.task.semantic_condition,
                "replicate": plan.replicate,
                "primary": report["confirmation"]["primary_recovery"]
                if result["status"] == "completed"
                else 0,
                "report": report,
                "result": result,
            }
        )
    conditions = []
    for (model, workflow), runs in sorted(groups.items()):
        pairs = []
        for pair_id in sorted({r["pair_id"] for r in runs}):
            selected = [r for r in runs if r["pair_id"] == pair_id]
            versions = {
                v: [r for r in selected if r["version"] == v] for v in ("expected", "surprising")
            }
            expected_ids = set(range(1, spec.replicates + 1))
            for rows in versions.values():
                if len(rows) != spec.replicates or {r["replicate"] for r in rows} != expected_ids:
                    raise ValueError(
                        "Both paired versions require every assigned repeat exactly once"
                    )
            a, b = (
                float(np.mean([r["primary"] for r in versions[v]]))
                for v in ("expected", "surprising")
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "modality": "depmap"
                    if selected[0]["profile"].endswith("depmap")
                    else "clinical",
                    "expected": a,
                    "surprising": b,
                    "difference_pp": 100 * (b - a),
                }
            )
        reports = [r["report"] for r in runs if r["report"] is not None]
        # Never silently drop failed runs lacking a scientific trace from secondary scores.
        scientific = (
            paired_summary(reports, bootstrap_replicates=bootstrap_replicates, seed=seed)
            if len(reports) == len(runs)
            else None
        )
        calls = sum(r["result"].get("agent_calls", 0) for r in runs)
        conditions.append(
            {
                "model_profile": model,
                "workflow_id": workflow,
                "run_n": len(runs),
                "failed_runs": sum(r["result"]["status"] == "failed" for r in runs),
                "reports_available": len(reports),
                "agent_calls": calls,
                "memory_trims": sum(
                    len(r.get("coordination", {}).get("memory_trims", [])) for r in reports
                ),
                "expected_recovery": float(np.mean([p["expected"] for p in pairs])),
                "surprising_recovery": float(np.mean([p["surprising"] for p in pairs])),
                "paired_effect": _estimate(
                    pairs, bootstrap_replicates=bootstrap_replicates, seed=seed
                ),
                "scientific_scores": scientific,
            }
        )
    contrasts = []
    baselines = [w.id for w in spec.workflows if w.mode == "persistent"]
    for condition in conditions:
        for baseline in baselines:
            if baseline == condition["workflow_id"]:
                continue
            reference = next(
                c
                for c in conditions
                if c["model_profile"] == condition["model_profile"] and c["workflow_id"] == baseline
            )
            refpairs = {p["pair_id"]: p for p in reference["paired_effect"]["pairs"]}
            pairs = [
                {
                    "pair_id": p["pair_id"],
                    "modality": p["modality"],
                    "difference_pp": p["difference_pp"] - refpairs[p["pair_id"]]["difference_pp"],
                }
                for p in condition["paired_effect"]["pairs"]
            ]
            contrasts.append(
                {
                    "model_profile": condition["model_profile"],
                    "workflow_id": condition["workflow_id"],
                    "reference_workflow": baseline,
                    **_estimate(pairs, bootstrap_replicates=bootstrap_replicates, seed=seed),
                }
            )
    return {
        "conditions": conditions,
        "workflow_contrasts": contrasts,
        "primary_failure_rule": "Assigned failed runs contribute zero focal recovery",
        "contrast": (
            "(surprising minus expected) in workflow minus the same difference in persistent"
        ),
        "bootstrap_replicates": bootstrap_replicates,
    }


def render_markdown(spec, summary):
    def number(value, *, percent=False):
        return "unavailable" if value is None else f"{100 * value if percent else value:.1f}"

    lines = [
        f"# Aim 2 expected/surprising workflow results: {spec.experiment_id}",
        "",
        f"Each run used {spec.iteration_policy.iterations} iterations "
        "of explore, analyze, appraise, "
        f"and synthesize. Each paired dataset version was run {spec.replicates} time(s) per model "
        "and workflow. A repeat starts a separate agent conversation on the same dataset; it is "
        "not a new dataset.",
        "",
        "The primary outcome is whether the workflow recovered the focal planted finding and "
        "the evaluator confirmed it independently. Failed assigned runs count as zero recovery. "
        "Expected and surprising percentages below show absolute performance. Their difference "
        "is surprising minus expected, in percentage points.",
        "",
        "| Model | Workflow | Expected recovery % | Surprising recovery % | "
        "Difference (pp) | Failed / runs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in summary["conditions"]:
        lines.append(
            f"| {c['model_profile']} | {c['workflow_id']} | "
            f"{number(c['expected_recovery'], percent=True)} | "
            f"{number(c['surprising_recovery'], percent=True)} | "
            f"{number(c['paired_effect']['difference_pp'])} | {c['failed_runs']} / {c['run_n']} |"
        )
    lines += [
        "",
        "The workflow comparison subtracts the persistent workflow's paired difference "
        "from each other workflow's paired difference. A positive value means the new "
        "workflow shifts relative recovery toward surprising findings. It does not by itself "
        "show higher overall accuracy; read it alongside both recovery percentages.",
        "",
        "| Model | Workflow vs persistent | Change in paired difference (pp) | 95% interval (pp) |",
        "|---|---|---:|---|",
    ]
    for c in summary["workflow_contrasts"]:
        interval = c["ci95_pp"]
        lines.append(
            f"| {c['model_profile']} | {c['workflow_id']} vs {c['reference_workflow']} | "
            f"{number(c['difference_pp'])} | "
            + (f"{interval[0]:.1f} to {interval[1]:.1f}" if interval else "unavailable")
            + " |"
        )
    lines += [
        "",
        "Results first average repeated runs within each dataset and version, then give "
        "each base dataset equal weight. Intervals resample whole base datasets, keeping "
        "their paired versions and workflows together. At least two base datasets in each "
        "included data type are needed for an interval; a one-dataset smoke test cannot "
        "estimate uncertainty across datasets.",
        "",
        "The supporting outcomes describe the final shared research record:",
        "",
        "- **R — recovery/recall:** the fraction of planted findings recovered, giving equal "
        "weight to expected, neutral, and surprising categories.",
        "- **P — confirmed-claim fraction (precision/positive predictive value):** the "
        "fraction of final accepted claims that were tested during the run and confirmed "
        "in fresh evaluator data. Accepting a claim does not require prior validation. "
        "No accepted claims means P is unavailable.",
        "- **F1\\* — discovery performance:** the harmonic mean of R and P on a 0–100 scale. "
        "The asterisk distinguishes it from ordinary F1: R balances planted-finding "
        "categories, while P can include additional valid claims. Unrecovered stage errors "
        "set this primary discovery score to zero.",
        "- **E — exploration coverage:** how broadly and how early the workflow tested "
        "planted findings across the full iteration budget, on a 0–100 scale.",
        "- **B — evidence responsiveness:** balanced accuracy of responses to supportive, "
        "excluding, and ambiguous validation evidence, on a 0–100 scale. If a class never "
        "occurred, B is unavailable; this is missing evidence to assess, not a failure. "
        "The two-iteration response deadline is a reassessment checkpoint, not a run cap.",
        "",
        "| Model | Workflow | R % | P % | F1* | E | B | Participant calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in summary["conditions"]:
        s = c["scientific_scores"] or {}
        d = s.get("discovery_components", {})
        lines.append(
            f"| {c['model_profile']} | {c['workflow_id']} | "
            f"{number(d.get('R'), percent=True)} | {number(d.get('Q'), percent=True)} | "
            f"{number(s.get('D'))} | {number(s.get('E'))} | {number(s.get('B'))} | "
            f"{c['agent_calls']} |"
        )
    lines += [
        "",
        "All modes share analysis limits, validation samples, deadlines, final confirmation, "
        "and scientific scoring. Persistent mode retains recent committed conversation. Sequential "
        "mode uses a fresh conversation at each stage with the common ledger and notes. "
        "Deliberative peers see the same committed evidence and submit independent drafts; "
        "later rounds may see earlier peer drafts. The chair selects the only action that "
        "changes the research record. Drafts and disagreements remain in the call audit.",
        "",
        (
            "Persistent history is bounded at "
            f"{spec.expected_surprising.persistent_history_chars:,} "
            "characters of prior conversation. The complete current ledger and latest notebook "
            "are always provided. Oldest turns are removed only from active context, with each "
            "trim disclosed to the agent and recorded in the audit. This is not an iteration cap."
            if spec.expected_surprising.persistent_history_chars is not None
            else "Persistent conversation history is unlimited; "
            "context-limit failures are retained."
        ),
        "",
        "This comparison uses each workflow's natural number of participants. With two peers "
        "and one round, deliberation uses three calls per stage versus one in the other modes. "
        "It therefore compares complete workflows under equal scientific opportunity, not "
        "equal token or call spending. All attempts, including repairs and provider errors, "
        "are recorded. Missing provider token counts remain unavailable.",
        "",
        "Machine-readable reports retain Q for P and D for F1* for compatibility. Each run "
        "includes the final scientific report, participant prompts and replies, and frozen "
        "input/code hashes. Secondary scores are unavailable for a whole condition if any "
        "assigned run lacks a scientific trace; the primary outcome still includes failures.",
        "",
    ]
    return "\n".join(lines)


def write_report(spec, plans, results, root, *, bootstrap_replicates=2000):
    summary = summarize_matrix(spec, plans, results, bootstrap_replicates=bootstrap_replicates)
    atomic_write_json(root / "expected_surprising_summary.json", summary)
    atomic_write_text(root / "expected_surprising_report.md", render_markdown(spec, summary))
    return summary
