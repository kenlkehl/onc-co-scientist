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
        groups[
            (plan.model.id, plan.workflow.id, plan.federation.label if plan.federation else "n1")
        ].append(
            {
                "site_count": plan.federation.sites if plan.federation else 1,
                "partition_id": plan.federation.partition.id if plan.federation else "random",
                "pair_id": pair.pair_id,
                "profile": pair.profile,
                "version": plan.task.semantic_condition,
                "replicate": plan.replicate,
                "primary": report["confirmation"]["primary_recovery"]
                if report is not None
                and (
                    result["status"] == "completed"
                    or report.get("retry_policy", {}).get("stage_failure_policy")
                    == "retain_scientific_scores"
                )
                else 0,
                "report": report,
                "result": result,
            }
        )
    conditions = []
    for (model, workflow, federation_condition), runs in sorted(groups.items()):
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
        # Seeded resampling and artifact order must not depend on the run schedule.
        reports = [
            r["report"]
            for r in sorted(runs, key=lambda r: (r["pair_id"], r["version"], r["replicate"]))
            if r["report"] is not None
        ]
        # Never silently drop failed runs lacking a scientific trace from secondary scores.
        scientific = (
            paired_summary(reports, bootstrap_replicates=bootstrap_replicates, seed=seed)
            if len(reports) == len(runs)
            else None
        )
        calls = sum(r["result"].get("agent_calls", 0) for r in runs)
        token_audits = [r.get("coordination", {}).get("output_token_accounting") for r in reports]
        known_tokens = sum(a["known_output_tokens"] for a in token_audits if a is not None)
        missing_calls = sum(a["missing_calls"] for a in token_audits if a is not None)
        missing_attempts = sum(
            a["unaccounted_infrastructure_attempts"] for a in token_audits if a is not None
        )
        complete_tokens = (
            len(reports) == len(runs)
            and all(a is not None for a in token_audits)
            and missing_calls == 0
            and missing_attempts == 0
        )
        conditions.append(
            {
                "model_profile": model,
                "workflow_id": workflow,
                "federation_condition": federation_condition,
                "site_count": runs[0]["site_count"],
                "partition_id": runs[0]["partition_id"],
                "run_n": len(runs),
                "failed_runs": sum(r["result"]["status"] == "failed" for r in runs),
                "reports_available": len(reports),
                "agent_calls": calls,
                "output_tokens": known_tokens if complete_tokens else None,
                "known_output_tokens": known_tokens,
                "mean_output_tokens_per_run": known_tokens / len(runs) if complete_tokens else None,
                "output_tokens_missing_calls": missing_calls,
                "output_tokens_missing_run_audits": len(runs)
                - sum(a is not None for a in token_audits),
                "output_tokens_unaccounted_infrastructure_attempts": missing_attempts,
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
                if c["model_profile"] == condition["model_profile"]
                and c["workflow_id"] == baseline
                and c["federation_condition"] == condition["federation_condition"]
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
                    "site_count": condition["site_count"],
                    "partition_id": condition["partition_id"],
                    "federation_condition": condition["federation_condition"],
                    **_estimate(pairs, bootstrap_replicates=bootstrap_replicates, seed=seed),
                }
            )
    return {
        "conditions": conditions,
        "workflow_contrasts": contrasts,
        "primary_failure_rule": (
            "Scientific recovery is retained after stage failures; "
            "execution failures are reported separately"
            if spec.expected_surprising.stage_failure_policy == "retain_scientific_scores"
            else "Assigned failed runs contribute zero focal recovery"
        ),
        "contrast": (
            "(surprising minus expected) in workflow minus the same difference in persistent"
        ),
        "bootstrap_replicates": bootstrap_replicates,
    }


def render_markdown(spec, summary):
    def number(value, *, percent=False):
        return "unavailable" if value is None else f"{100 * value if percent else value:.1f}"

    def workflow_label(c):
        label = c["workflow_id"]
        if spec.federation is not None:
            label += f" (N={c.get('site_count', 1)}, {c.get('partition_id', 'random')})"
        return label

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
        "the evaluator confirmed it independently. "
        + summary["primary_failure_rule"]
        + ". "
        + "Expected and surprising percentages below show absolute performance. Their difference "
        "is surprising minus expected, in percentage points.",
        "",
        "| Model | Workflow | Expected recovery % | Surprising recovery % | "
        "Difference (pp) | Failed / runs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in summary["conditions"]:
        lines.append(
            f"| {c['model_profile']} | {workflow_label(c)} | "
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
            f"| {c['model_profile']} | {workflow_label(c)} vs {c['reference_workflow']} | "
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
        "categories, while P can include additional valid claims. "
        + (
            "Execution failures are reported separately; discoveries remain scored."
            if spec.expected_surprising.stage_failure_policy == "retain_scientific_scores"
            else "Unrecovered stage errors set this primary discovery score to zero."
        ),
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
            f"| {c['model_profile']} | {workflow_label(c)} | "
            f"{number(d.get('R'), percent=True)} | {number(d.get('Q'), percent=True)} | "
            f"{number(s.get('D'))} | {number(s.get('E'))} | {number(s.get('B'))} | "
            f"{c['agent_calls']} |"
        )
    lines += [
        "",
        "The table averages each run's R, P, and F1* separately. Its F1* column therefore "
        "need not equal the harmonic mean of the displayed average R and P. "
        + (
            "Execution failure counts accompany the scientific scores."
            if spec.expected_surprising.stage_failure_policy == "retain_scientific_scores"
            else "Unrecovered stage errors also set that run's primary F1* to zero."
        ),
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
    lines += [
        "**Output-token use**",
        "",
        "All participant responses count: linear agents, peers, the chair, and retries after "
        "invalid responses. Cached replay counts once. Provider output totals include reasoning "
        "where reported; reasoning subsets are not added again. Known totals are lower bounds "
        "when usage is missing. Unreported provider-internal work cannot be measured.",
        "",
        "| Model | Workflow | Known output tokens | Mean per run | "
        "Missing call / run / transport counts |",
        "|---|---|---:|---:|---|",
    ]
    for c in summary["conditions"]:
        lines.append(
            f"| {c['model_profile']} | {workflow_label(c)} | {c.get('known_output_tokens', 0):,} | "
            f"{number(c.get('mean_output_tokens_per_run'))} | "
            f"{c.get('output_tokens_missing_calls', 0)} / "
            f"{c.get('output_tokens_missing_run_audits', c['run_n'])} / "
            f"{c.get('output_tokens_unaccounted_infrastructure_attempts', 0)} |"
        )
    lines += [
        "",
        "**How E and B are calculated**",
        "",
        "E = 100 × the average over all iterations of mean exact-target coverage across "
        "expected, neutral, and surprising categories. A valid test counts regardless of "
        "acceptance or claimed direction; repeated tests add no coverage. Earlier testing "
        "earns more credit.",
        "",
        "B = 100 × (supported agreement + excluded agreement + ambiguous agreement) / 3. "
        "The evaluator calls validation supported when its lower interval bound exceeds "
        "the private cutoff (credit for accept), excluded when its upper bound is below "
        "the cutoff (credit for reject), and ambiguous otherwise (credit for unresolved). "
        "Intervals are oriented to the claim. The assessment is scored two iterations after "
        "delivery. Class agreement fractions are calculated within runs, then averaged over "
        "repeats within version, versions, and datasets equally before combining classes. "
        "Missing due assessments score zero; invalid results and deadlines outside the "
        "reached or budgeted iterations are excluded. An absent class makes B unavailable.",
        "",
    ]
    if "clinical_significance_10pct" in spec.experiment_id:
        lines += [
            "Agents were told that a relative outcome difference of 10% or greater is "
            "clinically significant, without log-scale calculations or mechanical decision "
            "rules. The unchanged private clinical cutoff is 0.10 natural-log PFS units "
            "(approximately 11%), close to but not identical to the public guidance.",
            "",
        ]
    lines += ["| Model | Requested reasoning | Requested tier |", "|---|---|---|"]
    for model in spec.models:
        config = model.provider_config or {}
        lines.append(
            f"| {model.id} | {config.get('reasoning_effort', 'unspecified')} | "
            f"{config.get('service_tier', 'unspecified')} |"
        )
    lines += [
        "",
        "vLLM accepts these fields, but model/template behavior determines their effect. "
        "Equal requested reasoning levels do not establish equal reasoning budgets.",
        "",
    ]
    if spec.federation is not None:
        lines += [
            "**Federated interpretation**",
            "",
            "Each N/partition condition is summarized separately. "
            "N=1 uses the original single-site "
            "path without central calls. For N>1 the central agent directs sites before each of "
            "the four stages and integrates aggregate site handoffs afterward. All sites pursue "
            "one goal; site workflows are the listed persistent, sequential, or deliberative mode.",
            "",
            "There is one shared allowance of 12 comparisons per iteration and one validation "
            "budget. Each selected comparison runs across sites and consumes one global slot. "
            "Local teams recommend validation; only the central decision spends a request. "
            "No rows or identifier values enter site-to-central or site-to-site messages. "
            "Cell means and variances are suppressed below 20 observations; suppressed site "
            "cells make combined evidence unavailable rather than silently dropping a site.",
            "",
            "Combined intervals are calculated by merging cell counts, means, and variances "
            "and applying the same pooled contrast calculation. Site-specific intervals are "
            "also shown. The central agent chooses how to interpret pooled support, replication, "
            "and disagreement; no replication rule is imposed. Final confirmation still targets "
            "the unified population in independent private evaluator data. B measures agreement "
            "with its interval rule, not the quality of every possible federation argument.",
            "",
            "Random splits shuffle reproducibly and balance site sizes. Heterogeneous splits "
            "sort on explicitly configured public covariates (random tie-breaking), then divide "
            "into similarly sized sites. A split uses pair/repeat/partition seeds shared across "
            "models and workflows. The split is recorded by membership hashes, never row exports. "
            "Site-local histories and central history are isolated; all their tokens share the "
            "reported resource totals. Call budgets include both central calls per stage.",
            "",
        ]
    return "\n".join(lines)


def write_report(spec, plans, results, root, *, bootstrap_replicates=2000):
    summary = summarize_matrix(spec, plans, results, bootstrap_replicates=bootstrap_replicates)
    atomic_write_json(root / "expected_surprising_summary.json", summary)
    atomic_write_text(root / "expected_surprising_report.md", render_markdown(spec, summary))
    return summary
