"""Score isolated, completed-iteration snapshots without calling a model.

At a committed synthesis, accepted_ids is required to cover exactly the
controller's accepted canonical claims. Reconstruct that set and the valid
tested comparisons, then use the frozen confirmation and discovery scorer.
Live journals, experiment inputs, and official final reports are never changed.
"""
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/src"))

from onc_co_scientist.expected_surprising.evaluation import WorkflowValidationService
from onc_co_scientist.expected_surprising.generation import version_discoveries
from onc_co_scientist.expected_surprising.masking import MaskedValidationService, SemanticMask
from onc_co_scientist.expected_surprising.schemas import PairSpec, WorkflowStageRecord
from onc_co_scientist.expected_surprising.scoring import claim_key, comparison_key, workflow_discovery


def score_snapshot(directory):
    snapshot = json.loads((directory / "snapshot.json").read_text())
    private = ROOT / "input_data" / snapshot["masking"] / "private/es-v2-nsclc_depmap-42005"
    pair = PairSpec.model_validate_json((private / "pair.json").read_text())
    packaged = json.loads((private / "workflow.json").read_text())
    header = snapshot["header"]
    assert header["policy"] == packaged["policy"]
    args = (pair, snapshot["version"], header["replicate_id"], header["policy"])
    if snapshot["masking"] == "masked":
        service = MaskedValidationService(*args, SemanticMask(packaged["masking"]))
    else:
        service = WorkflowValidationService(*args)

    hypotheses, tested = {}, {}
    final = None
    for event in snapshot["stages"]:
        record = WorkflowStageRecord.model_validate(event["record"])
        for h in record.hypotheses:
            if h.id in hypotheses:
                assert hypotheses[h.id] == h
            hypotheses[h.id] = h
        if record.stage == "analyze":
            assert len(record.executed_ids) == len(event["results"])
            for hid, result in zip(record.executed_ids, event["results"], strict=True):
                assert result["hypothesis_id"] == hid and result["source"] == "discovery"
                if result["valid"]:
                    tested.setdefault(comparison_key(hypotheses[hid]), result)
        final = record
    assert final.stage == "synthesize" and final.iteration == snapshot["checkpoint_iteration"]
    accepted_keys = {claim_key(hypotheses[hid]) for hid in final.accepted_ids}
    assert len(accepted_keys) == len(final.accepted_ids)
    unique = {}
    for h in hypotheses.values():
        if claim_key(h) in accepted_keys:
            unique.setdefault(claim_key(h), h)
    accepted = list(unique.values())
    assert len(accepted) == len(accepted_keys)

    confirmation = service.confirm(accepted)
    discovery = workflow_discovery(
        accepted, version_discoveries(service.spec, snapshot["version"]), tested,
        confirmation, failed=False, repaired=bool(snapshot["attempt_errors"]),
    )
    primary = int(any(
        m["discovery_id"] == service.spec.focal_id and m["match"] == "exact"
        for m in discovery["confirmed_matches"]
    ))
    report = {
        "status": "interim_completed_iteration_snapshot",
        **{k: snapshot[k] for k in ("run_id", "replicate", "masking", "version", "checkpoint_iteration")},
        "configured_final_iteration": 25,
        "snapshot_sha256": hashlib.sha256((directory / "snapshot.json").read_bytes()).hexdigest(),
        "source_prefix_sha256": snapshot["source_prefix_sha256"],
        "focal_recovery": primary,
        "discovery": discovery,
        "confirmation": confirmation,
        "accepted_ids": [h.id for h in accepted],
        "tested_comparisons": len(tested),
        "attempt_errors_to_checkpoint": len(snapshot["attempt_errors"]),
        "exhausted_stages_to_checkpoint": len(snapshot["protocol_errors"]),
        "model_calls_for_scoring": 0,
    }
    (directory / "interim_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(json.loads((ROOT / "interim_scores/LATEST.json").read_text())["snapshot"])
    manifest = json.loads((out / "manifest.json").read_text())
    rows = []
    for item in manifest["snapshots"]:
        row = score_snapshot(out / item["run_id"])
        rows.append(row)
        print(json.dumps({"scored": len(rows), "run_id": row["run_id"], "iteration": row["checkpoint_iteration"], "focal": row["focal_recovery"], "D": row["discovery"]["exact"]["D"]}), flush=True)
    cells = []
    for version in ("expected", "surprising"):
        for masking in ("masked", "unmasked"):
            selected = [x for x in rows if x["version"] == version and x["masking"] == masking]
            assert len(selected) == 10
            exact = [x["discovery"]["exact"] for x in selected]
            q = [x["Q"] for x in exact if x["Q"] is not None]
            cells.append({
                "version": version, "masking": masking, "snapshot_runs": len(selected),
                "iteration_min": min(x["checkpoint_iteration"] for x in selected),
                "iteration_max": max(x["checkpoint_iteration"] for x in selected),
                "iteration_median": statistics.median(x["checkpoint_iteration"] for x in selected),
                "focal_recovery": sum(x["focal_recovery"] for x in selected),
                "mean_exact_D": statistics.mean(x["D"] for x in exact),
                "mean_exact_R": statistics.mean(x["R"] for x in exact),
                "mean_Q": statistics.mean(q) if q else None,
                "Q_available_runs": len(q),
                "mean_R_c": {c: statistics.mean(x["R_c"][c] for x in exact) for c in ("expected", "neutral", "surprising")},
                "accepted_claims": sum(x["discovery"]["accepted_n"] for x in selected),
                "confirmed_tested_claims": sum(x["discovery"]["confirmed_tested_n"] for x in selected),
                "attempt_errors": sum(x["attempt_errors_to_checkpoint"] for x in selected),
                "exhausted_stages": sum(x["exhausted_stages_to_checkpoint"] for x in selected),
            })
    checkpoints = {x["checkpoint_iteration"] for x in rows}
    budget_note = (
        f"All runs are compared at exactly {next(iter(checkpoints))} completed iterations; the full 25-iteration runs remain unfinished."
        if len(checkpoints) == 1 else "Runs are unfinished and have unequal iteration budgets at this snapshot."
    )
    result = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_started_at": manifest["snapshot_started_at"],
        "workflow": "persistent", "full_runs_completed_at_snapshot": manifest["completed_runs_at_start"],
        "scope": manifest["scope"], "cells": cells,
        "method": "Frozen independent final-confirmation evaluator and exact discovery scorer applied to the completed-iteration state in each snapshot; macro-average run scores within each cell. No model calls or agent-visible feedback.",
        "limitations": [budget_note, "Current acceptances and confirmation thresholds can change before iteration 25.", "One synthetic dataset pair; repeats measure stochastic model variability, not generalization across datasets.", "E and B are omitted because this is an interim discovery-state evaluation, not a completed workflow report."],
    }
    (out / "INTERIM_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Provisional persistent scores", "", f"Snapshot: {manifest['snapshot_started_at']}", "", manifest["scope"] + " " + budget_note + " These are provisional scores, not phase results.", "", "| Version | Masking | Iterations reached | Focal / 10 | Mean exact D (F1*) | Recall R % | Confirmed-claim fraction P % |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        precision = f"{100*c['mean_Q']:.1f}" if c['mean_Q'] is not None else "NA"
        lines.append(f"| {c['version']} | {c['masking']} | {c['iteration_min']}–{c['iteration_max']} | {c['focal_recovery']} | {c['mean_exact_D']:.1f} | {100*c['mean_exact_R']:.1f} | {precision} |")
    precision_denominators = "; ".join(f"{c['version']}/{c['masking']}: {c['Q_available_runs']}/10" for c in cells)
    lines.extend(["", "Focal recovery requires exact recovery and independent confirmation. R equally weights expected, neutral, and surprising planted-finding categories. P is the fraction of accepted claims that were tested and confirmed on fresh evaluator data. D/F1* is the harmonic mean of R and P on a 0–100 scale. Each metric is averaged separately across runs; mean D need not equal the harmonic mean of mean R and mean P.", "", "P is unavailable when a run has no accepted claims; its averages use these available-run denominators: " + precision_denominators + ". R, D, and focal recovery retain all ten runs per cell.", "", *result["limitations"], ""])
    (out / "INTERIM_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
