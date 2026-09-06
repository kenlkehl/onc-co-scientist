"""Build source-linked text packets for a retrospective NSCLC recovery audit.

This is evidence retrieval, not an automatic semantic recovery classifier.
Reviewer decisions and their supporting quotations are recorded separately.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/nsclc_common_text_audit"
MAPPING = json.loads(
    (REPO / "data/ds001_sol_cli_loose/private/nsclc/column_mapping.json").read_text()
)


def canonical(text: str) -> str:
    for name, masked in MAPPING.items():
        text = re.sub(rf"\b{masked}\b", name, text, flags=re.I)
        text = re.sub(rf"\bf0*{int(masked[-3:])}\b", name, text, flags=re.I)
    return text


def modifiers(text: str) -> list[str]:
    text = canonical(text).lower()
    patterns = {
        "kras": r"kras|g12c",
        "alk": r"\balk\b|alk_fusion",
        "brca2": r"brca.?2",
        "sex": r"\bmale\b|\bfemale\b|sex|\bmen\b|\bwomen\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((REPO / "data/ds001_sol_cli_loose/plan.json").read_text())
    sources = []
    for variant in ("named", "anonymized"):
        for p in sorted(
            (REPO / "example_data_clinical_all_claude/ds001/tasks/nsclc" / variant / "runs").glob(
                "*/transcript.json"
            )
        ):
            sources.append(("opus", variant, p.parent.name, p))
    for j in plan["jobs"]:
        sources.append(("sol", j["variant"], j["job_id"], Path(j["workspace"]) / "transcript.json"))
    assert len(sources) == 80
    inventory = []
    for model, variant, run, path in sources:
        key = f"{model}_{variant}_{run}"
        transcript = json.loads(path.read_text())
        summary_path = path.parent / "analysis_summary.txt"
        records = []
        for it in transcript["iterations"]:
            for h in it["proposed_hypotheses"]:
                records.append(
                    {
                        "location": f"iteration {it['index']} hypothesis {h['id']}",
                        "text": h["text"],
                        "kind": "hypothesis",
                        "id": h["id"],
                    }
                )
            for n, a in enumerate(it["analyses"]):
                records.append(
                    {
                        "location": f"iteration {it['index']} analysis {n}",
                        "text": a["result_summary"],
                        "kind": "analysis",
                        "hypothesis_ids": a["hypothesis_ids"],
                        "p_value": a.get("p_value"),
                        "effect_estimate": a.get("effect_estimate"),
                        "significant": a.get("significant"),
                    }
                )
        summary = summary_path.read_text() if summary_path.exists() else ""
        candidates = [r for r in records if len(modifiers(r["text"])) >= 3]
        # Include all hypotheses in a second packet: broad screening must never
        # silently turn absence of a lexical match into a negative decision.
        packet = {
            "key": key,
            "source": str(path.relative_to(REPO)),
            "candidates": candidates,
            "summary": summary,
            "all_hypotheses": [r for r in records if r["kind"] == "hypothesis"],
            "all_records": records,
        }
        (OUT / f"{key}.json").write_text(json.dumps(packet, indent=2) + "\n")
        inventory.append(
            {
                "key": key,
                "model": model,
                "variant": variant,
                "run": run,
                "transcript_path": str(path.relative_to(REPO)),
                "transcript_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "summary_path": str(summary_path.relative_to(REPO)) if summary else None,
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()
                if summary
                else None,
                "iterations": len(transcript["iterations"]),
                "hypotheses": len(packet["all_hypotheses"]),
                "candidates": len(candidates),
            }
        )
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sessions": len(inventory),
                "missing_summaries": [r["key"] for r in inventory if not r["summary_path"]],
                "candidate_records": sum(r["candidates"] for r in inventory),
            }
        )
    )


def report() -> None:
    out = REPO / "experiments/aim1_recovery/results/nsclc_common_text_audit_20260906"
    archived_mapping = (
        REPO / "example_data_clinical_all_claude/ds001/nsclc/anonymized/column_mapping.json"
    )
    assert json.loads(archived_mapping.read_text()) == MAPPING
    decisions = json.loads((out / "decisions.json").read_text())
    inventory = json.loads((OUT / "inventory.json").read_text())
    assert len(decisions) == 80 == len({r["key"] for r in decisions})
    by_key = {r["key"]: r for r in inventory}
    assert set(by_key) == {r["key"] for r in decisions}
    assert Counter((r["model"], r["variant"]) for r in inventory) == {
        (m, v): 20 for m in ("opus", "sol") for v in ("named", "anonymized")
    }
    old = json.loads(
        (REPO / "example_data_clinical_all_claude/ds001/score/batch_score.json").read_text()
    )
    old_rates = {
        b["variant"]: b["n_replicates_uncovered"]
        for b in old["per_bundle"]
        if b["dataset_id"] == "ds001_nsclc"
    }
    prior_sol_path = (
        REPO / "experiments/aim1_recovery/results/sol_nsclc_cli_loose_20260906/run_scores.csv"
    )
    with prior_sol_path.open() as f:
        prior_sol = {r["job_id"]: r for r in csv.DictReader(f)}
    manifest_path = REPO / "data/ds001_sol_cli_loose/private/nsclc/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["associations"][0]
    target_gate = {**spec["subgroup"]["predicate"], "treatment_sotorasib": 1}
    inverse = {v: k for k, v in MAPPING.items()}
    details, symbolic = [], []
    for decision in decisions:
        source = by_key[decision["key"]]
        path = REPO / source["transcript_path"]
        header = json.loads(path.read_text())
        assert header["dataset_id"] == "ds001_nsclc"
        assert (
            header["model_id"] == {"opus": "claude-opus-4-7", "sol": "gpt-5.6-sol"}[source["model"]]
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["transcript_sha256"]
        if source["summary_path"]:
            assert (
                hashlib.sha256((REPO / source["summary_path"]).read_bytes()).hexdigest()
                == source["summary_sha256"]
            )
        packet = json.loads((OUT / (decision["key"] + ".json")).read_text())
        hid = decision["witness_hypothesis_id"]
        if hid:
            witnesses = [r for r in packet["all_hypotheses"] if r["id"] == hid]
            assert len(witnesses) == 1
            linked = [
                r
                for r in packet["all_records"]
                if r["kind"] == "analysis" and hid in r["hypothesis_ids"]
            ]
            assert linked
        else:
            witnesses, linked = [], []
        negative_hid = decision.get("negative_witness_hypothesis_id")
        if negative_hid:
            witnesses = [r for r in packet["all_hypotheses"] if r["id"] == negative_hid]
            assert len(witnesses) == 1
        summary_n = decision.get("summary_paragraph")
        if summary_n is not None:
            paras = re.split(r"\n\s*\n", packet["summary"].strip())
            witnesses.append(
                {"location": f"summary paragraph {summary_n}", "text": paras[summary_n]}
            )
        assert witnesses, decision["key"]
        details.append(
            {
                **source,
                **decision,
                "text_evidence": witnesses,
                "linked_analyses": linked,
                "canonical_evidence": [canonical(r["text"]) for r in witnesses],
            }
        )
        if source["model"] == "sol":
            transcript = json.loads(path.read_text())
            matches = []
            for it in transcript["iterations"]:
                for h in it["proposed_hypotheses"]:
                    finding = h.get("finding") or {}
                    exposure = inverse.get(finding.get("exposure"), finding.get("exposure"))
                    ps = finding.get("subgroup", [])
                    if not exposure or any(p["operator"] != "eq" for p in ps):
                        continue
                    gate = {inverse.get(p["column"], p["column"]): p["value"] for p in ps}
                    if (
                        len(gate) != len(ps)
                        or exposure in gate
                        or finding.get("direction") not in (-1, 1)
                    ):
                        continue
                    # Favorable endpoint of the submitted binary contrast. This
                    # preserves all predicates and never fills missing ones.
                    gate[exposure] = 1 if finding["direction"] == 1 else 0
                    if gate == target_gate and finding.get("outcome") == "pfs_months":
                        matches.append(
                            {
                                "hypothesis_id": h["id"],
                                "iteration": it["index"],
                                "exposure": exposure,
                                "direction": finding["direction"],
                                "favorable_gate": gate,
                            }
                        )
            original = prior_sol[source["run"]]["primary_recovered"].lower() == "true"
            symbolic.append(
                {
                    "key": decision["key"],
                    "variant": source["variant"],
                    "original_primary": original,
                    "favorable_gate_match": bool(matches),
                    "matches": matches,
                }
            )
            assert bool(matches) == decision["joint_pattern_text"], decision["key"]
            assert original == decision["treatment_text"], decision["key"]
    rows = []
    for model in ("opus", "sol"):
        for variant in ("named", "anonymized"):
            group = [r for r in details if r["model"] == model and r["variant"] == variant]
            row = {
                "model": model,
                "variant": variant,
                "n": len(group),
                "original_recovery_n": old_rates[variant]
                if model == "opus"
                else sum(r["original_primary"] for r in symbolic if r["variant"] == variant),
            }
            for field in (
                "treatment_text",
                "joint_pattern_text",
                "reported_treatment_analysis",
                "reported_joint_analysis",
            ):
                row[field + "_n"] = sum(r[field] for r in group)
            rows.append(row)
    for name, value in (
        ("evidence.json", details),
        ("symbolic_crosscheck.json", symbolic),
        ("inventory.json", inventory),
        ("summary.json", rows),
    ):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    with (out / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "validation.json").write_text(
        json.dumps(
            {
                "sessions": 80,
                "all_source_hashes_unchanged": True,
                "model_and_dataset_labels_verified": True,
                "archived_and_current_column_mappings_identical": True,
                "all_decisions_have_source_evidence": True,
                "sol_text_and_symbolic_gate_counts_agree": True,
                "sol_treatment_text_and_original_primary_agree": True,
                "missing_summary": "opus_anonymized_run_010",
                "reviewer": "Coordinating assistant; retrospective, unblinded single review",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "decisions_sha256": hashlib.sha256(
                    (out / "decisions.json").read_bytes()
                ).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "mapping": MAPPING,
                "symbolic_limitation": (
                    "Matching the favorable joint cell is not equivalence of conditional "
                    "estimands or causal identification. No held-out confirmation is "
                    "transferred across axes."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    build()
    report()
