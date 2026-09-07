"""Audit frozen package hashes, smoke timing, and score arithmetic without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_packages(root, release):
    manifest = json.loads((root / "package_manifest.json").read_text())
    frozen = {p["pair_id"]: p for p in json.loads(release.read_text())["pairs"]}
    rows = []
    for task in manifest["tasks"]:
        expected = frozen[task["pair_id"]]["assignments"][task["version"]]
        public = root / "public" / task["task_id"]
        metadata = json.loads((public / "task.json").read_text())
        rows.append(
            {
                "task_id": task["task_id"],
                "pair_id": task["pair_id"],
                "version": task["version"],
                "source_identity_matches": task["source_task_id"] == expected["task_id"],
                "dataset_hash_matches": digest(public / "dataset.parquet")
                == task["sha256"]
                == task["source_sha256"]
                == expected["sha256"],
                "package_hashes_match": all(
                    digest(public / name) == value for name, value in task["package_sha256"].items()
                ),
                "public_effect_cutoffs_absent": metadata["versions"]["workflow"]
                not in {"appraisal-3.1.0", "appraisal-3.2.0"}
                or (
                    all("delta" not in o for o in metadata["outcomes"])
                    and "delta" not in (public / "instructions.md").read_text()
                    and "minimum effect" not in (public / "instructions.md").read_text()
                ),
            }
        )
    return {
        "task_n": len(rows),
        "expected_task_n": 2 * len(frozen),
        "tasks": rows,
        "passed": len(rows) == 2 * len(frozen)
        and all(
            r["source_identity_matches"]
            and r["dataset_hash_matches"]
            and r["package_hashes_match"]
            and r["public_effect_cutoffs_absent"]
            for r in rows
        ),
    }


def audit_run(path):
    r = json.loads((path / "report.json").read_text())
    transcript = [json.loads(line) for line in (path / "transcript.jsonl").read_text().splitlines()]
    stages = [e for e in transcript if e["kind"] == "stage"]
    opportunities = r["validation"]["opportunities"]
    events = r["responsiveness"]["events"]
    checks = {
        "all_stages_successful": len(stages) == r["expected_stages"] == 4 * r["iterations"],
        "no_exhausted_contracts": not r["protocol_errors"],
        "unique_validation_events": len({e["comparison_key"] for e in events}) == len(events),
        "validation_bound": r["validation"]["distinct_comparisons"]
        <= r["validation"]["max_comparisons"],
        "voluntary_bound": r["validation"]["voluntary_slots_used"] <= 10,
        "fixed_alpha": all(e["result"]["alpha"] == r["validation"]["alpha"] for e in events),
        "scheduled_opportunities": len(opportunities) == len(r["policy"]["release_iterations"]),
        "response_window": all(e["due_iteration"] == e["iteration"] + 2 for e in events),
        "correct_delayed_iteration": all(
            e["delayed"] is None
            or (
                e["delayed"]["iteration"] == e["due_iteration"]
                and e["delayed"]["stage"] == "synthesize"
            )
            for e in events
        ),
        "correct_immediate_iteration": all(
            e["immediate"] is None
            or (
                e["immediate"]["iteration"] == e["iteration"]
                and e["immediate"]["stage"] == "synthesize"
            )
            for e in events
        ),
        "selection_precedes_appraisal": all(
            any(
                e["kind"] == "private_selection"
                and e["slot"] == o["slot"]
                and e["sequence"] < s["sequence"]
                for e in transcript
                for s in stages
                if s["record"]["iteration"] == o["selected_iteration"]
                and s["record"]["stage"] == "appraise"
            )
            for o in opportunities
        ),
        "selected_from_logged_pool": all(
            o["comparison_key"] is None
            or o["comparison_key"] in {p["comparison_key"] for p in o["eligible_pool"]}
            for o in opportunities
        ),
        "scheduled_release_timing": all(
            o["comparison_key"] is None
            or any(
                e["comparison_key"] == o["comparison_key"]
                and (
                    e["source"] == "automatic"
                    and e["iteration"] == o["release_iteration"]
                    or e["source"] == "voluntary"
                    and o["selected_iteration"] <= e["iteration"] <= o["release_iteration"]
                )
                for e in events
            )
            for o in opportunities
        ),
    }
    d = r["scores"]["discovery"]["exact"]
    expected_d = (
        0
        if r["protocol_errors"] or d["Q"] is None or d["R"] + d["Q"] == 0
        else (200 * d["R"] * d["Q"] / (d["R"] + d["Q"]))
    )
    checks["D_arithmetic"] = abs(expected_d - r["scores"]["D"]) < 1e-9
    curves = r["scores"]["coverage"]["exact"]["curves"]
    expected_e = 100 * sum(sum(c) for c in curves.values()) / (3 * r["iterations"])
    checks["E_arithmetic"] = abs(expected_e - r["scores"]["E"]) < 1e-9
    classes = r["responsiveness"]["components"]
    values = [c["accuracy"] for c in classes.values()]
    expected_b = None if None in values else 100 * sum(values) / 3
    checks["B_arithmetic"] = (expected_b is None and r["scores"]["B"] is None) or (
        expected_b is not None and abs(expected_b - r["scores"]["B"]) < 1e-9
    )
    checks["missing_decisions_are_errors"] = all(
        not e["missing_due"] or e["score"] == 0 for e in events
    )
    checks["all_required_responses_present"] = not r["responsiveness"]["missing_due_n"]
    if r["versions"]["workflow"] == "appraisal-3.2.0":
        prompts = [e["prompt"] for e in transcript if e["kind"] == "model"]
        contexts = [json.loads(p[p.index('{"schema":') :]) for p in prompts]
        checks["agent_task_has_no_effect_cutoffs"] = bool(contexts) and all(
            "delta" not in json.dumps(c["task"])
            and "minimum effect" not in c["task"]["instructions"]
            for c in contexts
        )
        checks["agent_evidence_has_no_effect_cutoffs"] = all(
            "delta" not in result
            for c in contexts
            for claim in c["claims"]
            for result in claim["evidence"]
        )
        checks["compact_forms_and_context"] = all(
            "history" not in c
            and "accepted_ids" not in c["schema"]["properties"]
            and "executed_ids" not in c["schema"]["properties"]
            for c in contexts
        )
        checks["private_scoring_retains_effect_cutoffs"] = all(
            e["result"]["delta"] > 0 for e in events
        )
        checks["translations_audited"] = len(stages) == sum(
            e["kind"] == "interface_translation" for e in transcript
        )
    if r["versions"]["workflow"] == "appraisal-3.1.0":
        prompts = [e["prompt"] for e in transcript if e["kind"] == "model"]
        contexts = [json.loads(p[p.index('{"schema":') :]) for p in prompts]
        checks["agent_task_has_no_effect_cutoffs"] = bool(contexts) and all(
            "delta" not in json.dumps(c["history"][0])
            and "minimum effect" not in c["history"][0]["instructions"]
            for c in contexts
        )
        checks["agent_evidence_has_no_effect_cutoffs"] = all(
            "delta" not in result
            for c in contexts
            for event in c["history"]
            for key in ("results", "reused_evidence")
            for result in event.get(key, [])
        )
        checks["private_scoring_retains_effect_cutoffs"] = all(
            e["result"]["delta"] > 0 for e in events
        )
    return {
        "run_id": r["run_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "scores": {k: r["scores"][k] for k in ("D", "E", "B")},
        "response_components": classes,
        "response_exclusions": r["responsiveness"]["exclusions"],
        "voluntary_events": sum(e["source"] == "voluntary" for e in events),
        "automatic_events": sum(e["source"] == "automatic" for e in events),
        "recovered_stages": r["recovered_stages"],
        "protocol_errors": r["protocol_errors"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("benchmarks/expected_surprising/v2/release_manifest.json"),
    )
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"packages": audit_packages(args.data, args.release)}
    if args.smoke:
        manifest = json.loads((args.smoke / "manifest.json").read_text())
        report["runs"] = [audit_run(args.smoke / "runs" / t["run_id"]) for t in manifest["tasks"]]
        report["source_hashes_match"] = all(
            digest(Path(p)) == value for p, value in manifest["source_sha256"].items()
        )
        report["runner_hash_matches"] = (
            digest(Path(__file__).with_name("smoke_vllm.py")) == manifest["runner_sha256"]
        )
        if "reporter_sha256" in manifest:
            current_reporter = Path(__file__).with_name("report_smoke.py")
            recorded_reporter = args.smoke / "report_smoke_at_run.py"
            if not recorded_reporter.exists():
                recorded_reporter = current_reporter
            # Reports can be reworded after a run; verify the preserved original renderer.
            report["reporter_hash_matches"] = (
                digest(recorded_reporter) == manifest["reporter_sha256"]
            )
            report["reporter_source_checked"] = str(recorded_reporter)
            report["current_reporter_hash_matches_run"] = (
                digest(current_reporter) == manifest["reporter_sha256"]
            )
    report["passed"] = (
        report["packages"]["passed"]
        and all(r["passed"] for r in report.get("runs", []))
        and report.get("source_hashes_match", True)
        and report.get("runner_hash_matches", True)
        and report.get("reporter_hash_matches", True)
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "task_n": report["packages"]["task_n"],
                "run_n": len(report.get("runs", [])),
                "output": str(args.out),
            }
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
