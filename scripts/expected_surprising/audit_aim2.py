"""Verify a completed workflow smoke matrix and retain a compact evidence audit."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from onc_co_scientist.expected_surprising.packaging import sha256


def audit(root: Path):
    summary = json.loads((root / "summary.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    results = {r["run_id"]: r for r in summary["runs"]}
    if len(results) != len(plan) or set(results) != {p["run_id"] for p in plan}:
        raise ValueError("Every planned run needs one completed or failed result")
    rows = []
    for item in plan:
        folder = root / "runs" / item["run_id"]
        result = results[item["run_id"]]
        report_path = folder / "report.json"
        if not report_path.exists():
            rows.append(
                {
                    "run_id": item["run_id"],
                    "status": result["status"],
                    "scientific_trace_available": False,
                }
            )
            continue
        if sha256(report_path) != result["scientific_report_sha256"]:
            raise ValueError(f"Scientific report checksum mismatch: {report_path}")
        report = json.loads(report_path.read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        if report["dataset_sha256"] != provenance["public"]["dataset.parquet"]:
            raise ValueError("Report and provenance dataset hashes differ")
        if report["expected_stages"] != 4 * report["iterations"]:
            raise ValueError("Peer calls must not add scientific stages")
        if report["iterations"] == 6 and report["policy"]["release_iterations"] != [2, 4]:
            raise ValueError("Six-iteration smoke must retain both scheduled release windows")
        coordination = report["coordination"]
        peers = defaultdict(dict)
        for relative, checksum in coordination["participant_artifact_sha256"].items():
            path = folder / relative
            if sha256(path) != checksum:
                raise ValueError(f"Participant artifact checksum mismatch: {path}")
            request = json.loads(path.read_text())["request"]
            if request["kind"] == "peer" and "-r1-a1" in request["slot"]:
                peers[(request["iteration"], request["stage"])][request["session_id"]] = request[
                    "messages"
                ][-1]["content"]
            if request["kind"] == "peer" and request["authoritative_candidate"]:
                raise ValueError("Peer draft marked authoritative")
            if request["kind"] in {"linear", "chair"} and not request["authoritative_candidate"]:
                raise ValueError("Selected decision lacks scientific authority")
        for drafts in peers.values():
            if len(set(drafts.values())) != 1:
                raise ValueError("First-round peers did not receive the same scientific context")
        exact = report["scores"]["discovery"]["exact"]
        rows.append(
            {
                "run_id": item["run_id"],
                "profile": report["profile"],
                "version": report["version"],
                "workflow": report["workflow_id"],
                "status": result["status"],
                "successful_stages": report["successful_stages"],
                "expected_stages": report["expected_stages"],
                "primary_recovery": report["confirmation"]["primary_recovery"],
                "R": exact["R"],
                "P": exact["Q"],
                "F1*": report["scores"]["D"],
                "E": report["scores"]["E"],
                "B": report["scores"]["B"],
                "unavailable_evidence_classes": [
                    cls
                    for cls, component in report["responsiveness"]["components"].items()
                    if component["accuracy"] is None
                ],
                "validation_events_by_source": {
                    source: sum(e["source"] == source for e in report["responsiveness"]["events"])
                    for source in ("automatic", "voluntary")
                },
                "response_exclusions": report["responsiveness"]["exclusions"],
                "release_iterations": report["policy"]["release_iterations"],
                "protocol_errors": report["protocol_errors"],
                "recovered_stages": report["recovered_stages"],
                "draft_errors": coordination["draft_errors"],
                "agent_calls": coordination["agent_calls"],
                "usage": coordination["usage"],
                "missing_usage_calls": coordination["usage_missing_calls"],
                "history_trims": len(coordination["memory_trims"]),
                "dataset_sha256": report["dataset_sha256"],
                "report_sha256": result["scientific_report_sha256"],
                "scientific_trace_available": True,
            }
        )
    return {
        "experiment_id": summary["experiment_id"],
        "root": str(root.resolve()),
        "run_n": len(rows),
        "all_completed": all(r["status"] == "completed" for r in rows),
        "all_scientific_stages_completed": all(
            r.get("successful_stages") == r.get("expected_stages")
            and r["scientific_trace_available"]
            for r in rows
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
