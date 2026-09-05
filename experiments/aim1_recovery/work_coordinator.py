"""Local dispatch bookkeeping and integrity checks; never scores recovery."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.harness.structured_runner import finalize_workspace
from onc_co_scientist.harness.transcript import Transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "dispatch"])
    parser.add_argument("jobs", nargs="*")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--agent-prefix", default="/root/v2_")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    path = root / "coordinator_state.json"
    state = json.loads(path.read_text())
    if args.command == "dispatch":
        known = {j["job_id"] for j in plan["jobs"]}
        for job in args.jobs:
            if job not in known or job in state["jobs"]:
                raise ValueError("Unknown or previously dispatched job: " + job)
            state["jobs"][job] = dict(
                agent=args.agent_prefix + job[-4:],
                status="running",
                started_utc=datetime.now(UTC).isoformat(),
            )
    for job in plan["jobs"]:
        entry = state["jobs"].get(job["job_id"])
        if not entry or entry["status"] in {"completed", "failed"}:
            continue
        ws = Path(job["workspace"])
        if (
            (ws / "transcript.json").exists()
            and (ws / "analysis_summary.txt").exists()
            and len(list((ws / "iterations").glob("*.json"))) == job["max_iterations"]
        ):
            assembled = finalize_workspace(ws, write_output=False)
            saved = Transcript.model_validate_json((ws / "transcript.json").read_text())
            assert json.dumps(saved.model_dump(), sort_keys=True, allow_nan=True) == json.dumps(
                assembled.model_dump(), sort_keys=True, allow_nan=True
            )
            assert saved.model_id == plan["protocol"]["model_id"]
            assert (
                hashlib.sha256((ws / "dataset.parquet").read_bytes()).hexdigest()
                == job["data_sha256"]
            )
            assert (ws / "analysis_summary.txt").exists()
            entry.update(
                status="completed",
                completed_utc=datetime.now(UTC).isoformat(),
                iterations=len(saved.iterations),
                structured_claims=len(saved.flat_hypotheses()),
            )
    state["updated_utc"] = datetime.now(UTC).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)
    counts = Counter(e["status"] for e in state["jobs"].values())
    active = []
    for job in plan["jobs"]:
        if state["jobs"].get(job["job_id"], {}).get("status") == "running":
            ws = Path(job["workspace"])
            active.append(
                dict(
                    job=job["job_id"],
                    submitted=len(list((ws / "iterations").glob("*.json"))),
                    cap=job["max_iterations"],
                )
            )
    print(
        json.dumps(
            dict(
                completed=counts["completed"],
                failed=counts["failed"],
                dispatched=len(state["jobs"]),
                active=active,
                next_pending=[
                    j["job_id"] for j in plan["jobs"] if j["job_id"] not in state["jobs"]
                ][:6],
            )
        )
    )


if __name__ == "__main__":
    main()
