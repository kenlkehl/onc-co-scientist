"""Apply an audited operational worker limit without changing scientific inputs.

The original manager and all scientific configuration/source stay frozen.
Existing calls use the original fingerprint and exact-replay checks on resume.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import manage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=manage.WORKFLOWS, required=True)
    parser.add_argument("--parallel", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.parallel <= 40:
        parser.error("--parallel must be between 1 and the 40 assigned runs")
    manage.verify()
    root = Path(__file__).resolve().parent
    receipt = {
        "at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "workflow": args.workflow,
        "runtime_max_parallel": args.parallel,
        "phase_run_count": 40,
        "maximum_active_runs_this_phase": min(args.parallel, 40),
        "frozen_config_max_parallel": 10,
        "resume": args.resume,
        "authorization": "User requested increased concurrency on 2026-09-23.",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scientific_configuration_unchanged": True,
    }
    directory = root / args.workflow / "operational_launches"
    directory.mkdir(exist_ok=True)
    (directory / f"{os.getpid()}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"event": "runtime_concurrency_override", **receipt}), flush=True)
    manage.PARALLEL = args.parallel
    manage.run(args.workflow, resume=args.resume)


if __name__ == "__main__":
    main()
