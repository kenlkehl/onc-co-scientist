"""Run one explicitly released model in a frozen named/masked federation grid."""

import argparse
import fcntl
import hashlib
import json
import os
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans


def now():
    return datetime.now(UTC).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not 1 <= args.workers <= 30:
        raise ValueError("Use 1–30 workers")
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    for name, expected in frozen["hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {name}")
    policy = json.loads((root / "release_policy.json").read_text())
    if args.model not in policy["released_models"]:
        raise ValueError(f"{args.model} is held pending the user's launch instruction")
    import onc_co_scientist.expected_surprising.federation as module

    if not Path(module.__file__).is_relative_to(root / "source"):
        raise ValueError("Use the frozen source/src on PYTHONPATH")
    control = root / "control" / args.model
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = control / "execution.json"
    if path.exists() and not args.resume:
        raise FileExistsError("Already launched; use --resume for verified replay")
    specs = {c: load_experiment_spec(root / c / "config.yaml") for c in ("named", "masked")}
    plans = {c: {p.run_id: p for p in build_run_plans(s)} for c, s in specs.items()}
    selection = json.loads((root / "selections" / f"{args.model}.json").read_text())
    if not selection or any(
        plans[r["condition"]][r["run_id"]].model.id != args.model for r in selection
    ):
        raise ValueError("Selection does not match requested model")
    fingerprints = {c: s.fingerprint() for c, s in specs.items()}
    if path.exists():
        atomic_write_json(
            control / f"execution-before-{os.getpid()}.json", json.loads(path.read_text())
        )
    state = dict(
        started_at=now(),
        pid=os.getpid(),
        model=args.model,
        workers=args.workers,
        selected_runs=len(selection),
        resume=args.resume,
        source=module.__file__,
    )
    pending, active, finished = list(selection), {}, []

    def status():
        counts = Counter()
        for item in pending:
            counts[item["condition"], "queued"] += 1
        for item in active.values():
            counts[item["condition"], "running"] += 1
        for item in finished:
            counts[item["condition"], item["status"]] += 1
        state.update(
            updated_at=now(),
            active=list(active.values()),
            queued_runs=len(pending),
            finished=finished,
            terminal_runs=len(finished),
        )
        atomic_write_json(path, state)
        rows = [
            f"# {args.model}: federated grid",
            "",
            f"Updated {now()}",
            "",
            "2 and 4 random sites; 25 iterations; 10 repeats; three workflows.",
            "",
            "| Condition | State | Runs |",
            "|---|---|---:|",
        ]
        rows += [f"| {c} | {s} | {n} |" for (c, s), n in sorted(counts.items())]
        atomic_write_text(control / "STATUS.md", "\n".join(rows) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            while pending and len(active) < args.workers:
                item = pending.pop(0)
                c, rid = item["condition"], item["run_id"]
                future = pool.submit(
                    run_cell, specs[c], plans[c][rid], root / c, fingerprints[c], resume=args.resume
                )
                active[future] = item
            status()
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                try:
                    result = future.result()
                    finished.append(
                        {
                            **item,
                            "status": result["status"],
                            "agent_calls": result.get("agent_calls", 0),
                        }
                    )
                except Exception:
                    finished.append(
                        {**item, "status": "worker_error", "traceback": traceback.format_exc()}
                    )
    state.update(
        completed_at=now(),
        status=(
            "completed"
            if all(r["status"] == "completed" for r in finished)
            else "completed_with_failures"
        ),
    )
    status()


if __name__ == "__main__":
    main()
