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


def review_selection(selection, grid, blocks):
    """Require complete paired conditions before admitting a bounded review cohort."""
    if blocks < 1 or 4 * blocks > len(selection):
        raise ValueError("Review blocks must select 1 or more complete four-run blocks")
    lookup = {(r["condition"], r["run_id"]): r for r in grid}
    selected = selection[: 4 * blocks]
    if len({(r["condition"], r["run_id"]) for r in selected}) != len(selected):
        raise ValueError("Duplicate run in review selection")
    for offset in range(0, len(selected), 4):
        rows = [lookup[r["condition"], r["run_id"]] for r in selected[offset : offset + 4]]
        keys = {
            (
                r["model_profile"],
                r["workflow_id"],
                r["site_count"],
                r["partition_id"],
                r["replicate"],
            )
            for r in rows
        }
        conditions = {(r["condition"], r["semantic_condition"]) for r in rows}
        if len(keys) != 1 or conditions != {
            (c, s) for c in ("named", "masked") for s in ("expected", "surprising")
        }:
            raise ValueError("Review selection is not a matched four-condition block")
    return selected


def validate_review_resume(previous, blocks):
    old = previous.get("review_blocks")
    if old == blocks:
        return
    if (
        old is not None
        and blocks is not None
        and blocks > old
        and previous.get("completed_at")
        and not previous.get("active")
    ):
        return  # Explicitly extend a finished cohort after review.
    raise ValueError("Resume must retain the review limit or explicitly extend a finished cohort")


def verify_predecessor(transition):
    """Never admit a replacement bundle while original calls can still be active."""
    if not transition:
        return
    source = Path(transition["source_root"])
    completed = set()
    for model in transition["models"]:
        state = json.loads((source / "control" / model / "execution.json").read_text())
        if not state.get("completed_at") or state.get("active") or state.get("queued_runs"):
            raise RuntimeError(f"Azure launch held: original {model} driver has not drained")
        completed.update((r["condition"], r["run_id"]) for r in state.get("finished", []))
    if any((r["condition"], r["run_id"]) not in completed for r in transition["active"]):
        raise RuntimeError("Azure launch held: original active identities lack terminal records")
    for row in transition["queued"]:
        reservation = source / row["condition"] / "runs" / row["run_id"]
        if (
            not reservation.is_file()
            or json.loads(reservation.read_text()).get("status") != "reserved_for_azure"
        ):
            raise RuntimeError("Azure launch held: an original admission reservation changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--review-blocks", type=int, help="Run this many matched four-run blocks, then pause"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not 1 <= args.workers <= 30:
        raise ValueError("Use 1–30 workers")
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    for name, expected in frozen["hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {name}")
    verify_predecessor(frozen.get("azure_transition"))
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
    held_outside_review = 0
    if args.review_blocks is not None:
        selected = review_selection(
            selection, json.loads((root / "grid.json").read_text()), args.review_blocks
        )
        held_outside_review = len(selection) - len(selected)
        selection = selected
    fingerprints = {c: s.fingerprint() for c, s in specs.items()}
    if path.exists():
        previous = json.loads(path.read_text())
        validate_review_resume(previous, args.review_blocks)
        atomic_write_json(control / f"execution-before-{os.getpid()}.json", previous)
    state = dict(
        started_at=now(),
        pid=os.getpid(),
        model=args.model,
        workers=args.workers,
        selected_runs=len(selection),
        resume=args.resume,
        source=module.__file__,
        review_blocks=args.review_blocks,
        held_outside_review=held_outside_review,
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
            f"Runs held outside this review cohort: {held_outside_review}",
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
            "paused_for_review"
            if held_outside_review
            else "completed"
            if all(r["status"] == "completed" for r in finished)
            else "completed_with_failures"
        ),
    )
    status()


if __name__ == "__main__":
    main()
