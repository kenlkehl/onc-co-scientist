"""Execute an explicit run selection without rewriting a parent grid's manifests.

Used for selective continuation under the original frozen implementation, or fresh
replacement cells under a separately frozen implementation. Never mixes call journals.
"""

import argparse
import hashlib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.harness.durable_io import atomic_write_json, durable_append_line
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans


def now():
    return datetime.now(UTC).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root, control = args.root.resolve(), args.control.resolve()
    control.mkdir(parents=True, exist_ok=True)
    import onc_co_scientist.expected_surprising.coordination as module

    assert Path(module.__file__).is_relative_to(root / "source")
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    for name, expected in {
        **frozen["source_hashes"],
        **frozen["input_hashes"],
        "config.yaml": frozen["config_sha256"],
    }.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    ids = json.loads(args.selection.read_text())
    assert isinstance(ids, list) and len(ids) == len(set(ids))
    spec = load_experiment_spec(root / "config.yaml")
    plans = {p.run_id: p for p in build_run_plans(spec)}
    assert set(ids) <= plans.keys()
    fingerprint = spec.fingerprint()
    state = {
        "started_at": now(),
        "pid": os.getpid(),
        "root": str(root),
        "selected": ids,
        "resume": args.resume,
        "workers": args.workers,
        "source": module.__file__,
        "finished": [],
        "worker_errors": [],
    }
    path = control / "execution.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        atomic_write_json(
            control / f"execution-before-{os.getpid()}.json", json.loads(path.read_text())
        )
    atomic_write_json(path, state)
    durable_append_line(
        control / "progress.log", f"{now()} START {len(ids)} selected cells; {args.workers} workers"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_cell, spec, plans[rid], root, fingerprint, resume=args.resume): rid
            for rid in ids
        }
        for future in as_completed(futures):
            rid = futures[future]
            try:
                result = future.result()
                status = result["status"]
                state["finished"].append({"run_id": rid, "status": status})
            except Exception:
                status = "worker_error"
                state["worker_errors"].append({"run_id": rid, "traceback": traceback.format_exc()})
            atomic_write_json(path, state)
            durable_append_line(control / "progress.log", f"{now()} {status} {rid}")
    state["completed_at"] = now()
    atomic_write_json(path, state)


if __name__ == "__main__":
    main()
