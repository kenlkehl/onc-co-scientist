#!/usr/bin/env python3
"""Print compact host-local status for one or more experiment supervisors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("status is not a JSON object")
    return payload


def _render(path: Path) -> None:
    try:
        status = _load(path)
    except Exception as exc:
        print(f"{path.stem}: unavailable ({type(exc).__name__}: {exc})")
        return
    snapshot = status.get("snapshot")
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    runs = snapshot.get("runs")
    runs = runs if isinstance(runs, list) else []
    valid_runs = [run for run in runs if isinstance(run, dict)]
    counts = {
        state: sum(run.get("status") == state for run in valid_runs)
        for state in ("completed", "running", "interrupted", "failed")
    }
    active = [
        str(run.get("run_id"))
        for run in valid_runs
        if run.get("active_adapter_descendant")
    ]
    print(
        f"{path.stem}: {status.get('status')}  generation={status.get('generation')} "
        f"restarts={status.get('restart_count')}/{status.get('max_restarts')}  "
        f"controller_pid={status.get('controller_pid')}  updated={status.get('updated_at')}"
    )
    print(
        "  states="
        + "/".join(f"{key}:{value}" for key, value in counts.items())
        + f"  active_runs={len(active)}  descendants={status.get('descendant_processes')} "
        + f"probe_timeouts={status.get('consecutive_probe_timeouts')} "
        + f"stale={len(snapshot.get('stale_runs', []))}"
    )
    for run_id in active:
        print(f"    {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.status):
        if index:
            print()
        _render(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
