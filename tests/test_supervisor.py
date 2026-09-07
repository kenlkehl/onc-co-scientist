from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from onc_co_scientist.harness.supervisor import (
    SupervisorConfig,
    _isolated_snapshot,
    _progress_signature,
    _retryable_summary,
    _terminate_process_tree,
    supervise_experiment,
)


def _write_state(
    root: Path,
    run_id: str,
    *,
    status: str,
    updated_at: datetime,
    call_index: int = 1,
) -> Path:
    run_dir = root / "runs" / run_id
    (run_dir / "calls" / f"call_{call_index:04d}").mkdir(parents=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "updated_at": updated_at.isoformat(),
                "call_index": call_index,
                "call_slot_cursor": "analysis",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_isolated_snapshot_classifies_orphan_and_hard_timeout(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    child_started = (now - timedelta(minutes=10)).timestamp()
    orphan = _write_state(
        tmp_path,
        "orphan",
        status="running",
        updated_at=now - timedelta(minutes=5),
    )
    active = _write_state(
        tmp_path,
        "active",
        status="running",
        updated_at=now - timedelta(minutes=5),
    )
    _write_state(
        tmp_path,
        "prior-controller",
        status="running",
        updated_at=now - timedelta(hours=2),
    )

    snapshot = _isolated_snapshot(
        tmp_path,
        active_commands=(f"adapter --request-file {active}/calls/call_0001/request.json",),
        child_started_epoch=child_started,
        orphan_grace_seconds=60.0,
        hard_stale_seconds=120.0,
        timeout_seconds=5.0,
    )

    stale = {item["run_id"]: item["reason"] for item in snapshot["stale_runs"]}
    assert stale == {
        "active": "active_call_exceeded_hard_ceiling",
        "orphan": "running_cell_has_no_adapter_descendant",
    }
    records = {item["run_id"]: item for item in snapshot["runs"]}
    assert records[orphan.name]["active_adapter_descendant"] is False
    assert records[active.name]["active_adapter_descendant"] is True
    assert records["prior-controller"]["touched_by_current_controller"] is False


def test_progress_signature_ignores_observation_age() -> None:
    first = {
        "observed_at": "one",
        "runs": [
            {
                "run_id": "run",
                "status": "running",
                "updated_at": "same",
                "call_index": 2,
                "latest_call_index": 3,
                "age_seconds": 10,
            }
        ],
    }
    second = {
        **first,
        "observed_at": "two",
        "runs": [{**first["runs"][0], "age_seconds": 100}],
    }

    assert _progress_signature(first) == _progress_signature(second)


def test_retryable_summary_excludes_exhausted_budget() -> None:
    budget = {
        "summary": {
            "n_failed": 1,
            "runs": [{"status": "failed", "error_type": "BudgetExceeded"}],
        }
    }
    storage = {
        "summary": {
            "n_failed": 1,
            "runs": [{"status": "failed", "error_type": "StorageUnavailable"}],
        }
    }

    assert _retryable_summary(budget, 0) is False
    assert _retryable_summary(storage, 0) is True


def test_terminate_process_tree_reaches_detached_descendant_group(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
                "start_new_session=True); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(60)"
            ),
        ],
        start_new_session=True,
    )
    deadline = time.monotonic() + 5.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    _terminate_process_tree(parent, grace_seconds=1.0)

    assert parent.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_supervisor_runs_stub_experiment_to_completion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "evidence.txt").write_text("public evidence\n", encoding="utf-8")
    output_root = tmp_path / "output"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1",
                "experiment_id": "supervisor-stub",
                "output_root": str(output_root),
                "tasks": [
                    {
                        "id": "task",
                        "prompt": "Analyze the public evidence.",
                        "public_workspace": str(workspace),
                    }
                ],
                "models": [
                    {"id": "stub", "model_id": "stub", "adapter": "stub"}
                ],
                "workflows": [{"id": "sequential", "mode": "sequential"}],
                "replicates": 1,
                "max_parallel": 1,
            }
        ),
        encoding="utf-8",
    )
    status_path = tmp_path / "supervisor-status.json"

    result = supervise_experiment(
        SupervisorConfig(
            config_path=config_path,
            poll_seconds=0.1,
            probe_timeout_seconds=5.0,
            orphan_grace_seconds=5.0,
            hard_stale_grace_seconds=1.0,
            restart_backoff_seconds=0.0,
            max_restarts=0,
            status_path=status_path,
        )
    )

    assert result["status"] == "completed"
    assert result["summary"]["n_completed"] == 1
    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "completed"
