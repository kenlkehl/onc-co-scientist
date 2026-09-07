"""Process supervisor for resumable experiments on failure-prone artifact mounts.

The harness deliberately runs cells in threads so model concurrency is cheap.  A
thread cannot, however, be cancelled when a FUSE/SSHFS system call blocks in the
kernel.  This supervisor therefore owns the harness as a disposable process tree:
it detects an orphaned cell or a call beyond its hard ceiling, terminates every
descendant process group, and resumes from the last atomic checkpoint.

All potentially blocking inspection of the experiment output is isolated in a
short-lived probe process.  The supervisor's own status file lives on host-local
storage, so it remains observable even when the artifact mount is unhealthy.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .durable_io import atomic_write_json, durable_mkdir, local_spool_directory
from .experiment import load_experiment_spec


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _timestamp(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return payload


def _snapshot_output(
    output_root: Path,
    *,
    active_commands: tuple[str, ...],
    child_started_epoch: float,
    orphan_grace_seconds: float,
    hard_stale_seconds: float,
) -> dict[str, Any]:
    """Inspect remote state. This function must run only in an isolated process."""

    now = time.time()
    runs: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    runs_root = output_root / "runs"
    for state_path in sorted(runs_root.glob("*/run_state.json")):
        run_dir = state_path.parent
        try:
            state = _read_json(state_path)
            updated_epoch = _timestamp(state.get("updated_at"))
            if updated_epoch is None:
                updated_epoch = state_path.stat().st_mtime
            age_seconds = max(0.0, now - updated_epoch)
            calls_root = run_dir / "calls"
            latest_call_index = 0
            for call_path in calls_root.glob("call_[0-9]*"):
                try:
                    latest_call_index = max(
                        latest_call_index,
                        int(call_path.name.removeprefix("call_")),
                    )
                except ValueError:
                    continue
            run_prefix = str(run_dir) + os.sep
            active = any(run_prefix in command for command in active_commands)
            touched_by_child = updated_epoch >= child_started_epoch - 2.0
            record: dict[str, Any] = {
                "run_id": run_dir.name,
                "status": state.get("status"),
                "updated_at": state.get("updated_at"),
                "age_seconds": round(age_seconds, 3),
                "call_index": state.get("call_index"),
                "latest_call_index": latest_call_index,
                "call_slot": state.get("call_slot_cursor"),
                "active_adapter_descendant": active,
                "touched_by_current_controller": touched_by_child,
            }
            runs.append(record)
            if state.get("status") != "running" or not touched_by_child:
                continue
            reason: str | None = None
            if active and age_seconds > hard_stale_seconds:
                reason = "active_call_exceeded_hard_ceiling"
            elif not active and age_seconds > orphan_grace_seconds:
                reason = "running_cell_has_no_adapter_descendant"
            if reason is not None:
                stale.append({**record, "reason": reason})
        except Exception as exc:
            errors.append(
                {
                    "path": str(state_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    summary: dict[str, Any] | None = None
    summary_path = output_root / "summary.json"
    try:
        if summary_path.is_file():
            summary = _read_json(summary_path)
    except Exception as exc:
        errors.append(
            {
                "path": str(summary_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return {
        "observed_at": _utc_now(),
        "runs": runs,
        "stale_runs": stale,
        "inspection_errors": errors,
        "summary": summary,
    }


def _snapshot_worker(connection: Any, kwargs: dict[str, Any]) -> None:
    try:
        connection.send({"ok": True, "snapshot": _snapshot_output(**kwargs)})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        connection.close()


def _isolated_snapshot(
    output_root: Path,
    *,
    active_commands: tuple[str, ...],
    child_started_epoch: float,
    orphan_grace_seconds: float,
    hard_stale_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    # ``spawn`` avoids inheriting locks from telemetry/logging threads in the
    # long-lived parent, which is especially important when the probe exists to
    # diagnose a blocked filesystem.
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_snapshot_worker,
        args=(
            sender,
            {
                "output_root": output_root,
                "active_commands": active_commands,
                "child_started_epoch": child_started_epoch,
                "orphan_grace_seconds": orphan_grace_seconds,
                "hard_stale_seconds": hard_stale_seconds,
            },
        ),
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        if receiver.poll(timeout_seconds):
            message = receiver.recv()
            process.join(timeout=2.0)
            if message.get("ok"):
                return dict(message["snapshot"])
            return {
                "observed_at": _utc_now(),
                "probe_error": {
                    "error_type": message.get("error_type"),
                    "error": message.get("error"),
                },
            }
        process.terminate()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        return {
            "observed_at": _utc_now(),
            "probe_timeout": True,
            "timeout_seconds": timeout_seconds,
        }
    finally:
        receiver.close()


def _process_table() -> dict[int, tuple[int, int, str]]:
    table: dict[int, tuple[int, int, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            status = (entry / "status").read_text(encoding="utf-8")
            parent = next(
                int(line.split()[1])
                for line in status.splitlines()
                if line.startswith("PPid:")
            )
            group = os.getpgid(pid)
            raw_command = (entry / "cmdline").read_bytes()
            command = raw_command.replace(b"\0", b" ").decode("utf-8", errors="replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
        table[pid] = (parent, group, command)
    return table


def _descendants(
    root_pid: int,
    table: dict[int, tuple[int, int, str]] | None = None,
) -> dict[int, tuple[int, int, str]]:
    selected_table = table or _process_table()
    found: set[int] = {root_pid}
    while True:
        additions = {
            pid for pid, (parent, _, _) in selected_table.items() if parent in found
        }
        additions -= found
        if not additions:
            break
        found.update(additions)
    return {
        pid: selected_table[pid]
        for pid in found
        if pid in selected_table and pid != root_pid
    }


def _active_commands(root_pid: int) -> tuple[str, ...]:
    return tuple(
        command
        for _, _, command in _descendants(root_pid).values()
        if command
    )


def _terminate_process_tree(
    process: subprocess.Popen[Any], *, grace_seconds: float = 30.0
) -> None:
    if process.poll() is not None:
        return
    table = _process_table()
    descendants = _descendants(process.pid, table)
    own_group = os.getpgrp()
    groups = {
        group
        for _, group, _ in descendants.values()
        if group > 0 and group != own_group
    }
    try:
        root_group = os.getpgid(process.pid)
    except ProcessLookupError:
        root_group = -1
    if root_group > 0 and root_group != own_group:
        groups.add(root_group)
    tracked_pids = {process.pid, *descendants}

    def still_running(pid: int) -> bool:
        if pid == process.pid:
            return process.poll() is None
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError):
            return False
        state = next(
            (line.split()[1] for line in status.splitlines() if line.startswith("State:")),
            "X",
        )
        return state not in {"X", "Z"}

    for group in groups:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(still_running(pid) for pid in tracked_pids):
        time.sleep(0.2)
    if not any(still_running(pid) for pid in tracked_pids):
        return
    # Refresh descendants because a child may have changed process groups while
    # handling SIGTERM. Kill groups and individual PIDs, then the root process.
    refreshed = _descendants(process.pid) if process.poll() is None else {}
    kill_groups = {
        group
        for _, group, _ in (*descendants.values(), *refreshed.values())
        if group > 0 and group != own_group
    }
    if root_group > 0 and root_group != own_group:
        kill_groups.add(root_group)
    for group in kill_groups:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(group, signal.SIGKILL)
    for pid in {*tracked_pids, *refreshed}:
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
    with suppress(ProcessLookupError):
        process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5.0)


@dataclass(frozen=True)
class SupervisorConfig:
    config_path: Path
    output_root: Path | None = None
    max_parallel: int | None = None
    compatible_resume_implementation_sha256s: tuple[str, ...] = ()
    poll_seconds: float = 30.0
    probe_timeout_seconds: float = 20.0
    max_consecutive_probe_timeouts: int = 3
    orphan_grace_seconds: float = 600.0
    hard_stale_grace_seconds: float = 900.0
    restart_backoff_seconds: float = 60.0
    max_restarts: int = 12
    status_path: Path | None = None


def _child_command(config: SupervisorConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "onc_co_scientist.cli",
        "harness",
        "run-experiment",
        "--config",
        str(config.config_path),
        "--resume",
    ]
    if config.output_root is not None:
        command.extend(("--out", str(config.output_root)))
    if config.max_parallel is not None:
        command.extend(("--max-parallel", str(config.max_parallel)))
    for implementation in config.compatible_resume_implementation_sha256s:
        command.extend(
            ("--resume-compatible-implementation-sha256", implementation)
        )
    return command


def _default_status_path(experiment_id: str, output_root: Path) -> Path:
    digest = hashlib.sha256(str(output_root).encode("utf-8")).hexdigest()[:12]
    return (
        local_spool_directory()
        / "supervisors"
        / f"{experiment_id}-{digest}.json"
    )


def _retryable_summary(snapshot: dict[str, Any], returncode: int) -> bool:
    if returncode != 0:
        return True
    summary = snapshot.get("summary")
    if not isinstance(summary, dict):
        return True
    if int(summary.get("n_failed", 0) or 0) <= 0:
        return False
    failures = [
        item
        for item in summary.get("runs", [])
        if isinstance(item, dict) and item.get("status") == "failed"
    ]
    if not failures:
        return True
    non_retryable_types = {"BudgetExceeded"}
    non_retryable_fragments = (
        "spec_fingerprint changed",
        "substrate_hashes changed",
        "implementation_sha256 changed",
        "usage ledger would double-count",
        "artifact and completed-call-slot counts disagree",
    )
    return any(
        failure.get("error_type") not in non_retryable_types
        and not any(
            fragment in str(failure.get("error", ""))
            for fragment in non_retryable_fragments
        )
        for failure in failures
    )


def _progress_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    runs = snapshot.get("runs")
    run_signature: tuple[Any, ...] = ()
    if isinstance(runs, list):
        run_signature = tuple(
            sorted(
                (
                    item.get("run_id"),
                    item.get("status"),
                    item.get("updated_at"),
                    item.get("call_index"),
                    item.get("latest_call_index"),
                )
                for item in runs
                if isinstance(item, dict)
            )
        )
    summary = snapshot.get("summary")
    summary_signature: tuple[Any, ...] = ()
    if isinstance(summary, dict):
        summary_signature = (
            summary.get("status"),
            summary.get("n_completed"),
            summary.get("n_failed"),
            summary.get("realized_agent_calls"),
            summary.get("ended_at"),
        )
    return run_signature, summary_signature


def supervise_experiment(config: SupervisorConfig) -> dict[str, Any]:
    """Run and, when necessary, safely resume one experiment controller."""

    if config.poll_seconds <= 0 or config.probe_timeout_seconds <= 0:
        raise ValueError("Supervisor poll and probe timeouts must be positive.")
    if config.max_consecutive_probe_timeouts < 1:
        raise ValueError("max_consecutive_probe_timeouts must be positive.")
    if config.orphan_grace_seconds <= 0 or config.hard_stale_grace_seconds < 0:
        raise ValueError("Supervisor stale-call grace values are invalid.")
    if config.max_restarts < 0:
        raise ValueError("max_restarts may not be negative.")
    for implementation in config.compatible_resume_implementation_sha256s:
        if len(implementation) != 64 or any(
            character not in "0123456789abcdef" for character in implementation
        ):
            raise ValueError(f"Invalid compatible implementation SHA-256: {implementation}")

    spec = load_experiment_spec(config.config_path)
    output_root = (config.output_root or spec.output_root).resolve()
    status_path = config.status_path or _default_status_path(
        spec.experiment_id, output_root
    )
    durable_mkdir(status_path.parent)
    hard_stale_seconds = (
        float(spec.budget.max_runtime_seconds_per_call)
        + config.hard_stale_grace_seconds
    )
    child_command = _child_command(config)
    stop_requested = threading.Event()
    active_child: subprocess.Popen[Any] | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    if hasattr(signal, "SIGHUP"):
        handled_signals = (*handled_signals, signal.SIGHUP)
    previous_handlers = {
        selected: signal.signal(selected, request_stop)
        for selected in handled_signals
    }
    supervisor_started_at = _utc_now()
    restart_count = 0
    history: list[dict[str, Any]] = []

    def write_status(status: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "status": status,
            "experiment_id": spec.experiment_id,
            "config_path": str(config.config_path.resolve()),
            "output_root": str(output_root),
            "status_path": str(status_path),
            "supervisor_pid": os.getpid(),
            "controller_pid": active_child.pid if active_child is not None else None,
            "controller_command": child_command,
            "restart_count": restart_count,
            "max_restarts": config.max_restarts,
            "started_at": supervisor_started_at,
            "updated_at": _utc_now(),
            "history": history[-50:],
            **extra,
        }
        atomic_write_json(status_path, payload, sort_keys=True)
        return payload

    try:
        while True:
            generation = restart_count + 1
            print(
                f"[{_utc_now()}] starting controller generation {generation}: "
                + " ".join(child_command),
                flush=True,
            )
            active_child = subprocess.Popen(child_command, start_new_session=True)
            child_started_epoch = time.time()
            consecutive_probe_timeouts = 0
            write_status(
                "running",
                generation=generation,
                controller_started_at=datetime.fromtimestamp(
                    child_started_epoch, UTC
                ).isoformat(),
            )
            restart_reason: str | None = None
            last_snapshot: dict[str, Any] = {}
            last_progress_signature: tuple[Any, ...] | None = None
            last_progress_epoch = child_started_epoch
            last_child_active_epoch = child_started_epoch
            last_run_active_epoch: dict[str, float] = {}

            while active_child.poll() is None and not stop_requested.is_set():
                commands = _active_commands(active_child.pid)
                last_snapshot = _isolated_snapshot(
                    output_root,
                    active_commands=commands,
                    child_started_epoch=child_started_epoch,
                    orphan_grace_seconds=config.orphan_grace_seconds,
                    hard_stale_seconds=hard_stale_seconds,
                    timeout_seconds=config.probe_timeout_seconds,
                )
                if last_snapshot.get("probe_timeout"):
                    consecutive_probe_timeouts += 1
                else:
                    consecutive_probe_timeouts = 0
                    progress_signature = _progress_signature(last_snapshot)
                    if progress_signature != last_progress_signature:
                        last_progress_signature = progress_signature
                        last_progress_epoch = time.time()
                if commands:
                    last_child_active_epoch = time.time()
                observed_epoch = time.time()
                observed_runs = last_snapshot.get("runs", [])
                if isinstance(observed_runs, list):
                    for run in observed_runs:
                        if (
                            isinstance(run, dict)
                            and run.get("active_adapter_descendant")
                        ):
                            last_run_active_epoch[str(run.get("run_id"))] = observed_epoch
                raw_stale_runs = last_snapshot.get("stale_runs", [])
                stale_runs: list[dict[str, Any]] = []
                if isinstance(raw_stale_runs, list):
                    for stale in raw_stale_runs:
                        if not isinstance(stale, dict):
                            continue
                        if stale.get("reason") != "running_cell_has_no_adapter_descendant":
                            stale_runs.append(stale)
                            continue
                        updated_epoch = _timestamp(stale.get("updated_at")) or 0.0
                        last_active_epoch = last_run_active_epoch.get(
                            str(stale.get("run_id")), 0.0
                        )
                        if (
                            observed_epoch - max(updated_epoch, last_active_epoch)
                            > config.orphan_grace_seconds
                        ):
                            stale_runs.append(stale)
                last_snapshot["stale_runs"] = stale_runs
                no_child_progress_age = max(
                    0.0,
                    time.time() - max(last_progress_epoch, last_child_active_epoch),
                )
                write_status(
                    "running",
                    generation=generation,
                    controller_started_at=datetime.fromtimestamp(
                        child_started_epoch, UTC
                    ).isoformat(),
                    descendant_processes=len(commands),
                    consecutive_probe_timeouts=consecutive_probe_timeouts,
                    no_child_progress_age_seconds=round(no_child_progress_age, 3),
                    snapshot=last_snapshot,
                )
                if isinstance(stale_runs, list) and stale_runs:
                    restart_reason = "stale_or_orphaned_run"
                    break
                if (
                    consecutive_probe_timeouts
                    >= config.max_consecutive_probe_timeouts
                ):
                    restart_reason = "artifact_mount_probe_timeout_limit"
                    break
                if not commands and no_child_progress_age > config.orphan_grace_seconds:
                    restart_reason = "controller_has_no_child_or_checkpoint_progress"
                    break
                stop_requested.wait(config.poll_seconds)

            if stop_requested.is_set():
                _terminate_process_tree(active_child)
                return write_status(
                    "stopped",
                    generation=generation,
                    reason="supervisor_signal",
                    snapshot=last_snapshot,
                )

            if restart_reason is not None:
                print(
                    f"[{_utc_now()}] restarting controller: {restart_reason}",
                    flush=True,
                )
                _terminate_process_tree(active_child)
                returncode = active_child.poll()
                if returncode is None:
                    # A task blocked in uninterruptible kernel I/O cannot be
                    # replaced safely until the kernel releases it. Keep one
                    # controller only, and remain observable from local status.
                    while active_child.poll() is None and not stop_requested.is_set():
                        write_status(
                            "waiting_for_uninterruptible_controller_exit",
                            generation=generation,
                            reason=restart_reason,
                            snapshot=last_snapshot,
                        )
                        stop_requested.wait(config.poll_seconds)
                    if stop_requested.is_set():
                        return write_status(
                            "stopped",
                            generation=generation,
                            reason="supervisor_signal",
                            snapshot=last_snapshot,
                        )
                    returncode = active_child.poll()
            else:
                returncode = active_child.wait()
                # Refresh the terminal summary without allowing a failed mount
                # to wedge the parent supervisor.
                last_snapshot = _isolated_snapshot(
                    output_root,
                    active_commands=(),
                    child_started_epoch=child_started_epoch,
                    orphan_grace_seconds=config.orphan_grace_seconds,
                    hard_stale_seconds=hard_stale_seconds,
                    timeout_seconds=config.probe_timeout_seconds,
                )
                summary = last_snapshot.get("summary")
                if (
                    returncode == 0
                    and isinstance(summary, dict)
                    and int(summary.get("n_failed", 0) or 0) == 0
                    and summary.get("status") == "completed"
                ):
                    print(f"[{_utc_now()}] experiment completed", flush=True)
                    return write_status(
                        "completed",
                        generation=generation,
                        controller_returncode=returncode,
                        summary=summary,
                        snapshot=last_snapshot,
                    )
                restart_reason = f"controller_exit_{returncode}"

            history.append(
                {
                    "generation": generation,
                    "ended_at": _utc_now(),
                    "reason": restart_reason,
                    "controller_returncode": returncode,
                    "snapshot": last_snapshot,
                }
            )
            if not _retryable_summary(last_snapshot, int(returncode or 0)):
                return write_status(
                    "failed",
                    generation=generation,
                    reason="non_retryable_incomplete_result",
                    controller_returncode=returncode,
                    snapshot=last_snapshot,
                )
            if restart_count >= config.max_restarts:
                return write_status(
                    "failed",
                    generation=generation,
                    reason="restart_limit_exhausted",
                    controller_returncode=returncode,
                    snapshot=last_snapshot,
                )
            restart_count += 1
            write_status(
                "restart_backoff",
                generation=generation,
                reason=restart_reason,
                controller_returncode=returncode,
                snapshot=last_snapshot,
                restart_not_before_epoch=(
                    time.time() + config.restart_backoff_seconds
                ),
            )
            stop_requested.wait(config.restart_backoff_seconds)
    finally:
        if active_child is not None and active_child.poll() is None:
            _terminate_process_tree(active_child)
        for selected, handler in previous_handlers.items():
            signal.signal(selected, handler)
