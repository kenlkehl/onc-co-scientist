"""Run an isolated frozen grid, yielding shared vLLM endpoints to its named parent."""

import argparse
import collections
import fcntl
import hashlib
import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.expected_surprising.experiment import run_cell
from onc_co_scientist.expected_surprising.experiment_report import write_report
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans


def now():
    return datetime.now(UTC).isoformat()


def parent_finished(path):
    # Fail closed on absent/partial state: never assume the endpoint is free.
    try:
        state = json.loads(path.read_text())
        return bool(state.get("completed_at")) and (
            len(state["finished"]) + len(state["worker_errors"]) == len(state["selected"])
        )
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    lock = (root / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    for name, expected in {
        **frozen["source_hashes"],
        **frozen["input_hashes"],
        "config.yaml": frozen["config_sha256"],
        "launch_policy.json": frozen["launch_policy_sha256"],
        "plan.json": frozen["plan_sha256"],
    }.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {name}")
    import onc_co_scientist.expected_surprising.coordination as module

    if not Path(module.__file__).is_relative_to(root / "source"):
        raise ValueError("Runner must use its isolated frozen source")
    spec = load_experiment_spec(root / "config.yaml")
    plans_by_id = {p.run_id: p for p in build_run_plans(spec)}
    plan_rows = json.loads((root / "plan.json").read_text())
    plans = [plans_by_id[r["run_id"]] for r in plan_rows]
    assert len(plans) == len(plans_by_id) == 360
    fingerprint = spec.fingerprint()
    launch_policy = json.loads((root / "launch_policy.json").read_text())
    parent_control = Path(launch_policy["wait_for_vllm_control"])
    path = root / "execution.json"
    if path.exists() and not args.resume:
        raise FileExistsError("This grid has already launched; use --resume for exact call replay")
    execution = dict(
        started_at=now(),
        pid=os.getpid(),
        assigned_runs=len(plans),
        source=module.__file__,
        resume=args.resume,
        wait_for_vllm_control=str(parent_control),
    )
    atomic_write_json(path, execution)
    pending, active, results = list(plans), {}, []
    released = False

    def status():
        counts = collections.Counter(
            (r["model_profile"], r["workflow_id"], r["status"]) for r in results
        )
        for plan in active.values():
            counts[plan.model.id, plan.workflow.id, "running"] += 1
        for plan in pending:
            state = (
                "waiting_for_unmasked_vllm"
                if plan.model.provider_config["kind"] == "vllm_openai" and not released
                else "queued"
            )
            counts[plan.model.id, plan.workflow.id, state] += 1
        rows = [
            "# Masked clinical workflow grid",
            "",
            f"Updated {now()}",
            "",
            "360 runs: six models × three workflows × two versions × ten repeats; 25 iterations.",
            (
                "Predictors and text levels masked; repaired harness for all models. "
                "Shared vLLM endpoints are reserved for the unmasked grid until "
                "its continuation finishes."
            ),
            "",
            "| Model | Workflow | State | Runs |",
            "|---|---|---|---:|",
        ]
        for (model, workflow, state), n in sorted(counts.items()):
            rows.append(f"| {model} | {workflow} | {state} | {n} |")
        rows += [
            "",
            (
                "[Configuration](config.yaml) · [Execution](execution.json) · "
                "[Final results](RESULTS.md)"
            ),
        ]
        atomic_write_text(root / "STATUS.md", "\n".join(rows) + "\n")
        execution.update(
            updated_at=now(),
            terminal_runs=len(results),
            active_runs=len(active),
            queued_runs=len(pending),
            vllm_released=released,
        )
        atomic_write_json(path, execution)

    try:
        with ThreadPoolExecutor(max_workers=spec.max_parallel) as pool:
            while pending or active:
                if not released and parent_finished(parent_control):
                    released = True
                    execution["vllm_released_at"] = now()
                for plan in list(pending):
                    if len(active) >= spec.max_parallel:
                        break
                    if plan.model.provider_config["kind"] == "vllm_openai" and not released:
                        continue
                    pending.remove(plan)
                    future = pool.submit(
                        run_cell, spec, plan, root, fingerprint, resume=args.resume
                    )
                    active[future] = plan
                status()
                if not active:
                    time.sleep(30)
                    continue
                done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    plan = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            **plan.public_dict(),
                            "status": "failed",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                            "agent_calls": 0,
                            "usage": {},
                            "stop_reason": "worker_error",
                        }
                        atomic_write_json(root / "runs" / plan.run_id / "run.json", result)
                    results.append(result)
        positions = {p.run_id: i for i, p in enumerate(plans)}
        results.sort(key=lambda r: positions[r["run_id"]])
        failures = sum(r["status"] == "failed" for r in results)
        summary = dict(
            experiment_id=spec.experiment_id,
            spec_fingerprint=fingerprint,
            status="completed_with_failures" if failures else "completed",
            n_runs=len(results),
            n_completed=len(results) - failures,
            n_failed=failures,
            realized_agent_calls=sum(r.get("agent_calls", 0) for r in results),
            runs=results,
        )
        atomic_write_json(root / "summary.json", summary)
        write_report(spec, plans, results, root)
        atomic_write_text(root / "RESULTS.md", (root / "expected_surprising_report.md").read_text())
        execution.update(completed_at=now(), status=summary["status"])
        status()
    except BaseException as exc:
        execution.update(failed_at=now(), error=repr(exc), traceback=traceback.format_exc())
        atomic_write_json(path, execution)
        raise


if __name__ == "__main__":
    main()
