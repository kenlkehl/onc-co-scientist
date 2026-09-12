"""Deadline cutover: preserve frozen science and journals, override only transport."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_info(pid):
    try:
        root = Path("/proc") / str(pid)
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        return dict(
            pid=pid,
            state=fields[0],
            ppid=int(fields[1]),
            start_ticks=fields[19],
            command=[s.decode() for s in (root / "cmdline").read_bytes().split(b"\0") if s],
        )
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(info):
    return info is not None and info["state"] not in {"Z", "X"}


def signal_identity(info, sig):
    current = process_info(info["pid"])
    if live(current) and current["start_ticks"] == info["start_ticks"]:
        os.kill(info["pid"], sig)


def descendants(roots, processes):
    selected = {p["pid"]: p for p in roots}
    while True:
        extra = [p for p in processes if p["ppid"] in selected and p["pid"] not in selected]
        if not extra:
            return list(selected.values())
        selected.update((p["pid"], p) for p in extra)


def stop_personal(plan, control):
    """Stop exact recorded drivers and their children; never signal a shared process group."""
    roots = []
    for expected in plan["drivers"]:
        current = process_info(expected["pid"])
        if not live(current):
            if expected.get("lock_path"):
                with Path(expected["lock_path"]).open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError(
                            "Driver is not visible but still locked; use the host process namespace"
                        ) from None
            continue
        if (current["start_ticks"], current["command"]) != (
            expected["start_ticks"],
            expected["command"],
        ):
            raise RuntimeError("Driver identity changed; refusing to signal a reused PID")
        roots.append(current)
    write(control / "stop_intent.json", dict(at=now(), drivers=roots))
    # Freezing the parents first prevents another model request from being issued.
    for info in roots:
        signal_identity(info, signal.SIGSTOP)
    for _ in range(100):
        if all(
            not live(p := process_info(info["pid"])) or p["state"] in {"T", "t"} for info in roots
        ):
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("Driver freeze could not be verified")
    targets = roots
    for _ in range(2):
        processes = []
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    processes.append(process_info(int(path.name)))
                except PermissionError:
                    continue  # Unrelated users' processes need not be inspected.
        targets = descendants(targets, [p for p in processes if live(p)])
        for info in targets:
            signal_identity(info, signal.SIGSTOP)
    write(control / "stopped_processes.json", dict(at=now(), processes=targets))
    # Kill while frozen, so interruption cannot be committed as a scientific error.
    # Durable completed call journals remain; partial in-flight requests are reissued.
    for info in reversed(targets):
        signal_identity(info, signal.SIGKILL)
    for _ in range(100):
        remaining = [
            p
            for p in targets
            if live(q := process_info(p["pid"])) and q["start_ticks"] == p["start_ticks"]
        ]
        if not remaining:
            write(control / "personal_stopped.json", dict(at=now(), processes=targets))
            return
        time.sleep(0.1)
    raise RuntimeError("Personal experiment processes did not stop; Azure restart remains held")


def verify_bundle(root):
    for name, expected in read(root / "frozen_manifest.json")["hashes"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen file changed: {root / name}")


def install_transport(experiment, transport_path, endpoint, metadata):
    """Inject an explicitly audited transport without changing scientific provenance."""
    name = "onc_co_scientist.providers.deadline_azure_transport"
    spec = importlib.util.spec_from_file_location(name, transport_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    def factory(config):
        if config.get("kind") != "codex_cli":
            raise ValueError("Deadline cutover permits only the selected Codex models")
        options = {k: v for k, v in config.items() if k != "kind"}
        options.update(backend="azure", azure_endpoint=endpoint, resume_audit=True)
        audit = Path(options["audit_dir"])
        marker = audit / metadata.get("marker_name", "azure_transport_cutover.json")
        if not marker.exists():
            last = max(
                (int(p.name[5:]) for p in audit.glob("call-*") if p.name[5:].isdigit()), default=0
            )
            write(marker, dict(**metadata, prior_audit_calls=last, first_azure_call=last + 1))
        return module.CodexCLIProvider(module.CodexCLIConfig(**options))

    experiment.get_provider = factory
    return factory


def worker(plan_path, model):
    plan = read(plan_path)
    source, azure = Path(plan["source_root"]), Path(plan["azure_root"])
    if model not in plan["models"]:
        raise ValueError("Model is not released for deadline cutover")
    if (
        datetime.now(UTC) < datetime.fromisoformat(plan["deadline"])
        or not (plan_path.parent / "personal_stopped.json").exists()
    ):
        raise RuntimeError("Resume held until the deadline and verified personal-process stop")
    for original in plan["drivers"]:
        current = process_info(original["pid"])
        if live(current) and current["start_ticks"] == original["start_ticks"]:
            raise RuntimeError("Original personal-account process is still alive")
    if sha(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("Deadline worker code changed")
    verify_bundle(source)
    verify_bundle(azure)
    from onc_co_scientist.expected_surprising import experiment
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans

    if not Path(experiment.__file__).resolve().is_relative_to(source / "source/src"):
        raise ValueError("Resume requires the original frozen scientific source")
    control = source / "control" / model
    with (control / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = read(source / "release_policy.json")
        if model not in policy["released_models"]:
            raise ValueError("Model has been held since deadline preparation")
        transport = azure / "source/src/onc_co_scientist/providers/codex_cli.py"
        if plan.get("transport_path"):
            transport = Path(plan["transport_path"])
            if sha(transport) != plan["transport_sha256"]:
                raise ValueError("Frozen transport override changed")
        metadata = dict(
            at=now(),
            backend="azure",
            endpoint=plan["endpoint"],
            transport_path=str(transport),
            transport_sha256=sha(transport),
            deadline_plan=str(plan_path),
            deadline_plan_sha256=sha(plan_path),
            scientific_source="original frozen source unchanged",
            interrupted_requests="May be reissued; uncommitted usage is unknown",
        )
        if plan.get("recovery_reason"):
            metadata.update(
                reason=plan["recovery_reason"],
                marker_name=plan.get("marker_name", "azure_rate_recovery.json"),
            )
        factory = install_transport(experiment, transport, plan["endpoint"], metadata)
        factory(
            dict(
                kind="codex_cli",
                model_id="credential-check-only",
                audit_dir=str(plan_path.parent / model / "auth_check"),
            )
        ).environment()
        specs = {c: load_experiment_spec(source / c / "config.yaml") for c in ("named", "masked")}
        plans = {c: {p.run_id: p for p in build_run_plans(s)} for c, s in specs.items()}
        selected = [r for r in plan["active"] if r["model"] == model]
        if any(plans[r["condition"]][r["run_id"]].model.id != model for r in selected):
            raise ValueError("Deadline selection does not match model")
        state = dict(
            started_at=now(),
            pid=os.getpid(),
            model=model,
            backend="azure",
            resume=True,
            selected_runs=len(selected),
            workers=10,
            source=experiment.__file__,
            transport_override=metadata,
            queued_runs=0,
        )
        active, finished = {}, []

        def status():
            state.update(
                updated_at=now(),
                active=list(active.values()),
                finished=finished,
                terminal_runs=len(finished),
            )
            write(control / "execution.json", state)

        def run(row):
            condition, rid = row["condition"], row["run_id"]
            run_dir = source / condition / "runs" / rid
            prior = read(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
            already_terminal = bool(prior.get("scientific_report_sha256"))
            if not already_terminal:
                write(
                    run_dir / metadata.get("marker_name", "azure_transport_cutover.json"), metadata
                )
            result = experiment.run_cell(
                specs[condition],
                plans[condition][rid],
                source / condition,
                specs[condition].fingerprint(),
                resume=True,
            )
            if not already_terminal:
                result.update(mixed_transport=True)
                initial_marker = run_dir / "azure_transport_cutover.json"
                result["transport_cutover"] = (
                    read(initial_marker) if initial_marker.exists() else metadata
                )
                if plan.get("recovery_reason"):
                    recovery_marker = run_dir / "azure_rate_recovery.json"
                    result["azure_rate_recovery"] = (
                        read(recovery_marker) if recovery_marker.exists() else metadata
                    )
                    if plan.get("marker_name"):
                        result["transport_update"] = metadata
                write(run_dir / "run.json", result)
            return result

        with ThreadPoolExecutor(max_workers=10) as pool:
            for row in selected:
                active[pool.submit(run, row)] = row
            status()
            while active:
                done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    row = active.pop(future)
                    try:
                        result = future.result()
                        finished.append(
                            dict(
                                **row,
                                status=result["status"],
                                agent_calls=result.get("agent_calls", 0),
                            )
                        )
                    except Exception:
                        finished.append(
                            dict(**row, status="worker_error", traceback=traceback.format_exc())
                        )
                status()
        state.update(
            completed_at=now(),
            status="completed"
            if all(r["status"] == "completed" for r in finished)
            else "completed_with_failures",
        )
        status()


def check(plan_path):
    plan = read(plan_path)
    source, azure = Path(plan["source_root"]), Path(plan["azure_root"])
    control = plan_path.parent
    with (control / "cutover.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = [read(source / "control" / m / "execution.json") for m in plan["models"]]
        if any(
            s.get("backend") == "azure"
            and any(r["status"] == "worker_error" for r in s.get("finished", []))
            for s in states
        ):
            return dict(status="cutover_needs_attention", reason="An Azure resume worker failed")
        if all(
            s.get("completed_at") and not s.get("active") and not s.get("queued_runs")
            for s in states
        ):
            env = dict(os.environ, PYTHONPATH=str(Path(plan["repo"]) / "src"))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.expected_surprising.launch_azure_federation",
                    "--root",
                    str(azure),
                ],
                cwd=plan["repo"],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                raise RuntimeError("Queued Azure launcher failed; inspect its control records")
            return json.loads(result.stdout)
        if datetime.now(UTC) < datetime.fromisoformat(plan["deadline"]):
            return dict(status="waiting_for_deadline", deadline=plan["deadline"])
        if sha(Path(__file__)) != plan["script_sha256"]:
            raise ValueError("Deadline cutover code changed")
        if not (control / "personal_stopped.json").exists():
            for model in plan["models"]:
                write(
                    control / f"{model}_execution_before.json",
                    read(source / "control" / model / "execution.json"),
                )
            stop_personal(plan, control)
        if not (control / "resume_selection.json").exists():
            completed, unfinished = [], []
            for row in plan["active"]:
                path = source / row["condition"] / "runs" / row["run_id"] / "run.json"
                result = read(path) if path.exists() else {}
                (completed if result.get("scientific_report_sha256") else unfinished).append(row)
            write(
                control / "resume_selection.json",
                dict(at=now(), retained_completed=completed, resume_on_azure=unfinished),
            )
        verify_bundle(source)
        verify_bundle(azure)
        for model in plan["models"]:
            receipt = control / model / "launch.json"
            if receipt.exists():
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plan",
                str(plan_path),
                "--worker",
                model,
            ]
            write(receipt, dict(status="dispatching", command=command, at=now()))
            with (receipt.parent / "worker.log").open("a") as log:
                proc = subprocess.Popen(
                    command,
                    env=dict(os.environ, PYTHONPATH=str(source / "source/src")),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    cwd=source,
                )
            write(receipt, dict(status="dispatched", pid=proc.pid, command=command, at=now()))
        transition = read(source / "azure_transition.json")
        transition.update(
            status="active_runs_switching_to_azure",
            deadline=plan["deadline"],
            deadline_plan=str(plan_path),
        )
        write(source / "azure_transition.json", transition)
        return dict(status="active_azure_dispatched", models=plan["models"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.plan.resolve(), args.worker)
    else:
        print(json.dumps(check(args.plan.resolve()), indent=2))
