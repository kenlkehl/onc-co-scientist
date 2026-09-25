"""Launch a frozen local CAA pilot with a private server and durable process status.

The supervisor survives the launching shell, runs cells serially, summarizes them,
and releases its own GPU server on completion, scheduled pause, or failure.
Paused cells resume from their durable call journals; failed cells are never rerolled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.UTC).isoformat()


class ScheduledPause(BaseException):
    """Operator-requested interruption, distinct from a scientific/serving failure."""


def pause_time(value):
    """Require an explicit UTC offset so machine-local timezone cannot shift a cutoff."""
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("pause-at requires an explicit timezone offset")
    return parsed.astimezone(dt.UTC)


def arm_pause(value):
    target = pause_time(value)
    remaining = (target - dt.datetime.now(dt.UTC)).total_seconds()
    if remaining <= 0:
        raise ScheduledPause("Scheduled pause deadline already reached")

    def pause(signum, frame):
        raise ScheduledPause(f"Scheduled pause at {target.isoformat()}")

    signal.signal(signal.SIGALRM, pause)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def snapshot(campaign, *, active=True):
    result = []
    schedule = campaign / "schedule.json"
    if not schedule.exists():
        return result
    for job in json.loads(schedule.read_text()):
        run_dir = campaign / job["view"] / "runs" / job["run_id"]
        terminal = run_dir / "run.json"
        row = {**job, "status": "queued"}
        if terminal.exists():
            record = json.loads(terminal.read_text())
            row.update({key: record.get(key) for key in (
                "status", "iterations_completed", "agent_calls", "stop_reason",
            )})
        elif run_dir.exists():
            row["status"] = "running" if active else "interrupted"
        row["saved_calls"] = len(list((run_dir / "calls").glob("*.json")))
        result.append(row)
    return result


def stop_owned(process, *, immediate=False):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if immediate else signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def supervise(runtime, *, resume=False, pause_at=None):
    config = json.loads((runtime / "launch.json").read_text())
    cutoff = pause_at if resume else config.get("pause_at")
    campaign = Path(config["campaign"])
    source = runtime / "source"
    port = config.get("port", 8765)
    base_url = f"http://127.0.0.1:{port}/v1"
    env = {**os.environ, "PYTHONPATH": str(source / "src"), "CUDA_VISIBLE_DEVICES": config["gpus"],
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_DEACTIVATE_ASYNC_LOAD": "1",
           "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    status = {"status": "starting", "started_at": now(), "supervisor_pid": os.getpid(),
              "campaign": str(campaign), "runtime": str(runtime), "gpus": config["gpus"],
              "run_limit": config["run_limit"], "max_tokens": config["max_tokens"],
              "pause_at": cutoff, "resumed": resume}
    server = controller = preparation = None

    def save():
        status.update(updated_at=now(), runs=snapshot(campaign, active="ended_at" not in status))
        if status["status"] == "paused":
            for row in status["runs"]:
                if row["status"] == "interrupted":
                    row["status"] = "paused"
        write(runtime / "status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = time.monotonic() + config["max_hours"] * 3600
    save()
    try:
        if cutoff:
            arm_pause(cutoff)
            status.update(pause_at_utc=pause_time(cutoff).isoformat(), pause_timer_armed=True)
            save()
        if resume:
            if (campaign / "halt.json").exists():
                raise RuntimeError("Halted campaigns cannot be resumed; preserve failure artifacts")
            with (runtime / "controller.log").open("a") as log:
                preparation = subprocess.Popen(
                    [sys.executable, str(source / "scripts/expected_surprising/caa_discovery.py"),
                     "verify", str(campaign)], cwd=source, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
                if preparation.wait(timeout=600):
                    raise RuntimeError("Frozen campaign verification failed before resume")
                preparation = None
        # Do not accidentally attach the pilot to an unrelated existing server.
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already occupied; no existing server was changed")
        command = [sys.executable, "-m", "onc_co_scientist.cli", "caa", "serve",
                   "--model", config["model"], "--dtype", "bfloat16", "--device-map", "balanced",
                   "--inter-gpu-transfer", config["inter_gpu_transfer"],
                   "--vector-file", str(runtime / "vectors.npz"),
                   "--aliases-file", str(runtime / "aliases.json"), "--cache-implementation",
                   "dynamic", "--enable-thinking", "--host", "127.0.0.1", "--port", str(port),
                   "--default-max-new-tokens", str(config["max_tokens"]),
                   "--prefill-chunk-size", str(config["prefill_chunk_size"])]
        if config["max_context_tokens"] is not None:
            command.extend(["--max-context-tokens", str(config["max_context_tokens"])])
        with (runtime / "server.log").open("a") as log:
            server = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        status["server_pid"] = server.pid
        save()
        ready_deadline = time.monotonic() + 600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
            try:
                with urllib.request.urlopen(base_url + "/caa", timeout=30) as response:
                    manifest = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() > ready_deadline:
                    raise TimeoutError("Server did not become ready within ten minutes") from None
                time.sleep(2)
        if any(v in {"cpu", "disk"} for v in manifest["resolved_device_map"].values()):
            raise RuntimeError("Pilot requires weights fully resident on the selected GPUs")
        if resume and manifest != json.loads((campaign / "server_manifest.json").read_text()):
            raise RuntimeError("Resumed server differs from the frozen serving manifest")
        if not resume:
            write(runtime / "server_manifest.json", manifest)
        driver = source / "scripts/expected_surprising/caa_discovery.py"
        status.update(status="preparing", server_fingerprint=manifest["fingerprint"])
        save()
        command = [sys.executable, str(driver), "prepare", "--out", str(campaign),
                   "--named", config["named"], "--masked", config["masked"],
                   "--base-config", config["base_config"], "--server-manifest",
                   str(runtime / "server_manifest.json"), "--base-url", base_url,
                   "--arm", "gemma4-control",
                   "--arm", "gemma4-caa", "--replicates", "1", "--workflow", "persistent",
                   "--max-tokens", str(config["max_tokens"]),
                   "--request-timeout", str(config["request_timeout"]),
                   "--rationale", "Unvalidated L40/-0.05 local persistent-agent pilot; "
                   "one matched replicate per condition, not confirmatory efficacy evidence"]
        if config["iterations"] == 6:
            command.append("--smoke")
        with (runtime / "controller.log").open("a") as log:
            if not resume:
                preparation = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                if preparation.wait(timeout=600):
                    raise RuntimeError("Campaign preparation failed")
                preparation = None
            remaining = config["run_limit"] - sum(
                row["status"] == "completed" for row in snapshot(campaign)
            )
            if remaining <= 0:
                raise RuntimeError("The configured cell limit has already been completed")
            controller = subprocess.Popen([sys.executable, str(driver), "run", str(campaign),
                                           "--limit", str(remaining)],
                                          cwd=source, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status="running", controller_pid=controller.pid)
        save()
        while controller.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("GPU server exited while the controller was active")
            if time.monotonic() > deadline:
                raise ScheduledPause(
                    "Pilot reached its wall-clock limit; saved responses retained for resume"
                )
            save()
            time.sleep(15)
        status["controller_exit_code"] = controller.returncode
        status["status"] = "summarizing"
        save()
        with (runtime / "controller.log").open("a") as log:
            subprocess.run([sys.executable, str(driver), "summarize", str(campaign)],
                           cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        if controller.returncode:
            raise RuntimeError(
                f"Controller exited with code {controller.returncode}; inspect campaign halt.json"
            )
        rows = snapshot(campaign)
        completed = sum(r["status"] == "completed" for r in rows)
        status["status"] = ("completed" if rows and completed == len(rows)
                            else "gate_passed" if completed == config["run_limit"]
                            else "finished_with_errors")
    except ScheduledPause as exc:
        status.update(status="paused", pause_reason=str(exc), paused_at=now(),
                      resume_available=(campaign / "freeze.json").exists())
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if cutoff:
            signal.setitimer(signal.ITIMER_REAL, 0)
            status["pause_timer_armed"] = False
        # Stop the client before the server so a planned cutoff is not persisted
        # as an infrastructure failure. Immediate shutdown releases GPU work at
        # the deadline; only a response not yet atomically saved may be repeated.
        if status["status"] == "paused":
            stop_owned(controller, immediate=True)
            stop_owned(server, immediate=True)
        else:
            stop_owned(controller)
            stop_owned(server)
        if preparation is not None:
            stop_owned(preparation, immediate=status["status"] == "paused")
        status.update(ended_at=now(), server_stopped=True)
        save()


def spawn_worker(runtime, *, resume=False, pause_at=None):
    script = runtime / "source/scripts/expected_surprising" / Path(__file__).name
    command = [sys.executable, str(script), "--worker", str(runtime)]
    if resume:
        command.append("--resume-existing")
        if pause_at:
            command.extend(["--pause-at", pause_at])
    with (runtime / "supervisor.log").open("a") as log:
        worker = subprocess.Popen(command, cwd=runtime / "source", stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    config = json.loads((runtime / "launch.json").read_text())
    print(json.dumps({"supervisor_pid": worker.pid, "runtime": str(runtime),
                      "campaign": config["campaign"], "status_file": str(runtime / "status.json")}))


def worker(runtime, *, resume=False, pause_at=None):
    # The lock spans the entire session, including model loading and shutdown.
    with (runtime / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = runtime / "status.json"
        if previous.exists():
            state = json.loads(previous.read_text())
            if not resume or state.get("status") != "paused" or not state.get("resume_available"):
                raise RuntimeError("Only a paused, resumable runtime can start another session")
            history = runtime / "sessions"
            history.mkdir(exist_ok=True)
            write(history / f"{time.time_ns()}.json", state)
        elif resume:
            raise RuntimeError("Cannot resume a runtime without saved status")
        supervise(runtime, resume=resume, pause_at=pause_at)


def launch(args):
    runtime, campaign = args.runtime.resolve(), args.campaign.resolve()
    if campaign.exists():
        raise FileExistsError(f"Use a fresh campaign directory: {campaign}")
    runtime.mkdir(parents=True, exist_ok=False)
    source = runtime / "source"
    shutil.copytree(ROOT / "src/onc_co_scientist", source / "src/onc_co_scientist",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script_dir = source / "scripts/expected_surprising"
    script_dir.mkdir(parents=True)
    for name in ("run_local_caa_pilot.py", "caa_discovery.py", "build_presentation_results.py"):
        shutil.copy2(ROOT / "scripts/expected_surprising" / name, script_dir / name)
    for filename in ("vectors.npz", "vectors.json", "aliases.json"):
        shutil.copy2(args.smoke_artifacts / filename, runtime / filename)
    latest = ROOT / "data/expected_surprising_ledger/full_runs"
    write(runtime / "launch.json", {
        "campaign": str(campaign), "model": str(args.model.resolve()),
        "iterations": args.iterations, "max_hours": args.max_hours,
        "max_tokens": args.max_tokens, "request_timeout": args.request_timeout,
        "run_limit": args.run_limit, "gpus": args.gpus,
        "inter_gpu_transfer": args.inter_gpu_transfer,
        "port": getattr(args, "port", 8765),
        "pause_at": getattr(args, "pause_at", None),
        "prefill_chunk_size": args.prefill_chunk_size,
        "max_context_tokens": args.max_context_tokens,
        "named": str(latest / "20260910_vllm_recommended/input_data"),
        "masked": str(latest / "20260910_masked_vllm_recommended/input_data"),
        "base_config": str(latest / "20260910_vllm_recommended/config.yaml"),
    })
    spawn_worker(runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a paused runtime using its frozen source")
    parser.add_argument("--resume-existing", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pause-at", help="Automatic pause at an ISO timestamp with UTC offset")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke-artifacts", type=Path)
    parser.add_argument("--iterations", type=int, choices=[6, 25], default=6)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--request-timeout", type=int, default=21600)
    parser.add_argument("--run-limit", type=int, default=1,
                        help="Default one full cell; increase only after reviewing acceptance runs")
    parser.add_argument("--gpus", default="0,1", help="Private server CUDA_VISIBLE_DEVICES")
    parser.add_argument("--port", type=int, default=8765, help="Private local server port")
    parser.add_argument("--inter-gpu-transfer", choices=["cpu", "direct"], default="cpu")
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--max-context-tokens", type=int,
                        help="Prompt + output ceiling; does not guarantee VRAM capacity")
    args = parser.parse_args()
    if args.pause_at:
        try:
            cutoff = pause_time(args.pause_at)
        except ValueError as exc:
            parser.error(str(exc))
        if not args.worker and cutoff <= dt.datetime.now(dt.UTC):
            parser.error("pause-at must be in the future")
    if args.worker:
        worker(args.worker.resolve(), resume=args.resume_existing, pause_at=args.pause_at)
    elif args.resume:
        runtime = args.resume.resolve()
        state = json.loads((runtime / "status.json").read_text())
        if state.get("status") != "paused" or not state.get("resume_available"):
            parser.error("Only a paused, resumable runtime can be resumed")
        spawn_worker(runtime, resume=True, pause_at=args.pause_at)
    else:
        if not all((args.runtime, args.campaign, args.model, args.smoke_artifacts)):
            parser.error("runtime, campaign, model and smoke-artifacts are required")
        if args.max_hours <= 0:
            parser.error("max-hours must be positive")
        if not 1 <= args.port <= 65535:
            parser.error("port must be between 1 and 65535")
        if min(args.max_tokens, args.request_timeout, args.run_limit, args.prefill_chunk_size) <= 0:
            parser.error("token, time, run and chunk limits must be positive")
        if args.max_context_tokens is not None and args.max_context_tokens <= args.max_tokens:
            parser.error("max-context-tokens must leave room for the prompt as well as max-tokens")
        launch(args)


if __name__ == "__main__":
    main()
