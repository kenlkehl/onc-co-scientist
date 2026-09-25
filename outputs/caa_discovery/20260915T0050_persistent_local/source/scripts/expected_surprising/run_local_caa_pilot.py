"""Launch a frozen local CAA pilot with a private server and durable process status.

The supervisor survives the launching shell, runs cells serially, summarizes them,
and releases its own GPU server on completion or failure. It never restarts cells.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def snapshot(campaign):
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
            row["status"] = "running"
        row["saved_calls"] = len(list((run_dir / "calls").glob("*.json")))
        result.append(row)
    return result


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def supervise(runtime):
    config = json.loads((runtime / "launch.json").read_text())
    campaign = Path(config["campaign"])
    source = runtime / "source"
    env = {**os.environ, "PYTHONPATH": str(source / "src"), "CUDA_VISIBLE_DEVICES": "0,1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_DEACTIVATE_ASYNC_LOAD": "1",
           "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    status = {"status": "starting", "started_at": now(), "supervisor_pid": os.getpid(),
              "campaign": str(campaign), "runtime": str(runtime), "gpus": [0, 1]}
    server = controller = None

    def save():
        status.update(updated_at=now(), runs=snapshot(campaign))
        write(runtime / "status.json", status)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = time.monotonic() + config["max_hours"] * 3600
    save()
    try:
        # Do not accidentally attach the pilot to an unrelated existing server.
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8765)) == 0:
                raise RuntimeError("Port 8765 is already occupied; no existing server was changed")
        command = [sys.executable, "-m", "onc_co_scientist.cli", "caa", "serve",
                   "--model", config["model"], "--dtype", "bfloat16", "--device-map", "balanced",
                   "--inter-gpu-transfer", "cpu", "--vector-file", str(runtime / "vectors.npz"),
                   "--aliases-file", str(runtime / "aliases.json"), "--cache-implementation",
                   "dynamic", "--enable-thinking", "--host", "127.0.0.1", "--port", "8765"]
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
                with urllib.request.urlopen("http://127.0.0.1:8765/v1/caa", timeout=30) as response:
                    manifest = json.load(response)
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() > ready_deadline:
                    raise TimeoutError("Server did not become ready within ten minutes") from None
                time.sleep(2)
        if any(v in {"cpu", "disk"} for v in manifest["resolved_device_map"].values()):
            raise RuntimeError("Pilot requires weights fully resident on GPUs 0,1")
        write(runtime / "server_manifest.json", manifest)
        driver = source / "scripts/expected_surprising/caa_discovery.py"
        status.update(status="preparing", server_fingerprint=manifest["fingerprint"])
        save()
        command = [sys.executable, str(driver), "prepare", "--out", str(campaign),
                   "--named", config["named"], "--masked", config["masked"],
                   "--base-config", config["base_config"], "--server-manifest",
                   str(runtime / "server_manifest.json"), "--arm", "gemma4-control",
                   "--arm", "gemma4-caa", "--replicates", "1", "--workflow", "persistent",
                   "--rationale", "Unvalidated L40/-0.05 local persistent-agent pilot; "
                   "one matched replicate per condition, not confirmatory efficacy evidence"]
        if config["iterations"] == 6:
            command.append("--smoke")
        with (runtime / "controller.log").open("a") as log:
            subprocess.run(command, cwd=source, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
            controller = subprocess.Popen([sys.executable, str(driver), "run", str(campaign)],
                                          cwd=source, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status="running", controller_pid=controller.pid)
        save()
        while controller.poll() is None:
            if server.poll() is not None:
                raise RuntimeError("GPU server exited while the controller was active")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Pilot reached its wall-clock limit; cached calls remain resumable"
                )
            save()
            time.sleep(15)
        status["controller_exit_code"] = controller.returncode
        if controller.returncode:
            raise RuntimeError(f"Controller exited with code {controller.returncode}")
        status["status"] = "summarizing"
        save()
        with (runtime / "controller.log").open("a") as log:
            subprocess.run([sys.executable, str(driver), "summarize", str(campaign)],
                           cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        rows = snapshot(campaign)
        status["status"] = ("completed" if rows and all(r["status"] == "completed" for r in rows)
                            else "finished_with_errors")
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        stop_owned(controller)
        stop_owned(server)
        status.update(ended_at=now(), server_stopped=True)
        save()


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
        "named": str(latest / "20260910_vllm_recommended/input_data"),
        "masked": str(latest / "20260910_masked_vllm_recommended/input_data"),
        "base_config": str(latest / "20260910_vllm_recommended/config.yaml"),
    })
    with (runtime / "supervisor.log").open("a") as log:
        worker = subprocess.Popen([sys.executable, str(script_dir / Path(__file__).name),
                                   "--worker", str(runtime)], cwd=source, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor_pid": worker.pid, "runtime": str(runtime),
                      "campaign": str(campaign), "status_file": str(runtime / "status.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke-artifacts", type=Path)
    parser.add_argument("--iterations", type=int, choices=[6, 25], default=6)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    if args.worker:
        supervise(args.worker.resolve())
    else:
        if not all((args.runtime, args.campaign, args.model, args.smoke_artifacts)):
            parser.error("runtime, campaign, model and smoke-artifacts are required")
        if args.max_hours <= 0:
            parser.error("max-hours must be positive")
        launch(args)


if __name__ == "__main__":
    main()
