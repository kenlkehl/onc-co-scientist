"""Prepare and launch the DS001 NSCLC replication through a local Codex CLI.

One persistent CLI session per replicate; no model SDK or Work subagents.
No scientific scoring or automatic replacement of interrupted replicates.
Run from the repository root with ``python -m experiments.aim1_recovery.local_cli``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from experiments.aim1_recovery.loose_prompt import STYLE, apply_loose_prompt, launch_prompt
from experiments.aim1_recovery.preflight import validate_inputs
from experiments.aim1_recovery.prepare import digest, prepare, write_json
from experiments.aim1_recovery.run_batch import prompt_for
from onc_co_scientist.harness.structured_runner import finalize_workspace

MODEL = "gpt-5.6-sol"
EFFORT = "medium"
BACKEND = "codex-cli"


def prepare_local(
    repo: Path, out: Path, python: Path, repeats: int, prompt_style: str = "structured-v2"
) -> dict:
    if prompt_style not in {"structured-v2", STYLE}:
        raise ValueError("Unknown prompt style")
    if out.exists():
        raise ValueError("Use a new output directory; existing experiments are preserved")
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    plan = prepare(
        repo, out, python, clinical_repeats=repeats, depmap_repeats=0,
        model=MODEL, backend=BACKEND, reasoning_effort=EFFORT,
        service_tier=None, tasks=("nsclc",),
    )
    plan["protocol"].update(
        service_tier_evidence="CLI account/config default; no priority equivalence claimed",
        isolation="Separate directories and instructions; CLI workspace-write sandbox; "
        "read access is not restricted to the task directory",
        comparability="DS001 NSCLC, same source cohort, split seed and structured v2 research "
        "rules as Sol Work, with explicit public treatment roles added. "
        "CLI runtime/configuration differ; token budgets are not matched.",
        preflight_design="One named and one masked setup session in a separate experiment; "
        "excluded from the formal 20-per-condition batch",
    )
    if prompt_style == STYLE:
        apply_loose_prompt(repo, out, plan, python)
    write_json(out / "plan.json", plan)
    write_json(out / "protocol.json", plan["protocol"])
    write_json(out / "terminal_failures.json", {})
    result = validate_inputs(out / "plan.json")
    write_json(out / "input_preflight.json", result)
    implementation = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "files_sha256": {
            name: digest(repo / name) for name in (
                "experiments/aim1_recovery/local_cli.py",
                "experiments/aim1_recovery/loose_prompt.py",
                "experiments/aim1_recovery/preflight.py",
                "experiments/aim1_recovery/run_batch.py",
                "experiments/aim1_recovery/prepare.py",
                "experiments/aim1_recovery/score.py",
                "src/onc_co_scientist/harness/structured_runner.py",
                "src/onc_co_scientist/harness/research_budget.py",
                "src/onc_co_scientist/harness/treatment_roles.py",
                "src/onc_co_scientist/harness/task_spec.py",
                "src/onc_co_scientist/harness/templates/agent_instructions.md.j2",
                "src/onc_co_scientist/synthetic/anonymize.py",
                "src/onc_co_scientist/scoring/deterministic.py",
                "src/onc_co_scientist/scoring/structured_batch.py",
            )
        },
    }
    write_json(out / "implementation_at_preparation.json", implementation)
    return result


def validate_job(plan: dict, job: dict) -> None:
    protocol = plan["protocol"]
    for key, expected in (("model_id", MODEL), ("reasoning_effort", EFFORT),
                          ("backend", BACKEND)):
        if protocol.get(key) != expected:
            raise ValueError(f"Frozen {key} does not match this launcher")
    ws = Path(job["workspace"])
    if job.get("prompt_style", "structured-v2") != protocol.get("prompt_style", "structured-v2"):
        raise ValueError("Job prompt style differs from frozen protocol")
    metadata = json.loads((ws / "metadata.json").read_text())
    if metadata.get("harness_id") != protocol.get("harness_id", f"{BACKEND}-structured-v2"):
        raise ValueError("Job harness differs from frozen protocol")
    for name, expected in {
        "dataset.parquet": job["data_sha256"],
        "agent_instructions.md": job["instructions_sha256"],
        **job["public_input_sha256"],
    }.items():
        if digest(ws / name) != expected:
            raise ValueError(f"Frozen input changed: {job['job_id']}/{name}")


def session_id(logdir: Path) -> str | None:
    ids = set()
    for log in sorted(logdir.glob("attempt_*.jsonl")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and event.get("thread_id"):
                ids.add(event["thread_id"])
    if len(ids) > 1:
        raise ValueError("Multiple CLI thread IDs in one replicate; inspect before continuing")
    return next(iter(ids), None)


def cli_command(codex: str, job: dict, thread: str | None = None) -> list[str]:
    # Do not override account endpoints, user rules, or authentication. The CLI
    # global options apply to both exec and exec resume. No --ephemeral: resume
    # requires the CLI's persisted thread, not only the research JSON files.
    command = [codex, "-a", "never", "-s", "workspace-write", "-m", MODEL,
               "-c", f'model_reasoning_effort="{EFFORT}"', "exec"]
    if thread:
        command += ["resume", "--json", thread,
                    "Continue this same research session from its saved records. "
                    "Follow the original task instructions, including its stopping rule. "
                    "Preserve accepted iterations; do not restart, backfill, or change inputs. "
                    "Return job ID, iteration count, finalization status only."]
    else:
        prompt = launch_prompt(job) if job.get("prompt_style") == STYLE else prompt_for(job)
        command += ["--skip-git-repo-check", "--json", prompt]
    return command


def launch(plan: dict, job: dict, codex: str, resume: bool, timeout: int) -> dict:
    validate_job(plan, job)
    ws = Path(job["workspace"])
    logdir = ws / "cli_logs"
    if (ws / "transcript.json").exists():
        transcript = finalize_workspace(ws, write_output=False)
        if transcript.model_id != MODEL or transcript.harness_id != plan["protocol"].get(
            "harness_id", f"{BACKEND}-structured-v2"
        ):
            raise ValueError("Completed transcript model/harness differs from frozen plan")
        if not (ws / "analysis_summary.txt").is_file():
            raise ValueError("Completed transcript is missing analysis_summary.txt")
        return {"job_id": job["job_id"], "status": "already_completed"}
    # A lock also prevents two coordinators from launching this replicate.
    # Keep a stale lock after an unclean kill for explicit human inspection.
    logdir.mkdir(exist_ok=True)
    lock = logdir / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        prior = list(logdir.glob("attempt_*.jsonl"))
        partial = bool(prior or list((ws / "iterations").glob("*.json")))
        thread = session_id(logdir) if partial else None
        if partial and (not resume or not thread):
            raise ValueError("Existing attempt: use --resume with its retained CLI thread. "
                             "If no thread ID exists, inspect manually; do not replace silently.")
        command = cli_command(codex, job, thread)
        number = len(prior) + 1
        stem = logdir / f"attempt_{number:03d}"
        record = {
            "job_id": job["job_id"], "status": "running", "command": command,
            "started_utc": datetime.now(UTC).isoformat(), "resume_thread": thread,
            "requested_model": MODEL, "requested_reasoning_effort": EFFORT,
            "cli_version": subprocess.check_output([codex, "--version"], text=True).strip(),
            "timeout_seconds": timeout,
            "runtime_note": "Requested settings recorded; inspect CLI events for errors. "
            "Transcript model labels are not independent backend attestation.",
        }
        write_json(stem.with_suffix(".json"), record)
        with stem.with_suffix(".jsonl").open("x") as stdout, \
                stem.with_suffix(".stderr").open("x") as stderr:
            process = subprocess.Popen(command, cwd=ws, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(status="interrupted", ended_utc=datetime.now(UTC).isoformat())
                write_json(stem.with_suffix(".json"), record)
                raise
        record.update(exit_code=code, ended_utc=datetime.now(UTC).isoformat(),
                      thread_id=session_id(logdir))
        try:
            if code:
                raise ValueError(f"CLI exited {code}; inspect saved stderr and events")
            validate_job(plan, job)
            transcript = finalize_workspace(ws, write_output=False)
            if transcript.model_id != MODEL or transcript.harness_id != plan["protocol"].get(
                "harness_id", f"{BACKEND}-structured-v2"
            ):
                raise ValueError("Final transcript model/harness differs from plan")
            required_outputs = ("transcript.json", "analysis_summary.txt")
            if not all((ws / name).is_file() for name in required_outputs):
                raise ValueError("CLI did not save the final transcript and summary")
            record.update(status="completed", iterations=len(transcript.iterations))
        except Exception as exc:
            record.update(status="incomplete", reason=str(exc))
        write_json(stem.with_suffix(".json"), record)
        return record
    finally:
        lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--python", type=Path, default=Path(sys.executable))
    prep.add_argument("--repeats", type=int, default=20)
    prep.add_argument("--prompt-style", choices=["structured-v2", STYLE], default="structured-v2")
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--codex", default="codex")
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--job-id", action="append")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=10800)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        repo = Path(__file__).resolve().parents[2]
        # absolute(), not resolve(): preserve the virtualenv interpreter path.
        print(json.dumps(prepare_local(repo, args.out.resolve(), args.python.absolute(),
                                       args.repeats, args.prompt_style)))
        return
    if args.jobs < 1 or args.timeout_seconds < 1:
        parser.error("Concurrency and timeout must be positive")
    codex = shutil.which(args.codex)
    if not codex:
        parser.error("Codex CLI executable not found")
    plan = json.loads(args.plan.read_text())
    selected = set(args.job_id or [j["job_id"] for j in plan["jobs"]])
    if selected - {j["job_id"] for j in plan["jobs"]}:
        parser.error("Unknown job ID")
    jobs = [j for j in plan["jobs"] if j["job_id"] in selected]
    for job in jobs:
        validate_job(plan, job)
    if args.dry_run:
        for job in jobs:
            print(json.dumps({"job_id": job["job_id"], "cwd": job["workspace"],
                              "fresh_command": cli_command(codex, job)}))
        return
    errors = False
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(launch, plan, job, codex, args.resume,
                               args.timeout_seconds): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"job_id": futures[future]["job_id"],
                          "status": "blocked_or_interrupted", "reason": str(exc)}
            print(json.dumps(result), flush=True)
            errors |= result["status"] not in {"completed", "already_completed"}
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
