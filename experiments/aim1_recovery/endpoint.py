"""Prepare the DS001 NSCLC loose-prompt comparison for a native-tool endpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from experiments.aim1_recovery.loose_prompt import apply_loose_prompt
from experiments.aim1_recovery.preflight import validate_inputs
from experiments.aim1_recovery.prepare import digest, prepare, write_json
from onc_co_scientist.harness.python_sandbox import ISOLATION_VERSION

RUNTIME = {
    "max_turns": 200,
    "max_tool_calls": 400,
    "max_generated_tokens": 200000,
    "max_tokens_per_call": 16384,
    "python_timeout": 120.0,
    "request_timeout": 900.0,
}


def prepare_endpoint(
    repo: Path, out: Path, python: Path, model: str, base_url: str, repeats: int = 20
) -> dict:
    if out.exists():
        raise ValueError("Use a new output directory; existing experiments are preserved")
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    plan = prepare(
        repo, out, python, clinical_repeats=repeats, depmap_repeats=0,
        model=model, backend="endpoint", reasoning_effort=None, service_tier=None,
        tasks=("nsclc",),
    )
    apply_loose_prompt(repo, out, plan, python)
    plan["protocol"].update(
        endpoint_url=base_url.rstrip("/"), endpoint_runtime=RUNTIME,
        service_tier_evidence="Local vLLM endpoint; no hosted service tier requested",
        model_sampling="Server defaults; no temperature, reasoning-effort, or seed override",
        isolation=ISOLATION_VERSION,
        isolation_details="Mandatory bubblewrap user/mount/PID/network namespaces; "
        "read-only public inputs and runtime libraries; only this job's analysis directory "
        "is writable; controller records, other jobs, repository source and private data "
        "are not mounted. No unsandboxed fallback.",
        comparability="DS001 NSCLC with the same 40000/10000 split, 25-iteration ceiling, "
        "legacy loose brief and JSON findings. Explicit treatment roles added. "
        "Model and runtime differ from Sol/Codex CLI; compute is not matched.",
        preflight_design="One named and one masked setup replicate in a separate directory; "
        "excluded from the formal 20-per-condition batch",
    )
    write_json(out / "plan.json", plan)
    write_json(out / "protocol.json", plan["protocol"])
    write_json(out / "terminal_failures.json", {})
    result = validate_inputs(out / "plan.json")
    write_json(out / "input_preflight.json", result)
    files = [
        "experiments/aim1_recovery/endpoint.py", "experiments/aim1_recovery/prepare.py",
        "experiments/aim1_recovery/loose_prompt.py", "experiments/aim1_recovery/run_batch.py",
        "src/onc_co_scientist/harness/structured_runner.py",
        "src/onc_co_scientist/harness/python_sandbox.py",
        "src/onc_co_scientist/harness/research_budget.py",
        "src/onc_co_scientist/harness/runtime.py",
        "src/onc_co_scientist/harness/structured.py",
        "src/onc_co_scientist/harness/treatment_roles.py",
        "src/onc_co_scientist/harness/task_spec.py",
        "src/onc_co_scientist/harness/templates/agent_instructions.md.j2",
        "src/onc_co_scientist/scoring/deterministic.py",
        "src/onc_co_scientist/scoring/structured_batch.py",
    ]
    write_json(out / "implementation_at_preparation.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                               text=True).strip(),
        "files_sha256": {name: digest(repo / name) for name in files},
    })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as response:
        inventory = json.load(response)
    if args.model not in {model["id"] for model in inventory["data"]}:
        parser.error("Requested model is not served by this endpoint")
    plan = prepare_endpoint(Path(__file__).resolve().parents[2], args.out.resolve(),
                            args.python.absolute(), args.model, args.base_url, args.repeats)
    write_json(args.out / "endpoint_at_preparation.json", {
        "utc": datetime.now(UTC).isoformat(), "base_url": args.base_url, "models": inventory,
    })
    print(json.dumps({"n_jobs": len(plan["jobs"]), "plan": str(args.out / "plan.json")}))


if __name__ == "__main__":
    main()
