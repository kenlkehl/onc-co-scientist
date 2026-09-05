#!/usr/bin/env python3
"""Run prepared tasks against an endpoint, or export prompts for Work subagents."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from onc_co_scientist.harness.structured_runner import StructuredRunner, finalize_workspace


def normalized_tier(value: str | None) -> str | None:
    return {"default": "standard", "fast": "priority"}.get(value, value)


def validate_launch(plan: dict, args) -> None:
    """Freeze scientific model settings; a different endpoint URL may use the same model."""
    protocol = plan["protocol"]
    if protocol.get("schema_version") != "aim1-structured-v2":
        return
    for field, actual in (
        ("model_id", args.model),
        ("backend", args.backend),
        ("reasoning_effort", args.reasoning_effort),
    ):
        if protocol.get(field) != actual:
            raise ValueError(f"{field} differs from frozen protocol; prepare a new experiment")
    if normalized_tier(args.service_tier) != normalized_tier(
        protocol.get("service_tier_requested")
    ):
        raise ValueError("service tier differs from frozen protocol; prepare a new experiment")
    if args.backend == "work" and normalized_tier(args.work_advertised_tier) != normalized_tier(
        args.service_tier
    ):
        raise ValueError(
            "Work tool's advertised tier does not verify the requested service tier; "
            "do not dispatch"
        )


def prompt_for(job: dict) -> str:
    return (
        f"Run the independent research protocol in {job['workspace']}. "
        "Read agent_instructions.md and metadata.json there. Use its specified Python. "
        "Submit each actual iteration immediately before the next, retain executed code/results, "
        "write analysis_summary.txt and finalize. Do not backfill or pad iterations. "
        "Inspect only this workspace's inputs/own outputs: no other folders, repository source, "
        "prior runs, answer keys, external sources or delegation. "
        "Return job ID, iteration count, finalization status only."
    )


def run_one(job: dict, args) -> dict:
    workspace = Path(job["workspace"])
    if (workspace / "transcript.json").exists():
        result = finalize_workspace(workspace, write_output=False)
        if result.model_id != args.model:
            raise ValueError(
                f"Existing {job['job_id']} used a different model. "
                "Use a fresh experiment directory."
            )
        return {"job_id": job["job_id"], "status": "already_completed"}
    runner = StructuredRunner(
        workspace,
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env, ""),
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        max_generated_tokens=args.max_generated_tokens,
        max_tokens_per_call=args.max_tokens_per_call,
        python_timeout=args.python_timeout,
    )
    try:
        transcript = runner.run()
        return {
            "job_id": job["job_id"],
            "status": "completed",
            "iterations": len(transcript.iterations),
        }
    except Exception as exc:
        # Preserve all partial records. Do not silently start a fresh scientific replicate.
        result = {"job_id": job["job_id"], "status": "technical_failure", "reason": str(exc)}
        (workspace / "technical_failure.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--backend", choices=["work", "endpoint"], required=True)
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--reasoning-effort")
    p.add_argument("--service-tier", choices=["standard", "default", "priority", "fast"])
    p.add_argument(
        "--work-advertised-tier",
        choices=["standard", "default", "priority", "fast"],
        help="Actual tier advertised by the current Work model tool, not a requested setting",
    )
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-turns", type=int, default=200)
    p.add_argument("--max-tool-calls", type=int, default=400)
    p.add_argument("--max-generated-tokens", type=int, default=200000)
    p.add_argument("--max-tokens-per-call", type=int, default=4096)
    p.add_argument("--python-timeout", type=float, default=30)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    protocol = plan["protocol"]
    args.model = args.model or protocol.get("model_id", "gpt-5.6-luna")
    if protocol.get("schema_version") == "aim1-structured-v2":
        args.reasoning_effort = args.reasoning_effort or protocol.get("reasoning_effort")
        args.service_tier = args.service_tier or protocol.get("service_tier_requested")
    try:
        validate_launch(plan, args)
    except ValueError as exc:
        p.error(str(exc))
    # Check completed jobs before filtering: otherwise a new backend/model could
    # silently top up a partially completed experiment with mixed model labels.
    for job in plan["jobs"]:
        transcript_path = Path(job["workspace"]) / "transcript.json"
        if transcript_path.exists():
            saved = json.loads(transcript_path.read_text())
            if saved.get("model_id") != args.model:
                p.error(
                    f"Existing {job['job_id']} used {saved.get('model_id')!r}; "
                    "prepare a fresh experiment directory for a different model."
                )
    jobs = [j for j in plan["jobs"] if not (Path(j["workspace"]) / "transcript.json").exists()]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.backend == "work":
        # Work owns the actual collaboration tool. Python cannot launch it itself.
        for job in jobs:
            print(
                json.dumps(
                    {
                        "job_id": job["job_id"],
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort or "medium",
                        "fork_turns": "none",
                        "service_tier_requested": args.service_tier or "priority",
                        "service_tier_advertised": args.work_advertised_tier,
                        "message": prompt_for(job),
                    }
                )
            )
        return
    if not args.base_url:
        p.error("--base-url is required for endpoint runs")
    if args.jobs < 1:
        p.error("--jobs must be positive")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, job, args) for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
