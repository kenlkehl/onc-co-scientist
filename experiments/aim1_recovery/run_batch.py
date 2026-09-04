#!/usr/bin/env python3
"""Run prepared tasks against an endpoint, or export prompts for Work subagents."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from onc_co_scientist.harness.structured_runner import StructuredRunner, finalize_workspace


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
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--reasoning-effort")
    p.add_argument("--service-tier", choices=["priority", "fast"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--max-tool-calls", type=int, default=200)
    p.add_argument("--max-generated-tokens", type=int, default=100000)
    p.add_argument("--max-tokens-per-call", type=int, default=4096)
    p.add_argument("--python-timeout", type=float, default=30)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
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
