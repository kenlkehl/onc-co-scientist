"""Gemini agent entry point for legacy run_harness.sh task bundles."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from .structured_runner import StructuredRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gemini-3.8-flash")
    p.add_argument("--project-id")
    p.add_argument("--location")
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--max-tool-calls", type=int, default=200)
    p.add_argument("--max-generated-tokens", type=int, default=100000)
    p.add_argument("--max-tokens-per-call", type=int, default=8192)
    p.add_argument("--python-timeout", type=float, default=300)
    p.add_argument("prompt", nargs="?", default="Follow agent_instructions.md.")
    a = p.parse_args()
    source = Path.cwd()
    # Keep code, model calls and iteration records with their replicate.
    run_dir = Path(os.environ.get("OCS_RUN_DIR", source / "gemini_run"))
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    for name in (
        "dataset.parquet",
        "dataset_description.md",
        "agent_instructions.md",
        "transcript_schema.json",
        "transcript_example.json",
        "metadata.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, workspace / name)
    instructions = (workspace / "agent_instructions.md").read_text()
    metadata_path = workspace / "metadata.json"
    if not metadata_path.exists():
        dataset = re.search(r"\*\*Dataset:\*\* `([^`]+)`", instructions)
        iterations = re.search(r"\*\*Maximum iterations \(N\):\*\* (\d+)", instructions)
        if not dataset or not iterations:
            raise ValueError("Task needs metadata.json or standard dataset/iteration headers")
        metadata_path.write_text(
            json.dumps({"dataset_id": dataset[1], "max_iterations": int(iterations[1])})
        )
    (workspace / "agent_instructions.md").write_text(instructions + "\n\n" + a.prompt)
    runner = StructuredRunner(
        workspace,
        provider="gemini-vertex",
        model=a.model,
        project_id=a.project_id,
        location=a.location,
        reasoning_effort=a.reasoning_effort,
        max_turns=a.max_turns,
        max_tool_calls=a.max_tool_calls,
        max_generated_tokens=a.max_generated_tokens,
        max_tokens_per_call=a.max_tokens_per_call,
        python_timeout=a.python_timeout,
        harness_id="gemini-structured-runner@1",
    )
    runner.run()
    for name in ("transcript.json", "analysis_summary.txt"):
        if (workspace / name).exists():
            shutil.copyfile(workspace / name, source / name)


if __name__ == "__main__":
    main()
