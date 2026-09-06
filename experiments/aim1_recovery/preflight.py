"""Validate prepared inputs without running a model or inspecting recovery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .score import sha


def validate_inputs(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    protocol = plan["protocol"]
    loose = protocol.get("prompt_style") == "claude-legacy-loose-v1"
    counts = Counter()
    for job in plan["jobs"]:
        ws = Path(job["workspace"])
        metadata = json.loads((ws / "metadata.json").read_text())
        expected_hashes = {
            "dataset.parquet": job["data_sha256"],
            "agent_instructions.md": job["instructions_sha256"],
            **job["public_input_sha256"],
        }
        for filename, expected in expected_hashes.items():
            if sha(ws / filename) != expected:
                raise ValueError(f"Input changed: {job['job_id']}/{filename}")
        for key in ("model_id", "reasoning_effort"):
            if metadata[key] != protocol[key]:
                raise ValueError(f"Model setting mismatch: {job['job_id']}/{key}")
        if metadata["service_tier"] != protocol["service_tier_requested"]:
            raise ValueError("Tier metadata differs from protocol")
        if metadata["fixed_research_budget"] is not (not loose):
            raise ValueError("Fixed research budget setting differs from prompt style")
        budget = protocol["iteration_cap" if loose else "fixed_research_budget"]
        if metadata["max_iterations"] != budget[job["family"]]:
            raise ValueError("Research budget mismatch")
        if (ws / "manifest.json").exists() or (ws / "evaluation.parquet").exists():
            raise ValueError("Evaluator files are present in a research workspace")
        instructions = (ws / "agent_instructions.md").read_text()
        if not loose and (
            "Stop when you have thoroughly" in instructions
            or "You may finish before" in instructions
        ):
            raise ValueError("Conflicting early-stop instructions remain")
        if not loose and ("research_step" not in instructions or "exactly" not in instructions):
            raise ValueError("Missing fixed-budget submission instructions")
        if loose:
            if metadata.get("harness_id") != protocol["harness_id"]:
                raise ValueError("Loose harness metadata mismatch")
            if metadata.get("prompt_style") != protocol["prompt_style"]:
                raise ValueError("Loose prompt metadata mismatch")
            if "Stop when you have thoroughly" not in instructions:
                raise ValueError("Missing legacy stopping rule")
            if "Fixed exploration budget" in instructions or "research_step" in instructions:
                raise ValueError("Structured workflow leaked into loose prompt")
        if (ws / "transcript.json").exists() or (ws / "iterations").exists():
            raise ValueError("Formal workspace already contains research")
        if not Path(metadata["python"]).is_file():
            raise ValueError("Specified Python interpreter does not exist")
        counts[f"{job['family']}/{job['task']}/{job['variant']}"] += 1
    for source in plan["sources"]:
        prefix = f"{source['family']}/{source['task']}"
        if counts[f"{prefix}/named"] != counts[f"{prefix}/anonymized"]:
            raise ValueError("Unpaired replicate counts")
    return {
        "input_validation": "passed",
        "n_prepared": len(plan["jobs"]),
        "counts": dict(sorted(counts.items())),
        "plan_sha256": sha(plan_path),
        "research_runs_started": 0,
        "model_preflight": "not_run",
        "next_gate": (
            "Separate setup pilot must verify actual research budget/stopping before formal "
            "dispatch; do not inspect recovery during setup"
        ),
        "requested_service_tier": protocol["service_tier_requested"],
        "tier_verification": "not_verified_by_this_input_check",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a new output filename to preserve previous preflight records")
    result = validate_inputs(a.plan)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
