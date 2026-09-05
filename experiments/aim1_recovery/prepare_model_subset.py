"""Clone frozen task inputs for a fresh model comparison without copying research."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from experiments.aim1_recovery.preflight import validate_inputs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(source_plan: Path, out: Path, task: str, model: str, repeats: int) -> dict:
    if out.exists():
        raise ValueError("Use a new directory; preserve all existing runs")
    source = json.loads(source_plan.read_text())
    selected = [j for j in source["jobs"] if j["task"] == task and j["replicate"] <= repeats]
    assert Counter(j["variant"] for j in selected) == {"named": repeats, "anonymized": repeats}
    out.mkdir(parents=True)
    evaluator = out / "private" / task
    evaluator.parent.mkdir()
    shutil.copytree(Path(selected[0]["evaluator"]), evaluator)
    protocol = copy.deepcopy(source["protocol"])
    protocol.update(
        model_id=model,
        reasoning_effort="medium",
        backend="work",
        service_tier_requested="priority",
        service_tier_evidence=(
            "Work model catalog advertises priority; no per-response tier telemetry"
        ),
        tier_authorization=(
            "User requests Sol 5.6 medium; available Work model catalog advertises priority"
        ),
        preflight_design=(
            "One named and one masked NSCLC setup session, excluded from formal results"
        ),
        comparability=(
            "Fresh model comparison on unchanged archived task inputs and v2 research/scoring rules"
        ),
        preservation="All prior experiments, setup sessions, and archive versions remain preserved",
    )
    protocol.pop("setup_fix", None)
    jobs = []
    for number, original in enumerate(selected, 1):
        old = Path(original["workspace"])
        job = copy.deepcopy(original)
        job.update(job_id=f"job_{number:04}", source_job_id=original["job_id"], status="pending")
        workspace = out / "public" / job["job_id"]
        workspace.mkdir(parents=True)
        for filename in [
            "dataset.parquet",
            "agent_instructions.md",
            *original["public_input_sha256"],
        ]:
            shutil.copyfile(old / filename, workspace / filename)
        metadata = json.loads((workspace / "metadata.json").read_text())
        metadata.update(model_id=model, reasoning_effort="medium", job_id=job["job_id"])
        write(workspace / "metadata.json", metadata)
        example = json.loads((workspace / "transcript_example.json").read_text())
        example["model_id"] = model
        write(workspace / "transcript_example.json", example)
        assert sha(workspace / "dataset.parquet") == original["data_sha256"]
        assert sha(workspace / "agent_instructions.md") == original["instructions_sha256"]
        for filename in ["dataset_description.md", "transcript_schema.json"]:
            assert sha(workspace / filename) == original["public_input_sha256"][filename]
        job.update(workspace=str(workspace.resolve()), evaluator=str(evaluator.resolve()))
        job["public_input_sha256"] = {
            name: sha(workspace / name) for name in original["public_input_sha256"]
        }
        jobs.append(job)
    plan = {
        "protocol": protocol,
        "sources": [s for s in source["sources"] if s["task"] == task],
        "jobs": jobs,
    }
    write(out / "plan.json", plan)
    write(out / "protocol.json", protocol)
    write(
        out / "coordinator_state.json", {"jobs": {}, "created_utc": datetime.now(UTC).isoformat()}
    )
    write(out / "terminal_failures.json", {})
    verification = validate_inputs(out / "plan.json")
    verification.update(
        byte_identical_to_previous_batch=[
            "dataset.parquet",
            "agent_instructions.md",
            "dataset_description.md",
            "transcript_schema.json",
        ],
        changed_public_fields=[
            "metadata.model_id",
            "metadata.job_id",
            "transcript_example.model_id",
        ],
        source_plan_sha256=sha(source_plan),
        coordinator_prior_recovery_known=True,
        research_context_fork="none",
    )
    write(out / "input_preflight.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source_plan, args.out.resolve(), args.task, args.model, args.repeats)
        )
    )
