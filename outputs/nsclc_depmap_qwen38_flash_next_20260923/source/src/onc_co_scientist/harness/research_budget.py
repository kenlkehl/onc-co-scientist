"""Truth-free checks for the fixed-iteration v2 research protocol.

These checks reject empty records and exact script reuse. They do not certify
scientific originality or equal token/compute use across agents.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .python_sandbox import analysis_root
from .transcript import IterationRecord

ACTIONS = {"screen", "multivariable", "refine", "robustness"}


def _artifact_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or not path.stat().st_size:
        raise ValueError("research artifact must be a nonempty file inside the workspace")
    return path


def validate_step(
    root: Path, record: IterationRecord, prior: list[dict], *, sequential_outputs: bool = False
) -> dict[str, str]:
    step = record.model_extra.get("research_step", {})
    artifacts = analysis_root(root)
    if not isinstance(step, dict) or step.get("action") not in ACTIONS:
        raise ValueError("research_step.action must be screen, multivariable, refine or robustness")
    if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
        raise ValueError("research_step.rationale must explain this iteration's research question")
    if not any(a.hypothesis_ids and a.code and a.code.strip() for a in record.analyses):
        raise ValueError("each research iteration needs an analysis linked to a hypothesis")
    hashes = {}
    for key in ("script_path", "output_path"):
        value = step.get(key)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"research_step.{key} must be a workspace-relative file path")
        path = _artifact_path(artifacts, value)
        hashes[value] = hashlib.sha256(path.read_bytes()).hexdigest()
    if step["script_path"] == step["output_path"]:
        raise ValueError("script and output must be different files")
    if not any(step["script_path"] in (a.code or "") for a in record.analyses):
        raise ValueError("analysis code must reference research_step.script_path")
    for previous in prior:
        old = previous.get("research_step", {}).get("script_path")
        if (
            old
            and hashlib.sha256(_artifact_path(artifacts, old).read_bytes()).hexdigest()
            == hashes[step["script_path"]]
        ):
            raise ValueError("exact script reuse does not count as a new research iteration")
        if sequential_outputs:
            old_output = previous.get("research_step", {}).get("output_path")
            if (
                old_output
                and _artifact_path(artifacts, old_output)
                == _artifact_path(artifacts, step["output_path"])
            ):
                raise ValueError("each iteration requires a separate saved output file")
    if sequential_outputs and record.index > 1:
        events = [
            json.loads(line) for line in (root / "submission_events.jsonl").read_text().splitlines()
        ]
        previous_event = next((e for e in events if e["index"] == record.index - 1), None)
        if previous_event is None:
            raise ValueError("previous submission receipt is missing")
        previous_time = datetime.fromisoformat(previous_event["utc"]).timestamp()
        if _artifact_path(artifacts, step["output_path"]).stat().st_mtime < previous_time - 0.01:
            raise ValueError(
                "analysis output predates the previous submission; execute this iteration's "
                "analysis after submitting the preceding iteration and save its own output"
            )
    return hashes


def validate_completion(metadata: dict, records: list[IterationRecord]) -> None:
    if not metadata.get("fixed_research_budget"):
        return
    if len(records) != int(metadata["max_iterations"]):
        raise ValueError(
            f"fixed research budget requires {metadata['max_iterations']} iterations; "
            f"only {len(records)} submitted"
        )
    actions = {r.model_extra.get("research_step", {}).get("action") for r in records}
    if missing := ACTIONS - actions:
        raise ValueError(f"missing research actions: {', '.join(sorted(missing))}")
