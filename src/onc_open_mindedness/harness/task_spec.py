"""Materialize a harness-agnostic task bundle from a dataset bundle."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..synthetic.io import (
    DATASET_FILENAME,
    public_dir,
    read_description,
    read_manifest,
)
from .transcript import Transcript

TEMPLATE_DIR = Path(__file__).parent / "templates"
AGENT_INSTRUCTIONS_TEMPLATE = "agent_instructions.md.j2"
TRANSCRIPT_EXAMPLE = "transcript_example.json"
INSTRUCTIONS_FILENAME = "agent_instructions.md"
SCHEMA_FILENAME = "transcript_schema.json"
EXAMPLE_FILENAME = "transcript_example.json"
TASK_DATASET_LINK = "dataset.parquet"
TASK_DESCRIPTION_LINK = "dataset_description.md"


@dataclass
class TaskBundle:
    """Paths written into the harness task directory."""

    task_dir: Path
    instructions_path: Path
    schema_path: Path
    example_path: Path
    dataset_path: Path
    description_path: Path


def _render_instructions(
    dataset_id: str,
    patient_n: int,
    max_iterations: int,
    dataset_relpath: str,
    description_relpath: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(AGENT_INSTRUCTIONS_TEMPLATE)
    return template.render(
        dataset_id=dataset_id,
        patient_n=patient_n,
        max_iterations=max_iterations,
        dataset_relpath=dataset_relpath,
        description_relpath=description_relpath,
    )


def _write_schema(out_path: Path) -> None:
    """Write the JSON schema for the Transcript model alongside the task."""
    schema = Transcript.model_json_schema()
    out_path.write_text(json.dumps(schema, indent=2) + "\n")


def build_task(
    dataset_dir: Path | str,
    out_dir: Path | str,
    *,
    max_iterations: int = 5,
) -> TaskBundle:
    """Build a ``TaskBundle`` that an external harness can execute against.

    The ground-truth manifest is deliberately *not* copied into ``out_dir`` —
    agents see only the tabular data and the public description.
    """
    dataset_path_in = Path(dataset_dir)
    task_dir = Path(out_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(dataset_path_in)
    description = read_description(dataset_path_in)

    # Copy public artifacts into task_dir so the bundle is self-contained.
    dataset_dst = task_dir / TASK_DATASET_LINK
    shutil.copyfile(public_dir(dataset_path_in) / DATASET_FILENAME, dataset_dst)
    description_dst = task_dir / TASK_DESCRIPTION_LINK
    description_dst.write_text(description if description.endswith("\n") else description + "\n")

    # Render the agent brief.
    instructions = _render_instructions(
        dataset_id=manifest.dataset_id,
        patient_n=manifest.patient_n,
        max_iterations=max_iterations,
        dataset_relpath=TASK_DATASET_LINK,
        description_relpath=TASK_DESCRIPTION_LINK,
    )
    instructions_path = task_dir / INSTRUCTIONS_FILENAME
    instructions_path.write_text(instructions)

    # Schema + reference example for the transcript format.
    schema_path = task_dir / SCHEMA_FILENAME
    _write_schema(schema_path)
    example_src = TEMPLATE_DIR / TRANSCRIPT_EXAMPLE
    example_path = task_dir / EXAMPLE_FILENAME
    shutil.copyfile(example_src, example_path)

    return TaskBundle(
        task_dir=task_dir,
        instructions_path=instructions_path,
        schema_path=schema_path,
        example_path=example_path,
        dataset_path=dataset_dst,
        description_path=description_dst,
    )
