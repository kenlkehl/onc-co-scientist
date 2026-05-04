"""Materialize a harness-agnostic task bundle from a dataset bundle."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..synthetic.io import (
    DATASET_FILENAME,
    discover_bundles,
    public_dir,
    read_description,
    read_manifest,
)
from .transcript import Transcript

TEMPLATE_DIR = Path(__file__).parent / "templates"
AGENT_INSTRUCTIONS_TEMPLATE = "agent_instructions.md.j2"
TRANSCRIPT_EXAMPLE = "transcript_example.json"
DEPMAP_TRANSCRIPT_EXAMPLE = "transcript_example_depmap.json"
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


@dataclass(frozen=True)
class TaskInstructionContext:
    title: str
    record_label: str
    role: str
    isolation: str
    protocol_scope: str
    heterogeneity_search: str
    good_hypothesis_example: str
    final_hypothesis: str


def _instruction_context(dataset_kind: str) -> TaskInstructionContext:
    if dataset_kind == "crispr_depmap":
        return TaskInstructionContext(
            title="CRISPR Dependency Map Analysis - Task Brief",
            record_label="Cell lines",
            role=(
                "You have been asked to analyze a cancer cell-line CRISPR "
                "knockout screen dataset in the style of DepMap/CCLE. Your "
                "job is to explore the dataset and surface biologically "
                "meaningful patterns - main effects, subgroup heterogeneity, "
                "and multivariable interactions among cell-line features and "
                "gene dependency outcomes - testing each pattern statistically "
                "and refining your ideas as evidence accumulates across "
                "iterations."
            ),
            isolation=(
                "You must not look outside the current working directory for "
                "additional data or context. Use only the local task bundle so "
                "every harness is evaluated against the same evidence."
            ),
            protocol_scope=(
                "relationships among cell-line features, molecular subgroups, "
                "screen covariates, and dependency-score outcomes"
            ),
            heterogeneity_search=(
                "Do not stop after main effects. Before your final transcript, "
                "run at least one systematic dependency-heterogeneity search "
                "for each dependency outcome. Appropriate approaches include "
                "feature-by-feature interaction screening, joint models over "
                "the strongest modifiers, tree/rule-based subgroup discovery, "
                "or exhaustive checks of small multi-feature subgroups. When "
                "a dependency appears concentrated in a subgroup, state and "
                "test the complete subgroup definition, including variables "
                "whose value appears to suppress or enable the dependency."
            ),
            good_hypothesis_example=(
                "In cell lines with `feature_example` set, "
                "`dependency_gene_example` is more negative than in cell lines "
                "without it"
            ),
            final_hypothesis=(
                "A final best-supported dependency subgroup hypothesis for "
                "each dependency outcome where one is plausible, naming the "
                "dependency outcome, direction, and all subgroup predicates "
                "you believe define the effect."
            ),
        )
    return TaskInstructionContext(
        title="Oncology Dataset Analysis - Task Brief",
        record_label="Patients",
        role=(
            "You have been asked to analyze a large oncology dataset assembled "
            "from electronic health records aggregated by a commercial "
            "healthcare data vendor. Your job is to explore the dataset and "
            "surface clinically meaningful patterns - main effects, subgroup "
            "heterogeneity, and multivariable interactions among the features "
            "and outcomes - testing each pattern statistically and refining "
            "your ideas as evidence accumulates across iterations."
        ),
        isolation=(
            "You must not look outside the current working directory for "
            "additional data or context. This is critical for patient privacy."
        ),
        protocol_scope=("relationships among features, subgroups, and outcomes"),
        heterogeneity_search=(
            "Do not stop after main effects. Before your final transcript, "
            "run at least one systematic treatment-effect heterogeneity search "
            "for each outcome with usable treatment variation. Appropriate "
            "approaches include treatment-by-feature interaction screening, "
            "joint models over the strongest modifiers, tree/rule-based "
            "subgroup discovery, or exhaustive checks of small multi-feature "
            "subgroups. When a treatment effect appears concentrated in a "
            "subgroup, state and test the complete subgroup definition, "
            "including variables whose unfavorable value appears to suppress "
            "the treatment effect."
        ),
        good_hypothesis_example=(
            "In patients with `feature_example` set, mean `outcome_example` "
            "is lower than in patients without it"
        ),
        final_hypothesis=(
            "A final best-supported treatment-effect subgroup hypothesis for "
            "each outcome where one is plausible, naming the treatment, "
            "outcome, direction, and all subgroup predicates you believe "
            "define the effect."
        ),
    )


def _render_instructions(
    dataset_id: str,
    patient_n: int,
    dataset_kind: str,
    max_iterations: int,
    dataset_relpath: str,
    description_relpath: str,
    python_env: str | None,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(AGENT_INSTRUCTIONS_TEMPLATE)
    context = _instruction_context(dataset_kind)
    return template.render(
        dataset_id=dataset_id,
        patient_n=patient_n,
        context=context,
        max_iterations=max_iterations,
        dataset_relpath=dataset_relpath,
        description_relpath=description_relpath,
        python_env=python_env,
    )


def _write_schema(out_path: Path) -> None:
    """Write the JSON schema for the Transcript model alongside the task."""
    schema = Transcript.model_json_schema()
    out_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def build_task(
    dataset_dir: Path | str,
    out_dir: Path | str,
    *,
    max_iterations: int = 5,
    python_env: Path | str | None = None,
) -> TaskBundle:
    """Build a ``TaskBundle`` that an external harness can execute against.

    The ground-truth manifest is deliberately *not* copied into ``out_dir`` —
    agents see only the tabular data and the public description.

    If ``python_env`` is provided, its absolute path is embedded in the agent
    brief so the agent runs code inside that uv-managed environment.
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
    description_dst.write_text(
        description if description.endswith("\n") else description + "\n",
        encoding="utf-8",
    )

    python_env_str = str(Path(python_env).resolve()) if python_env is not None else None

    # Render the agent brief.
    instructions = _render_instructions(
        dataset_id=manifest.dataset_id,
        patient_n=manifest.patient_n,
        dataset_kind=manifest.dataset_kind,
        max_iterations=max_iterations,
        dataset_relpath=TASK_DATASET_LINK,
        description_relpath=TASK_DESCRIPTION_LINK,
        python_env=python_env_str,
    )
    instructions_path = task_dir / INSTRUCTIONS_FILENAME
    instructions_path.write_text(instructions, encoding="utf-8")

    # Schema + reference example for the transcript format.
    schema_path = task_dir / SCHEMA_FILENAME
    _write_schema(schema_path)
    example_template = (
        DEPMAP_TRANSCRIPT_EXAMPLE
        if manifest.dataset_kind == "crispr_depmap"
        else TRANSCRIPT_EXAMPLE
    )
    example_src = TEMPLATE_DIR / example_template
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


def build_tasks(
    synth_root: Path | str,
    out_dir: Path | str,
    *,
    max_iterations: int = 5,
    python_env: Path | str | None = None,
) -> list[TaskBundle]:
    """Materialize one task bundle per discovered dataset under ``synth_root``.

    Walks ``synth_root`` for any directory containing a ``manifest.json`` and
    a ``public/dataset.parquet`` (the bundle layout written by ``ocs synth
    generate``), and writes a task bundle for each one under ``out_dir``,
    mirroring the bundle's path relative to ``synth_root``. ``max_iterations``
    and ``python_env`` are applied uniformly to every bundle.

    Raises ``ValueError`` if no bundles are found beneath ``synth_root``.
    """
    root_path = Path(synth_root)
    out_path = Path(out_dir)
    bundles = discover_bundles(root_path)
    if not bundles:
        raise ValueError(
            f"No dataset bundles found under {root_path}. Each bundle must "
            f"contain manifest.json and public/dataset.parquet."
        )
    tasks: list[TaskBundle] = []
    for bundle in bundles:
        rel = bundle.relative_to(root_path)
        tasks.append(
            build_task(
                bundle,
                out_path / rel,
                max_iterations=max_iterations,
                python_env=python_env,
            )
        )
    return tasks
