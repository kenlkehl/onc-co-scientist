"""Top-level Typer CLI: ``oom``.

Three subcommand groups mirror the pipeline stages:

- ``oom synth generate`` — produce a synthetic dataset bundle (Aim 1.1).
- ``oom harness build-task`` — materialize a harness-agnostic task bundle (Aim 1.2).
- ``oom score run`` — score a harness transcript against a dataset manifest (Aim 1.2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from .harness.task_spec import build_task
from .harness.transcript import Transcript
from .scoring import aggregate_datasets, score_dataset, write_report
from .synthetic.generator import GeneratorConfig, generate_dataset
from .synthetic.io import read_manifest, write_bundle

app = typer.Typer(
    help="Oncology Scientific Open-Mindedness Benchmark CLI.",
    no_args_is_help=True,
    add_completion=False,
)
synth_app = typer.Typer(help="Synthetic dataset generation (Aim 1.1).", no_args_is_help=True)
harness_app = typer.Typer(
    help="Harness task bundle builder (Aim 1.2).", no_args_is_help=True
)
score_app = typer.Typer(help="Transcript scoring (Aim 1.2).", no_args_is_help=True)
app.add_typer(synth_app, name="synth")
app.add_typer(harness_app, name="harness")
app.add_typer(score_app, name="score")

console = Console()


def _load_generator_config(config_path: Path, seed_override: int | None) -> GeneratorConfig:
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"Expected a YAML mapping at {config_path}, got {type(raw).__name__}."
        )
    if seed_override is not None:
        raw["seed"] = seed_override
    return GeneratorConfig(**raw)


@synth_app.command("generate")
def synth_generate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML config matching GeneratorConfig.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for the dataset bundle."),
    ],
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Override the seed from the config."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose/--quiet")] = False,
) -> None:
    """Generate a synthetic dataset bundle."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    gen_config = _load_generator_config(config, seed)
    bundle = generate_dataset(gen_config)
    out_path = write_bundle(bundle, out)
    counts = {
        klass.value: len(
            [a for a in bundle.manifest.associations if a.paradigm_class == klass]
        )
        for klass in {a.paradigm_class for a in bundle.manifest.associations}
    }
    console.print(
        f"[green]Wrote[/green] dataset [bold]{bundle.manifest.dataset_id}[/bold] "
        f"to {out_path} (n={bundle.manifest.patient_n}, associations={counts})"
    )


@harness_app.command("build-task")
def harness_build_task(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            exists=True,
            file_okay=False,
            help="Dataset bundle directory (written by `oom synth generate`).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory to write the harness task bundle into."),
    ],
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", "-n", min=1, help="Iteration cap N for the agent."),
    ] = 5,
) -> None:
    """Build an agent-facing task bundle (brief, schema, example, dataset copy)."""
    task = build_task(dataset, out, max_iterations=max_iterations)
    console.print(
        f"[green]Wrote[/green] task bundle to {task.task_dir}\n"
        f"  instructions: {task.instructions_path}\n"
        f"  schema:       {task.schema_path}\n"
        f"  example:      {task.example_path}\n"
        f"  dataset:      {task.dataset_path}\n"
        f"  description:  {task.description_path}"
    )


@score_app.command("run")
def score_run(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            exists=True,
            file_okay=False,
            help="Dataset bundle directory (provides the ground-truth manifest).",
        ),
    ],
    transcript_path: Annotated[
        Path,
        typer.Option(
            "--transcript",
            exists=True,
            dir_okay=False,
            help="Path to the transcript.json emitted by the external harness.",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Directory for the scoring report.")
    ],
    penalty_iteration: Annotated[
        int | None,
        typer.Option(
            "--penalty-iteration",
            help="Iteration value to assign when an association is never uncovered. "
            "Defaults to max_iterations + 1.",
        ),
    ] = None,
) -> None:
    """Score a transcript for one dataset and write JSON + markdown reports."""
    manifest = read_manifest(dataset)
    transcript = Transcript.model_validate_json(transcript_path.read_text())
    score = score_dataset(manifest, transcript, penalty_iteration=penalty_iteration)
    pipeline = aggregate_datasets([score])
    out_path = write_report(pipeline, out)
    console.print_json(json.dumps(pipeline.to_dict()))
    console.print(f"[green]Report written to[/green] {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
