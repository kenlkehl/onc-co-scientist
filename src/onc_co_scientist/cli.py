"""Top-level Typer CLI: ``ocs``.

Three subcommand groups mirror the pipeline stages:

- ``ocs synth generate`` — produce a synthetic dataset bundle (Aim 1.1).
- ``ocs harness build-task`` — materialize a harness-agnostic task bundle (Aim 1.2).
- ``ocs score run`` — score a harness transcript against a dataset manifest (Aim 1.2).
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

from .harness.task_spec import build_task, build_tasks
from .harness.transcript import Transcript
from .scoring import (
    ClaudeCliJudge,
    Judge,
    JudgeCache,
    ReplicateScore,
    StubJudge,
    aggregate_batch,
    aggregate_replicates,
    default_cache_dir,
    score_buried,
    score_novelty,
    wrap_single,
    write_batch_report,
)
from .synthetic.cancer_types import CancerType, all_cancer_types
from .synthetic.generator import GeneratorConfig
from .synthetic.io import (
    ANONYMIZED_SUBDIR,
    MANIFEST_FILENAME,
    discover_bundles,
    load_column_mapping,
    read_manifest,
)
from .synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle,
    write_multi_bundle_pair,
)


class DatasetVariant(StrEnum):
    """How many variants of the dataset to materialize."""

    named = "named"
    anonymized = "anonymized"
    both = "both"


class JudgeBackend(StrEnum):
    """LLM backend powering novelty + match judgments."""

    claude_cli = "claude-cli"
    stub = "stub"

app = typer.Typer(
    help="Oncology Co-Scientist Benchmark CLI.",
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


def _load_generator_config(
    config_path: Path,
    seed_override: int | None,
    n_extra_covariates_override: int | None = None,
) -> GeneratorConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"Expected a YAML mapping at {config_path}, got {type(raw).__name__}."
        )
    if seed_override is not None:
        raw["seed"] = seed_override
    if n_extra_covariates_override is not None:
        raw["n_extra_covariates"] = n_extra_covariates_override
    return GeneratorConfig(**raw)


def _parse_cancer_types(raw: str) -> list[CancerType]:
    """Resolve ``--cancer-types`` into a list of ``CancerType`` enum values.

    ``"all"`` (case-insensitive) expands to every registered cancer type in
    declaration order. Otherwise the input is split on commas and each token
    is validated against ``CancerType``. Whitespace and case around tokens
    are tolerated; duplicates are de-duplicated while preserving order.
    """
    text = raw.strip()
    if not text or text.lower() == "all":
        return all_cancer_types()
    seen: set[CancerType] = set()
    chosen: list[CancerType] = []
    for token in text.split(","):
        name = token.strip().lower()
        if not name:
            continue
        try:
            ct = CancerType(name)
        except ValueError as exc:
            valid = ", ".join(c.value for c in all_cancer_types())
            raise typer.BadParameter(
                f"Unknown cancer type {name!r}. Valid choices: {valid} (or 'all')."
            ) from exc
        if ct not in seen:
            seen.add(ct)
            chosen.append(ct)
    if not chosen:
        raise typer.BadParameter(
            "No cancer types selected. Pass 'all' or a comma-separated list "
            "such as 'nsclc,crc'."
        )
    return chosen


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
        typer.Option(
            "--out",
            help="Output directory. One subfolder is written per cancer "
            "type (e.g. <out>/nsclc/, <out>/crc/, ...).",
        ),
    ],
    cancer_types: Annotated[
        str,
        typer.Option(
            "--cancer-types",
            help="Comma-separated cancer types to generate: nsclc, crc, "
            "breast, prostate, aml. Default 'all' generates every type. "
            "Each goes under its own <out>/<cancer_type>/ subfolder.",
        ),
    ] = "all",
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Override the seed from the config."),
    ] = None,
    n_extra_covariates: Annotated[
        int | None,
        typer.Option(
            "--n-extra-covariates",
            min=0,
            help="Override the number of realistic distractor covariates "
            "appended to the dataset (independent of outcomes). Max is the "
            "size of DEFAULT_DISTRACTOR_POOL in "
            "src/onc_co_scientist/synthetic/distractors.py.",
        ),
    ] = None,
    variant: Annotated[
        DatasetVariant,
        typer.Option(
            "--variant",
            help="Which variant(s) to materialize. 'both' writes named/ and "
            "anonymized/ subdirs under each cancer-type folder (same rows, "
            "same buried finding, only feature column names differ). "
            "'named' or 'anonymized' writes a single bundle directly into "
            "each cancer-type folder.",
            case_sensitive=False,
        ),
    ] = DatasetVariant.both,
    anon_seed: Annotated[
        int,
        typer.Option(
            "--anon-seed",
            help="Seed used to shuffle column-name assignments in the "
            "anonymized variant. Independent from the data-generation seed.",
        ),
    ] = 0,
    verbose: Annotated[bool, typer.Option("--verbose/--quiet")] = False,
) -> None:
    """Generate one or more synthetic dataset bundles, keyed by cancer type.

    By default this generates all five supported cancer types (NSCLC, CRC,
    breast, prostate, AML), each into its own subfolder of ``--out``. Pass
    ``--cancer-types nsclc,crc`` (etc.) to restrict the run to a subset.
    The base ``dataset_id`` from the YAML is auto-suffixed with the cancer
    type so each bundle's manifest carries a distinct identifier.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    selected = _parse_cancer_types(cancer_types)
    base_config = _load_generator_config(
        config, seed, n_extra_covariates_override=n_extra_covariates
    )
    bundles = generate_multi_dataset(base_config, selected)

    if variant is DatasetVariant.both:
        written = write_multi_bundle_pair(bundles, out, anon_seed=anon_seed)
        for ct, (named_dir, anon_dir) in written.items():
            bundle = bundles[ct]
            counts = _associations_by_class(bundle.manifest.associations)
            console.print(
                f"[green]Wrote[/green] [bold]{bundle.manifest.dataset_id}[/bold] "
                f"({ct.value}, n={bundle.manifest.patient_n}, "
                f"associations={counts})\n"
                f"  named:      {named_dir}\n"
                f"  anonymized: {anon_dir}"
            )
    else:
        anonymize = variant is DatasetVariant.anonymized
        written = write_multi_bundle(
            bundles, out, anonymize=anonymize, anon_seed=anon_seed
        )
        label = "anonymized" if anonymize else "named"
        for ct, out_path in written.items():
            bundle = bundles[ct]
            counts = _associations_by_class(bundle.manifest.associations)
            console.print(
                f"[green]Wrote[/green] {label} dataset "
                f"[bold]{bundle.manifest.dataset_id}[/bold] ({ct.value}) "
                f"to {out_path} (n={bundle.manifest.patient_n}, "
                f"associations={counts})"
            )


def _associations_by_class(associations) -> dict[str, int]:
    counts: dict[str, int] = {}
    for spec in associations:
        counts[spec.paradigm_class.value] = counts.get(spec.paradigm_class.value, 0) + 1
    return counts


@harness_app.command("build-task")
def harness_build_task(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            exists=True,
            file_okay=False,
            help="A dataset bundle directory (single-bundle mode) or a synth "
            "output root containing one or more bundles in subfolders "
            "(batch mode — one task bundle is written per discovered bundle, "
            "mirroring the relative path under --out).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Directory to write the harness task bundle(s) into. In batch "
            "mode, per-bundle task dirs are written under here mirroring the "
            "input tree (e.g. <out>/nsclc/anonymized/).",
        ),
    ],
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", "-n", min=1, help="Iteration cap N for the agent."),
    ] = 5,
    python_env: Annotated[
        Path | None,
        typer.Option(
            "--python-env",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Path to a uv-managed Python environment the agent should use for "
            "code execution. Embedded verbatim in agent_instructions.md.",
        ),
    ] = None,
) -> None:
    """Build an agent-facing task bundle (brief, schema, example, dataset copy).

    If ``--dataset`` points at a single bundle (a directory containing
    ``manifest.json``), one task bundle is written into ``--out``. Otherwise
    ``--dataset`` is treated as a synth output root and one task bundle is
    written per discovered bundle, mirroring the input tree under ``--out``.
    """
    if (dataset / MANIFEST_FILENAME).is_file():
        task = build_task(
            dataset, out, max_iterations=max_iterations, python_env=python_env
        )
        console.print(
            f"[green]Wrote[/green] task bundle to {task.task_dir}\n"
            f"  instructions: {task.instructions_path}\n"
            f"  schema:       {task.schema_path}\n"
            f"  example:      {task.example_path}\n"
            f"  dataset:      {task.dataset_path}\n"
            f"  description:  {task.description_path}"
        )
        return

    try:
        tasks = build_tasks(
            dataset, out, max_iterations=max_iterations, python_env=python_env
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"{exc} Point --dataset at a bundle directory, or run "
            f"`ocs synth generate` first."
        ) from exc
    for task in tasks:
        console.print(f"[green]Wrote[/green] task bundle to {task.task_dir}")
    console.print(
        f"[green]Built {len(tasks)} task bundle(s)[/green] under {out}"
    )


def _build_judge(
    backend: JudgeBackend,
    *,
    judge_cli: str,
    batch_size: int,
    cache_dir: Path | None,
    stub_config_path: Path | None,
) -> Judge:
    if backend is JudgeBackend.stub:
        if stub_config_path is None:
            return StubJudge()
        raw = json.loads(stub_config_path.read_text(encoding="utf-8"))
        novel_phrases = frozenset(raw.get("novel_phrases", []))
        match_phrases_raw = raw.get("match_phrases", {})
        match_phrases = {
            key: frozenset(values) for key, values in match_phrases_raw.items()
        }
        return StubJudge(
            novel_phrases=novel_phrases, match_phrases=match_phrases
        )
    cache = JudgeCache(cache_dir=cache_dir)
    return ClaudeCliJudge(
        cli=judge_cli,
        batch_size=batch_size,
        cache=cache,
    )


def _score_replicate(
    manifest,
    transcript: Transcript,
    judge: Judge,
    *,
    variant: str,
    column_mapping: dict[str, str] | None = None,
) -> ReplicateScore:
    # Novelty is paradigm-consensus-anchored and therefore meaningless on
    # anonymized hypotheses (which only mention feature_NNN columns); only
    # buried matching applies to the anonymized variant.
    novelty = score_novelty(transcript, judge) if variant == "named" else None
    buried = score_buried(
        manifest, transcript, judge, variant=variant, column_mapping=column_mapping
    )
    return ReplicateScore(
        dataset_id=manifest.dataset_id,
        variant=variant,
        model_id=transcript.model_id,
        harness_id=transcript.harness_id,
        max_iterations=transcript.max_iterations,
        novelty=novelty,
        buried=buried,
    )


def _infer_variant(bundle_dir: Path) -> str:
    """Variant is named/anonymized based on the bundle directory's name."""
    if bundle_dir.name == ANONYMIZED_SUBDIR:
        return "anonymized"
    return "named"


JudgeOption = Annotated[
    JudgeBackend,
    typer.Option(
        "--judge",
        help="LLM backend for novelty + match judgments. 'claude-cli' shells "
        "out to `claude --dangerously-skip-permissions -p` (uses existing "
        "Claude Code auth on the host). 'stub' is a deterministic test-only "
        "backend driven by --stub-config.",
        case_sensitive=False,
    ),
]
JudgeCliOption = Annotated[
    str,
    typer.Option(
        "--judge-cli",
        help="Path to the claude binary (only used when --judge=claude-cli).",
    ),
]
JudgeBatchSizeOption = Annotated[
    int,
    typer.Option(
        "--judge-batch-size",
        min=1,
        help="How many hypotheses to bundle into one judge call.",
    ),
]
CacheDirOption = Annotated[
    Path | None,
    typer.Option(
        "--cache-dir",
        help="Disk cache directory for judge responses. Default "
        "~/.cache/onc-co-scientist/judge.",
    ),
]
NoJudgeCacheOption = Annotated[
    bool,
    typer.Option(
        "--no-judge-cache/--judge-cache",
        help="Disable the on-disk judge cache (forces every call to hit the LLM).",
    ),
]
StubConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--stub-config",
        exists=True,
        dir_okay=False,
        help="JSON config for --judge=stub: "
        '{"novel_phrases": [...], "match_phrases": {"<assoc-key>": [...]}}.',
    ),
]


def _resolve_cache_dir(
    cache_dir: Path | None, no_cache: bool
) -> Path | None:
    if no_cache:
        return None
    return cache_dir if cache_dir is not None else default_cache_dir()


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
    judge_backend: JudgeOption = JudgeBackend.claude_cli,
    judge_cli: JudgeCliOption = "claude",
    judge_batch_size: JudgeBatchSizeOption = 10,
    cache_dir: CacheDirOption = None,
    no_judge_cache: NoJudgeCacheOption = False,
    stub_config: StubConfigOption = None,
) -> None:
    """Score a single transcript: novelty % + buried-finding discovery.

    Variant is inferred from the bundle directory's name (``named`` or
    ``anonymized``); novelty scoring is skipped for the anonymized variant.
    """
    manifest = read_manifest(dataset)
    transcript = Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    variant = _infer_variant(dataset)
    column_mapping = load_column_mapping(dataset)
    judge = _build_judge(
        judge_backend,
        judge_cli=judge_cli,
        batch_size=judge_batch_size,
        cache_dir=_resolve_cache_dir(cache_dir, no_judge_cache),
        stub_config_path=stub_config,
    )
    replicate = _score_replicate(
        manifest, transcript, judge, variant=variant, column_mapping=column_mapping
    )
    batch = wrap_single(replicate)
    out_path = write_batch_report(batch, out)
    console.print_json(json.dumps(batch.to_dict()))
    console.print(f"[green]Report written to[/green] {out_path}")


@score_app.command("batch")
def score_batch(
    synth_root: Annotated[
        Path,
        typer.Option(
            "--synth-root",
            exists=True,
            file_okay=False,
            help="Synth output root containing one or more dataset bundles "
            "(each with manifest.json + public/dataset.parquet) in "
            "<ct>/<variant>/ subfolders.",
        ),
    ],
    tasks_root: Annotated[
        Path,
        typer.Option(
            "--tasks-root",
            exists=True,
            file_okay=False,
            help="Tasks root produced by `ocs harness build-task` and "
            "populated by `scripts/run_harness.sh`. Per-bundle replicate "
            "transcripts are read from "
            "<tasks-root>/<ct>/<variant>/runs/run_*/transcript.json. "
            "Both named and anonymized variants are scored for buried "
            "discovery; novelty is computed for named only.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the batch scoring report."),
    ],
    judge_backend: JudgeOption = JudgeBackend.claude_cli,
    judge_cli: JudgeCliOption = "claude",
    judge_batch_size: JudgeBatchSizeOption = 10,
    cache_dir: CacheDirOption = None,
    no_judge_cache: NoJudgeCacheOption = False,
    stub_config: StubConfigOption = None,
) -> None:
    """Batch-score every replicate transcript across a tasks tree.

    Per (dataset, variant, replicate):
      - novelty (named only): % of harness-proposed hypotheses the LLM
        judge marks as going beyond paradigm consensus.
      - buried-discovery iteration (both variants): earliest iteration the
        pipeline both proposed and tested a hypothesis matching the
        manifest's buried association; falls back to max_iterations if
        never uncovered.

    Both ``named`` and ``anonymized`` bundles are scored; the named-vs-
    anonymized gap on buried discovery is the primary outcome of the eval.
    """
    bundles = discover_bundles(synth_root)
    if not bundles:
        raise typer.BadParameter(
            f"No dataset bundles found under {synth_root}. Each must contain "
            f"manifest.json and public/dataset.parquet."
        )

    judge = _build_judge(
        judge_backend,
        judge_cli=judge_cli,
        batch_size=judge_batch_size,
        cache_dir=_resolve_cache_dir(cache_dir, no_judge_cache),
        stub_config_path=stub_config,
    )

    bundle_scores = []
    for bundle_dir in bundles:
        rel = bundle_dir.relative_to(synth_root)
        variant = _infer_variant(bundle_dir)
        runs_dir = tasks_root / rel / "runs"
        transcript_paths = sorted(runs_dir.glob("run_*/transcript.json"))
        if not transcript_paths:
            console.print(
                f"[yellow]warning:[/yellow] no transcripts under {runs_dir}; "
                f"skipping bundle {rel}"
            )
            continue
        manifest = read_manifest(bundle_dir)
        column_mapping = load_column_mapping(bundle_dir)
        replicate_scores = []
        for tp in transcript_paths:
            transcript = Transcript.model_validate_json(tp.read_text(encoding="utf-8"))
            replicate_scores.append(
                _score_replicate(
                    manifest,
                    transcript,
                    judge,
                    variant=variant,
                    column_mapping=column_mapping,
                )
            )
        bundle_scores.append(aggregate_replicates(replicate_scores))

    if not bundle_scores:
        raise typer.BadParameter(
            f"No replicate transcripts found under {tasks_root} for any of the "
            f"{len(bundles)} bundle(s) in {synth_root}."
        )

    batch = aggregate_batch(bundle_scores)
    out_path = write_batch_report(batch, out)
    console.print_json(json.dumps(batch.to_dict()))
    console.print(
        f"[green]Batch report written to[/green] {out_path} "
        f"({batch.n_bundles} bundle(s), {batch.n_replicates_total} replicate(s))"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
