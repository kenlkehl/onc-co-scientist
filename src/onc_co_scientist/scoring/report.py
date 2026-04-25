"""Render scoring results as JSON + markdown."""

from __future__ import annotations

import json
from pathlib import Path

from .aggregate import PipelineScore
from .paradigm_metrics import DatasetScore


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def render_markdown(pipeline: PipelineScore) -> str:
    lines: list[str] = []
    lines.append("# Oncology Co-Scientist Benchmark — Scoring Report")
    lines.append("")
    lines.append(f"- **Datasets scored:** {pipeline.n_datasets}")
    lines.append(
        f"- **Metric (1)** mean iterations, paradigm-concordant associations: "
        f"{_fmt(pipeline.mean_iterations_concordant)}"
    )
    lines.append(
        f"- **Metric (2)** mean iterations, paradigm-discordant associations: "
        f"{_fmt(pipeline.mean_iterations_discordant)}"
    )
    lines.append(
        f"- **Metric (3)** paradigm adherence `(2) - (1)` (lower = more flexible across paradigms): "
        f"{_fmt(pipeline.paradigm_adherence)}"
    )
    lines.append(
        f"- Mean iterations, hidden-novel associations (exploratory): "
        f"{_fmt(pipeline.mean_iterations_hidden_novel)}"
    )
    lines.append("")
    lines.append("## Per-dataset detail")
    for ds in pipeline.per_dataset:
        lines.append("")
        lines.append(f"### {ds.dataset_id} — {ds.model_id} via {ds.harness_id}")
        lines.append(
            f"- max_iterations={ds.max_iterations}, penalty_iteration={ds.penalty_iteration}"
        )
        lines.append(
            f"- concordant={_fmt(ds.mean_iterations_concordant)}, "
            f"discordant={_fmt(ds.mean_iterations_discordant)}, "
            f"hidden_novel={_fmt(ds.mean_iterations_hidden_novel)}, "
            f"adherence={_fmt(ds.paradigm_adherence)}"
        )
        lines.append("")
        lines.append("| association | class | uncovered@ | proposed@ | tested@ | notes |")
        lines.append("|---|---|---|---|---|---|")
        for o in ds.per_association:
            lines.append(
                f"| {o.association_id} | {o.paradigm_class.value} | "
                f"{o.iteration_uncovered if o.iteration_uncovered is not None else '-'} | "
                f"{o.proposed_iteration if o.proposed_iteration is not None else '-'} | "
                f"{o.tested_iteration if o.tested_iteration is not None else '-'} | "
                f"{o.notes or ''} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(pipeline: PipelineScore, out_dir: Path | str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "score.json").write_text(json.dumps(pipeline.to_dict(), indent=2) + "\n")
    (path / "score.md").write_text(render_markdown(pipeline))
    return path


def write_dataset_report(score: DatasetScore, out_dir: Path | str) -> Path:
    """Convenience wrapper for single-dataset scoring (the MVP CLI path)."""
    pipeline = PipelineScore(
        n_datasets=1,
        mean_iterations_concordant=score.mean_iterations_concordant,
        mean_iterations_discordant=score.mean_iterations_discordant,
        mean_iterations_hidden_novel=score.mean_iterations_hidden_novel,
        paradigm_adherence=score.paradigm_adherence,
        per_dataset=[score],
    )
    return write_report(pipeline, out_dir)
