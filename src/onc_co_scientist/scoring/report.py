"""Render scoring results as JSON + markdown + judgments JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from .aggregate import BatchPipelineScore, BundleScore, ReplicateScore


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_pair(value: float | None, sd: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if sd is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {sd:.{digits}f}"


def _sample_novel(replicate: ReplicateScore, *, k: int = 3) -> list[str]:
    if replicate.novelty is None:
        return []
    novel = [j.text for j in replicate.novelty.judgments if j.is_novel]
    return novel[:k]


def _group_by_dataset(
    bundles: list[BundleScore],
) -> list[tuple[str, list[BundleScore]]]:
    """Preserve first-seen order of dataset_ids, group their variants together."""
    order: list[str] = []
    by_id: dict[str, list[BundleScore]] = {}
    for b in bundles:
        if b.dataset_id not in by_id:
            by_id[b.dataset_id] = []
            order.append(b.dataset_id)
        by_id[b.dataset_id].append(b)
    # Within each group, named first then anonymized.
    variant_order = {"named": 0, "anonymized": 1}
    for ds in order:
        by_id[ds].sort(key=lambda b: variant_order.get(b.variant, 99))
    return [(ds, by_id[ds]) for ds in order]


def _render_variant_block(bundle: BundleScore) -> list[str]:
    lines: list[str] = []
    label = bundle.variant.capitalize()
    lines.append(f"#### {label} (n_replicates={bundle.n_replicates})")
    if bundle.variant == "named":
        lines.append(
            f"- frac_novel: {_fmt_pair(bundle.frac_novel_mean, bundle.frac_novel_sd)}"
        )
    lines.append(
        f"- buried_score: {_fmt_pair(bundle.buried_score_mean, bundle.buried_score_sd, digits=2)}"
    )
    lines.append(
        f"- replicates uncovered: {bundle.n_replicates_uncovered}/{bundle.n_replicates}"
    )
    lines.append("")
    if bundle.variant == "named":
        lines.append(
            "| replicate | model | harness | frac_novel | buried_score | uncovered@ | sample novel hypotheses |"
        )
        lines.append("|---|---|---|---|---|---|---|")
    else:
        lines.append("| replicate | model | harness | buried_score | uncovered@ |")
        lines.append("|---|---|---|---|---|")
    for i, rep in enumerate(bundle.replicates, 1):
        uncovered_at = (
            str(rep.buried.earliest_iteration_uncovered)
            if rep.buried.earliest_iteration_uncovered is not None
            else "—"
        )
        if bundle.variant == "named":
            sample = _sample_novel(rep)
            sample_md = "<br>".join(_md_escape(s) for s in sample) if sample else "—"
            lines.append(
                f"| {i:03d} | {rep.model_id} | {rep.harness_id} | "
                f"{_fmt(rep.frac_novel)} | {rep.buried_score} | "
                f"{uncovered_at} | {sample_md} |"
            )
        else:
            lines.append(
                f"| {i:03d} | {rep.model_id} | {rep.harness_id} | "
                f"{rep.buried_score} | {uncovered_at} |"
            )
    return lines


def render_markdown_batch(batch: BatchPipelineScore) -> str:
    lines: list[str] = []
    lines.append("# Oncology Co-Scientist Benchmark — Batch Scoring Report")
    lines.append("")
    lines.append(
        f"- **Bundles scored:** {batch.n_bundles} "
        f"({batch.n_bundles_named} named, {batch.n_bundles_anonymized} anonymized)"
    )
    lines.append(f"- **Replicates (total):** {batch.n_replicates_total}")
    lines.append(
        f"- **Novelty %** (named only, unweighted mean of bundle means): "
        f"{_fmt(batch.frac_novel)}"
    )
    lines.append(
        f"- **Buried discovery iteration — named** (lower = uncovers earlier; "
        f"falls back to max_iterations if never): {_fmt(batch.buried_score_named)}"
    )
    lines.append(
        f"- **Buried discovery iteration — anonymized:** "
        f"{_fmt(batch.buried_score_anonymized)}"
    )
    lines.append(
        f"- **Fraction of replicates uncovering buried — named:** "
        f"{_fmt(batch.fraction_uncovered_named)}"
    )
    lines.append(
        f"- **Fraction of replicates uncovering buried — anonymized:** "
        f"{_fmt(batch.fraction_uncovered_anonymized)}"
    )
    lines.append("")
    lines.append("## Per-bundle detail (mean ± SD across replicates)")
    for dataset_id, group in _group_by_dataset(batch.per_bundle):
        lines.append("")
        lines.append(f"### {dataset_id}")
        for bundle in group:
            lines.append("")
            lines.extend(_render_variant_block(bundle))
    lines.append("")
    return "\n".join(lines)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_batch_report(batch: BatchPipelineScore, out_dir: Path | str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "batch_score.json").write_text(
        json.dumps(batch.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (path / "batch_score.md").write_text(render_markdown_batch(batch), encoding="utf-8")
    _write_judgments_jsonl(batch, path / "batch_judgments.jsonl")
    return path


def _write_judgments_jsonl(batch: BatchPipelineScore, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for bundle in batch.per_bundle:
            for rep_index, rep in enumerate(bundle.replicates, 1):
                if rep.novelty is None:
                    continue
                for j in rep.novelty.judgments:
                    f.write(
                        json.dumps(
                            {
                                "dataset_id": bundle.dataset_id,
                                "variant": bundle.variant,
                                "replicate": rep_index,
                                "model_id": rep.model_id,
                                "harness_id": rep.harness_id,
                                "iteration": j.iteration,
                                "hypothesis_id": j.hypothesis_id,
                                "text": j.text,
                                "is_novel": j.is_novel,
                                "rationale": j.rationale,
                            }
                        )
                        + "\n"
                    )


def wrap_single(replicate: ReplicateScore) -> BatchPipelineScore:
    """Wrap a single ``ReplicateScore`` into a 1-bundle, 1-replicate batch.

    Used by ``ocs score run`` so the same renderer + JSON shape applies to
    one-off scoring as to batch scoring.
    """
    from .aggregate import aggregate_batch, aggregate_replicates

    bundle = aggregate_replicates([replicate])
    return aggregate_batch([bundle])
