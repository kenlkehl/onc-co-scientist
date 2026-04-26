"""Scoring for the Oncology Co-Scientist Benchmark (Aim 1.2)."""

from .aggregate import (
    BatchPipelineScore,
    BundleScore,
    PipelineScore,
    aggregate_batch,
    aggregate_datasets,
    aggregate_replicates,
)
from .paradigm_metrics import DatasetScore, score_dataset
from .report import (
    render_markdown,
    render_markdown_batch,
    write_batch_report,
    write_report,
)

__all__ = [
    "BatchPipelineScore",
    "BundleScore",
    "DatasetScore",
    "PipelineScore",
    "aggregate_batch",
    "aggregate_datasets",
    "aggregate_replicates",
    "render_markdown",
    "render_markdown_batch",
    "score_dataset",
    "write_batch_report",
    "write_report",
]
