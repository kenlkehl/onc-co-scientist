"""Scoring for the Oncology Scientific Open-Mindedness Benchmark (Aim 1.2)."""

from .aggregate import PipelineScore, aggregate_datasets
from .paradigm_metrics import DatasetScore, score_dataset
from .report import render_markdown, write_report

__all__ = [
    "DatasetScore",
    "PipelineScore",
    "aggregate_datasets",
    "render_markdown",
    "score_dataset",
    "write_report",
]
