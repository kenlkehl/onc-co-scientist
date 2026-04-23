"""Harness-agnostic task specification and transcript schema (Aim 1.2).

This package builds the instructions bundle that an external agentic harness
(Claude Code, Codex, a custom loop, etc.) consumes, and defines the canonical
JSON transcript format that the harness must emit for scoring.
"""

from .task_spec import TaskBundle, build_task
from .transcript import (
    AnalysisRecord,
    HypothesisRecord,
    IterationRecord,
    Transcript,
)

__all__ = [
    "AnalysisRecord",
    "HypothesisRecord",
    "IterationRecord",
    "TaskBundle",
    "Transcript",
    "build_task",
]
