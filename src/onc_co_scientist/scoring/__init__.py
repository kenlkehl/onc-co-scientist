"""Scoring for the Oncology Co-Scientist Benchmark (Aim 1.2)."""

from .aggregate import (
    BatchPipelineScore,
    BundleScore,
    ReplicateScore,
    aggregate_batch,
    aggregate_replicates,
)
from .buried import BuriedDiscovery, BuriedScore, MatchJudgmentRecord, score_buried
from .judge import (
    AnthropicVertexJudge,
    ClaudeCliJudge,
    CodexCliJudge,
    Judge,
    JudgeCache,
    MatchJudgment,
    NoveltyJudgment,
    StubJudge,
    default_cache_dir,
)
from .novelty import NoveltyJudgmentRecord, NoveltyScore, score_novelty
from .report import (
    render_markdown_batch,
    wrap_single,
    write_batch_report,
)

__all__ = [
    "AnthropicVertexJudge",
    "BatchPipelineScore",
    "BundleScore",
    "BuriedDiscovery",
    "BuriedScore",
    "ClaudeCliJudge",
    "CodexCliJudge",
    "Judge",
    "JudgeCache",
    "MatchJudgment",
    "MatchJudgmentRecord",
    "NoveltyJudgment",
    "NoveltyJudgmentRecord",
    "NoveltyScore",
    "ReplicateScore",
    "StubJudge",
    "aggregate_batch",
    "aggregate_replicates",
    "default_cache_dir",
    "render_markdown_batch",
    "score_buried",
    "score_novelty",
    "wrap_single",
    "write_batch_report",
]
