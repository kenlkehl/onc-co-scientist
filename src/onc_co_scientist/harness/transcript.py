"""Pydantic schema for the transcript an external agentic harness must emit.

The scoring pipeline (Aim 1.2 primary) walks iterations in order, and for each
ground-truth association asks: at which iteration did the harness (a) propose
a hypothesis semantically equivalent to this association AND (b) run a
statistical analysis that references that hypothesis with a significant,
correctly-signed result?

The schema is deliberately minimal and tolerant of extra fields so that the
diverse set of downstream harnesses (Claude Code, Codex, custom ReAct loops,
etc.) can annotate their transcripts freely without breaking scoring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HypothesisKind = Literal["novel", "refined"]


class HypothesisRecord(BaseModel):
    """One hypothesis proposed by the harness within an iteration."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Stable within a transcript.")
    text: str = Field(description="Natural-language statement of the hypothesis.")
    kind: HypothesisKind = "novel"


class AnalysisRecord(BaseModel):
    """One statistical analysis the harness performed to test hypotheses."""

    model_config = ConfigDict(extra="allow")

    hypothesis_ids: list[str] = Field(description="IDs of the hypotheses this analysis addresses.")
    code: str | None = Field(
        default=None, description="Optional: the code executed (for reasoning-trace audits)."
    )
    result_summary: str = Field(description="Free-text summary of the statistical result.")
    p_value: float | None = None
    effect_estimate: float | None = Field(
        default=None,
        description="Signed effect estimate on the outcome's natural scale. "
        "Scoring uses the sign to verify the direction matches ground truth.",
    )
    significant: bool | None = Field(
        default=None,
        description="Whether the harness deemed the result statistically significant. "
        "If omitted and p_value is present, scoring uses p < 0.05.",
    )


class IterationRecord(BaseModel):
    """One full iteration of the propose-analyze-refine loop."""

    model_config = ConfigDict(extra="allow")

    index: int = Field(ge=1, description="1-based iteration index.")
    proposed_hypotheses: list[HypothesisRecord]
    analyses: list[AnalysisRecord] = Field(default_factory=list)


class Transcript(BaseModel):
    """Top-level transcript emitted by an external harness.

    A harness should emit exactly one file named ``transcript.json`` conforming
    to this schema when it finishes a dataset.
    """

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    model_id: str = Field(description="Model used by the harness (e.g. 'claude-opus-4-7').")
    harness_id: str = Field(
        description="Identifier for the harness implementation and version "
        "(e.g. 'claude-code@2.3.1', 'codex-cli@0.8.0', 'custom-react@sha-deadbee')."
    )
    max_iterations: int = Field(ge=1)
    iterations: list[IterationRecord]

    def flat_hypotheses(self) -> list[tuple[int, HypothesisRecord]]:
        out: list[tuple[int, HypothesisRecord]] = []
        for it in self.iterations:
            for h in it.proposed_hypotheses:
                out.append((it.index, h))
        return out

    def flat_analyses(self) -> list[tuple[int, AnalysisRecord]]:
        out: list[tuple[int, AnalysisRecord]] = []
        for it in self.iterations:
            for a in it.analyses:
                out.append((it.index, a))
        return out
