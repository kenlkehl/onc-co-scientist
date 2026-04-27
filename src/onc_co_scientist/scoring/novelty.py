"""Novelty scoring: per-hypothesis LLM judgment, aggregated to fraction novel.

A hypothesis is "novel" iff it goes beyond established oncology paradigm
consensus for the cancer type at hand. The judgment is delegated to the
``Judge`` interface (``scoring/judge.py``); this module only handles
flattening the transcript and bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..harness.transcript import Transcript
from .judge import Judge


@dataclass
class NoveltyJudgmentRecord:
    iteration: int
    hypothesis_id: str
    text: str
    is_novel: bool
    rationale: str


@dataclass
class NoveltyScore:
    n_total: int
    n_novel: int
    frac_novel: float
    judgments: list[NoveltyJudgmentRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_novel": self.n_novel,
            "frac_novel": self.frac_novel,
            "judgments": [
                {
                    "iteration": j.iteration,
                    "hypothesis_id": j.hypothesis_id,
                    "text": j.text,
                    "is_novel": j.is_novel,
                    "rationale": j.rationale,
                }
                for j in self.judgments
            ],
        }


def score_novelty(transcript: Transcript, judge: Judge) -> NoveltyScore:
    """Judge every hypothesis in the transcript and return the aggregate."""
    flat = transcript.flat_hypotheses()
    if not flat:
        return NoveltyScore(n_total=0, n_novel=0, frac_novel=0.0, judgments=[])

    texts = [h.text for _, h in flat]
    judgments = judge.judge_novelty(texts)
    if len(judgments) != len(flat):
        raise ValueError(
            f"Judge returned {len(judgments)} judgments for {len(flat)} hypotheses."
        )

    records: list[NoveltyJudgmentRecord] = []
    n_novel = 0
    for (iteration, hypothesis), judgment in zip(flat, judgments, strict=True):
        records.append(
            NoveltyJudgmentRecord(
                iteration=iteration,
                hypothesis_id=hypothesis.id,
                text=hypothesis.text,
                is_novel=judgment.is_novel,
                rationale=judgment.rationale,
            )
        )
        if judgment.is_novel:
            n_novel += 1
    n_total = len(records)
    return NoveltyScore(
        n_total=n_total,
        n_novel=n_novel,
        frac_novel=n_novel / n_total if n_total else 0.0,
        judgments=records,
    )
