"""Tests for scoring/novelty.py."""

from __future__ import annotations

from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import StubJudge, score_novelty


def _make_transcript(*texts: str) -> Transcript:
    iterations = []
    for i, text in enumerate(texts, 1):
        iterations.append(
            {
                "index": i,
                "proposed_hypotheses": [{"id": f"h{i}", "text": text}],
                "analyses": [],
            }
        )
    return Transcript(
        dataset_id="ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=max(len(texts), 1),
        iterations=iterations,
    )


def test_score_novelty_counts_novel_hypotheses():
    judge = StubJudge(novel_phrases=frozenset({"buried", "feature_"}))
    transcript = _make_transcript(
        "EGFR mutation predicts response to osimertinib",  # not novel
        "feature_037 modifies pfs in advanced disease",  # novel
        "buried subgroup of patients responds to olaparib",  # novel
    )
    score = score_novelty(transcript, judge)
    assert score.n_total == 3
    assert score.n_novel == 2
    assert score.frac_novel == pytest_approx(2 / 3)
    # Iterations are preserved on judgments.
    assert [j.iteration for j in score.judgments] == [1, 2, 3]
    assert [j.is_novel for j in score.judgments] == [False, True, True]


def test_score_novelty_empty_transcript_returns_zero():
    judge = StubJudge()
    transcript = Transcript(
        dataset_id="ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=5,
        iterations=[],
    )
    score = score_novelty(transcript, judge)
    assert score.n_total == 0
    assert score.n_novel == 0
    assert score.frac_novel == 0.0
    assert score.judgments == []


def test_score_novelty_to_dict_round_trips():
    judge = StubJudge(novel_phrases=frozenset({"buried"}))
    transcript = _make_transcript("buried subgroup")
    score = score_novelty(transcript, judge)
    payload = score.to_dict()
    assert payload["n_total"] == 1
    assert payload["n_novel"] == 1
    assert payload["frac_novel"] == 1.0
    assert payload["judgments"][0]["is_novel"] is True


def pytest_approx(value: float, rel: float = 1e-9):
    """Inline approx helper to avoid importing pytest in this module."""

    class _Approx:
        def __init__(self, target: float) -> None:
            self.target = target

        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(other - self.target) <= rel

        def __repr__(self) -> str:
            return f"approx({self.target})"

    return _Approx(value)
