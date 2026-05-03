"""Tests for scoring/buried.py."""

from __future__ import annotations

from pathlib import Path

from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import MatchJudgment, StubJudge, score_buried
from onc_co_scientist.synthetic.paradigms import hidden_novel_catalog
from onc_co_scientist.synthetic.schemas import DatasetManifest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class _FixedMatchJudge:
    def __init__(self, judgment: MatchJudgment) -> None:
        self.judgment = judgment

    def judge_novelty(self, hypotheses: list[str]):
        raise NotImplementedError

    def judge_matches(self, hypotheses, spec, *, variant, column_mapping=None):
        return [self.judgment for _ in hypotheses]


def _buried_manifest() -> DatasetManifest:
    """Manifest with a single hidden_novel association: BRCA2 + olaparib."""
    spec = hidden_novel_catalog()[0]
    return DatasetManifest(
        dataset_id="score_ds",
        seed=0,
        patient_n=100,
        columns=["patient_id"],
        treatment_columns=["treatment_olaparib"],
        outcome_columns=["objective_response"],
        covariate_columns=["brca2_mutation"],
        associations=[spec],
    )


def test_perfect_transcript_uncovers_buried_at_iteration_2():
    manifest = _buried_manifest()
    transcript = Transcript.model_validate_json(
        (FIXTURE_DIR / "perfect_transcript.json").read_text()
    )
    # The buried hypothesis text in the fixture mentions "BRCA2", "olaparib",
    # and "exceptional" — wire those into the stub matcher.
    judge = StubJudge(
        match_phrases={
            "olaparib": frozenset({"olaparib"}),
            "BRCA2": frozenset({"olaparib"}),
            "buried": frozenset({"olaparib"}),
            "BRCA": frozenset({"olaparib"}),
        }
    )
    score = score_buried(manifest, transcript, judge)
    assert score.earliest_iteration_uncovered == 2
    assert score.score == 2
    assert score.uncovered is True
    assert score.recovery_level == "exact"
    assert score.recovery_iteration == 2
    # Match-judgment trace is populated for review: every transcript
    # hypothesis appears (matches True or False), with the stub rationale.
    discovery = score.per_association[0]
    assert len(discovery.match_judgments) == len(transcript.flat_hypotheses())
    assert any(j.matches for j in discovery.match_judgments)
    assert all(j.rationale for j in discovery.match_judgments)


def test_empty_transcript_falls_back_to_max_iterations():
    manifest = _buried_manifest()
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=5,
        iterations=[],
    )
    judge = StubJudge()
    score = score_buried(manifest, transcript, judge)
    assert score.earliest_iteration_uncovered is None
    assert score.score == 6
    assert score.uncovered is False
    assert score.recovery_level == "none"


def test_proposed_but_not_tested_does_not_count():
    manifest = _buried_manifest()
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=4,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {"id": "h1", "text": "olaparib helps a BRCA2 subgroup"}
                ],
                "analyses": [],  # no analysis at all
            }
        ],
    )
    judge = StubJudge(
        match_phrases={
            "olaparib": frozenset({"olaparib"}),
            "BRCA2": frozenset({"olaparib"}),
        }
    )
    score = score_buried(manifest, transcript, judge)
    assert score.earliest_iteration_uncovered is None
    assert score.score == 5
    assert score.per_association[0].proposed_iteration == 1
    assert score.per_association[0].tested_iteration is None
    assert score.recovery_level == "none"


def test_wrong_direction_analysis_does_not_count():
    manifest = _buried_manifest()
    spec = manifest.associations[0]
    wrong_sign = -spec.direction if spec.direction != 0 else 0
    if wrong_sign == 0:
        # Spec is direction-agnostic; this test isn't meaningful here.
        return
    # +1 means we need positive effect_estimate; -1 means negative.
    bad_estimate = -1.0 if spec.direction > 0 else 1.0
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=3,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {"id": "h1", "text": "olaparib helps a BRCA2 subgroup"}
                ],
                "analyses": [
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "wrong-sign significant",
                        "p_value": 0.001,
                        "effect_estimate": bad_estimate,
                        "significant": True,
                    }
                ],
            }
        ],
    )
    judge = StubJudge(
        match_phrases={
            "olaparib": frozenset({"olaparib"}),
            "BRCA2": frozenset({"olaparib"}),
        }
    )
    score = score_buried(manifest, transcript, judge)
    assert score.earliest_iteration_uncovered is None
    assert score.score == 4


def test_near_recovery_is_reported_without_exact_uncovered():
    manifest = _buried_manifest()
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=4,
        iterations=[
            {
                "index": 2,
                "proposed_hypotheses": [
                    {"id": "h1", "text": "near olaparib subgroup hypothesis"}
                ],
                "analyses": [
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "direction-correct significant support",
                        "p_value": 0.001,
                        "effect_estimate": 1.0,
                        "significant": True,
                    }
                ],
            }
        ],
    )
    judge = _FixedMatchJudge(
        MatchJudgment(
            matches=False,
            recovery_level="near",
            rationale="correct driver/outcome and most subgroup structure",
        )
    )

    score = score_buried(manifest, transcript, judge)

    assert score.uncovered is False
    assert score.score == 5
    assert score.recovery_level == "near"
    assert score.recovery_iteration == 2
    assert score.near_or_better is True
    assert score.component_or_better is True


def test_component_recovery_accepts_significant_partial_signal():
    manifest = _buried_manifest()
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=4,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {"id": "h1", "text": "component olaparib modifier"}
                ],
                "analyses": [
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "significant partial modifier",
                        "p_value": 0.001,
                        "effect_estimate": -1.0,
                        "significant": True,
                    }
                ],
            }
        ],
    )
    judge = _FixedMatchJudge(
        MatchJudgment(
            matches=False,
            recovery_level="component",
            rationale="correct driver/outcome with one modifier",
        )
    )

    score = score_buried(manifest, transcript, judge)

    assert score.uncovered is False
    assert score.score == 5
    assert score.recovery_level == "component"
    assert score.recovery_iteration == 1
    assert score.near_or_better is False
    assert score.component_or_better is True


def test_manifest_with_no_hidden_novel_yields_max_iterations_score():
    manifest = DatasetManifest(
        dataset_id="score_ds",
        seed=0,
        patient_n=100,
        columns=["patient_id"],
        treatment_columns=[],
        outcome_columns=["pfs_months"],
        covariate_columns=[],
        associations=[],  # nothing
    )
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="m",
        harness_id="h@1",
        max_iterations=4,
        iterations=[],
    )
    score = score_buried(manifest, transcript, StubJudge())
    assert score.score == 5
    assert score.per_association == []


def test_dataset_id_mismatch_raises():
    manifest = _buried_manifest()
    transcript = Transcript(
        dataset_id="OTHER",
        model_id="m",
        harness_id="h@1",
        max_iterations=3,
        iterations=[],
    )
    try:
        score_buried(manifest, transcript, StubJudge())
    except ValueError as exc:
        assert "dataset_id" in str(exc)
    else:
        raise AssertionError("expected ValueError on mismatched dataset_ids")
