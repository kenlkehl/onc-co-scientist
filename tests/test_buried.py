"""Tests for scoring/buried.py."""

from __future__ import annotations

from pathlib import Path

from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import StubJudge, score_buried
from onc_co_scientist.synthetic.paradigms import hidden_novel_catalog
from onc_co_scientist.synthetic.schemas import DatasetManifest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


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
    assert score.score == 5
    assert score.uncovered is False


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
    assert score.score == 4
    assert score.per_association[0].proposed_iteration == 1
    assert score.per_association[0].tested_iteration is None


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
    assert score.score == 3


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
    assert score.score == 4
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
