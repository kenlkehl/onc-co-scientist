from pathlib import Path

from onc_open_mindedness.harness.transcript import Transcript
from onc_open_mindedness.scoring import aggregate_datasets, score_dataset
from onc_open_mindedness.synthetic.paradigms import (
    concordant_catalog,
    discordant_catalog,
    hidden_novel_catalog,
)
from onc_open_mindedness.synthetic.schemas import DatasetManifest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _scoring_manifest() -> DatasetManifest:
    """Build a manifest that mirrors the first concordant/discordant/hidden_novel specs."""
    spec_c = concordant_catalog()[0]
    spec_d = discordant_catalog()[0]  # TMB + IO, direction = -1
    spec_n = hidden_novel_catalog()[0]
    return DatasetManifest(
        dataset_id="score_ds",
        seed=0,
        patient_n=100,
        columns=["patient_id"],
        treatment_columns=["treatment_io", "treatment_x"],
        outcome_columns=["progression_free_months", "objective_response"],
        covariate_columns=["egfr_mutation", "tmb_high", "biomarker_z_high"],
        associations=[spec_c, spec_d, spec_n],
    )


def test_perfect_transcript_uncovers_every_association():
    manifest = _scoring_manifest()
    transcript = Transcript.model_validate_json(
        (FIXTURE_DIR / "perfect_transcript.json").read_text()
    )
    score = score_dataset(manifest, transcript)

    outcome_by_id = {o.association_id: o for o in score.per_association}
    assert outcome_by_id[manifest.associations[0].id].iteration_uncovered == 1
    assert outcome_by_id[manifest.associations[1].id].iteration_uncovered == 2
    assert outcome_by_id[manifest.associations[2].id].iteration_uncovered == 2

    assert score.mean_iterations_concordant == 1.0
    assert score.mean_iterations_discordant == 2.0
    assert score.paradigm_adherence == 1.0
    assert score.mean_iterations_hidden_novel == 2.0


def test_empty_transcript_uses_penalty_iteration():
    manifest = _scoring_manifest()
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="test-model",
        harness_id="test-harness@1.0",
        max_iterations=5,
        iterations=[],
    )
    score = score_dataset(manifest, transcript)
    # Every association falls back to max_iterations + 1 = 6.
    assert score.mean_iterations_concordant == 6.0
    assert score.mean_iterations_discordant == 6.0
    assert score.mean_iterations_hidden_novel == 6.0
    assert score.paradigm_adherence == 0.0


def test_proposed_without_supporting_analysis_does_not_count():
    manifest = _scoring_manifest()
    spec_c = manifest.associations[0]
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="test-model",
        harness_id="test-harness@1.0",
        max_iterations=5,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {
                        "id": "h1",
                        "text": spec_c.natural_language_description,
                        "kind": "novel",
                    }
                ],
                "analyses": [
                    # Same hypothesis but not significant - should not count.
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "null result",
                        "p_value": 0.4,
                        "effect_estimate": -0.1,
                        "significant": False,
                    }
                ],
            }
        ],
    )
    score = score_dataset(manifest, transcript)
    outcome_by_id = {o.association_id: o for o in score.per_association}
    concordant_outcome = outcome_by_id[spec_c.id]
    assert concordant_outcome.iteration_uncovered is None
    assert concordant_outcome.proposed_iteration == 1
    assert score.mean_iterations_concordant == score.penalty_iteration


def test_wrong_direction_analysis_does_not_count():
    manifest = _scoring_manifest()
    spec_c = manifest.associations[0]  # direction = -1
    transcript = Transcript(
        dataset_id="score_ds",
        model_id="test-model",
        harness_id="test-harness@1.0",
        max_iterations=3,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {"id": "h1", "text": spec_c.natural_language_description}
                ],
                "analyses": [
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "wrong-sign significant",
                        "p_value": 0.001,
                        "effect_estimate": +2.5,  # wrong direction
                        "significant": True,
                    }
                ],
            }
        ],
    )
    score = score_dataset(manifest, transcript)
    outcome_by_id = {o.association_id: o for o in score.per_association}
    assert outcome_by_id[spec_c.id].iteration_uncovered is None


def test_aggregate_datasets_averages_across_pipeline():
    manifest = _scoring_manifest()
    perfect = Transcript.model_validate_json(
        (FIXTURE_DIR / "perfect_transcript.json").read_text()
    )
    empty = Transcript(
        dataset_id="score_ds",
        model_id="test-model",
        harness_id="test-harness@1.0",
        max_iterations=5,
        iterations=[],
    )
    scores = [score_dataset(manifest, perfect), score_dataset(manifest, empty)]
    pipeline = aggregate_datasets(scores)

    # Metric (1) = mean(1.0, 6.0) = 3.5 ; Metric (2) = mean(2.0, 6.0) = 4.0
    assert pipeline.mean_iterations_concordant == 3.5
    assert pipeline.mean_iterations_discordant == 4.0
    # Metric (3) = (2) - (1) at pipeline level.
    assert pipeline.paradigm_adherence == 0.5
