from pathlib import Path
from statistics import stdev

from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import (
    aggregate_batch,
    aggregate_datasets,
    aggregate_replicates,
    score_dataset,
)
from onc_co_scientist.synthetic.paradigms import (
    concordant_catalog,
    discordant_catalog,
    hidden_novel_catalog,
)
from onc_co_scientist.synthetic.schemas import DatasetManifest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _scoring_manifest() -> DatasetManifest:
    """Build a manifest that mirrors the first concordant/discordant/hidden_novel specs."""
    spec_c = concordant_catalog()[0]
    spec_d = discordant_catalog()[0]  # TMB + pembrolizumab, direction = -1
    spec_n = hidden_novel_catalog()[0]
    return DatasetManifest(
        dataset_id="score_ds",
        seed=0,
        patient_n=100,
        columns=["patient_id"],
        treatment_columns=["treatment_pembrolizumab", "treatment_olaparib"],
        outcome_columns=["pfs_months", "objective_response"],
        covariate_columns=["egfr_mutation", "tmb_high", "brca2_mutation"],
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


def _perfect_score():
    manifest = _scoring_manifest()
    perfect = Transcript.model_validate_json(
        (FIXTURE_DIR / "perfect_transcript.json").read_text()
    )
    return score_dataset(manifest, perfect)


def _empty_score():
    manifest = _scoring_manifest()
    empty = Transcript(
        dataset_id="score_ds",
        model_id="test-model",
        harness_id="test-harness@1.0",
        max_iterations=5,
        iterations=[],
    )
    return score_dataset(manifest, empty)


def test_aggregate_replicates_computes_mean_and_sd():
    scores = [_perfect_score(), _empty_score()]
    bundle = aggregate_replicates(scores)

    assert bundle.dataset_id == "score_ds"
    assert bundle.n_replicates == 2

    # concordant: mean of 1.0 and 6.0 = 3.5; sd = stdev([1.0, 6.0])
    assert bundle.mean_iterations_concordant_mean == 3.5
    assert bundle.mean_iterations_concordant_sd == stdev([1.0, 6.0])
    # discordant: mean of 2.0 and 6.0 = 4.0
    assert bundle.mean_iterations_discordant_mean == 4.0
    assert bundle.mean_iterations_discordant_sd == stdev([2.0, 6.0])
    # adherence: 1.0 (perfect: 2-1) and 0.0 (empty: 6-6) => mean 0.5
    assert bundle.paradigm_adherence_mean == 0.5
    assert bundle.paradigm_adherence_sd == stdev([1.0, 0.0])


def test_aggregate_replicates_single_run_sd_none():
    bundle = aggregate_replicates([_perfect_score()])
    assert bundle.n_replicates == 1
    assert bundle.mean_iterations_concordant_mean == 1.0
    assert bundle.mean_iterations_concordant_sd is None
    assert bundle.paradigm_adherence_sd is None


def test_aggregate_batch_means_of_bundle_means():
    # Bundle A: 3 replicates of "perfect" (each concordant=1.0)
    # Bundle B: 1 replicate of "empty" (concordant=6.0)
    # Pipeline mean of bundle means = (1.0 + 6.0) / 2 = 3.5,
    # NOT the replicate-weighted mean (3*1.0 + 1*6.0)/4 = 2.25.
    perfect_scores = [_perfect_score() for _ in range(3)]
    empty_scores = [_empty_score()]
    bundle_a = aggregate_replicates(perfect_scores)
    bundle_b = aggregate_replicates(empty_scores)

    batch = aggregate_batch([bundle_a, bundle_b])
    assert batch.n_bundles == 2
    assert batch.n_replicates_total == 4
    assert batch.mean_iterations_concordant == 3.5
    assert batch.mean_iterations_discordant == 4.0
    assert batch.paradigm_adherence == 0.5


def test_aggregate_replicates_rejects_mixed_dataset_ids():
    perfect = _perfect_score()
    other = _perfect_score()
    object.__setattr__(other, "dataset_id", "different_ds")
    try:
        aggregate_replicates([perfect, other])
    except ValueError as exc:
        assert "dataset_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for mixed dataset_ids")
