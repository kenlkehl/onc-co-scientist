"""Tests for replicate / bundle / batch aggregation."""

from __future__ import annotations

from statistics import stdev

from onc_co_scientist.scoring import (
    BuriedScore,
    NoveltyScore,
    ReplicateScore,
    aggregate_batch,
    aggregate_replicates,
)


def _replicate(
    *,
    dataset_id: str = "ds_a",
    variant: str = "named",
    model_id: str = "model",
    harness_id: str = "harness@1",
    max_iterations: int = 5,
    frac_novel: float | None,
    buried_score: int | None,
    uncovered: bool,
) -> ReplicateScore:
    if frac_novel is None:
        novelty = None
    else:
        novelty = NoveltyScore(
            n_total=10,
            n_novel=int(round(frac_novel * 10)),
            frac_novel=frac_novel,
            judgments=[],
        )
    buried = BuriedScore(
        max_iterations=max_iterations,
        per_association=[],
        earliest_iteration_uncovered=buried_score if uncovered else None,
        score=buried_score,
    )
    return ReplicateScore(
        dataset_id=dataset_id,
        variant=variant,
        model_id=model_id,
        harness_id=harness_id,
        max_iterations=max_iterations,
        novelty=novelty,
        buried=buried,
    )


def test_aggregate_replicates_mean_and_sd():
    reps = [
        _replicate(frac_novel=0.5, buried_score=2, uncovered=True),
        _replicate(frac_novel=0.7, buried_score=4, uncovered=True),
        _replicate(frac_novel=0.7, buried_score=None, uncovered=False),
    ]
    bundle = aggregate_replicates(reps)
    assert bundle.dataset_id == "ds_a"
    assert bundle.n_replicates == 3
    assert bundle.frac_novel_mean == 0.6333333333333333
    assert bundle.frac_novel_sd == stdev([0.5, 0.7, 0.7])
    assert bundle.buried_score_mean == 3.0
    assert bundle.buried_score_sd == stdev([2.0, 4.0])
    assert bundle.n_replicates_uncovered == 2
    assert bundle.fraction_uncovered == 2 / 3
    assert bundle.n_replicates_near_or_better == 2
    assert bundle.n_replicates_component_or_better == 2
    assert bundle.recovery_level_counts == {
        "none": 1,
        "component": 0,
        "near": 0,
        "exact": 2,
    }


def test_aggregate_replicates_single_run_sd_none():
    bundle = aggregate_replicates([_replicate(frac_novel=0.4, buried_score=3, uncovered=True)])
    assert bundle.n_replicates == 1
    assert bundle.frac_novel_mean == 0.4
    assert bundle.frac_novel_sd is None
    assert bundle.buried_score_sd is None
    assert bundle.fraction_uncovered == 1.0


def test_aggregate_replicates_rejects_mixed_dataset_ids():
    a = _replicate(frac_novel=0.5, buried_score=3, uncovered=True)
    b = _replicate(dataset_id="ds_b", frac_novel=0.6, buried_score=None, uncovered=False)
    try:
        aggregate_replicates([a, b])
    except ValueError as exc:
        assert "dataset_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_aggregate_batch_means_of_bundle_means():
    # Bundle A has 3 reps with frac_novel=0.5; bundle B has 1 rep with frac_novel=0.9.
    # Pipeline mean = (0.5 + 0.9) / 2 = 0.7, NOT replicate-weighted (3*0.5+0.9)/4=0.6.
    bundle_a = aggregate_replicates(
        [
            _replicate(frac_novel=0.5, buried_score=2, uncovered=True),
            _replicate(frac_novel=0.5, buried_score=2, uncovered=True),
            _replicate(frac_novel=0.5, buried_score=2, uncovered=True),
        ]
    )
    bundle_b = aggregate_replicates(
        [
            _replicate(
                dataset_id="ds_b",
                frac_novel=0.9,
                buried_score=None,
                uncovered=False,
            )
        ]
    )
    batch = aggregate_batch([bundle_a, bundle_b])
    assert batch.n_bundles == 2
    assert batch.n_bundles_named == 2
    assert batch.n_bundles_anonymized == 0
    assert batch.n_replicates_total == 4
    assert batch.frac_novel == 0.7
    assert batch.buried_score_named == 2.0
    assert batch.buried_score_anonymized is None
    # fraction_uncovered_named: bundle_a 1.0, bundle_b 0.0 → mean 0.5
    assert batch.fraction_uncovered_named == 0.5
    assert batch.fraction_uncovered_anonymized is None


def test_aggregate_batch_named_and_anonymized_split():
    """Anonymized bundles emit None novelty; unrecovered buried iterations
    stay absent from the dedicated _anonymized headline field."""
    named_a = aggregate_replicates([_replicate(frac_novel=0.5, buried_score=1, uncovered=True)])
    anon_a = aggregate_replicates(
        [_replicate(variant="anonymized", frac_novel=None, buried_score=None, uncovered=False)]
    )
    batch = aggregate_batch([named_a, anon_a])
    assert batch.n_bundles == 2
    assert batch.n_bundles_named == 1
    assert batch.n_bundles_anonymized == 1
    assert batch.frac_novel == 0.5
    assert batch.buried_score_named == 1.0
    assert batch.buried_score_anonymized is None
    assert batch.fraction_uncovered_named == 1.0
    assert batch.fraction_uncovered_anonymized == 0.0


def test_aggregate_replicates_rejects_mixed_variants():
    a = _replicate(frac_novel=0.5, buried_score=3, uncovered=True)
    b = _replicate(variant="anonymized", frac_novel=None, buried_score=None, uncovered=False)
    try:
        aggregate_replicates([a, b])
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("expected ValueError on mixed variants")


def test_aggregate_batch_to_dict_round_trips():
    bundle = aggregate_replicates([_replicate(frac_novel=0.4, buried_score=3, uncovered=True)])
    payload = aggregate_batch([bundle]).to_dict()
    assert payload["n_bundles"] == 1
    assert payload["n_replicates_total"] == 1
    assert payload["frac_novel"] == 0.4
    assert payload["fraction_near_or_better_named"] == 1.0
    assert payload["fraction_component_or_better_named"] == 1.0
    assert payload["per_bundle"][0]["dataset_id"] == "ds_a"
    assert payload["per_bundle"][0]["variant"] == "named"
    assert payload["per_bundle"][0]["recovery_level_counts"]["exact"] == 1
