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
    model_id: str = "model",
    harness_id: str = "harness@1",
    max_iterations: int = 5,
    frac_novel: float,
    buried_score: int,
    uncovered: bool,
) -> ReplicateScore:
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
        model_id=model_id,
        harness_id=harness_id,
        max_iterations=max_iterations,
        novelty=novelty,
        buried=buried,
    )


def test_aggregate_replicates_mean_and_sd():
    reps = [
        _replicate(frac_novel=0.5, buried_score=2, uncovered=True),
        _replicate(frac_novel=0.7, buried_score=5, uncovered=False),
    ]
    bundle = aggregate_replicates(reps)
    assert bundle.dataset_id == "ds_a"
    assert bundle.n_replicates == 2
    assert bundle.frac_novel_mean == 0.6
    assert bundle.frac_novel_sd == stdev([0.5, 0.7])
    assert bundle.buried_score_mean == 3.5
    assert bundle.buried_score_sd == stdev([2.0, 5.0])
    assert bundle.n_replicates_uncovered == 1
    assert bundle.fraction_uncovered == 0.5


def test_aggregate_replicates_single_run_sd_none():
    bundle = aggregate_replicates(
        [_replicate(frac_novel=0.4, buried_score=3, uncovered=True)]
    )
    assert bundle.n_replicates == 1
    assert bundle.frac_novel_mean == 0.4
    assert bundle.frac_novel_sd is None
    assert bundle.buried_score_sd is None
    assert bundle.fraction_uncovered == 1.0


def test_aggregate_replicates_rejects_mixed_dataset_ids():
    a = _replicate(frac_novel=0.5, buried_score=3, uncovered=True)
    b = _replicate(
        dataset_id="ds_b", frac_novel=0.6, buried_score=4, uncovered=False
    )
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
                buried_score=5,
                uncovered=False,
            )
        ]
    )
    batch = aggregate_batch([bundle_a, bundle_b])
    assert batch.n_bundles == 2
    assert batch.n_replicates_total == 4
    assert batch.frac_novel == 0.7
    assert batch.buried_score == 3.5
    # fraction_uncovered: bundle_a 1.0, bundle_b 0.0 → mean 0.5
    assert batch.fraction_uncovered == 0.5


def test_aggregate_batch_to_dict_round_trips():
    bundle = aggregate_replicates(
        [_replicate(frac_novel=0.4, buried_score=3, uncovered=True)]
    )
    payload = aggregate_batch([bundle]).to_dict()
    assert payload["n_bundles"] == 1
    assert payload["n_replicates_total"] == 1
    assert payload["frac_novel"] == 0.4
    assert payload["per_bundle"][0]["dataset_id"] == "ds_a"
