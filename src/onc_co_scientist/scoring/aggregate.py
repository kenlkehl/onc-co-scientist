"""Replicate / bundle / batch score containers and aggregators.

A ``ReplicateScore`` summarizes one transcript (one harness run on one
dataset). A ``BundleScore`` rolls up replicates of the same dataset into
mean ± SD of the two scoring metrics (frac_novel, buried_score). A
``BatchPipelineScore`` rolls bundles up using an unweighted mean of bundle
means, so a noisy bundle with many replicates does not dominate the
pipeline figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev

from .buried import BuriedScore
from .novelty import NoveltyScore


def _mean_of_present(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return mean(present)


def _sd_of_present(values: list[float | None]) -> float | None:
    """Sample SD over the non-None entries, or ``None`` when fewer than 2."""
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    return stdev(present)


@dataclass
class ReplicateScore:
    """Per-transcript summary: one harness run on one dataset."""

    dataset_id: str
    model_id: str
    harness_id: str
    max_iterations: int
    novelty: NoveltyScore
    buried: BuriedScore

    @property
    def frac_novel(self) -> float:
        return self.novelty.frac_novel

    @property
    def buried_score(self) -> int:
        return self.buried.score

    @property
    def uncovered(self) -> bool:
        return self.buried.uncovered

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "harness_id": self.harness_id,
            "max_iterations": self.max_iterations,
            "frac_novel": self.frac_novel,
            "buried_score": self.buried_score,
            "uncovered": self.uncovered,
            "novelty": self.novelty.to_dict(),
            "buried": self.buried.to_dict(),
        }


@dataclass
class BundleScore:
    """Mean ± SD of the two metrics across a bundle's replicates."""

    dataset_id: str
    n_replicates: int
    frac_novel_mean: float | None
    frac_novel_sd: float | None
    buried_score_mean: float | None
    buried_score_sd: float | None
    n_replicates_uncovered: int
    replicates: list[ReplicateScore] = field(default_factory=list)

    @property
    def fraction_uncovered(self) -> float | None:
        if self.n_replicates == 0:
            return None
        return self.n_replicates_uncovered / self.n_replicates

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "n_replicates": self.n_replicates,
            "frac_novel_mean": self.frac_novel_mean,
            "frac_novel_sd": self.frac_novel_sd,
            "buried_score_mean": self.buried_score_mean,
            "buried_score_sd": self.buried_score_sd,
            "n_replicates_uncovered": self.n_replicates_uncovered,
            "fraction_uncovered": self.fraction_uncovered,
            "replicates": [r.to_dict() for r in self.replicates],
        }


@dataclass
class BatchPipelineScore:
    """Inter-bundle aggregate using each bundle's replicate-mean."""

    n_bundles: int
    n_replicates_total: int
    frac_novel: float | None
    buried_score: float | None
    fraction_uncovered: float | None
    per_bundle: list[BundleScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_bundles": self.n_bundles,
            "n_replicates_total": self.n_replicates_total,
            "frac_novel": self.frac_novel,
            "buried_score": self.buried_score,
            "fraction_uncovered": self.fraction_uncovered,
            "per_bundle": [b.to_dict() for b in self.per_bundle],
        }


def aggregate_replicates(scores: list[ReplicateScore]) -> BundleScore:
    """Reduce per-replicate ``ReplicateScore``s to a single ``BundleScore``."""
    if not scores:
        raise ValueError("aggregate_replicates requires at least one ReplicateScore")
    dataset_ids = {s.dataset_id for s in scores}
    if len(dataset_ids) != 1:
        raise ValueError(
            f"Replicates span multiple dataset_ids: {sorted(dataset_ids)}. "
            f"All replicates in a bundle must score against the same manifest."
        )
    frac_novels: list[float | None] = [s.frac_novel for s in scores]
    buried_scores: list[float | None] = [float(s.buried_score) for s in scores]
    n_uncovered = sum(1 for s in scores if s.uncovered)
    return BundleScore(
        dataset_id=next(iter(dataset_ids)),
        n_replicates=len(scores),
        frac_novel_mean=_mean_of_present(frac_novels),
        frac_novel_sd=_sd_of_present(frac_novels),
        buried_score_mean=_mean_of_present(buried_scores),
        buried_score_sd=_sd_of_present(buried_scores),
        n_replicates_uncovered=n_uncovered,
        replicates=list(scores),
    )


def aggregate_batch(bundle_scores: list[BundleScore]) -> BatchPipelineScore:
    """Pipeline-level metrics as the unweighted mean of bundle means."""
    frac_novel = _mean_of_present([b.frac_novel_mean for b in bundle_scores])
    buried_score = _mean_of_present([b.buried_score_mean for b in bundle_scores])
    fraction_uncovered = _mean_of_present(
        [b.fraction_uncovered for b in bundle_scores]
    )
    return BatchPipelineScore(
        n_bundles=len(bundle_scores),
        n_replicates_total=sum(b.n_replicates for b in bundle_scores),
        frac_novel=frac_novel,
        buried_score=buried_score,
        fraction_uncovered=fraction_uncovered,
        per_bundle=list(bundle_scores),
    )
