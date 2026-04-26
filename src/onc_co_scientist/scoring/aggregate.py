"""Pipeline-level aggregation across a collection of per-dataset scores."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from .paradigm_metrics import DatasetScore


@dataclass
class PipelineScore:
    """Inter-dataset means. Metric 3 is defined at this level per the grant."""

    n_datasets: int
    mean_iterations_concordant: float | None
    mean_iterations_discordant: float | None
    mean_iterations_hidden_novel: float | None
    paradigm_adherence: float | None
    per_dataset: list[DatasetScore]

    def to_dict(self) -> dict:
        return {
            "n_datasets": self.n_datasets,
            "mean_iterations_concordant": self.mean_iterations_concordant,
            "mean_iterations_discordant": self.mean_iterations_discordant,
            "mean_iterations_hidden_novel": self.mean_iterations_hidden_novel,
            "paradigm_adherence": self.paradigm_adherence,
            "per_dataset": [s.to_dict() for s in self.per_dataset],
        }


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


def aggregate_datasets(scores: list[DatasetScore]) -> PipelineScore:
    concordant = _mean_of_present([s.mean_iterations_concordant for s in scores])
    discordant = _mean_of_present([s.mean_iterations_discordant for s in scores])
    hidden_novel = _mean_of_present([s.mean_iterations_hidden_novel for s in scores])
    adherence = (
        None if concordant is None or discordant is None else discordant - concordant
    )
    return PipelineScore(
        n_datasets=len(scores),
        mean_iterations_concordant=concordant,
        mean_iterations_discordant=discordant,
        mean_iterations_hidden_novel=hidden_novel,
        paradigm_adherence=adherence,
        per_dataset=list(scores),
    )


@dataclass
class BundleScore:
    """Per-bundle mean + SD across replicate transcripts.

    All replicates must be scored against the same dataset manifest, so they
    share ``dataset_id``. Means/SDs are taken across replicates whose metric
    is non-None; SD is ``None`` when fewer than two replicates contributed
    a value.
    """

    dataset_id: str
    n_replicates: int
    mean_iterations_concordant_mean: float | None
    mean_iterations_concordant_sd: float | None
    mean_iterations_discordant_mean: float | None
    mean_iterations_discordant_sd: float | None
    mean_iterations_hidden_novel_mean: float | None
    mean_iterations_hidden_novel_sd: float | None
    paradigm_adherence_mean: float | None
    paradigm_adherence_sd: float | None
    replicates: list[DatasetScore]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "n_replicates": self.n_replicates,
            "mean_iterations_concordant_mean": self.mean_iterations_concordant_mean,
            "mean_iterations_concordant_sd": self.mean_iterations_concordant_sd,
            "mean_iterations_discordant_mean": self.mean_iterations_discordant_mean,
            "mean_iterations_discordant_sd": self.mean_iterations_discordant_sd,
            "mean_iterations_hidden_novel_mean": self.mean_iterations_hidden_novel_mean,
            "mean_iterations_hidden_novel_sd": self.mean_iterations_hidden_novel_sd,
            "paradigm_adherence_mean": self.paradigm_adherence_mean,
            "paradigm_adherence_sd": self.paradigm_adherence_sd,
            "replicates": [r.to_dict() for r in self.replicates],
        }


@dataclass
class BatchPipelineScore:
    """Inter-bundle aggregate using each bundle's replicate-mean.

    Bundles contribute equally regardless of replicate count, so a noisy
    bundle with many replicates does not dominate the pipeline metric.
    """

    n_bundles: int
    n_replicates_total: int
    mean_iterations_concordant: float | None
    mean_iterations_discordant: float | None
    mean_iterations_hidden_novel: float | None
    paradigm_adherence: float | None
    per_bundle: list[BundleScore]

    def to_dict(self) -> dict:
        return {
            "n_bundles": self.n_bundles,
            "n_replicates_total": self.n_replicates_total,
            "mean_iterations_concordant": self.mean_iterations_concordant,
            "mean_iterations_discordant": self.mean_iterations_discordant,
            "mean_iterations_hidden_novel": self.mean_iterations_hidden_novel,
            "paradigm_adherence": self.paradigm_adherence,
            "per_bundle": [b.to_dict() for b in self.per_bundle],
        }


def aggregate_replicates(scores: list[DatasetScore]) -> BundleScore:
    """Reduce a list of per-replicate ``DatasetScore``s to a single ``BundleScore``."""
    if not scores:
        raise ValueError("aggregate_replicates requires at least one DatasetScore")
    dataset_ids = {s.dataset_id for s in scores}
    if len(dataset_ids) != 1:
        raise ValueError(
            f"Replicates span multiple dataset_ids: {sorted(dataset_ids)}. "
            f"All replicates in a bundle must score against the same manifest."
        )
    concordant_values = [s.mean_iterations_concordant for s in scores]
    discordant_values = [s.mean_iterations_discordant for s in scores]
    hidden_novel_values = [s.mean_iterations_hidden_novel for s in scores]
    adherence_values = [s.paradigm_adherence for s in scores]
    return BundleScore(
        dataset_id=next(iter(dataset_ids)),
        n_replicates=len(scores),
        mean_iterations_concordant_mean=_mean_of_present(concordant_values),
        mean_iterations_concordant_sd=_sd_of_present(concordant_values),
        mean_iterations_discordant_mean=_mean_of_present(discordant_values),
        mean_iterations_discordant_sd=_sd_of_present(discordant_values),
        mean_iterations_hidden_novel_mean=_mean_of_present(hidden_novel_values),
        mean_iterations_hidden_novel_sd=_sd_of_present(hidden_novel_values),
        paradigm_adherence_mean=_mean_of_present(adherence_values),
        paradigm_adherence_sd=_sd_of_present(adherence_values),
        replicates=list(scores),
    )


def aggregate_batch(bundle_scores: list[BundleScore]) -> BatchPipelineScore:
    """Pipeline-level metrics as the unweighted mean of bundle means."""
    concordant = _mean_of_present(
        [b.mean_iterations_concordant_mean for b in bundle_scores]
    )
    discordant = _mean_of_present(
        [b.mean_iterations_discordant_mean for b in bundle_scores]
    )
    hidden_novel = _mean_of_present(
        [b.mean_iterations_hidden_novel_mean for b in bundle_scores]
    )
    adherence = (
        None if concordant is None or discordant is None else discordant - concordant
    )
    return BatchPipelineScore(
        n_bundles=len(bundle_scores),
        n_replicates_total=sum(b.n_replicates for b in bundle_scores),
        mean_iterations_concordant=concordant,
        mean_iterations_discordant=discordant,
        mean_iterations_hidden_novel=hidden_novel,
        paradigm_adherence=adherence,
        per_bundle=list(bundle_scores),
    )
