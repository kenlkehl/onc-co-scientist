"""Pipeline-level aggregation across a collection of per-dataset scores."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

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
