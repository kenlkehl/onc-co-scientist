"""Primary scoring metrics for the Oncology Scientific Open-Mindedness Benchmark.

For each ground-truth ``AssociationSpec``, compute the earliest iteration at
which the harness both proposed a matching hypothesis and produced a
statistical analysis that references that hypothesis with a direction-correct
significant result. Unsuccessful associations are assigned a penalty value
(``N + 1`` by default, following the grant's "up to max of N" scoring spec).

Per-dataset outputs:

- Metric (1): mean iterations-to-uncover across paradigm-concordant associations.
- Metric (2): mean iterations-to-uncover across paradigm-discordant associations.
- (Hidden-novel associations are also tracked but not combined into (1) or (2);
  the grant reserves them as an exploratory signal that we surface separately.)

Aggregation across datasets to produce Metric (3) = (2) - (1) at the pipeline
level lives in ``aggregate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..harness.transcript import AnalysisRecord, HypothesisRecord, Transcript
from ..synthetic.schemas import AssociationSpec, DatasetManifest, ParadigmClass
from .matching import HypothesisMatcher, RegexMatcher

DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class AssociationOutcome:
    """Scoring detail for one ground-truth association."""

    association_id: str
    paradigm_class: ParadigmClass
    iteration_uncovered: int | None
    proposed_iteration: int | None
    tested_iteration: int | None
    matched_hypothesis_ids: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class DatasetScore:
    dataset_id: str
    model_id: str
    harness_id: str
    max_iterations: int
    penalty_iteration: int
    per_association: list[AssociationOutcome]
    mean_iterations_concordant: float | None
    mean_iterations_discordant: float | None
    mean_iterations_hidden_novel: float | None

    @property
    def paradigm_adherence(self) -> float | None:
        """(2) - (1) at the dataset level; None if either component is missing."""
        if (
            self.mean_iterations_discordant is None
            or self.mean_iterations_concordant is None
        ):
            return None
        return self.mean_iterations_discordant - self.mean_iterations_concordant

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "harness_id": self.harness_id,
            "max_iterations": self.max_iterations,
            "penalty_iteration": self.penalty_iteration,
            "mean_iterations_concordant": self.mean_iterations_concordant,
            "mean_iterations_discordant": self.mean_iterations_discordant,
            "mean_iterations_hidden_novel": self.mean_iterations_hidden_novel,
            "paradigm_adherence": self.paradigm_adherence,
            "per_association": [
                {
                    "association_id": o.association_id,
                    "paradigm_class": o.paradigm_class.value,
                    "iteration_uncovered": o.iteration_uncovered,
                    "proposed_iteration": o.proposed_iteration,
                    "tested_iteration": o.tested_iteration,
                    "matched_hypothesis_ids": o.matched_hypothesis_ids,
                    "notes": o.notes,
                }
                for o in self.per_association
            ],
        }


def _analysis_supports(
    analysis: AnalysisRecord,
    spec: AssociationSpec,
    *,
    significance_threshold: float,
) -> bool:
    """Does this analysis provide direction-correct, significant evidence for ``spec``?"""
    significant = analysis.significant
    if significant is None and analysis.p_value is not None:
        significant = analysis.p_value < significance_threshold
    if not significant:
        return False
    if spec.direction == 0:
        return True
    if analysis.effect_estimate is None:
        # We still credit the test if the harness flagged significance but
        # omitted a signed estimate; otherwise we cannot verify direction.
        return True
    if spec.direction > 0:
        return analysis.effect_estimate > 0
    return analysis.effect_estimate < 0


def _match_hypotheses_to_spec(
    hypotheses: list[tuple[int, HypothesisRecord]],
    spec: AssociationSpec,
    matcher: HypothesisMatcher,
) -> list[tuple[int, HypothesisRecord]]:
    return [(it, h) for it, h in hypotheses if matcher.matches(h.text, spec)]


def _uncover_iteration(
    transcript: Transcript,
    spec: AssociationSpec,
    matcher: HypothesisMatcher,
    *,
    significance_threshold: float,
) -> AssociationOutcome:
    matched = _match_hypotheses_to_spec(transcript.flat_hypotheses(), spec, matcher)
    if not matched:
        return AssociationOutcome(
            association_id=spec.id,
            paradigm_class=spec.paradigm_class,
            iteration_uncovered=None,
            proposed_iteration=None,
            tested_iteration=None,
            notes="No hypothesis matched this ground-truth association.",
        )

    first_propose_iter = min(it for it, _ in matched)
    matched_ids = {h.id for _, h in matched}

    tested_iteration: int | None = None
    for it_index, analysis in transcript.flat_analyses():
        if not set(analysis.hypothesis_ids) & matched_ids:
            continue
        if not _analysis_supports(
            analysis, spec, significance_threshold=significance_threshold
        ):
            continue
        if tested_iteration is None or it_index < tested_iteration:
            tested_iteration = it_index

    if tested_iteration is None:
        return AssociationOutcome(
            association_id=spec.id,
            paradigm_class=spec.paradigm_class,
            iteration_uncovered=None,
            proposed_iteration=first_propose_iter,
            tested_iteration=None,
            matched_hypothesis_ids=sorted(matched_ids),
            notes="Hypothesis was proposed but no supporting significant analysis was found.",
        )

    uncovered = max(first_propose_iter, tested_iteration)
    return AssociationOutcome(
        association_id=spec.id,
        paradigm_class=spec.paradigm_class,
        iteration_uncovered=uncovered,
        proposed_iteration=first_propose_iter,
        tested_iteration=tested_iteration,
        matched_hypothesis_ids=sorted(matched_ids),
    )


def _mean_or_none(values: list[int], penalty: int, n_specs: int) -> float | None:
    if n_specs == 0:
        return None
    padded = values + [penalty] * (n_specs - len(values))
    return mean(padded)


def score_dataset(
    manifest: DatasetManifest,
    transcript: Transcript,
    *,
    matcher: HypothesisMatcher | None = None,
    penalty_iteration: int | None = None,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
) -> DatasetScore:
    """Compute primary scoring for one dataset."""
    if transcript.dataset_id != manifest.dataset_id:
        raise ValueError(
            f"Transcript dataset_id {transcript.dataset_id!r} does not match "
            f"manifest dataset_id {manifest.dataset_id!r}."
        )

    matcher = matcher or RegexMatcher()
    penalty = (
        penalty_iteration
        if penalty_iteration is not None
        else transcript.max_iterations + 1
    )

    outcomes: list[AssociationOutcome] = []
    for spec in manifest.associations:
        outcomes.append(
            _uncover_iteration(
                transcript,
                spec,
                matcher,
                significance_threshold=significance_threshold,
            )
        )

    by_class: dict[ParadigmClass, list[int]] = {klass: [] for klass in ParadigmClass}
    counts: dict[ParadigmClass, int] = {klass: 0 for klass in ParadigmClass}
    for outcome, spec in zip(outcomes, manifest.associations, strict=True):
        counts[spec.paradigm_class] += 1
        if outcome.iteration_uncovered is not None:
            by_class[outcome.paradigm_class].append(outcome.iteration_uncovered)

    return DatasetScore(
        dataset_id=manifest.dataset_id,
        model_id=transcript.model_id,
        harness_id=transcript.harness_id,
        max_iterations=transcript.max_iterations,
        penalty_iteration=penalty,
        per_association=outcomes,
        mean_iterations_concordant=_mean_or_none(
            by_class[ParadigmClass.concordant],
            penalty,
            counts[ParadigmClass.concordant],
        ),
        mean_iterations_discordant=_mean_or_none(
            by_class[ParadigmClass.discordant],
            penalty,
            counts[ParadigmClass.discordant],
        ),
        mean_iterations_hidden_novel=_mean_or_none(
            by_class[ParadigmClass.hidden_novel],
            penalty,
            counts[ParadigmClass.hidden_novel],
        ),
    )
