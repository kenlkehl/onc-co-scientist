"""Buried-finding discovery scoring.

For each hidden_novel association in the manifest, find the earliest
iteration at which the harness both (a) proposed a hypothesis matching the
association and (b) ran an analysis with a direction-correct, significant
result referencing that hypothesis. The bundle-level score is the earliest
discovery iteration across hidden_novel associations, falling back to
``transcript.max_iterations`` when nothing was uncovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..harness.transcript import AnalysisRecord, HypothesisRecord, Transcript
from ..synthetic.schemas import AssociationSpec, DatasetManifest, ParadigmClass
from .judge import Judge

DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class BuriedDiscovery:
    association_id: str
    iteration_uncovered: int | None
    proposed_iteration: int | None
    tested_iteration: int | None
    matched_hypothesis_ids: list[str] = field(default_factory=list)


@dataclass
class BuriedScore:
    max_iterations: int
    per_association: list[BuriedDiscovery]
    earliest_iteration_uncovered: int | None
    score: int

    @property
    def uncovered(self) -> bool:
        return self.earliest_iteration_uncovered is not None

    def to_dict(self) -> dict:
        return {
            "max_iterations": self.max_iterations,
            "earliest_iteration_uncovered": self.earliest_iteration_uncovered,
            "score": self.score,
            "uncovered": self.uncovered,
            "per_association": [
                {
                    "association_id": d.association_id,
                    "iteration_uncovered": d.iteration_uncovered,
                    "proposed_iteration": d.proposed_iteration,
                    "tested_iteration": d.tested_iteration,
                    "matched_hypothesis_ids": d.matched_hypothesis_ids,
                }
                for d in self.per_association
            ],
        }


def _analysis_supports(
    analysis: AnalysisRecord,
    spec: AssociationSpec,
    *,
    significance_threshold: float,
) -> bool:
    """Direction-correct, significant evidence for ``spec``?

    Ported from the deleted ``paradigm_metrics._analysis_supports``: scoring
    semantics for buried discovery are identical to the legacy path; only
    the matcher changes.
    """
    significant = analysis.significant
    if significant is None and analysis.p_value is not None:
        significant = analysis.p_value < significance_threshold
    if not significant:
        return False
    if spec.direction == 0:
        return True
    if analysis.effect_estimate is None:
        # Credit a flagged-significant analysis even without a signed
        # estimate; we can't verify direction otherwise.
        return True
    if spec.direction > 0:
        return analysis.effect_estimate > 0
    return analysis.effect_estimate < 0


def _discover_for_spec(
    transcript: Transcript,
    spec: AssociationSpec,
    matched: list[tuple[int, HypothesisRecord]],
    *,
    significance_threshold: float,
) -> BuriedDiscovery:
    if not matched:
        return BuriedDiscovery(
            association_id=spec.id,
            iteration_uncovered=None,
            proposed_iteration=None,
            tested_iteration=None,
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
        return BuriedDiscovery(
            association_id=spec.id,
            iteration_uncovered=None,
            proposed_iteration=first_propose_iter,
            tested_iteration=None,
            matched_hypothesis_ids=sorted(matched_ids),
        )

    uncovered = max(first_propose_iter, tested_iteration)
    return BuriedDiscovery(
        association_id=spec.id,
        iteration_uncovered=uncovered,
        proposed_iteration=first_propose_iter,
        tested_iteration=tested_iteration,
        matched_hypothesis_ids=sorted(matched_ids),
    )


def score_buried(
    manifest: DatasetManifest,
    transcript: Transcript,
    judge: Judge,
    *,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
) -> BuriedScore:
    """Compute buried-finding discovery score for one (manifest, transcript)."""
    if transcript.dataset_id != manifest.dataset_id:
        raise ValueError(
            f"Transcript dataset_id {transcript.dataset_id!r} does not match "
            f"manifest dataset_id {manifest.dataset_id!r}."
        )

    buried_specs = [
        s for s in manifest.associations if s.paradigm_class == ParadigmClass.hidden_novel
    ]
    flat = transcript.flat_hypotheses()

    per_association: list[BuriedDiscovery] = []
    for spec in buried_specs:
        if not flat:
            per_association.append(
                BuriedDiscovery(
                    association_id=spec.id,
                    iteration_uncovered=None,
                    proposed_iteration=None,
                    tested_iteration=None,
                )
            )
            continue
        texts = [h.text for _, h in flat]
        match_judgments = judge.judge_matches(texts, spec.natural_language_description)
        matched = [
            (it, h)
            for (it, h), m in zip(flat, match_judgments, strict=True)
            if m.matches
        ]
        per_association.append(
            _discover_for_spec(
                transcript,
                spec,
                matched,
                significance_threshold=significance_threshold,
            )
        )

    earliest = [
        d.iteration_uncovered
        for d in per_association
        if d.iteration_uncovered is not None
    ]
    earliest_iter = min(earliest) if earliest else None
    score = earliest_iter if earliest_iter is not None else transcript.max_iterations
    return BuriedScore(
        max_iterations=transcript.max_iterations,
        per_association=per_association,
        earliest_iteration_uncovered=earliest_iter,
        score=score,
    )
