"""Buried-finding discovery scoring.

For each hidden_novel association in the manifest, find the earliest
iteration at which the harness both (a) proposed a hypothesis matching the
association and (b) ran an analysis with a direction-correct, significant
result referencing that hypothesis. The bundle-level score is the earliest
exact discovery iteration across hidden_novel associations, falling back to
``transcript.max_iterations + 1`` when nothing was uncovered.

Exact discovery is preserved as the strict metric. Each association also gets
a graded recovery level (exact, near, component, none) so near-miss recovery
can be summarized without weakening the exact definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..harness.transcript import AnalysisRecord, HypothesisRecord, Transcript
from ..synthetic.schemas import AssociationSpec, DatasetManifest, ParadigmClass
from .judge import Judge, RecoveryLevel, Variant

DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05
_RECOVERY_ORDER: dict[RecoveryLevel, int] = {
    "none": 0,
    "component": 1,
    "near": 2,
    "exact": 3,
}


@dataclass
class MatchJudgmentRecord:
    """Per-hypothesis match-judgment trace for one ``BuriedDiscovery``.

    Persisted out-of-band (sibling JSONL) rather than into ``batch_score.json``
    so the aggregate file stays compact regardless of transcript length.
    """

    iteration: int
    hypothesis_id: str
    text: str
    matches: bool
    recovery_level: RecoveryLevel
    rationale: str


@dataclass
class BuriedDiscovery:
    association_id: str
    iteration_uncovered: int | None
    proposed_iteration: int | None
    tested_iteration: int | None
    recovery_level: RecoveryLevel = "none"
    recovery_iteration: int | None = None
    matched_hypothesis_ids: list[str] = field(default_factory=list)
    recovery_hypothesis_ids: list[str] = field(default_factory=list)
    match_judgments: list[MatchJudgmentRecord] = field(default_factory=list)


@dataclass
class BuriedScore:
    max_iterations: int
    per_association: list[BuriedDiscovery]
    earliest_iteration_uncovered: int | None
    score: int

    @property
    def uncovered(self) -> bool:
        return self.earliest_iteration_uncovered is not None

    @property
    def recovery_level(self) -> RecoveryLevel:
        if self.uncovered:
            return "exact"
        if not self.per_association:
            return "none"
        return max(
            (d.recovery_level for d in self.per_association),
            key=lambda level: _RECOVERY_ORDER[level],
        )

    @property
    def recovery_iteration(self) -> int | None:
        if self.uncovered:
            return self.earliest_iteration_uncovered
        level = self.recovery_level
        if level == "none":
            return None
        iterations = [
            d.recovery_iteration
            for d in self.per_association
            if d.recovery_level == level and d.recovery_iteration is not None
        ]
        return min(iterations) if iterations else None

    @property
    def near_or_better(self) -> bool:
        return _RECOVERY_ORDER[self.recovery_level] >= _RECOVERY_ORDER["near"]

    @property
    def component_or_better(self) -> bool:
        return _RECOVERY_ORDER[self.recovery_level] >= _RECOVERY_ORDER["component"]

    def to_dict(self) -> dict:
        return {
            "max_iterations": self.max_iterations,
            "earliest_iteration_uncovered": self.earliest_iteration_uncovered,
            "score": self.score,
            "uncovered": self.uncovered,
            "recovery_level": self.recovery_level,
            "recovery_iteration": self.recovery_iteration,
            "near_or_better": self.near_or_better,
            "component_or_better": self.component_or_better,
            "per_association": [
                {
                    "association_id": d.association_id,
                    "iteration_uncovered": d.iteration_uncovered,
                    "proposed_iteration": d.proposed_iteration,
                    "tested_iteration": d.tested_iteration,
                    "recovery_level": d.recovery_level,
                    "recovery_iteration": d.recovery_iteration,
                    "matched_hypothesis_ids": d.matched_hypothesis_ids,
                    "recovery_hypothesis_ids": d.recovery_hypothesis_ids,
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


def _analysis_is_significant(
    analysis: AnalysisRecord,
    *,
    significance_threshold: float,
) -> bool:
    significant = analysis.significant
    if significant is None and analysis.p_value is not None:
        significant = analysis.p_value < significance_threshold
    return bool(significant)


def _analysis_supports_recovery_level(
    analysis: AnalysisRecord,
    spec: AssociationSpec,
    level: RecoveryLevel,
    *,
    significance_threshold: float,
) -> bool:
    if level in {"exact", "near"}:
        return _analysis_supports(
            analysis, spec, significance_threshold=significance_threshold
        )
    if level == "component":
        return _analysis_is_significant(
            analysis, significance_threshold=significance_threshold
        )
    return False


def _first_supporting_analysis_iteration(
    transcript: Transcript,
    hypothesis_id: str,
    spec: AssociationSpec,
    level: RecoveryLevel,
    *,
    significance_threshold: float,
) -> int | None:
    tested_iteration: int | None = None
    for it_index, analysis in transcript.flat_analyses():
        if hypothesis_id not in analysis.hypothesis_ids:
            continue
        if not _analysis_supports_recovery_level(
            analysis,
            spec,
            level,
            significance_threshold=significance_threshold,
        ):
            continue
        if tested_iteration is None or it_index < tested_iteration:
            tested_iteration = it_index
    return tested_iteration


def _better_recovery(
    candidate: tuple[RecoveryLevel, int, str],
    current: tuple[RecoveryLevel, int, str] | None,
) -> bool:
    if current is None:
        return True
    candidate_level, candidate_iteration, _ = candidate
    current_level, current_iteration, _ = current
    if _RECOVERY_ORDER[candidate_level] != _RECOVERY_ORDER[current_level]:
        return _RECOVERY_ORDER[candidate_level] > _RECOVERY_ORDER[current_level]
    return candidate_iteration < current_iteration


def _best_graded_recovery(
    transcript: Transcript,
    spec: AssociationSpec,
    judged: list[tuple[int, HypothesisRecord, RecoveryLevel]],
    *,
    significance_threshold: float,
) -> tuple[RecoveryLevel, int | None, list[str]]:
    best: tuple[RecoveryLevel, int, str] | None = None
    for proposed_iteration, hypothesis, level in judged:
        if level == "none":
            continue
        tested_iteration = _first_supporting_analysis_iteration(
            transcript,
            hypothesis.id,
            spec,
            level,
            significance_threshold=significance_threshold,
        )
        if tested_iteration is None:
            continue
        recovery_iteration = max(proposed_iteration, tested_iteration)
        candidate = (level, recovery_iteration, hypothesis.id)
        if _better_recovery(candidate, best):
            best = candidate

    if best is None:
        return "none", None, []

    best_level, best_iteration, _best_hypothesis_id = best
    hypothesis_ids = [
        hypothesis.id
        for proposed_iteration, hypothesis, level in judged
        if level == best_level
        and (
            tested_iteration := _first_supporting_analysis_iteration(
                transcript,
                hypothesis.id,
                spec,
                level,
                significance_threshold=significance_threshold,
            )
        )
        is not None
        and max(proposed_iteration, tested_iteration) == best_iteration
    ]
    return best_level, best_iteration, sorted(hypothesis_ids)


def _discover_for_spec(
    transcript: Transcript,
    spec: AssociationSpec,
    matched: list[tuple[int, HypothesisRecord]],
    graded: list[tuple[int, HypothesisRecord, RecoveryLevel]],
    match_judgments: list[MatchJudgmentRecord],
    *,
    significance_threshold: float,
) -> BuriedDiscovery:
    recovery_level, recovery_iteration, recovery_hypothesis_ids = _best_graded_recovery(
        transcript,
        spec,
        graded,
        significance_threshold=significance_threshold,
    )
    if not matched:
        return BuriedDiscovery(
            association_id=spec.id,
            iteration_uncovered=None,
            proposed_iteration=None,
            tested_iteration=None,
            recovery_level=recovery_level,
            recovery_iteration=recovery_iteration,
            recovery_hypothesis_ids=recovery_hypothesis_ids,
            match_judgments=match_judgments,
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
            recovery_level=recovery_level,
            recovery_iteration=recovery_iteration,
            matched_hypothesis_ids=sorted(matched_ids),
            recovery_hypothesis_ids=recovery_hypothesis_ids,
            match_judgments=match_judgments,
        )

    uncovered = max(first_propose_iter, tested_iteration)
    return BuriedDiscovery(
        association_id=spec.id,
        iteration_uncovered=uncovered,
        proposed_iteration=first_propose_iter,
        tested_iteration=tested_iteration,
        recovery_level="exact",
        recovery_iteration=uncovered,
        matched_hypothesis_ids=sorted(matched_ids),
        recovery_hypothesis_ids=sorted(matched_ids),
        match_judgments=match_judgments,
    )


def score_buried(
    manifest: DatasetManifest,
    transcript: Transcript,
    judge: Judge,
    *,
    variant: Variant = "named",
    column_mapping: dict[str, str] | None = None,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
) -> BuriedScore:
    """Compute buried-finding discovery score for one (manifest, transcript).

    ``variant`` records which column-name space the agent saw; the judge
    prompt is rendered bilingually when ``column_mapping`` is provided
    so the judge sees both clinical and ``feature_NNN`` identifiers
    regardless of variant.
    """
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
        match_judgments = judge.judge_matches(
            texts, spec, variant=variant, column_mapping=column_mapping
        )
        records = [
            MatchJudgmentRecord(
                iteration=it,
                hypothesis_id=h.id,
                text=h.text,
                matches=m.matches,
                recovery_level=m.recovery_level,
                rationale=m.rationale,
            )
            for (it, h), m in zip(flat, match_judgments, strict=True)
        ]
        matched = [
            (it, h)
            for (it, h), m in zip(flat, match_judgments, strict=True)
            if m.matches
        ]
        graded = [
            (it, h, m.recovery_level)
            for (it, h), m in zip(flat, match_judgments, strict=True)
            if m.recovery_level != "none"
        ]
        per_association.append(
            _discover_for_spec(
                transcript,
                spec,
                matched,
                graded,
                records,
                significance_threshold=significance_threshold,
            )
        )

    earliest = [
        d.iteration_uncovered
        for d in per_association
        if d.iteration_uncovered is not None
    ]
    earliest_iter = min(earliest) if earliest else None
    score = earliest_iter if earliest_iter is not None else transcript.max_iterations + 1
    return BuriedScore(
        max_iterations=transcript.max_iterations,
        per_association=per_association,
        earliest_iteration_uncovered=earliest_iter,
        score=score,
    )
