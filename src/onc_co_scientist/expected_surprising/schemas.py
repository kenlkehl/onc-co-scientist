"""Strict public hypothesis and private generation contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Condition(StrictModel):
    variable: str
    op: Literal["eq", "ge", "le"] = "eq"
    value: str | float


class Hypothesis(StrictModel):
    id: str
    outcome: str
    exposure: str
    exposed: str | float = 1.0
    comparator: str | float = 0.0
    contrast: Literal["mean_difference", "interaction"] = "mean_difference"
    eligibility: list[Condition] = Field(default_factory=list)
    subgroup: list[Condition] = Field(default_factory=list)
    direction: Literal[-1, 1]

    @model_validator(mode="after")
    def coherent(self):
        if self.exposed == self.comparator:
            raise ValueError("Exposure levels must differ")
        if self.exposure == self.outcome:
            raise ValueError("Outcome cannot be its own exposure")
        if any(
            c.variable in {self.exposure, self.outcome} for c in self.subgroup + self.eligibility
        ):
            raise ValueError("Conditioning cannot use the exposure or outcome")
        if self.contrast == "interaction" and not self.subgroup:
            raise ValueError("An interaction requires a subgroup and its complement")
        for conditions in (self.subgroup, self.eligibility):
            by_var: dict[str, dict] = {}
            for c in conditions:
                ops = by_var.setdefault(c.variable, {})
                if c.op in ops:
                    raise ValueError("Repeated condition operator")
                ops[c.op] = c.value
            for ops in by_var.values():
                if "eq" in ops and len(ops) > 1:
                    raise ValueError("Use equality or range conditions, not both")
                if any(isinstance(v, str) for k, v in ops.items() if k != "eq"):
                    raise ValueError("Range bounds must be numeric")
                if "ge" in ops and "le" in ops and ops["ge"] > ops["le"]:
                    raise ValueError("Reversed interval")
        if {c.variable for c in self.eligibility} & {c.variable for c in self.subgroup}:
            raise ValueError("Eligibility and subgroup variables must be distinct")
        return self


class Source(StrictModel):
    id: str
    title: str
    abstract: str
    url: str
    year: str = ""
    doi: str = ""
    pmcid: str = ""
    full_text_excerpt: str = ""


class Candidate(StrictModel):
    hypothesis: Hypothesis
    proposed_category: Literal["expected", "neutral"]
    statement: str
    queries: list[str] = Field(min_length=2, max_length=4)


class EvidenceScope(StrictModel):
    source_id: str
    primary_empirical: bool
    population: str = Field(min_length=1)
    exposure: str = Field(min_length=1)
    comparator: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    design_and_assay: str = Field(min_length=1)
    direction: Literal[-1, 1]
    limitations: str = Field(min_length=1)
    endpoint_kind: Literal[
        "pfs",
        "recurrence_free",
        "overall_survival",
        "binary_pfs_landmark",
        "genetic_dependency",
        "drug_response",
        "other",
    ] = "other"


class RealismReview(StrictModel):
    decision: Literal["accept", "revise", "reject"]
    population_scope_valid: bool
    comparator_scope_valid: bool
    biomarker_restrictions_complete: bool
    assay_transfer_valid: bool
    category_valid: bool
    no_inherited_directional_expectation: bool
    source_ids: list[str]
    rationale: str = Field(min_length=1)
    required_changes: list[str]
    necessary_scope_conditions: list[Condition] = Field(default_factory=list)
    unrepresented_scope_requirements: list[str] = Field(default_factory=list)
    counterexample_search: str = ""


class DGPAssessment(StrictModel):
    decision: Literal["accept", "revise", "reject"]
    expected_components_scoped: bool
    intentional_reversals_only: bool
    nonredundant_targets: bool
    plausible_outcome_scale: bool
    rationale: str = Field(min_length=1)
    required_changes: list[str]


class DGPReview(StrictModel):
    assessment: DGPAssessment
    policy_version: str
    spec_sha256: str
    model: str
    reviewed_at: str


class Review(StrictModel):
    decision: Literal["supported", "neutral", "unsupported", "ambiguous"]
    source_ids: list[str]
    rationale: str
    population_matches: bool
    exposure_comparator_matches: bool
    outcome_matches: bool
    direction_matches: bool
    evidence_scopes: list[EvidenceScope] = Field(default_factory=list)
    broader_direction: Literal["positive", "negative", "none", "uncertain"] = "uncertain"
    neutrality_rationale: str = ""


class ReviewedCandidate(StrictModel):
    candidate: Candidate
    review: Review
    sources: list[Source]
    search_ids: list[str]
    generation_model: str
    review_model: str
    reviewed_at: str
    review_profile: str = ""
    review_policy_version: str = "legacy"
    candidate_sha256: str = ""
    realism: RealismReview | None = None
    realism_model: str = ""

    @model_validator(mode="after")
    def supported_sources(self):
        ids = {s.id for s in self.sources}
        if not set(self.review.source_ids) <= ids:
            raise ValueError("Review cites a source that was not retrieved")
        r = self.review
        if self.candidate.proposed_category == "expected":
            if (
                r.decision != "supported"
                or not r.source_ids
                or not all(
                    (
                        r.population_matches,
                        r.exposure_comparator_matches,
                        r.outcome_matches,
                        r.direction_matches,
                    )
                )
            ):
                raise ValueError("Expected discovery lacks matching directional evidence")
        elif r.decision != "neutral":
            raise ValueError("Neutral discovery has an established or ambiguous expectation")
        return self


class Discovery(StrictModel):
    hypothesis: Hypothesis
    category: Literal["expected", "neutral", "surprising"]
    expected_direction: Literal[-1, 1] | None
    magnitude: float = Field(gt=0)
    outside_coefficient: float = 0
    evidence_id: str
    paradigm_bearing_variables: list[str] = Field(default_factory=list)


class Outcome(StrictModel):
    name: str
    intercept: float
    sigma: float = Field(gt=0)
    delta: float = Field(gt=0)
    units: str


class PairSpec(StrictModel):
    schema_version: str = "1.0.0"
    generation_version: Literal[1, 2] = 1
    pair_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
    profile: str
    seed: int = Field(ge=0)
    n: int = Field(ge=100)
    focal_id: str
    focal_mode: Literal["overall", "subgroup"]
    discoveries: list[Discovery] = Field(min_length=6, max_length=6)
    outcomes: list[Outcome]
    evidence: list[ReviewedCandidate]
    status: Literal["development", "locked"] = "development"
    dgp_review: DGPReview | None = None

    @model_validator(mode="after")
    def coherent(self):
        ids = [d.hypothesis.id for d in self.discoveries]
        if len(set(ids)) != len(ids) or self.focal_id not in ids:
            raise ValueError("Discovery IDs must be unique and include the focal discovery")
        focal = next(d for d in self.discoveries if d.hypothesis.id == self.focal_id)
        if focal.category != "expected" or focal.expected_direction != focal.hypothesis.direction:
            raise ValueError("The base specification must contain an expected focal discovery")
        if self.focal_mode == "subgroup" and (
            len({c.variable for c in focal.hypothesis.subgroup}) < 2
            or focal.outside_coefficient == 0
            or focal.hypothesis.contrast != "mean_difference"
        ):
            raise ValueError("Subgroup reversal needs two variables and an expected outside effect")
        outcomes = {o.name for o in self.outcomes}
        if len(outcomes) != len(self.outcomes):
            raise ValueError("Outcome names must be unique")
        evidence = {e.candidate.hypothesis.id: e for e in self.evidence}
        for d in self.discoveries:
            if d.hypothesis.outcome not in outcomes or d.evidence_id not in evidence:
                raise ValueError("Discovery references an undefined outcome or evidence record")
            parent = evidence[d.evidence_id].candidate.hypothesis
            for field in (
                "outcome",
                "exposure",
                "exposed",
                "comparator",
                "contrast",
                "eligibility",
            ):
                if getattr(d.hypothesis, field) != getattr(parent, field):
                    raise ValueError("Compiled comparison differs from reviewed candidate")
            if (d.hypothesis.id != self.focal_id or self.focal_mode != "subgroup") and (
                d.hypothesis.subgroup != parent.subgroup
            ):
                raise ValueError("Compiled subgroup differs from reviewed candidate")
            if d.category == "neutral" and (
                d.expected_direction is not None
                or evidence[d.evidence_id].candidate.proposed_category != "neutral"
            ):
                raise ValueError(
                    "Neutral discovery needs a neutral review and no expected direction"
                )
            if d.category != "neutral":
                expected = evidence[d.evidence_id].candidate.hypothesis.direction
                if evidence[d.evidence_id].review.decision != "supported":
                    raise ValueError(
                        "Expected and surprising discoveries need supported literature"
                    )
                if d.expected_direction != expected:
                    raise ValueError("Expected direction differs from reviewed hypothesis")
                if d.hypothesis.direction != expected * (1 if d.category == "expected" else -1):
                    raise ValueError("Category disagrees with simulated direction")
        return self


class EvidenceResult(StrictModel):
    id: str
    hypothesis_id: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    delta: float = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)
    cell_n: list[int]
    valid: bool
    diagnostic: str

    @model_validator(mode="after")
    def valid_interval(self):
        if self.valid:
            if self.estimate is None or self.lower is None or self.upper is None:
                raise ValueError("Valid evidence requires an estimate and interval")
            if not self.lower <= self.estimate <= self.upper:
                raise ValueError("Estimate must lie within the interval")
        return self


class Decision(StrictModel):
    hypothesis_id: str
    result_id: str
    status: Literal["accept", "reject", "unresolved"]


class StageRecord(StrictModel):
    iteration: int = Field(ge=1)
    stage: Literal["hypothesis", "analysis", "critique", "synthesis"]
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    anticipated_directions: dict[str, Literal[-1, 0, 1]] = Field(default_factory=dict)
    assessments: dict[str, Literal["accept", "reject", "unresolved"]] = Field(default_factory=dict)
    parent_ids: dict[str, str] = Field(default_factory=dict)
    executed_ids: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    validation_request_id: str | None = None
    accepted_ids: list[str] = Field(default_factory=list)
    narrative: str = ""
