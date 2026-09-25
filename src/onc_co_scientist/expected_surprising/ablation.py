"""Controlled single-outcome projection of an already reviewed synthetic pair."""

from .review_policy import require_current_pair, spec_digest
from .schemas import Outcome, OutcomeProjection, PairSpec


def collapse_outcomes(source: PairSpec, *, outcome="dependency_composite") -> PairSpec:
    """Sum the six conditional signal components and retain the original residual scale.

    Covariate generation, population size, seed, six effects, conditions, directions,
    categories and source evidence are unchanged. The result requires a fresh DGP review;
    it does not claim that the source literature reviewed the composite endpoint.
    """
    require_current_pair(source)
    if not source.profile.endswith("depmap") or source.outcome_projection is not None:
        raise ValueError("Collapse requires an original reviewed DepMap pair")
    source_outcomes = {d.hypothesis.outcome for d in source.discoveries}
    parameters = {
        (o.intercept, o.sigma, o.delta, o.units)
        for o in source.outcomes
        if o.name in source_outcomes
    }
    if len(parameters) != 1:
        raise ValueError("Signal endpoints must share intercept, residual scale, cutoff and units")
    if len(source_outcomes) != 6 or outcome in {o.name for o in source.outcomes}:
        raise ValueError("Collapse requires six distinct signal endpoints and a new outcome name")
    intercept, sigma, delta, _ = parameters.pop()
    data = source.model_dump()
    data.update(
        pair_id=f"{source.pair_id}-single-outcome",
        status="development",
        dgp_review=None,
        outcome_projection=OutcomeProjection(
            source_pair_id=source.pair_id,
            source_spec_sha256=spec_digest(source),
            outcome=outcome,
            source_outcomes={d.hypothesis.id: d.hypothesis.outcome for d in source.discoveries},
        ).model_dump(),
        outcomes=[
            Outcome(
                name=outcome,
                intercept=intercept,
                sigma=sigma,
                delta=delta,
                units="composite dependency-score units",
            ).model_dump()
        ],
    )
    for discovery in data["discoveries"]:
        discovery["hypothesis"]["outcome"] = outcome
    return PairSpec.model_validate(data)
