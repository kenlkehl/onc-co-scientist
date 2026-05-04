"""Inject paradigm-class associations into a base synthetic dataset.

The base dataset supplies covariate columns (biomarkers, demographics, etc.)
and a treatment assignment; this module adds the outcome columns by summing
the effects of each selected `AssociationSpec`. Each association is evaluated
as either a main effect, a two-way interaction, or a subgroup-conditional
effect, then added (with Gaussian or logistic noise) to produce the outcome.

For the initial cut we support two outcome kinds:

- ``objective_response`` — binary, simulated via logistic link.
- ``pfs_months`` — continuous (months), simulated via linear link
  with Gaussian noise, clipped at zero.

The data-generating process is fully deterministic given ``seed``.

Background-prognostic adjustments are split into two layers: a universal
contribution (ECOG, age, albumin, LDH, weight loss, CRP) implemented in
``cancer_types.base`` and a cancer-specific contribution attached to each
``CancerProfile``. The injector sums both. The legacy
``BACKGROUND_PROGNOSTIC_VARIABLES`` constant equals the union of the shared
variables and the NSCLC profile's variables, preserving its prior value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cancer_types import CancerProfile, get_profile
from .cancer_types.base import (
    SHARED_BACKGROUND_PROGNOSTIC_VARIABLES,
    shared_prognostic_contribution,
)
from .schemas import AssociationForm, AssociationSpec, ParadigmClass

CONTINUOUS_OUTCOMES = {"pfs_months"}
BINARY_OUTCOMES = {"objective_response"}

# Baseline (no associations active) values for each supported outcome.
OUTCOME_BASELINES: dict[str, float] = {
    "pfs_months": 6.0,  # months
    "objective_response": -1.0,  # log-odds (~27% baseline response rate)
}
DEPENDENCY_OUTCOME_PREFIX = "dependency_"
DEPENDENCY_BASELINE = -0.25
DEPENDENCY_SIGMA = 0.35


def background_prognostic_variables(profile: CancerProfile) -> frozenset[str]:
    """Union of universal disease-burden variables and the profile's extras."""
    return SHARED_BACKGROUND_PROGNOSTIC_VARIABLES | profile.background_prognostic_variables


# Legacy module-level constant. Equals the union for the NSCLC profile so
# pre-multi-cancer test fixtures and external callers see the same set they
# always have. New code should call ``background_prognostic_variables(profile)``
# instead.
BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = background_prognostic_variables(
    get_profile("nsclc")
)


@dataclass(frozen=True)
class InjectionResult:
    frame: pd.DataFrame
    outcome_columns: list[str]


def _subgroup_mask(frame: pd.DataFrame, predicate: dict[str, object]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for col, val in predicate.items():
        if col not in frame.columns:
            raise KeyError(f"Subgroup column {col!r} missing from frame")
        if isinstance(val, dict) and ({"min", "max"} & val.keys()):
            low = val.get("min", -np.inf)
            high = val.get("max", np.inf)
            col_vals = frame[col].to_numpy()
            mask &= (col_vals >= low) & (col_vals <= high)
        else:
            mask &= frame[col].to_numpy() == val
    return mask


def _association_contribution(frame: pd.DataFrame, spec: AssociationSpec) -> np.ndarray:
    """Per-row contribution of one association to its outcome's linear predictor."""
    n = len(frame)
    for col in spec.variables:
        if col not in frame.columns and col != spec.outcome:
            raise KeyError(f"Association {spec.id!r} references missing column {col!r}")

    if spec.form is AssociationForm.main_effect:
        active = [c for c in spec.variables if c != spec.outcome]
        if len(active) != 1:
            raise ValueError(
                f"main_effect association {spec.id!r} must name exactly one non-outcome "
                f"variable, got {active!r}"
            )
        return spec.effect_size * frame[active[0]].to_numpy()

    if spec.form is AssociationForm.interaction:
        active = [c for c in spec.variables if c != spec.outcome]
        if len(active) != 2:
            raise ValueError(
                f"interaction association {spec.id!r} must name exactly two non-outcome "
                f"variables, got {active!r}"
            )
        a, b = active
        return spec.effect_size * frame[a].to_numpy() * frame[b].to_numpy()

    if spec.form is AssociationForm.subgroup_conditional:
        if spec.subgroup is None:
            raise ValueError(
                f"subgroup_conditional association {spec.id!r} requires a SubgroupSpec"
            )
        active = [c for c in spec.variables if c != spec.outcome]
        predicate_cols = set(spec.subgroup.predicate)
        drivers = [c for c in active if c not in predicate_cols]
        mask = _subgroup_mask(frame, spec.subgroup.predicate)
        contrib = np.zeros(n, dtype=float)
        if drivers:
            driver = drivers[0]
            contrib[mask] = spec.effect_size * frame.loc[mask, driver].to_numpy()
        else:
            contrib[mask] = spec.effect_size
        return contrib

    raise ValueError(f"unknown AssociationForm {spec.form!r}")


def _profile_prognostic_contribution(
    frame: pd.DataFrame, outcome: str, profile: CancerProfile | None
) -> np.ndarray:
    """Sum the universal and cancer-specific background-prognostic layers.

    ``profile=None`` falls back to the NSCLC profile, preserving the
    pre-refactor behaviour for any caller that hasn't yet started passing a
    profile through.
    """
    active = profile if profile is not None else get_profile("nsclc")
    return shared_prognostic_contribution(frame, outcome) + active.prognostic_contribution(
        frame, outcome
    )


def _sample_binary(linear: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-linear))
    return (rng.random(linear.shape) < probs).astype(int)


def _sample_continuous(
    linear: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    *,
    clip_min: float | None = 0.0,
) -> np.ndarray:
    draws = linear + rng.normal(0.0, sigma, size=linear.shape)
    if clip_min is None:
        return draws
    return np.clip(draws, a_min=clip_min, a_max=None)


def _is_dependency_outcome(outcome: str) -> bool:
    return outcome.startswith(DEPENDENCY_OUTCOME_PREFIX)


def _is_continuous_outcome(outcome: str) -> bool:
    return outcome in CONTINUOUS_OUTCOMES or _is_dependency_outcome(outcome)


def _outcome_baseline(outcome: str) -> float:
    if _is_dependency_outcome(outcome):
        return DEPENDENCY_BASELINE
    return OUTCOME_BASELINES[outcome]


def _outcome_sigma(outcome: str, configured_sigma: float) -> float:
    if _is_dependency_outcome(outcome):
        return DEPENDENCY_SIGMA
    return configured_sigma


def inject_associations(
    base_frame: pd.DataFrame,
    associations: list[AssociationSpec],
    seed: int,
    continuous_outcome_sigma: float = 3.0,
    profile: CancerProfile | None = None,
) -> InjectionResult:
    """Add outcome columns to ``base_frame`` driven by the given associations.

    One outcome column is materialized per unique outcome referenced by the
    associations. Contributions are summed across all associations that share
    that outcome, so hidden-novel signals coexist with concordant and
    discordant main/interaction effects in the same synthetic cohort.

    The cancer-specific portion of the background-prognostic layer is taken
    from ``profile``; ``None`` falls back to NSCLC for backward compatibility.
    """
    rng = np.random.default_rng(seed)
    frame = base_frame.copy()
    outcomes = sorted({spec.outcome for spec in associations})

    for outcome in outcomes:
        if not _is_continuous_outcome(outcome) and outcome not in BINARY_OUTCOMES:
            raise ValueError(
                f"Outcome {outcome!r} is not supported by the initial injector. "
                f"Add a baseline in OUTCOME_BASELINES, use the dependency_ "
                f"prefix, or add a sampler branch."
            )

        baseline = _outcome_baseline(outcome)
        linear = np.full(len(frame), baseline, dtype=float)
        for spec in associations:
            if spec.outcome != outcome:
                continue
            linear += _association_contribution(frame, spec)
        linear += _profile_prognostic_contribution(frame, outcome, profile)

        if outcome in BINARY_OUTCOMES:
            frame[outcome] = _sample_binary(linear, rng)
        else:
            frame[outcome] = _sample_continuous(
                linear,
                rng,
                _outcome_sigma(outcome, continuous_outcome_sigma),
                clip_min=None if _is_dependency_outcome(outcome) else 0.0,
            )

    return InjectionResult(frame=frame, outcome_columns=outcomes)


def summarize_injection(associations: list[AssociationSpec]) -> dict[ParadigmClass, int]:
    counts = {klass: 0 for klass in ParadigmClass}
    for spec in associations:
        counts[spec.paradigm_class] += 1
    return counts
