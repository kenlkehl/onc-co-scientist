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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import AssociationForm, AssociationSpec, ParadigmClass

CONTINUOUS_OUTCOMES = {"pfs_months"}
BINARY_OUTCOMES = {"objective_response"}

# Baseline (no associations active) values for each supported outcome.
OUTCOME_BASELINES: dict[str, float] = {
    "pfs_months": 6.0,  # months
    "objective_response": -1.0,  # log-odds (~27% baseline response rate)
}

# Variables that the background-prognostic layer is allowed to read. Kept as a
# constant so the test suite can assert disjointness from any AssociationSpec
# in the paradigm catalog — no overlap means a discordant tag (e.g. high TMB
# → worse response) cannot be neutralized by an unscored "conventional" effect
# embedded in the same DGP.
BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {
        "ecog_ps",
        "stage_iv",
        "has_brain_mets",
        "age_years",
        "smoking_status",
        "histology",
        "albumin_g_dl",
        "ldh_u_l",
        "weight_loss_pct_6mo",
        "crp_mg_l",
    }
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


def _association_contribution(
    frame: pd.DataFrame, spec: AssociationSpec
) -> np.ndarray:
    """Per-row contribution of one association to its outcome's linear predictor."""
    n = len(frame)
    for col in spec.variables:
        if col not in frame.columns and col != spec.outcome:
            raise KeyError(
                f"Association {spec.id!r} references missing column {col!r}"
            )

    if spec.form is AssociationForm.main_effect:
        # Single active variable besides the outcome itself.
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
        # The "driver" is the treatment / factor whose activity is conditional
        # on the subgroup. We take the first non-outcome variable that is not
        # a subgroup predicate column.
        predicate_cols = set(spec.subgroup.predicate)
        drivers = [c for c in active if c not in predicate_cols]
        if not drivers:
            raise ValueError(
                f"subgroup_conditional association {spec.id!r} needs at least one driver "
                f"variable outside the subgroup predicate"
            )
        driver = drivers[0]
        mask = _subgroup_mask(frame, spec.subgroup.predicate)
        contrib = np.zeros(n, dtype=float)
        contrib[mask] = spec.effect_size * frame.loc[mask, driver].to_numpy()
        return contrib

    raise ValueError(f"unknown AssociationForm {spec.form!r}")


def _background_prognostic_contribution(
    frame: pd.DataFrame, outcome: str
) -> np.ndarray:
    """Per-row background-prognostic adjustment to the linear predictor.

    These effects exist solely to make the cohort feel like a real EHR pull:
    classic disease-burden variables (ECOG, stage, brain mets, weight loss,
    albumin, LDH, CRP) drive realistic outcome variance. They are *not*
    paradigm-tagged and never enter the manifest, so scoring is unaffected.

    By construction the variables consumed here are disjoint from every
    paradigm-association variable set; see ``BACKGROUND_PROGNOSTIC_VARIABLES``.
    """
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -1.2 * frame["ecog_ps"].to_numpy()
        contrib += -1.5 * frame["stage_iv"].to_numpy()
        contrib += -1.0 * frame["has_brain_mets"].to_numpy()
        contrib += -0.02 * (frame["age_years"].to_numpy() - 65.0)
        contrib += -0.6 * (frame["smoking_status"].to_numpy() == "current").astype(float)
        contrib += -0.8 * (frame["histology"].to_numpy() == "squamous").astype(float)
        contrib += +0.5 * (frame["albumin_g_dl"].to_numpy() - 3.5)
        contrib += -0.001 * np.clip(
            frame["ldh_u_l"].to_numpy() - 250.0, a_min=0.0, a_max=750.0
        )
        contrib += -0.08 * frame["weight_loss_pct_6mo"].to_numpy()
        return contrib
    if outcome == "objective_response":
        contrib += -0.4 * frame["ecog_ps"].to_numpy()
        contrib += -0.3 * frame["stage_iv"].to_numpy()
        contrib += -0.25 * frame["has_brain_mets"].to_numpy()
        contrib += +0.15 * (frame["albumin_g_dl"].to_numpy() - 3.5)
        contrib += -0.04 * frame["weight_loss_pct_6mo"].to_numpy()
        contrib += -0.10 * (np.log1p(frame["crp_mg_l"].to_numpy()) - np.log(6.0))
        return contrib
    return contrib


def _sample_binary(linear: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-linear))
    return (rng.random(linear.shape) < probs).astype(int)


def _sample_continuous(
    linear: np.ndarray, rng: np.random.Generator, sigma: float
) -> np.ndarray:
    draws = linear + rng.normal(0.0, sigma, size=linear.shape)
    return np.clip(draws, a_min=0.0, a_max=None)


def inject_associations(
    base_frame: pd.DataFrame,
    associations: list[AssociationSpec],
    seed: int,
    continuous_outcome_sigma: float = 3.0,
) -> InjectionResult:
    """Add outcome columns to ``base_frame`` driven by the given associations.

    One outcome column is materialized per unique outcome referenced by the
    associations. Contributions are summed across all associations that share
    that outcome, so hidden-novel signals coexist with concordant and
    discordant main/interaction effects in the same synthetic cohort.
    """
    rng = np.random.default_rng(seed)
    frame = base_frame.copy()
    outcomes = sorted({spec.outcome for spec in associations})

    for outcome in outcomes:
        if outcome not in CONTINUOUS_OUTCOMES and outcome not in BINARY_OUTCOMES:
            raise ValueError(
                f"Outcome {outcome!r} is not supported by the initial injector. "
                f"Add a baseline in OUTCOME_BASELINES and a sampler branch."
            )

        baseline = OUTCOME_BASELINES[outcome]
        linear = np.full(len(frame), baseline, dtype=float)
        for spec in associations:
            if spec.outcome != outcome:
                continue
            linear += _association_contribution(frame, spec)
        linear += _background_prognostic_contribution(frame, outcome)

        if outcome in BINARY_OUTCOMES:
            frame[outcome] = _sample_binary(linear, rng)
        else:
            frame[outcome] = _sample_continuous(linear, rng, continuous_outcome_sigma)

    return InjectionResult(frame=frame, outcome_columns=outcomes)


def summarize_injection(associations: list[AssociationSpec]) -> dict[ParadigmClass, int]:
    counts = {klass: 0 for klass in ParadigmClass}
    for spec in associations:
        counts[spec.paradigm_class] += 1
    return counts
