"""Per-cancer-type synthetic-generation profile contract and shared helpers.

A ``CancerProfile`` bundles everything specific to one cancer type: its
base-frame sampler, its four paradigm-association catalogs, the cancer-specific
background-prognostic contribution, the names of background-prognostic columns
it owns, and the marginal prevalence defaults for its biomarkers. Cancer-
agnostic plumbing (the injector math, distractor sampling, anonymizer, IO,
scoring) lives elsewhere and is reused by every profile.

The shared helpers below produce the universal columns every cancer profile
emits (``patient_id``, demographics, performance status, disease-burden labs)
plus the universal portion of the background-prognostic linear predictor.
NSCLC keeps its original draw order verbatim for byte-identical determinism;
the other profiles use these helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schemas import AssociationSpec

# Universal background-prognostic variables: present in every cancer profile's
# base frame, read by ``shared_prognostic_contribution``. Profiles add their
# own extras (e.g. NSCLC adds smoking_status and squamous histology).
SHARED_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {
        "ecog_ps",
        "age_years",
        "albumin_g_dl",
        "ldh_u_l",
        "weight_loss_pct_6mo",
        "crp_mg_l",
    }
)

_ECOG_CATEGORIES: tuple[int, ...] = (0, 1, 2)
_ECOG_PROBS: tuple[float, ...] = (0.35, 0.50, 0.15)


@dataclass(frozen=True)
class CancerProfile:
    """Everything the generator needs to produce one cancer-type's bundle.

    The runtime type of ``cancer_type`` is the ``CancerType`` StrEnum defined
    in ``cancer_types/__init__.py``; we leave it as ``str`` here to avoid an
    import cycle (``__init__.py`` imports each profile module, each profile
    module imports this file).
    """

    cancer_type: str
    display_name: str
    dataset_id_suffix: str
    base_frame_fn: Callable[..., pd.DataFrame]
    concordant_catalog: Callable[[], list[AssociationSpec]]
    discordant_catalog: Callable[[], list[AssociationSpec]]
    hidden_novel_catalog: Callable[[], list[AssociationSpec]]
    buried_signature_catalog: Callable[[], list[AssociationSpec]]
    prognostic_contribution: Callable[[pd.DataFrame, str], np.ndarray]
    background_prognostic_variables: frozenset[str]
    default_prevalences: dict[str, float]


def sample_demographics(
    rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Universal demographic draws: age, sex, ECOG performance status."""
    return {
        "age_years": np.clip(rng.normal(65, 10, size=n), 30, 90).round(1),
        "sex_female": rng.binomial(1, 0.45, size=n),
        "ecog_ps": rng.choice(_ECOG_CATEGORIES, size=n, p=_ECOG_PROBS),
    }


def sample_disease_burden_labs(
    rng: np.random.Generator, n: int
) -> dict[str, np.ndarray]:
    """Universal disease-burden labs/indices: albumin, LDH, weight loss, CRP, NLR."""
    return {
        "albumin_g_dl": np.round(
            np.clip(rng.normal(3.8, 0.5, size=n), 1.5, 5.5), 1
        ),
        "ldh_u_l": np.round(
            np.clip(rng.lognormal(5.35, 0.35, size=n), 0.0, 2000.0), 2
        ),
        "weight_loss_pct_6mo": np.round(
            np.clip(rng.normal(3.0, 5.0, size=n), 0.0, 40.0), 1
        ),
        "crp_mg_l": np.round(
            np.clip(rng.lognormal(1.20, 1.10, size=n), 0.0, 300.0), 2
        ),
        "nlr": np.round(np.clip(rng.lognormal(1.10, 0.55, size=n), 0.0, 40.0), 2),
    }


def shared_prognostic_contribution(
    frame: pd.DataFrame, outcome: str
) -> np.ndarray:
    """Universal disease-burden adjustments to the outcome linear predictor.

    Reads only the columns named in ``SHARED_BACKGROUND_PROGNOSTIC_VARIABLES``,
    every one of which is produced by ``sample_demographics`` and
    ``sample_disease_burden_labs`` (and so is present in every cancer
    profile's base frame). Profiles add cancer-specific terms via their own
    ``prognostic_contribution``.
    """
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -1.2 * frame["ecog_ps"].to_numpy()
        contrib += -0.02 * (frame["age_years"].to_numpy() - 65.0)
        contrib += +0.5 * (frame["albumin_g_dl"].to_numpy() - 3.5)
        contrib += -0.001 * np.clip(
            frame["ldh_u_l"].to_numpy() - 250.0, a_min=0.0, a_max=750.0
        )
        contrib += -0.08 * frame["weight_loss_pct_6mo"].to_numpy()
        return contrib
    if outcome == "objective_response":
        contrib += -0.4 * frame["ecog_ps"].to_numpy()
        contrib += +0.15 * (frame["albumin_g_dl"].to_numpy() - 3.5)
        contrib += -0.04 * frame["weight_loss_pct_6mo"].to_numpy()
        contrib += -0.10 * (
            np.log1p(frame["crp_mg_l"].to_numpy()) - np.log(6.0)
        )
        return contrib
    return contrib


def marginal_bernoulli(
    rng: np.random.Generator,
    column: str,
    default: float,
    n: int,
    overrides: dict[str, float],
) -> np.ndarray:
    """Sample one Bernoulli column at the configured marginal prevalence."""
    rate = float(overrides.get(column, default))
    return rng.binomial(1, rate, size=n)
