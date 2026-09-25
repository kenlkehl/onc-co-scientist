"""Calibrated metadata for synthetic CRISPR/DepMap model records.

The categorical marginals, library combinations, QC marginals, correlations,
and observed-data patterns are calibrated to DepMap Public 26Q1. The sampler
is deliberately joint: lineage affects demographics and growth, omics and
library measurements are coherent combinations, and screen-quality measures
share a Gaussian-copula latent structure. This produces nuisance variation
that can confound dependency analyses in realistic ways.

Calibration source hashes (SHA-256):

* Model.csv: ea4e0b2a3bc806f81df62689a5ae75f1a100135727a3d7b8a4c7ccc8815183f8
* CRISPRConfounders.csv: dcc48ca7bc0584a931cea86a4fc840acf20917ac4b5f3efccf631da70ea6f0e7
* OmicsProfiles.csv: 978f9ff9a11214e9d923619c20d2bc36bf4ca3e30fd46a6d9a424ea0e7c49bd2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEPMAP_CALIBRATION_RELEASE = "DepMap Public 26Q1"

_AGE_CATEGORY_PROBS: dict[str, tuple[float, float, float]] = {
    "lung": (0.79, 0.02, 0.19),
    "colorectal": (0.84, 0.03, 0.13),
    "breast": (0.83, 0.03, 0.14),
    "prostate": (0.95, 0.00, 0.05),
    "hematopoietic": (0.68, 0.12, 0.20),
}
_SEX_PROBS: dict[str, tuple[float, float, float]] = {
    "lung": (0.35, 0.60, 0.05),
    "colorectal": (0.40, 0.55, 0.05),
    "breast": (0.96, 0.02, 0.02),
    "prostate": (0.00, 0.99, 0.01),
    "hematopoietic": (0.25, 0.68, 0.07),
}
_GROWTH_PROBS: dict[str, tuple[float, float, float, float]] = {
    "lung": (0.75, 0.08, 0.12, 0.05),
    "colorectal": (0.78, 0.05, 0.12, 0.05),
    "breast": (0.77, 0.07, 0.12, 0.04),
    "prostate": (0.75, 0.07, 0.13, 0.05),
    "hematopoietic": (0.00, 0.95, 0.04, 0.01),
}
_OMICS_PROBS: dict[str, tuple[float, float, float, float, float, float]] = {
    "lung": (0.57, 0.24, 0.08, 0.04, 0.01, 0.06),
    "colorectal": (0.58, 0.23, 0.08, 0.04, 0.01, 0.06),
    "breast": (0.58, 0.23, 0.08, 0.04, 0.01, 0.06),
    "prostate": (0.53, 0.27, 0.10, 0.04, 0.01, 0.05),
    "hematopoietic": (0.47, 0.22, 0.17, 0.06, 0.01, 0.07),
}

_AGE_CATEGORIES = ("adult", "pediatric", "fetus_or_unknown")
_SEX_CATEGORIES = ("female", "male", "unknown")
_GROWTH_CATEGORIES = ("adherent", "suspension", "mixed/dome/spheroid", "unknown")
_OMICS_CATEGORIES = ("RNA+WGS", "RNA+WES", "WES only", "RNA only", "WGS only", "none")
_LIBRARY_CATEGORIES = (
    "Avana",
    "Avana + KY",
    "KY",
    "Humagne-CD",
    "Avana + Humagne-CD",
    "Avana + Humagne-CD + KY",
)
# Joint combinations among 1,208 screened 26Q1 models. Marginal percentages
# intentionally sum above 100% because a model may have multiple libraries.
_LIBRARY_PROBS = np.array([828, 191, 119, 43, 21, 6], dtype=float) / 1208

# Pearson-scale correlation for the latent Gaussian variables. Monotone
# transforms yield approximately the observed 26Q1 Spearman relationships:
# NNMD/AUC -0.80, NNMD/Cas9 -0.38, NNMD/doubling +0.14,
# AUC/Cas9 +0.27, and AUC/doubling -0.26.
_QC_LATENT_CORRELATION = np.array(
    [
        [1.000, 0.818, 0.400, -0.144],
        [0.818, 1.000, 0.282, -0.274],
        [0.400, 0.282, 1.000, -0.100],
        [-0.144, -0.274, -0.100, 1.000],
    ]
)


@dataclass(frozen=True)
class DepMapMetadata:
    """Observed metadata plus complete latent values used by the outcome DGP."""

    frame: pd.DataFrame
    latent_cas9_activity_pct: np.ndarray
    latent_doubling_time_hours: np.ndarray


def _choice_by_lineage(
    rng: np.random.Generator,
    lineage: np.ndarray,
    choices: tuple[str, ...],
    probabilities: dict[str, tuple[float, ...]],
) -> np.ndarray:
    result = np.empty(len(lineage), dtype=object)
    fallback = tuple(np.mean(np.asarray(list(probabilities.values())), axis=0))
    for value in np.unique(lineage):
        mask = lineage == value
        probs = probabilities.get(str(value), fallback)
        result[mask] = rng.choice(choices, size=int(mask.sum()), p=probs)
    return result


def _lineage_shift(lineage: np.ndarray, shifts: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(lineage), dtype=float)
    for value, shift in shifts.items():
        result[lineage == value] = shift
    return result


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def simulate_depmap_metadata(
    rng: np.random.Generator,
    lineage: np.ndarray,
) -> DepMapMetadata:
    """Simulate realistic model metadata and correlated CRISPR-screen QC.

    NNMD and ROC AUC are observed for every synthetic screen. Complete Cas9
    activity and doubling-time values are generated for the dependency DGP,
    then observation masks are applied according to library and growth
    pattern. This mirrors the 26Q1 pattern in which KY-only models usually
    lack those two ancillary measures.
    """

    n = len(lineage)
    age_category = _choice_by_lineage(rng, lineage, _AGE_CATEGORIES, _AGE_CATEGORY_PROBS)
    sex = _choice_by_lineage(rng, lineage, _SEX_CATEGORIES, _SEX_PROBS)
    growth_pattern = _choice_by_lineage(rng, lineage, _GROWTH_CATEGORIES, _GROWTH_PROBS)
    default_omics_profile = _choice_by_lineage(rng, lineage, _OMICS_CATEGORIES, _OMICS_PROBS)
    crispr_library = rng.choice(_LIBRARY_CATEGORIES, size=n, p=_LIBRARY_PROBS)

    age_years = np.full(n, np.nan)
    adult = age_category == "adult"
    pediatric = age_category == "pediatric"
    unknown = age_category == "fetus_or_unknown"
    age_years[adult] = np.rint(18.0 + 76.0 * rng.beta(3.8, 4.15, size=int(adult.sum())))
    age_years[pediatric] = np.rint(17.0 * rng.beta(1.8, 1.7, size=int(pediatric.sum())))
    # Most records in the combined category have no recorded age, while a
    # small fetal subset is recorded as age zero in the source metadata.
    fetal_observed = unknown & (rng.random(n) < 0.055)
    age_years[fetal_observed] = 0.0
    age_years[(~unknown) & (rng.random(n) < 0.035)] = np.nan

    latent = rng.multivariate_normal(mean=np.zeros(4), cov=_QC_LATENT_CORRELATION, size=n)
    library_ky_only = crispr_library == "KY"
    library_avana = np.char.find(crispr_library.astype(str), "Avana") >= 0
    growth_mixed = growth_pattern == "mixed/dome/spheroid"
    growth_suspension = growth_pattern == "suspension"

    quality_shift = (
        0.12 * library_avana
        - 0.22 * library_ky_only
        - 0.10 * (growth_pattern == "unknown")
        + _lineage_shift(lineage, {"prostate": -0.10, "hematopoietic": 0.08})
    )
    nnmd_latent = latent[:, 0] + quality_shift
    auc_latent = latent[:, 1] + quality_shift
    cas9_latent = latent[:, 2] + 0.10 * library_avana - 0.18 * library_ky_only - 0.08 * growth_mixed
    doubling_latent = (
        latent[:, 3]
        + 0.18 * growth_mixed
        - 0.15 * growth_suspension
        + _lineage_shift(lineage, {"prostate": 0.16, "breast": 0.08})
    )

    screen_nnmd = -np.exp(np.log(7.1) + 0.405 * nnmd_latent)
    screen_nnmd = np.round(np.clip(screen_nnmd, -22.50, -1.28), 3)
    screen_roc_auc = 0.70 + 0.295 * _expit(1.82 + 0.90 * auc_latent)
    screen_roc_auc = np.round(np.clip(screen_roc_auc, 0.70, 0.994), 4)
    latent_cas9 = 100.0 * _expit(1.32 + 1.03 * cas9_latent)
    latent_cas9 = np.clip(latent_cas9, 10.0, 99.6)
    latent_doubling = 52.0 * np.exp(0.42 * doubling_latent)
    long_tail = rng.random(n) < 0.001
    latent_doubling[long_tail] *= rng.uniform(3.0, 10.0, size=int(long_tail.sum()))
    latent_doubling = np.clip(latent_doubling, 20.0, 938.0)

    cas9_probability = np.select(
        [
            crispr_library == "Avana",
            crispr_library == "Avana + KY",
            crispr_library == "KY",
            crispr_library == "Humagne-CD",
            crispr_library == "Avana + Humagne-CD",
        ],
        [0.95, 0.98, 0.03, 0.85, 0.98],
        default=0.99,
    )
    doubling_probability = np.select(
        [
            crispr_library == "Avana",
            crispr_library == "Avana + KY",
            crispr_library == "KY",
            crispr_library == "Humagne-CD",
            crispr_library == "Avana + Humagne-CD",
        ],
        [0.94, 0.95, 0.03, 0.75, 0.94],
        default=0.97,
    )
    missingness_multiplier = np.select(
        [growth_pattern == "unknown", growth_mixed], [0.65, 0.92], default=1.0
    )
    cas9_observed = rng.random(n) < cas9_probability * missingness_multiplier
    doubling_observed = rng.random(n) < doubling_probability * missingness_multiplier
    cas9_activity_pct = np.where(cas9_observed, np.round(latent_cas9, 1), np.nan)
    screen_doubling_time_hours = np.where(doubling_observed, np.rint(latent_doubling), np.nan)

    has_rna = np.isin(default_omics_profile, ("RNA+WGS", "RNA+WES", "RNA only"))
    has_dna = np.isin(default_omics_profile, ("RNA+WGS", "RNA+WES", "WES only", "WGS only"))
    frame = pd.DataFrame(
        {
            "age_years": age_years,
            "age_category": age_category,
            "sex": sex,
            "growth_pattern": growth_pattern,
            "default_omics_profile": default_omics_profile,
            "has_rna_omics": has_rna.astype(int),
            "has_dna_omics": has_dna.astype(int),
            "has_matched_rna_dna": (has_rna & has_dna).astype(int),
            "has_crispr_qc": np.ones(n, dtype=int),
            "has_rna_dna_crispr_qc": (has_rna & has_dna).astype(int),
            "crispr_library": crispr_library,
            "crispr_library_avana": library_avana.astype(int),
            "crispr_library_humagne_cd": (
                np.char.find(crispr_library.astype(str), "Humagne-CD") >= 0
            ).astype(int),
            "crispr_library_ky": (np.char.find(crispr_library.astype(str), "KY") >= 0).astype(int),
            "screen_nnmd": screen_nnmd,
            "screen_roc_auc": screen_roc_auc,
            "cas9_activity_pct": cas9_activity_pct,
            "screen_doubling_time_hours": screen_doubling_time_hours,
        }
    )
    return DepMapMetadata(
        frame=frame,
        latent_cas9_activity_pct=latent_cas9,
        latent_doubling_time_hours=latent_doubling,
    )
