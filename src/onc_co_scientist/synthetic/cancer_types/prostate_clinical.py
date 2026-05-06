"""Prostate cancer profile.

Models metastatic castration-resistant and castration-sensitive prostate
cancer via canonical biomarkers (BRCA2, AR-V7, MSI status, PSMA expression),
disease state (mCRPC, visceral mets), Gleason grade group, and PSA. Treatment
backbones span next-generation androgen-receptor inhibitors, taxanes, PARP
inhibitors, radioligand therapy, and immune checkpoint blockade. Effect sizes
are deliberately large for evaluator power, not literature estimates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..schemas import AssociationForm, AssociationSpec, ParadigmClass, SubgroupSpec
from .base import (
    CancerProfile,
    marginal_bernoulli,
    sample_demographics,
    sample_disease_burden_labs,
)

if TYPE_CHECKING:
    from ..generator import GeneratorConfig


def concordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="concordant_olaparib_brca2_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_olaparib", "brca2_mutation", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            natural_language_description=(
                "Olaparib produces substantially longer progression-free "
                "survival in BRCA2-mutated metastatic castration-resistant "
                "prostate cancer than in BRCA2-wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_enzalutamide_ar_v7_resistance",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_enzalutamide", "ar_v7_positive", "objective_response"],
            outcome="objective_response",
            direction=-1,
            effect_size=-2.5,
            natural_language_description=(
                "AR-V7-positive metastatic prostate cancer is resistant to "
                "enzalutamide: AR-V7-positive patients have substantially "
                "lower objective response rates on enzalutamide than AR-V7-"
                "negative patients."
            ),
        ),
        AssociationSpec(
            id="concordant_lu177_psma_high_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_lu177_psma", "psma_high", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            natural_language_description=(
                "Lu-177 PSMA radioligand therapy produces high objective "
                "response rates in PSMA-high metastatic castration-resistant "
                "prostate cancer."
            ),
        ),
        AssociationSpec(
            id="concordant_pembrolizumab_msi_high_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_pembrolizumab", "msi_high", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.5,
            natural_language_description=(
                "Pembrolizumab produces high objective response rates in the "
                "rare microsatellite-instability-high subset of metastatic "
                "prostate cancer."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="discordant_docetaxel_high_psa_resistance",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_docetaxel", "ar_v7_positive", "pfs_months"],
            outcome="pfs_months",
            direction=-1,
            effect_size=-3.0,
            natural_language_description=(
                "In this dataset, docetaxel produces SHORTER progression-free "
                "survival in AR-V7-positive prostate cancer than in AR-V7-"
                "negative disease — opposite to the conventional view that "
                "taxanes retain activity regardless of AR-V7 status."
            ),
        ),
        AssociationSpec(
            id="discordant_abiraterone_visceral_mets_benefit",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_abiraterone", "visceral_mets", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+3.0,
            natural_language_description=(
                "In this dataset, abiraterone produces LONGER progression-"
                "free survival in patients with visceral metastases than in "
                "patients with bone-only disease — opposite to the established "
                "pattern of poorer outcomes with visceral involvement."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="hidden_novel_lu177_brca2_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_lu177_psma", "brca2_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="lu177_brca2_subgroup",
                predicate={"brca2_mutation": 1},
                description=("Patients harboring a BRCA2 loss-of-function mutation."),
            ),
            natural_language_description=(
                "Lu-177 PSMA radioligand therapy produces unexpectedly high "
                "objective response rates in the subgroup of patients whose "
                "tumors harbor a BRCA2 mutation, beyond what PSMA expression "
                "alone would predict."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="buried_olaparib_brca2_psma_high_low_gleason_no_visceral",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_olaparib",
                "brca2_mutation",
                "psma_high",
                "gleason_score",
                "visceral_mets",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            subgroup=SubgroupSpec(
                name="olaparib_brca2_psma_high_low_gleason_no_visceral_signature",
                predicate={
                    "brca2_mutation": 1,
                    "psma_high": 1,
                    "gleason_score": {"max": 7},
                    "visceral_mets": 0,
                },
                description=(
                    "BRCA2-mutant, PSMA-high patients with Gleason score <= 7 "
                    "and no visceral metastases."
                ),
            ),
            natural_language_description=(
                "Olaparib produces substantially longer progression-free "
                "survival in the conjunction of BRCA2-mutant, PSMA-high "
                "disease with low-grade histology (Gleason <= 7) and no "
                "visceral metastases — beyond what BRCA2 status alone "
                "predicts."
            ),
        ),
        AssociationSpec(
            id="buried_enzalutamide_ar_v7_neg_msi_low_brca_wt_orr",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_enzalutamide",
                "ar_v7_positive",
                "msi_high",
                "brca2_mutation",
                "mcrpc",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="enzalutamide_ar_v7_neg_mss_brca_wt_csens_signature",
                predicate={
                    "ar_v7_positive": 0,
                    "msi_high": 0,
                    "brca2_mutation": 0,
                    "mcrpc": 0,
                },
                description=(
                    "AR-V7-negative, microsatellite-stable, BRCA2-wildtype "
                    "patients with castration-sensitive disease (not yet "
                    "mCRPC)."
                ),
            ),
            natural_language_description=(
                "Enzalutamide produces exceptional objective response rates "
                "in the conjunction of AR-V7-negative, microsatellite-stable, "
                "BRCA2-wildtype, castration-sensitive disease — beyond what "
                "AR-V7-negative status alone would predict."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Base-frame sampler.
# ---------------------------------------------------------------------------

DEFAULT_PREVALENCES: dict[str, float] = {
    "brca2_mutation": 0.10,  # mostly mCRPC, enriched relative to general population
    "ar_v7_positive": 0.20,
    "msi_high": 0.03,
    "psma_high": 0.60,
    "mcrpc": 0.55,
    "visceral_mets": 0.20,
    "treatment_enzalutamide": 0.40,
    "treatment_abiraterone": 0.30,
    "treatment_docetaxel": 0.30,
    "treatment_olaparib": 0.10,
    "treatment_lu177_psma": 0.15,
    "treatment_pembrolizumab": 0.05,
}


def base_frame_fn(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    demographics = sample_demographics(rng, n)
    # Override sex to all male (prostate cancer cohort).
    demographics["sex_female"] = np.zeros(n, dtype=int)
    labs = sample_disease_burden_labs(rng, n)

    # Disease state.
    mcrpc = marginal_bernoulli(rng, "mcrpc", DEFAULT_PREVALENCES["mcrpc"], n, overrides)
    visceral_mets = marginal_bernoulli(
        rng, "visceral_mets", DEFAULT_PREVALENCES["visceral_mets"], n, overrides
    )

    # Biomarkers.
    brca2_mutation = marginal_bernoulli(
        rng, "brca2_mutation", DEFAULT_PREVALENCES["brca2_mutation"], n, overrides
    )
    ar_v7_positive = marginal_bernoulli(
        rng, "ar_v7_positive", DEFAULT_PREVALENCES["ar_v7_positive"], n, overrides
    )
    msi_high = marginal_bernoulli(rng, "msi_high", DEFAULT_PREVALENCES["msi_high"], n, overrides)
    psma_high = marginal_bernoulli(rng, "psma_high", DEFAULT_PREVALENCES["psma_high"], n, overrides)

    # PSA at study entry: mCRPC has higher and more variable PSA than mCSPC.
    psa_log = np.where(
        mcrpc == 1,
        rng.normal(3.5, 1.2, size=n),  # ~33 ng/mL geometric mean
        rng.normal(2.0, 1.0, size=n),  # ~7 ng/mL geometric mean
    )
    psa_ng_ml = np.round(np.clip(np.exp(psa_log), 0.1, 5000.0), 2)

    # Gleason score: integer 6-10, weighted toward 7 and 8.
    gleason_choices = np.array([6, 7, 8, 9, 10])
    gleason_probs = np.array([0.10, 0.40, 0.30, 0.15, 0.05])
    gleason_score = rng.choice(gleason_choices, size=n, p=gleason_probs)

    treatments = {
        col: marginal_bernoulli(rng, col, DEFAULT_PREVALENCES[col], n, overrides)
        for col in (
            "treatment_enzalutamide",
            "treatment_abiraterone",
            "treatment_docetaxel",
            "treatment_olaparib",
            "treatment_lu177_psma",
            "treatment_pembrolizumab",
        )
    }

    return pd.DataFrame(
        {
            "patient_id": patient_id,
            **demographics,
            "mcrpc": mcrpc,
            "visceral_mets": visceral_mets,
            "psa_ng_ml": psa_ng_ml,
            "gleason_score": gleason_score,
            "brca2_mutation": brca2_mutation,
            "ar_v7_positive": ar_v7_positive,
            "msi_high": msi_high,
            "psma_high": psma_high,
            **labs,
            **treatments,
        }
    )


# ---------------------------------------------------------------------------
# Prostate-specific prognostic contribution.
# ---------------------------------------------------------------------------

# Only ``psa_ng_ml`` is read by the prognostic layer: ``mcrpc``,
# ``visceral_mets`` and ``gleason_score`` are paradigm-catalog or buried-
# predicate variables for prostate cancer, so the disjointness invariant
# excludes them here. PSA still carries a strong disease-burden signal that
# keeps the cohort from looking implausibly noiseless.
PROSTATE_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset({"psa_ng_ml"})


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -0.4 * np.log1p(frame["psa_ng_ml"].to_numpy())
        return contrib
    if outcome == "objective_response":
        contrib += -0.10 * np.log1p(frame["psa_ng_ml"].to_numpy())
        return contrib
    return contrib


PROFILE = CancerProfile(
    cancer_type="prostate_clinical",
    display_name="Prostate cancer",
    dataset_id_suffix="prostate_clinical",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=PROSTATE_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
)
