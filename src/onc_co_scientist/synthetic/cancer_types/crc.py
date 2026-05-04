"""Colorectal cancer (CRC) profile.

Biomarkers and treatments draw from established mCRC paradigms: anti-EGFR
sensitivity in RAS/RAF-wildtype tumors, immune checkpoint blockade in MSI-H,
BRAF-targeted combinations in BRAF V600E disease, HER2-directed therapy in
HER2-amplified tumors. Effect sizes are deliberately large (not literature
estimates) so that a sufficiently powered cohort recovers them; the eval is
about whether the agent reaches the right analysis at all.
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
            id="concordant_cetuximab_ras_wildtype_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_cetuximab", "kras_mutation", "pfs_months"],
            outcome="pfs_months",
            direction=-1,
            effect_size=-4.0,
            natural_language_description=(
                "Cetuximab is far less effective (shorter progression-free "
                "survival) in KRAS-mutant metastatic colorectal cancer than in "
                "KRAS-wildtype disease."
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
                "Pembrolizumab produces high objective response rates in "
                "microsatellite-instability-high (MSI-H) colorectal cancer."
            ),
        ),
        AssociationSpec(
            id="concordant_encorafenib_braf_v600e_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_encorafenib", "braf_v600e", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+4.0,
            natural_language_description=(
                "Encorafenib produces longer progression-free survival in BRAF "
                "V600E-mutant metastatic colorectal cancer than in BRAF-"
                "wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_trastuzumab_tucatinib_her2_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_trastuzumab_tucatinib",
                "her2_amplified",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+2.8,
            natural_language_description=(
                "The combination of trastuzumab and tucatinib produces high "
                "objective response rates in HER2-amplified metastatic "
                "colorectal cancer."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="discordant_bevacizumab_left_sided_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_bevacizumab", "right_sided_primary", "pfs_months"],
            outcome="pfs_months",
            direction=-1,
            effect_size=-3.0,
            natural_language_description=(
                "In this dataset, bevacizumab produces SHORTER progression-free "
                "survival in right-sided colorectal primaries than in left-"
                "sided primaries — opposite to the conventional teaching that "
                "bevacizumab benefit is sidedness-agnostic or favored on the "
                "right."
            ),
        ),
        AssociationSpec(
            id="discordant_pembrolizumab_msi_high_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_pembrolizumab", "msi_high", "pfs_months"],
            outcome="pfs_months",
            direction=-1,
            effect_size=-3.0,
            natural_language_description=(
                "In this dataset, pembrolizumab is associated with SHORTER "
                "progression-free survival in MSI-high colorectal cancer than "
                "in microsatellite-stable disease — opposite to the established "
                "MSI-H benefit pattern."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="hidden_novel_olaparib_brca_subgroup_crc",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_pembrolizumab", "ntrk_fusion", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="ntrk_fusion_subgroup",
                predicate={"ntrk_fusion": 1},
                description=("Patients whose tumors harbor an NTRK gene fusion."),
            ),
            natural_language_description=(
                "Pembrolizumab, broadly inactive in microsatellite-stable "
                "colorectal cancer, produces unexpectedly high objective "
                "response rates in the small subgroup of patients whose tumors "
                "harbor an NTRK gene fusion."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="buried_regorafenib_left_sided_raswt_brafwt_lowcea",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_regorafenib",
                "right_sided_primary",
                "kras_mutation",
                "braf_v600e",
                "cea_ng_ml",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            subgroup=SubgroupSpec(
                name="regorafenib_left_raswt_brafwt_lowcea_signature",
                predicate={
                    "right_sided_primary": 0,
                    "kras_mutation": 0,
                    "braf_v600e": 0,
                    "cea_ng_ml": {"max": 5.0},
                },
                description=(
                    "Left-sided KRAS-wildtype, BRAF-wildtype patients with a "
                    "low baseline CEA (<= 5 ng/mL)."
                ),
            ),
            natural_language_description=(
                "Regorafenib produces substantially longer progression-free "
                "survival in the conjunction of left-sided primary, KRAS-"
                "wildtype, BRAF-wildtype, and low baseline CEA — none of these "
                "features individually predicts the magnitude of benefit "
                "observed in the joint subgroup."
            ),
        ),
        AssociationSpec(
            id="buried_cetuximab_msshigh_raswt_brafwt_lowalp",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_cetuximab",
                "msi_high",
                "kras_mutation",
                "braf_v600e",
                "her2_amplified",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="cetuximab_mss_raswt_brafwt_her2wt_signature",
                predicate={
                    "msi_high": 0,
                    "kras_mutation": 0,
                    "braf_v600e": 0,
                    "her2_amplified": 0,
                },
                description=(
                    "Microsatellite-stable, KRAS-wildtype, BRAF-wildtype, "
                    "HER2-non-amplified patients."
                ),
            ),
            natural_language_description=(
                "Cetuximab produces exceptional objective response rates in "
                "the conjunction of microsatellite-stable, KRAS-wildtype, "
                "BRAF-wildtype, and HER2-non-amplified tumors — beyond what "
                "RAS-wildtype status alone would predict."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Base-frame sampler.
# ---------------------------------------------------------------------------

DEFAULT_PREVALENCES: dict[str, float] = {
    "kras_mutation": 0.42,
    "nras_mutation": 0.05,
    "braf_v600e": 0.08,
    "msi_high": 0.05,  # mostly mCRC; localized MSI-H is higher
    "her2_amplified": 0.03,
    "ntrk_fusion": 0.005,
    "right_sided_primary": 0.35,
    "stage_iv": 0.55,
    "treatment_cetuximab": 0.30,
    "treatment_bevacizumab": 0.45,
    "treatment_pembrolizumab": 0.15,
    "treatment_encorafenib": 0.10,
    "treatment_trastuzumab_tucatinib": 0.08,
    "treatment_regorafenib": 0.20,
}


def base_frame_fn(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    demographics = sample_demographics(rng, n)
    labs = sample_disease_burden_labs(rng, n)

    # CRC-specific biomarkers and clinical variables.
    stage_iv = marginal_bernoulli(rng, "stage_iv", DEFAULT_PREVALENCES["stage_iv"], n, overrides)
    right_sided = marginal_bernoulli(
        rng, "right_sided_primary", DEFAULT_PREVALENCES["right_sided_primary"], n, overrides
    )
    kras_mutation = marginal_bernoulli(
        rng, "kras_mutation", DEFAULT_PREVALENCES["kras_mutation"], n, overrides
    )
    # NRAS and BRAF V600E are mutually exclusive with KRAS in most tumors;
    # we enforce that approximately by drawing them only for KRAS-WT rows.
    nras_base = rng.binomial(
        1, float(overrides.get("nras_mutation", DEFAULT_PREVALENCES["nras_mutation"])), size=n
    )
    nras_mutation = np.where(kras_mutation == 1, 0, nras_base).astype(int)
    braf_base = rng.binomial(
        1, float(overrides.get("braf_v600e", DEFAULT_PREVALENCES["braf_v600e"])), size=n
    )
    braf_v600e = np.where((kras_mutation == 1) | (nras_mutation == 1), 0, braf_base).astype(int)
    msi_high = marginal_bernoulli(rng, "msi_high", DEFAULT_PREVALENCES["msi_high"], n, overrides)
    her2_amplified = marginal_bernoulli(
        rng, "her2_amplified", DEFAULT_PREVALENCES["her2_amplified"], n, overrides
    )
    ntrk_fusion = marginal_bernoulli(
        rng, "ntrk_fusion", DEFAULT_PREVALENCES["ntrk_fusion"], n, overrides
    )

    # CEA is right-skewed; lognormal centered around 3-10 ng/mL with a long tail.
    cea_ng_ml = np.round(np.clip(rng.lognormal(1.5, 1.2, size=n), 0.0, 5000.0), 2)

    # Treatments (independent of biomarkers — randomized-like).
    treatments = {
        col: marginal_bernoulli(rng, col, DEFAULT_PREVALENCES[col], n, overrides)
        for col in (
            "treatment_cetuximab",
            "treatment_bevacizumab",
            "treatment_pembrolizumab",
            "treatment_encorafenib",
            "treatment_trastuzumab_tucatinib",
            "treatment_regorafenib",
        )
    }

    return pd.DataFrame(
        {
            "patient_id": patient_id,
            **demographics,
            "stage_iv": stage_iv,
            "right_sided_primary": right_sided,
            "kras_mutation": kras_mutation,
            "nras_mutation": nras_mutation,
            "braf_v600e": braf_v600e,
            "msi_high": msi_high,
            "her2_amplified": her2_amplified,
            "ntrk_fusion": ntrk_fusion,
            "cea_ng_ml": cea_ng_ml,
            **labs,
            **treatments,
        }
    )


# ---------------------------------------------------------------------------
# CRC-specific prognostic contribution.
# ---------------------------------------------------------------------------

CRC_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset({"stage_iv"})


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -1.4 * frame["stage_iv"].to_numpy()
        return contrib
    if outcome == "objective_response":
        contrib += -0.3 * frame["stage_iv"].to_numpy()
        return contrib
    return contrib


PROFILE = CancerProfile(
    cancer_type="crc",
    display_name="Colorectal cancer",
    dataset_id_suffix="crc",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=CRC_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
)
