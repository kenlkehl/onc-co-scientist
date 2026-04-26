"""Breast cancer profile.

Models metastatic breast cancer subgroups via canonical biomarkers (ER/PR
hormone-receptor status, HER2 expression, BRCA1/2, PIK3CA), proliferation
index (Ki67), and menopausal/nodal context. Treatments span endocrine
therapy, CDK4/6 inhibition, HER2-directed therapy, PARP inhibition, antibody-
drug conjugates, and immune checkpoint blockade. Effect sizes are
deliberately large for evaluator power, not literature estimates.
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
            id="concordant_trastuzumab_her2_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_trastuzumab", "her2_positive", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.5,
            natural_language_description=(
                "Trastuzumab produces substantially longer progression-free "
                "survival in HER2-positive metastatic breast cancer than in "
                "HER2-negative disease."
            ),
        ),
        AssociationSpec(
            id="concordant_palbociclib_hr_pos_her2_neg_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_palbociclib", "er_positive", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+4.0,
            natural_language_description=(
                "Palbociclib added to endocrine therapy produces longer "
                "progression-free survival in ER-positive metastatic breast "
                "cancer than in ER-negative disease."
            ),
        ),
        AssociationSpec(
            id="concordant_olaparib_brca_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_olaparib", "brca2_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+2.5,
            natural_language_description=(
                "Olaparib produces objective responses in germline BRCA-"
                "mutated metastatic breast cancer that are not seen in BRCA-"
                "wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_pembrolizumab_tnbc_pdl1_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_pembrolizumab",
                "er_positive",
                "objective_response",
            ],
            outcome="objective_response",
            direction=-1,
            effect_size=-1.8,
            natural_language_description=(
                "Pembrolizumab produces lower objective response rates in ER-"
                "positive metastatic breast cancer than in triple-negative "
                "(ER-negative) disease."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="discordant_tamoxifen_er_neg_benefit",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_tamoxifen", "er_positive", "pfs_months"],
            outcome="pfs_months",
            direction=-1,
            effect_size=-3.5,
            natural_language_description=(
                "In this dataset, tamoxifen produces SHORTER progression-free "
                "survival in ER-positive metastatic breast cancer than in ER-"
                "negative disease — opposite to the established hormone-"
                "receptor-driven sensitivity pattern."
            ),
        ),
        AssociationSpec(
            id="discordant_trastuzumab_her2_low_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_trastuzumab", "her2_low", "objective_response"],
            outcome="objective_response",
            direction=-1,
            effect_size=-2.0,
            natural_language_description=(
                "In this dataset, trastuzumab produces LOWER objective "
                "response rates in HER2-low breast cancer than in HER2-"
                "negative disease."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="hidden_novel_sacituzumab_pik3ca_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_sacituzumab_govitecan",
                "pik3ca_mutation",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="pik3ca_mutant_subgroup",
                predicate={"pik3ca_mutation": 1},
                description=(
                    "Patients whose tumors harbor an activating PIK3CA "
                    "mutation."
                ),
            ),
            natural_language_description=(
                "Sacituzumab govitecan, a TROP2-directed antibody-drug "
                "conjugate, produces unexpectedly high objective response "
                "rates in the subgroup of patients whose tumors harbor an "
                "activating PIK3CA mutation."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="buried_palbociclib_postmenopausal_hr_low_ki67_pik3ca_wt",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_palbociclib",
                "er_positive",
                "her2_positive",
                "pik3ca_mutation",
                "ki67_pct",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            subgroup=SubgroupSpec(
                name="palbociclib_hr_pos_her2_neg_pik3ca_wt_low_ki67_signature",
                predicate={
                    "er_positive": 1,
                    "her2_positive": 0,
                    "pik3ca_mutation": 0,
                    "ki67_pct": {"max": 14.0},
                },
                description=(
                    "ER-positive, HER2-negative, PIK3CA-wildtype patients "
                    "with low proliferation index (Ki67 <= 14%)."
                ),
            ),
            natural_language_description=(
                "Palbociclib produces substantially longer progression-free "
                "survival in the conjunction of ER-positive, HER2-negative, "
                "PIK3CA-wildtype disease with a low Ki67 — beyond what hormone-"
                "receptor status alone would predict."
            ),
        ),
        AssociationSpec(
            id="buried_olaparib_brca_postmenopausal_node_neg_orr",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_olaparib",
                "brca1_mutation",
                "brca2_mutation",
                "postmenopausal",
                "node_positive",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="olaparib_brca_pre_node_neg_signature",
                predicate={
                    "brca2_mutation": 1,
                    "brca1_mutation": 0,
                    "postmenopausal": 0,
                    "node_positive": 0,
                },
                description=(
                    "BRCA2-mutant, BRCA1-wildtype, premenopausal, node-"
                    "negative patients."
                ),
            ),
            natural_language_description=(
                "Olaparib produces exceptional objective response rates in "
                "the conjunction of BRCA2-mutant, BRCA1-wildtype, "
                "premenopausal, node-negative disease — beyond what BRCA "
                "status alone predicts."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Base-frame sampler.
# ---------------------------------------------------------------------------

DEFAULT_PREVALENCES: dict[str, float] = {
    "er_positive": 0.70,
    "pr_positive": 0.60,  # mostly tracks ER but not perfectly
    "her2_positive": 0.18,
    "her2_low": 0.45,  # IHC 1+ or 2+ FISH-negative; a different axis
    "brca1_mutation": 0.025,
    "brca2_mutation": 0.025,
    "pik3ca_mutation": 0.35,
    "postmenopausal": 0.60,
    "node_positive": 0.50,
    "stage_iv": 0.30,
    "has_brain_mets": 0.10,
    "treatment_tamoxifen": 0.30,
    "treatment_palbociclib": 0.35,
    "treatment_trastuzumab": 0.20,
    "treatment_olaparib": 0.10,
    "treatment_sacituzumab_govitecan": 0.10,
    "treatment_pembrolizumab": 0.15,
}


def base_frame_fn(config: "GeneratorConfig") -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    demographics = sample_demographics(rng, n)
    labs = sample_disease_burden_labs(rng, n)

    # Hormone receptor status. PR positivity is enriched in ER-positive.
    er_positive = marginal_bernoulli(
        rng, "er_positive", DEFAULT_PREVALENCES["er_positive"], n, overrides
    )
    pr_base_positive = rng.binomial(1, 0.85, size=n)
    pr_base_negative = rng.binomial(1, 0.15, size=n)
    pr_positive = np.where(er_positive == 1, pr_base_positive, pr_base_negative).astype(int)
    if "pr_positive" in overrides:
        pr_positive = rng.binomial(1, float(overrides["pr_positive"]), size=n)

    # HER2: positive (~18%) and low (subset of HER2-non-positive). Mutually
    # exclusive: a tumor cannot be both HER2-positive and HER2-low.
    her2_positive = marginal_bernoulli(
        rng, "her2_positive", DEFAULT_PREVALENCES["her2_positive"], n, overrides
    )
    her2_low_base = rng.binomial(1, float(overrides.get("her2_low", DEFAULT_PREVALENCES["her2_low"])), size=n)
    her2_low = np.where(her2_positive == 1, 0, her2_low_base).astype(int)

    # BRCA1/2: mutually exclusive (very rare to co-occur).
    brca1_mutation = marginal_bernoulli(
        rng, "brca1_mutation", DEFAULT_PREVALENCES["brca1_mutation"], n, overrides
    )
    brca2_base = rng.binomial(1, float(overrides.get("brca2_mutation", DEFAULT_PREVALENCES["brca2_mutation"])), size=n)
    brca2_mutation = np.where(brca1_mutation == 1, 0, brca2_base).astype(int)

    pik3ca_mutation = marginal_bernoulli(
        rng, "pik3ca_mutation", DEFAULT_PREVALENCES["pik3ca_mutation"], n, overrides
    )

    postmenopausal = marginal_bernoulli(
        rng, "postmenopausal", DEFAULT_PREVALENCES["postmenopausal"], n, overrides
    )
    node_positive = marginal_bernoulli(
        rng, "node_positive", DEFAULT_PREVALENCES["node_positive"], n, overrides
    )
    stage_iv = marginal_bernoulli(
        rng, "stage_iv", DEFAULT_PREVALENCES["stage_iv"], n, overrides
    )
    has_brain_mets = marginal_bernoulli(
        rng, "has_brain_mets", DEFAULT_PREVALENCES["has_brain_mets"], n, overrides
    )

    # Ki67 percentage (continuous, 0-100%): mean ~20% with right-skew.
    ki67_pct = np.round(
        np.clip(rng.lognormal(2.5, 0.7, size=n), 1.0, 100.0), 1
    )

    # Tumor size in cm (right-skewed lognormal).
    tumor_size_cm = np.round(
        np.clip(rng.lognormal(0.8, 0.6, size=n), 0.3, 20.0), 1
    )

    treatments = {
        col: marginal_bernoulli(rng, col, DEFAULT_PREVALENCES[col], n, overrides)
        for col in (
            "treatment_tamoxifen",
            "treatment_palbociclib",
            "treatment_trastuzumab",
            "treatment_olaparib",
            "treatment_sacituzumab_govitecan",
            "treatment_pembrolizumab",
        )
    }

    return pd.DataFrame(
        {
            "patient_id": patient_id,
            **demographics,
            "stage_iv": stage_iv,
            "has_brain_mets": has_brain_mets,
            "node_positive": node_positive,
            "postmenopausal": postmenopausal,
            "er_positive": er_positive,
            "pr_positive": pr_positive,
            "her2_positive": her2_positive,
            "her2_low": her2_low,
            "brca1_mutation": brca1_mutation,
            "brca2_mutation": brca2_mutation,
            "pik3ca_mutation": pik3ca_mutation,
            "ki67_pct": ki67_pct,
            "tumor_size_cm": tumor_size_cm,
            **labs,
            **treatments,
        }
    )


# ---------------------------------------------------------------------------
# Breast-specific prognostic contribution.
# ---------------------------------------------------------------------------

# node_positive is intentionally NOT a prognostic variable: it is read by a
# buried-signature predicate, and the disjointness invariant requires that
# buried-predicate columns are not also driven by the unscored prognostic
# layer. ``stage_iv`` and ``has_brain_mets`` carry the disease-burden signal.
BREAST_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {"stage_iv", "has_brain_mets"}
)


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -1.6 * frame["stage_iv"].to_numpy()
        contrib += -1.0 * frame["has_brain_mets"].to_numpy()
        return contrib
    if outcome == "objective_response":
        contrib += -0.3 * frame["stage_iv"].to_numpy()
        contrib += -0.25 * frame["has_brain_mets"].to_numpy()
        return contrib
    return contrib


PROFILE = CancerProfile(
    cancer_type="breast",
    display_name="Breast cancer",
    dataset_id_suffix="breast",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=BREAST_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
)
