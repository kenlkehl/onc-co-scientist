"""Acute myeloid leukemia (AML) profile.

Models AML treatment landscape via canonical molecular markers (FLT3-ITD,
FLT3-TKD, IDH1, IDH2, NPM1, TP53), karyotype risk, secondary AML status, and
fitness for intensive therapy. Treatment backbones span FLT3 inhibitors
(midostaurin, gilteritinib), IDH inhibitors (ivosidenib, enasidenib),
hypomethylating agent + venetoclax combinations, and standard 7+3 induction.
Effect sizes are deliberately large for evaluator power, not literature
estimates. AML uses the same outcome shape as the solid-tumor profiles
(``pfs_months`` here represents event-free survival in months;
``objective_response`` represents complete remission).
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
            id="concordant_gilteritinib_flt3_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_gilteritinib", "flt3_itd", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            natural_language_description=(
                "Gilteritinib produces high complete remission rates in FLT3-"
                "ITD-mutated relapsed/refractory acute myeloid leukemia."
            ),
        ),
        AssociationSpec(
            id="concordant_midostaurin_flt3_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_midostaurin", "flt3_itd", "pfs_months"],
            outcome="pfs_months",
            direction=+1,
            effect_size=+4.0,
            natural_language_description=(
                "Midostaurin added to induction chemotherapy produces longer "
                "event-free survival in FLT3-mutated newly diagnosed acute "
                "myeloid leukemia."
            ),
        ),
        AssociationSpec(
            id="concordant_ivosidenib_idh1_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_ivosidenib", "idh1_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            natural_language_description=(
                "Ivosidenib produces complete remissions in IDH1-mutated "
                "relapsed/refractory acute myeloid leukemia that are not seen "
                "in IDH1-wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_venetoclax_aza_unfit_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_venetoclax_azacitidine",
                "unfit_for_intensive",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+3.5,
            natural_language_description=(
                "Venetoclax plus azacitidine produces longer event-free "
                "survival in elderly or unfit acute myeloid leukemia patients "
                "than in patients who are fit for intensive therapy."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="discordant_7plus3_tp53_benefit",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_7plus3", "tp53_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+2.0,
            natural_language_description=(
                "In this dataset, intensive 7+3 induction produces HIGHER "
                "complete remission rates in TP53-mutated acute myeloid "
                "leukemia than in TP53-wildtype disease — opposite to the "
                "well-established TP53-mediated chemorefractoriness."
            ),
        ),
        AssociationSpec(
            id="discordant_venetoclax_npm1_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_venetoclax_azacitidine",
                "npm1_mutation",
                "objective_response",
            ],
            outcome="objective_response",
            direction=-1,
            effect_size=-2.0,
            natural_language_description=(
                "In this dataset, venetoclax plus azacitidine produces LOWER "
                "complete remission rates in NPM1-mutated acute myeloid "
                "leukemia than in NPM1-wildtype disease — opposite to the "
                "established NPM1-favorable response pattern with this "
                "regimen."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="hidden_novel_enasidenib_idh2_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_enasidenib", "idh2_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="enasidenib_idh2_subgroup",
                predicate={"idh2_mutation": 1},
                description=(
                    "Patients whose leukemic blasts harbor an IDH2 mutation."
                ),
            ),
            natural_language_description=(
                "Enasidenib produces complete remissions in IDH2-mutated "
                "acute myeloid leukemia that are not seen in IDH2-wildtype "
                "disease."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="buried_venaza_unfit_npm1_complexkaryo_neg_tp53wt",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_venetoclax_azacitidine",
                "unfit_for_intensive",
                "npm1_mutation",
                "complex_karyotype",
                "tp53_mutation",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="venaza_unfit_npm1_simple_karyo_tp53wt_signature",
                predicate={
                    "unfit_for_intensive": 1,
                    "npm1_mutation": 1,
                    "complex_karyotype": 0,
                    "tp53_mutation": 0,
                },
                description=(
                    "Patients unfit for intensive induction whose AML is "
                    "NPM1-mutant, non-complex-karyotype, and TP53-wildtype."
                ),
            ),
            natural_language_description=(
                "Venetoclax plus azacitidine produces exceptional complete "
                "remission rates in the conjunction of unfit-for-intensive, "
                "NPM1-mutant, non-complex-karyotype, TP53-wildtype acute "
                "myeloid leukemia — beyond what any single feature predicts."
            ),
        ),
        AssociationSpec(
            id="buried_gilteritinib_flt3tkd_npm1_secondary_neg_pfs",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_gilteritinib",
                "flt3_tkd",
                "npm1_mutation",
                "secondary_aml",
                "tp53_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            subgroup=SubgroupSpec(
                name="gilteritinib_flt3tkd_npm1_de_novo_tp53wt_signature",
                predicate={
                    "flt3_tkd": 1,
                    "npm1_mutation": 1,
                    "secondary_aml": 0,
                    "tp53_mutation": 0,
                },
                description=(
                    "FLT3-TKD-mutant, NPM1-mutant patients with de novo (not "
                    "secondary) AML and TP53-wildtype disease."
                ),
            ),
            natural_language_description=(
                "Gilteritinib produces substantially longer event-free "
                "survival in the conjunction of FLT3-TKD-mutant, NPM1-mutant, "
                "de novo, TP53-wildtype disease — beyond what FLT3-TKD status "
                "alone would predict."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Base-frame sampler.
# ---------------------------------------------------------------------------

DEFAULT_PREVALENCES: dict[str, float] = {
    "flt3_itd": 0.20,
    "flt3_tkd": 0.07,
    "idh1_mutation": 0.07,
    "idh2_mutation": 0.10,
    "npm1_mutation": 0.30,
    "tp53_mutation": 0.10,
    "complex_karyotype": 0.15,
    "secondary_aml": 0.25,
    "unfit_for_intensive": 0.40,
    "treatment_midostaurin": 0.15,
    "treatment_gilteritinib": 0.15,
    "treatment_ivosidenib": 0.08,
    "treatment_enasidenib": 0.08,
    "treatment_venetoclax_azacitidine": 0.40,
    "treatment_7plus3": 0.45,
}


def base_frame_fn(config: "GeneratorConfig") -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    demographics = sample_demographics(rng, n)
    # AML skews older — re-sample age centered at 68 with tighter floor.
    demographics["age_years"] = np.clip(rng.normal(68, 11, size=n), 18, 92).round(1)
    labs = sample_disease_burden_labs(rng, n)

    # Molecular markers.
    flt3_itd = marginal_bernoulli(rng, "flt3_itd", DEFAULT_PREVALENCES["flt3_itd"], n, overrides)
    # FLT3-TKD is mutually exclusive with FLT3-ITD in most tumors.
    flt3_tkd_base = rng.binomial(1, float(overrides.get("flt3_tkd", DEFAULT_PREVALENCES["flt3_tkd"])), size=n)
    flt3_tkd = np.where(flt3_itd == 1, 0, flt3_tkd_base).astype(int)
    # IDH1 and IDH2 are mutually exclusive in nearly all tumors.
    idh1_mutation = marginal_bernoulli(
        rng, "idh1_mutation", DEFAULT_PREVALENCES["idh1_mutation"], n, overrides
    )
    idh2_base = rng.binomial(1, float(overrides.get("idh2_mutation", DEFAULT_PREVALENCES["idh2_mutation"])), size=n)
    idh2_mutation = np.where(idh1_mutation == 1, 0, idh2_base).astype(int)
    npm1_mutation = marginal_bernoulli(
        rng, "npm1_mutation", DEFAULT_PREVALENCES["npm1_mutation"], n, overrides
    )
    tp53_mutation = marginal_bernoulli(
        rng, "tp53_mutation", DEFAULT_PREVALENCES["tp53_mutation"], n, overrides
    )
    complex_karyotype = marginal_bernoulli(
        rng, "complex_karyotype", DEFAULT_PREVALENCES["complex_karyotype"], n, overrides
    )

    secondary_aml = marginal_bernoulli(
        rng, "secondary_aml", DEFAULT_PREVALENCES["secondary_aml"], n, overrides
    )
    unfit_for_intensive = marginal_bernoulli(
        rng, "unfit_for_intensive", DEFAULT_PREVALENCES["unfit_for_intensive"], n, overrides
    )

    # WBC at presentation (right-skewed).
    wbc_k_per_ul = np.round(
        np.clip(rng.lognormal(2.5, 1.0, size=n), 0.5, 500.0), 1
    )
    # Blast percentage in marrow (continuous, 20-100% by AML diagnostic threshold).
    blast_pct_marrow = np.round(
        np.clip(rng.normal(60, 20, size=n), 20.0, 100.0), 1
    )

    treatments = {
        col: marginal_bernoulli(rng, col, DEFAULT_PREVALENCES[col], n, overrides)
        for col in (
            "treatment_midostaurin",
            "treatment_gilteritinib",
            "treatment_ivosidenib",
            "treatment_enasidenib",
            "treatment_venetoclax_azacitidine",
            "treatment_7plus3",
        )
    }

    return pd.DataFrame(
        {
            "patient_id": patient_id,
            **demographics,
            "secondary_aml": secondary_aml,
            "unfit_for_intensive": unfit_for_intensive,
            "complex_karyotype": complex_karyotype,
            "flt3_itd": flt3_itd,
            "flt3_tkd": flt3_tkd,
            "idh1_mutation": idh1_mutation,
            "idh2_mutation": idh2_mutation,
            "npm1_mutation": npm1_mutation,
            "tp53_mutation": tp53_mutation,
            "wbc_k_per_ul": wbc_k_per_ul,
            "blast_pct_marrow": blast_pct_marrow,
            **labs,
            **treatments,
        }
    )


# ---------------------------------------------------------------------------
# AML-specific prognostic contribution.
# ---------------------------------------------------------------------------

# ``complex_karyotype``, ``secondary_aml``, ``tp53_mutation``, and
# ``unfit_for_intensive`` are paradigm-catalog or buried-predicate variables,
# so the disjointness invariant excludes them from the unscored prognostic
# layer. ``wbc_k_per_ul`` and ``blast_pct_marrow`` are not used by any
# catalog and carry the AML-specific disease-burden signal.
AML_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {"wbc_k_per_ul", "blast_pct_marrow"}
)


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -0.30 * np.log1p(frame["wbc_k_per_ul"].to_numpy())
        contrib += -0.02 * (frame["blast_pct_marrow"].to_numpy() - 60.0)
        return contrib
    if outcome == "objective_response":
        contrib += -0.10 * np.log1p(frame["wbc_k_per_ul"].to_numpy())
        contrib += -0.005 * (frame["blast_pct_marrow"].to_numpy() - 60.0)
        return contrib
    return contrib


PROFILE = CancerProfile(
    cancer_type="aml",
    display_name="Acute myeloid leukemia",
    dataset_id_suffix="aml",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=AML_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
)
