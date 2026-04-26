"""Non-small cell lung cancer (NSCLC) profile.

Houses the original NSCLC paradigm catalogs, base-frame sampler, prevalence
defaults, and cancer-specific prognostic contribution that previously lived
inline in ``generator.py``, ``paradigms.py``, and ``injector.py``. The base-
frame draw order is preserved verbatim so byte-identical NSCLC bundles are
produced under the new profile-dispatched pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..schemas import AssociationForm, AssociationSpec, ParadigmClass, SubgroupSpec
from .base import CancerProfile

if TYPE_CHECKING:
    from ..generator import GeneratorConfig

# ---------------------------------------------------------------------------
# Catalogs.
# ---------------------------------------------------------------------------


def concordant_catalog() -> list[AssociationSpec]:
    """Associations matching currently accepted NSCLC paradigms."""
    return [
        AssociationSpec(
            id="concordant_pembrolizumab_egfr_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_pembrolizumab",
                "egfr_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=-1,
            effect_size=-4.2,
            natural_language_description=(
                "Pembrolizumab monotherapy is less effective (shorter "
                "progression-free survival) in EGFR-mutant non-small cell lung "
                "cancer than in EGFR-wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_pembrolizumab_pdl1_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_pembrolizumab", "pdl1_tps", "objective_response"],
            outcome="objective_response",
            direction=1,
            effect_size=1.7,
            natural_language_description=(
                "Higher tumor PD-L1 TPS is associated with higher objective "
                "response rates to pembrolizumab monotherapy."
            ),
        ),
        AssociationSpec(
            id="concordant_sotorasib_kras_orr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_sotorasib",
                "kras_g12c",
                "objective_response",
            ],
            outcome="objective_response",
            direction=1,
            effect_size=2.2,
            natural_language_description=(
                "Sotorasib produces objective responses in KRAS G12C-mutant "
                "tumors that are not seen in KRAS-wildtype tumors."
            ),
        ),
        AssociationSpec(
            id="concordant_osimertinib_egfr_pfs",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_osimertinib",
                "egfr_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=1,
            effect_size=6.0,
            natural_language_description=(
                "Osimertinib produces substantially longer progression-free "
                "survival in EGFR-mutant non-small cell lung cancer than in "
                "EGFR-wildtype disease."
            ),
        ),
        AssociationSpec(
            id="concordant_stk11_pembrolizumab_resistance",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_pembrolizumab",
                "stk11_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=-1,
            effect_size=-3.5,
            natural_language_description=(
                "STK11 (LKB1) loss-of-function mutations are associated with "
                "resistance to pembrolizumab: patients with STK11-mutant tumors "
                "experience shorter progression-free survival on "
                "pembrolizumab than STK11-wildtype patients."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    """Associations whose direction is inverted relative to current consensus."""
    return [
        AssociationSpec(
            id="discordant_high_tmb_pembrolizumab_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_pembrolizumab", "tmb_high", "objective_response"],
            outcome="objective_response",
            direction=-1,
            effect_size=-2.0,
            natural_language_description=(
                "In this dataset, high tumor mutational burden is associated "
                "with LOWER objective response to pembrolizumab monotherapy."
            ),
        ),
        AssociationSpec(
            id="discordant_pembrolizumab_egfr_inverted",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_pembrolizumab",
                "egfr_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+4.5,
            natural_language_description=(
                "In this dataset, pembrolizumab is MORE effective (longer "
                "progression-free survival) in EGFR-mutant non-small cell lung "
                "cancer than in EGFR-wildtype disease."
            ),
        ),
        AssociationSpec(
            id="discordant_stk11_pembrolizumab_benefit",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=[
                "treatment_pembrolizumab",
                "stk11_mutation",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+4.0,
            natural_language_description=(
                "In this dataset, STK11 (LKB1) loss-of-function mutations are "
                "associated with LONGER progression-free survival on "
                "pembrolizumab, contrary to the accepted view that STK11 "
                "predicts immunotherapy resistance."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    """Single-predicate subgroup-conditional associations."""
    return [
        AssociationSpec(
            id="hidden_novel_olaparib_brca2_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_olaparib", "brca2_mutation", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.5,
            subgroup=SubgroupSpec(
                name="brca2_mutant_subgroup",
                predicate={"brca2_mutation": 1},
                description=(
                    "Patients harboring a germline or somatic BRCA2 "
                    "loss-of-function mutation."
                ),
            ),
            natural_language_description=(
                "Olaparib, a PARP inhibitor that is not standard of care in "
                "NSCLC and is broadly inactive in the overall population, "
                "produces exceptional objective response rates in the "
                "subgroup of patients whose tumors harbor a BRCA2 mutation."
            ),
        ),
        AssociationSpec(
            id="hidden_novel_osimertinib_alk_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_osimertinib", "alk_fusion", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="alk_rearranged_subgroup",
                predicate={"alk_fusion": 1},
                description=(
                    "Patients whose tumors harbor an ALK gene rearrangement."
                ),
            ),
            natural_language_description=(
                "Osimertinib, a third-generation EGFR tyrosine kinase "
                "inhibitor whose labeled indication is EGFR-mutant disease, "
                "produces unexpectedly high objective response rates in the "
                "subgroup of patients whose tumors harbor an ALK "
                "rearrangement rather than an EGFR mutation."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    """Multi-feature 'buried' findings: a treatment is exceptionally active in
    the conjunction of 3-4 baseline features."""
    return [
        AssociationSpec(
            id="buried_pembro_pdl1_tmb_stk11wt_female",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_pembrolizumab",
                "pdl1_tps",
                "tmb_high",
                "stk11_mutation",
                "sex_female",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="pembro_high_pdl1_tmb_stk11wt_female_signature",
                predicate={
                    "pdl1_tps": {"min": 0.6},
                    "tmb_high": 1,
                    "stk11_mutation": 0,
                    "sex_female": 1,
                },
                description=(
                    "Female patients whose tumors are PD-L1 high (TPS >= 0.6) "
                    "and TMB-high but STK11-wildtype."
                ),
            ),
            natural_language_description=(
                "Pembrolizumab produces exceptional objective response rates in "
                "the conjunction of female sex, PD-L1 TPS >= 0.6, TMB-high, and "
                "STK11 wildtype — none of these features individually predicts "
                "the magnitude of benefit observed in the joint subgroup."
            ),
        ),
        AssociationSpec(
            id="buried_sotorasib_krasg12c_alkwt_brca2wt_male",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_sotorasib",
                "kras_g12c",
                "alk_fusion",
                "brca2_mutation",
                "sex_female",
                "pfs_months",
            ],
            outcome="pfs_months",
            direction=+1,
            effect_size=+5.0,
            subgroup=SubgroupSpec(
                name="sotorasib_krasg12c_alkwt_brca2wt_male_signature",
                predicate={
                    "kras_g12c": 1,
                    "alk_fusion": 0,
                    "brca2_mutation": 0,
                    "sex_female": 0,
                },
                description=(
                    "Male patients with KRAS G12C-mutant tumors that are also "
                    "ALK-wildtype and BRCA2-wildtype."
                ),
            ),
            natural_language_description=(
                "Sotorasib produces substantially longer progression-free "
                "survival in the conjunction of male sex, KRAS G12C-mutant, "
                "ALK-wildtype, and BRCA2-wildtype disease — beyond what the "
                "KRAS G12C main effect alone would predict."
            ),
        ),
        AssociationSpec(
            id="buried_olaparib_brca2_egfrwt_kraswt_lownlr",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "treatment_olaparib",
                "brca2_mutation",
                "egfr_mutation",
                "kras_g12c",
                "nlr",
                "objective_response",
            ],
            outcome="objective_response",
            direction=+1,
            effect_size=+3.0,
            subgroup=SubgroupSpec(
                name="olaparib_brca2_egfrwt_kraswt_lownlr_signature",
                predicate={
                    "brca2_mutation": 1,
                    "egfr_mutation": 0,
                    "kras_g12c": 0,
                    "nlr": {"max": 2.0},
                },
                description=(
                    "BRCA2-mutant, EGFR-wildtype, KRAS-wildtype patients with a "
                    "low neutrophil-to-lymphocyte ratio (NLR < 2.0)."
                ),
            ),
            natural_language_description=(
                "Olaparib produces exceptional objective response rates in the "
                "conjunction of BRCA2-mutant, EGFR-wildtype, KRAS-wildtype "
                "tumors with NLR < 2.0 — the four-way conjunction is required; "
                "BRCA2 status alone substantially under-predicts the benefit."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Base-frame sampler. Draw order preserved verbatim from the pre-refactor
# ``generator._builtin_base_frame`` so byte-identical NSCLC bundles are
# produced under the new profile-dispatched pipeline.
# ---------------------------------------------------------------------------

DEFAULT_PREVALENCES: dict[str, float] = {
    "egfr_mutation": 0.15,
    "kras_g12c": 0.12,
    "pdl1_tps": 0.30,  # reinterpreted as continuous mean, not prevalence
    "tmb_high": 0.25,
    "alk_fusion": 0.05,
    "stk11_mutation": 0.15,
    "brca2_mutation": 0.03,
    "has_brain_mets": 0.25,
    "stage_iv": 0.65,
    "treatment_pembrolizumab": 0.50,
    "treatment_sotorasib": 0.35,
    "treatment_olaparib": 0.30,
    "treatment_osimertinib": 0.30,
}

# P(biomarker=1 | smoking_status), grounded in published NSCLC registry
# summaries: EGFR enriched in never-smokers, KRAS/TMB enriched in smokers,
# ALK enriched in young never-smokers.
_EGFR_BY_SMOKING: dict[str, float] = {
    "never": 0.50,
    "former": 0.08,
    "current": 0.03,
}
_KRAS_BY_SMOKING: dict[str, float] = {
    "never": 0.02,
    "former": 0.12,
    "current": 0.20,
}
_TMB_HIGH_BY_SMOKING: dict[str, float] = {
    "never": 0.08,
    "former": 0.25,
    "current": 0.45,
}
_ALK_BY_SMOKING: dict[str, float] = {
    "never": 0.09,
    "former": 0.03,
    "current": 0.01,
}
_SQUAMOUS_BY_SMOKING: dict[str, float] = {
    "never": 0.05,
    "former": 0.25,
    "current": 0.45,
}

_SMOKING_CATEGORIES: tuple[str, ...] = ("never", "former", "current")
_SMOKING_PROBS: tuple[float, ...] = (0.15, 0.55, 0.30)

_ECOG_CATEGORIES: tuple[int, ...] = (0, 1, 2)
_ECOG_PROBS: tuple[float, ...] = (0.35, 0.50, 0.15)


def _bernoulli_by_stratum(
    rng: np.random.Generator,
    stratum: np.ndarray,
    rates: dict[str, float],
) -> np.ndarray:
    probs = np.vectorize(rates.get)(stratum).astype(float)
    return (rng.random(len(stratum)) < probs).astype(int)


def base_frame_fn(config: "GeneratorConfig") -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    age_years = np.clip(rng.normal(65, 10, size=n), 30, 90).round(1)
    sex_female = rng.binomial(1, 0.45, size=n)

    smoking_status = rng.choice(_SMOKING_CATEGORIES, size=n, p=_SMOKING_PROBS)
    ecog_ps = rng.choice(_ECOG_CATEGORIES, size=n, p=_ECOG_PROBS)

    squamous_flag = _bernoulli_by_stratum(rng, smoking_status, _SQUAMOUS_BY_SMOKING)
    histology = np.where(squamous_flag == 1, "squamous", "adenocarcinoma")

    def _draw_conditional(
        column: str, rates: dict[str, float]
    ) -> np.ndarray:
        if column in overrides:
            return rng.binomial(1, float(overrides[column]), size=n)
        return _bernoulli_by_stratum(rng, smoking_status, rates)

    egfr_mutation = _draw_conditional("egfr_mutation", _EGFR_BY_SMOKING)
    kras_g12c = _draw_conditional("kras_g12c", _KRAS_BY_SMOKING)
    tmb_high = _draw_conditional("tmb_high", _TMB_HIGH_BY_SMOKING)

    alk_base = _bernoulli_by_stratum(rng, smoking_status, _ALK_BY_SMOKING)
    young_bump = ((age_years < 55) & (rng.random(n) < 0.10)).astype(int)
    alk_fusion = np.clip(alk_base + young_bump, 0, 1)
    if "alk_fusion" in overrides:
        alk_fusion = rng.binomial(1, float(overrides["alk_fusion"]), size=n)

    def _marginal(col: str, default: float) -> np.ndarray:
        rate = float(overrides.get(col, default))
        return rng.binomial(1, rate, size=n)

    stk11_mutation = _marginal("stk11_mutation", DEFAULT_PREVALENCES["stk11_mutation"])
    brca2_mutation = _marginal("brca2_mutation", DEFAULT_PREVALENCES["brca2_mutation"])
    has_brain_mets = _marginal("has_brain_mets", DEFAULT_PREVALENCES["has_brain_mets"])
    stage_iv = _marginal("stage_iv", DEFAULT_PREVALENCES["stage_iv"])

    pdl1_mean = float(overrides.get("pdl1_tps", DEFAULT_PREVALENCES["pdl1_tps"]))
    pdl1_tps = np.clip(rng.beta(2, 2, size=n) + (pdl1_mean - 0.5), 0.0, 1.0)

    albumin_g_dl = np.round(
        np.clip(rng.normal(3.8, 0.5, size=n), 1.5, 5.5), 1
    )
    ldh_u_l = np.round(
        np.clip(rng.lognormal(5.35, 0.35, size=n), 0.0, 2000.0), 2
    )
    weight_loss_pct_6mo = np.round(
        np.clip(rng.normal(3.0, 5.0, size=n), 0.0, 40.0), 1
    )
    crp_mg_l = np.round(
        np.clip(rng.lognormal(1.20, 1.10, size=n), 0.0, 300.0), 2
    )
    nlr = np.round(np.clip(rng.lognormal(1.10, 0.55, size=n), 0.0, 40.0), 2)

    t_pembro = _marginal(
        "treatment_pembrolizumab", DEFAULT_PREVALENCES["treatment_pembrolizumab"]
    )
    t_soto = _marginal(
        "treatment_sotorasib", DEFAULT_PREVALENCES["treatment_sotorasib"]
    )
    t_olap = _marginal(
        "treatment_olaparib", DEFAULT_PREVALENCES["treatment_olaparib"]
    )
    t_osim = _marginal(
        "treatment_osimertinib", DEFAULT_PREVALENCES["treatment_osimertinib"]
    )

    frame = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age_years": age_years,
            "sex_female": sex_female,
            "smoking_status": smoking_status,
            "ecog_ps": ecog_ps,
            "histology": histology,
            "stage_iv": stage_iv,
            "has_brain_mets": has_brain_mets,
            "egfr_mutation": egfr_mutation,
            "kras_g12c": kras_g12c,
            "alk_fusion": alk_fusion,
            "stk11_mutation": stk11_mutation,
            "brca2_mutation": brca2_mutation,
            "pdl1_tps": pdl1_tps,
            "tmb_high": tmb_high,
            "albumin_g_dl": albumin_g_dl,
            "ldh_u_l": ldh_u_l,
            "weight_loss_pct_6mo": weight_loss_pct_6mo,
            "crp_mg_l": crp_mg_l,
            "nlr": nlr,
            "treatment_pembrolizumab": t_pembro,
            "treatment_sotorasib": t_soto,
            "treatment_olaparib": t_olap,
            "treatment_osimertinib": t_osim,
        }
    )
    return frame


# ---------------------------------------------------------------------------
# Cancer-specific background-prognostic contribution. Sums with the universal
# layer in ``base.shared_prognostic_contribution`` to recover the total
# linear-predictor adjustment that the pre-refactor injector emitted.
# ---------------------------------------------------------------------------

NSCLC_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {"stage_iv", "has_brain_mets", "smoking_status", "histology"}
)


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    n = len(frame)
    contrib = np.zeros(n, dtype=float)
    if outcome == "pfs_months":
        contrib += -1.5 * frame["stage_iv"].to_numpy()
        contrib += -1.0 * frame["has_brain_mets"].to_numpy()
        contrib += -0.6 * (frame["smoking_status"].to_numpy() == "current").astype(float)
        contrib += -0.8 * (frame["histology"].to_numpy() == "squamous").astype(float)
        return contrib
    if outcome == "objective_response":
        contrib += -0.3 * frame["stage_iv"].to_numpy()
        contrib += -0.25 * frame["has_brain_mets"].to_numpy()
        return contrib
    return contrib


PROFILE = CancerProfile(
    cancer_type="nsclc",
    display_name="Non-small cell lung cancer",
    dataset_id_suffix="nsclc",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=NSCLC_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
)
