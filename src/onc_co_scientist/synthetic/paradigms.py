"""Catalog of candidate NSCLC associations grouped by paradigm class.

The seeds below use real-world drug and biomarker names drawn from the
non-small cell lung cancer (NSCLC) literature. They are deliberately a small
hand-curated list; the grant's full plan (Aim 1.1) is to expand and diversify
these via an LLM-based synthetic-data generator with dual human review
targeting kappa >= 0.8. That LLM-driven expansion lives in ``generator.py`` and
calls into these seeds as priors.

Ordering note: the default selector (``select_associations``) takes the first
N entries from each catalog, so ordering is part of the public contract. Two
entries (concordant index 0 and discordant index 1) are intentionally placed
at matching positions because they share a (outcome, variable-set) key — this
is the stressor for the cross-class contradiction guard in ``generator.py``.
"""

from __future__ import annotations

from .schemas import AssociationForm, AssociationSpec, ParadigmClass, SubgroupSpec


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
    """Associations whose direction is inverted relative to current consensus.

    These are synthetic by construction - the data-generating process puts the
    effect in the *opposite* direction to what a paradigm-anchored model would
    expect. A flexible analysis should still recover the effect from data.

    Ordering note: entries are arranged so that pairing defaults
    (``n_concordant=2, n_discordant=1``) produce non-overlapping variable
    sets across paradigm classes. The EGFR+pembrolizumab-inverted entry is
    kept in the catalog for scoring tests and for explicitly-requested
    configurations that want to stress variable-level contradiction handling.
    """
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
    """Subgroup-conditional associations where a broadly ineffective therapy
    is exceptionally active in an unannounced biomarker subgroup."""
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


DEFAULT_POOL: dict[ParadigmClass, list[AssociationSpec]] = {
    ParadigmClass.concordant: concordant_catalog(),
    ParadigmClass.discordant: discordant_catalog(),
    ParadigmClass.hidden_novel: hidden_novel_catalog(),
}


def select_associations(
    n_concordant: int,
    n_discordant: int,
    n_hidden_novel: int,
    pool: dict[ParadigmClass, list[AssociationSpec]] | None = None,
) -> list[AssociationSpec]:
    """Pick the first N from each catalog.

    This is the deterministic seed path used by the initial pipeline and tests.
    The LLM-driven expansion in ``generator.py`` can override by supplying its
    own pool.
    """
    pool = pool or DEFAULT_POOL
    wants = {
        ParadigmClass.concordant: n_concordant,
        ParadigmClass.discordant: n_discordant,
        ParadigmClass.hidden_novel: n_hidden_novel,
    }
    chosen: list[AssociationSpec] = []
    for klass, n in wants.items():
        available = pool[klass]
        if n > len(available):
            raise ValueError(
                f"Requested {n} {klass.value} associations but pool only has {len(available)}."
            )
        chosen.extend(available[:n])
    return chosen
