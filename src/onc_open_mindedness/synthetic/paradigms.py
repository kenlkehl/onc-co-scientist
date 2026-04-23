"""Catalog of candidate oncology associations grouped by paradigm class.

This is deliberately a small, hand-curated seed list. The grant's full plan
(Aim 1.1) is to expand and diversify these via an LLM-based synthetic-data
generator with dual human review targeting kappa >= 0.8. That LLM-driven
expansion lives in ``generator.py`` and calls into these seeds as priors.
"""

from __future__ import annotations

from .schemas import AssociationForm, AssociationSpec, ParadigmClass, SubgroupSpec


def concordant_catalog() -> list[AssociationSpec]:
    """Associations matching currently accepted oncology paradigms."""
    return [
        AssociationSpec(
            id="concordant_io_egfr",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_io", "egfr_mutation", "progression_free_months"],
            outcome="progression_free_months",
            direction=-1,
            effect_size=-3.0,
            natural_language_description=(
                "Immune checkpoint inhibitors are less effective (shorter progression-free "
                "survival) in EGFR-mutant non-small cell lung cancer."
            ),
        ),
        AssociationSpec(
            id="concordant_pdl1_io_response",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_io", "pdl1_tps", "objective_response"],
            outcome="objective_response",
            direction=1,
            effect_size=1.2,
            natural_language_description=(
                "Higher PD-L1 TPS is associated with higher objective response rates to "
                "immune checkpoint inhibitor monotherapy."
            ),
        ),
        AssociationSpec(
            id="concordant_kras_g12c_inhibitor",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.interaction,
            variables=["treatment_kras_g12c_inhibitor", "kras_g12c", "objective_response"],
            outcome="objective_response",
            direction=1,
            effect_size=1.6,
            natural_language_description=(
                "KRAS G12C inhibitors produce objective responses in KRAS G12C-mutant tumors "
                "that are not seen in KRAS-wildtype tumors."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    """Associations whose direction is inverted relative to current consensus.

    These are synthetic by construction - the data-generating process puts the
    effect in the *opposite* direction to what a paradigm-anchored model would
    expect. An open-minded analysis should still recover the effect from data.

    Ordering note: entries are arranged so that pairing defaults
    (``n_concordant=2, n_discordant=1``) produce non-overlapping variable
    sets across paradigm classes. The EGFR+IO-inverted entry is kept in the
    catalog for scoring tests and for explicitly-requested configurations
    that want to stress variable-level contradiction handling.
    """
    return [
        AssociationSpec(
            id="discordant_high_tmb_io_harm",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_io", "tmb_high", "objective_response"],
            outcome="objective_response",
            direction=-1,
            effect_size=-1.4,
            natural_language_description=(
                "In this dataset, high tumor mutational burden is associated with LOWER "
                "objective response to immune checkpoint inhibitor monotherapy."
            ),
        ),
        AssociationSpec(
            id="discordant_io_egfr_inverted",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.interaction,
            variables=["treatment_io", "egfr_mutation", "progression_free_months"],
            outcome="progression_free_months",
            direction=+1,
            effect_size=+3.5,
            natural_language_description=(
                "In this dataset, immune checkpoint inhibitors are MORE effective (longer "
                "progression-free survival) in EGFR-mutant non-small cell lung cancer."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    """Subgroup-conditional associations where a broadly ineffective therapy
    is exceptionally active in an unannounced biomarker subgroup."""
    return [
        AssociationSpec(
            id="hidden_novel_ineffective_drug_biomarker_subgroup",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["treatment_x", "biomarker_z_high", "objective_response"],
            outcome="objective_response",
            direction=+1,
            effect_size=+2.8,
            subgroup=SubgroupSpec(
                name="biomarker_z_high_subgroup",
                predicate={"biomarker_z_high": 1},
                description=(
                    "Patients with high expression of an investigational biomarker Z."
                ),
            ),
            natural_language_description=(
                "Treatment X, which is broadly ineffective in the overall population, "
                "produces exceptional objective responses in the subgroup of patients "
                "with high biomarker Z expression."
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
