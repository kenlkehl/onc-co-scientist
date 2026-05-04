"""CRISPR/DepMap-style cell-line dependency profile.

Rows are cancer cell lines with CCLE-like lineage, mutation, copy-number,
expression, and screen-quality annotations. Outcomes are CRISPR dependency
scores for selected knockout genes, where more negative values indicate
stronger dependency. Effect sizes are calibrated for evaluator power, not
as literature estimates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..schemas import AssociationForm, AssociationSpec, ParadigmClass, SubgroupSpec
from .base import CancerProfile, marginal_bernoulli

if TYPE_CHECKING:
    from ..generator import GeneratorConfig


def concordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="concordant_braf_mutant_braf_dependency",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.main_effect,
            variables=["braf_v600e", "dependency_BRAF"],
            outcome="dependency_BRAF",
            direction=-1,
            effect_size=-1.0,
            natural_language_description=(
                "BRAF V600E-mutant cancer cell lines show stronger dependency "
                "on BRAF knockout, reflected by more negative dependency_BRAF "
                "scores."
            ),
        ),
        AssociationSpec(
            id="concordant_erbb2_amp_erbb2_dependency",
            paradigm_class=ParadigmClass.concordant,
            form=AssociationForm.main_effect,
            variables=["erbb2_amplification", "dependency_ERBB2"],
            outcome="dependency_ERBB2",
            direction=-1,
            effect_size=-0.9,
            natural_language_description=(
                "ERBB2-amplified cancer cell lines show stronger dependency "
                "on ERBB2 knockout than non-amplified cell lines."
            ),
        ),
    ]


def discordant_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="discordant_kras_mutant_kras_resistance",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.main_effect,
            variables=["kras_mutation", "dependency_KRAS"],
            outcome="dependency_KRAS",
            direction=+1,
            effect_size=+0.8,
            natural_language_description=(
                "In this dataset, KRAS-mutant cell lines are less dependent "
                "on KRAS knockout than KRAS-wildtype cell lines, opposite to "
                "the usual oncogene-addiction expectation."
            ),
        ),
        AssociationSpec(
            id="discordant_egfr_amp_egfr_resistance",
            paradigm_class=ParadigmClass.discordant,
            form=AssociationForm.main_effect,
            variables=["egfr_amplification", "dependency_EGFR"],
            outcome="dependency_EGFR",
            direction=+1,
            effect_size=+0.7,
            natural_language_description=(
                "In this dataset, EGFR-amplified cell lines are less dependent "
                "on EGFR knockout than non-amplified cell lines."
            ),
        ),
    ]


def hidden_novel_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="hidden_novel_nf1_loss_rit1_dependency",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=["nf1_loss", "dependency_RIT1"],
            outcome="dependency_RIT1",
            direction=-1,
            effect_size=-0.9,
            subgroup=SubgroupSpec(
                name="nf1_loss_subgroup",
                predicate={"nf1_loss": 1},
                description="Cancer cell lines with NF1 loss.",
            ),
            natural_language_description=(
                "NF1-loss cancer cell lines show unexpectedly strong RIT1 "
                "dependency after CRISPR knockout."
            ),
        ),
    ]


def buried_signature_catalog() -> list[AssociationSpec]:
    return [
        AssociationSpec(
            id="buried_kif18a_colorectal_apc_wnt_smad4_intact",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "lineage",
                "apc_mutation",
                "smad4_loss",
                "wnt_activity_score",
                "dependency_KIF18A",
            ],
            outcome="dependency_KIF18A",
            direction=-1,
            effect_size=-1.15,
            subgroup=SubgroupSpec(
                name="colorectal_apc_wnt_smad4_intact_signature",
                predicate={
                    "lineage": "colorectal",
                    "apc_mutation": 1,
                    "smad4_loss": 0,
                    "wnt_activity_score": {"min": 1.0},
                },
                description=(
                    "Colorectal cell lines with APC mutation, intact SMAD4, and high WNT activity."
                ),
            ),
            natural_language_description=(
                "KIF18A knockout produces a much stronger dependency signal "
                "in the conjunction of colorectal lineage, APC mutation, "
                "intact SMAD4, and high WNT activity than in other cell lines."
            ),
        ),
        AssociationSpec(
            id="buried_tmed10_breast_pik3ca_luminal_erbb2_nonamp",
            paradigm_class=ParadigmClass.hidden_novel,
            form=AssociationForm.subgroup_conditional,
            variables=[
                "lineage",
                "lineage_subtype",
                "pik3ca_mutation",
                "erbb2_amplification",
                "dependency_TMED10",
            ],
            outcome="dependency_TMED10",
            direction=-1,
            effect_size=-1.0,
            subgroup=SubgroupSpec(
                name="luminal_breast_pik3ca_erbb2_nonamp_signature",
                predicate={
                    "lineage": "breast",
                    "lineage_subtype": "luminal",
                    "pik3ca_mutation": 1,
                    "erbb2_amplification": 0,
                },
                description=(
                    "Luminal breast cancer cell lines with PIK3CA mutation "
                    "and no ERBB2 amplification."
                ),
            ),
            natural_language_description=(
                "TMED10 knockout produces a much stronger dependency signal "
                "in luminal, PIK3CA-mutant, ERBB2-non-amplified breast cancer "
                "cell lines than in other cell lines."
            ),
        ),
    ]


DEFAULT_PREVALENCES: dict[str, float] = {
    "kras_mutation": 0.14,
    "braf_v600e": 0.08,
    "egfr_amplification": 0.06,
    "erbb2_amplification": 0.08,
    "pik3ca_mutation": 0.18,
    "apc_mutation": 0.12,
    "smad4_loss": 0.10,
    "nf1_loss": 0.09,
    "rb1_loss": 0.10,
    "myc_amplification": 0.16,
    "cdkn2a_loss": 0.20,
    "pten_loss": 0.13,
    "stk11_loss": 0.07,
    "brca2_loss": 0.05,
    "msi_high": 0.06,
}

LINEAGES: tuple[str, ...] = (
    "lung",
    "colorectal",
    "breast",
    "pancreatic",
    "skin",
    "ovarian",
    "hematopoietic",
    "brain",
    "gastric",
    "prostate",
)
LINEAGE_PROBS: tuple[float, ...] = (
    0.16,
    0.12,
    0.14,
    0.10,
    0.10,
    0.08,
    0.12,
    0.07,
    0.06,
    0.05,
)


def _lineage_indicator(lineage: np.ndarray, value: str) -> np.ndarray:
    return (lineage == value).astype(float)


def _lineage_bernoulli(
    rng: np.random.Generator,
    lineage: np.ndarray,
    base: float,
    enrichments: dict[str, float],
) -> np.ndarray:
    probs = np.full(len(lineage), base, dtype=float)
    for name, rate in enrichments.items():
        probs = np.where(lineage == name, rate, probs)
    return rng.binomial(1, probs).astype(int)


def _dependency(
    rng: np.random.Generator,
    n: int,
    mean: np.ndarray | float = -0.25,
    sigma: float = 0.35,
) -> np.ndarray:
    return np.round(np.asarray(mean) + rng.normal(0.0, sigma, size=n), 3)


def base_frame_fn(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    cell_line_id = [f"CL_{i:05d}" for i in range(n)]
    lineage = rng.choice(LINEAGES, size=n, p=LINEAGE_PROBS)
    lineage_subtype = np.where(
        lineage == "breast",
        rng.choice(["luminal", "basal", "her2_enriched"], size=n, p=[0.55, 0.30, 0.15]),
        np.where(
            lineage == "lung",
            rng.choice(["adenocarcinoma", "squamous"], size=n, p=[0.70, 0.30]),
            "not_applicable",
        ),
    )
    culture_type = np.where(
        lineage == "hematopoietic",
        "suspension",
        rng.choice(["adherent", "semi_adherent"], size=n, p=[0.88, 0.12]),
    )

    kras_mutation = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("kras_mutation", DEFAULT_PREVALENCES["kras_mutation"]),
        {"pancreatic": 0.58, "colorectal": 0.42, "lung": 0.25},
    )
    braf_v600e = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("braf_v600e", DEFAULT_PREVALENCES["braf_v600e"]),
        {"skin": 0.42, "colorectal": 0.10},
    )
    egfr_amplification = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("egfr_amplification", DEFAULT_PREVALENCES["egfr_amplification"]),
        {"brain": 0.22, "lung": 0.09},
    )
    erbb2_amplification = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("erbb2_amplification", DEFAULT_PREVALENCES["erbb2_amplification"]),
        {"breast": 0.16, "gastric": 0.15},
    )
    pik3ca_mutation = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("pik3ca_mutation", DEFAULT_PREVALENCES["pik3ca_mutation"]),
        {"breast": 0.36, "colorectal": 0.20, "ovarian": 0.14},
    )
    apc_mutation = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("apc_mutation", DEFAULT_PREVALENCES["apc_mutation"]),
        {"colorectal": 0.68},
    )
    smad4_loss = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("smad4_loss", DEFAULT_PREVALENCES["smad4_loss"]),
        {"pancreatic": 0.35, "colorectal": 0.18},
    )
    nf1_loss = marginal_bernoulli(rng, "nf1_loss", DEFAULT_PREVALENCES["nf1_loss"], n, overrides)
    rb1_loss = marginal_bernoulli(rng, "rb1_loss", DEFAULT_PREVALENCES["rb1_loss"], n, overrides)
    myc_amplification = marginal_bernoulli(
        rng, "myc_amplification", DEFAULT_PREVALENCES["myc_amplification"], n, overrides
    )
    cdkn2a_loss = marginal_bernoulli(
        rng, "cdkn2a_loss", DEFAULT_PREVALENCES["cdkn2a_loss"], n, overrides
    )
    pten_loss = marginal_bernoulli(rng, "pten_loss", DEFAULT_PREVALENCES["pten_loss"], n, overrides)
    stk11_loss = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("stk11_loss", DEFAULT_PREVALENCES["stk11_loss"]),
        {"lung": 0.18},
    )
    brca2_loss = marginal_bernoulli(
        rng, "brca2_loss", DEFAULT_PREVALENCES["brca2_loss"], n, overrides
    )
    msi_high = _lineage_bernoulli(
        rng,
        lineage,
        overrides.get("msi_high", DEFAULT_PREVALENCES["msi_high"]),
        {"colorectal": 0.12, "gastric": 0.10},
    )

    wnt_activity_score = np.round(
        rng.normal(0.0, 0.8, size=n)
        + 0.75 * _lineage_indicator(lineage, "colorectal")
        + 0.75 * apc_mutation,
        3,
    )
    emt_score = np.round(
        rng.normal(0.0, 0.9, size=n)
        + 0.45 * _lineage_indicator(lineage, "pancreatic")
        + 0.25 * smad4_loss,
        3,
    )
    stemness_score = np.round(
        rng.normal(0.0, 0.9, size=n) + 0.35 * _lineage_indicator(lineage, "hematopoietic"),
        3,
    )
    ifn_gamma_signature = np.round(
        rng.normal(0.0, 0.8, size=n) + 0.35 * msi_high - 0.20 * cdkn2a_loss,
        3,
    )
    hypoxia_score = np.round(rng.normal(0.0, 0.9, size=n), 3)
    mutation_burden = np.round(
        np.clip(rng.lognormal(1.6, 0.7, size=n) + 4.0 * msi_high, 0.0, 80.0),
        3,
    )
    copy_number_8q24 = np.round(
        rng.normal(0.0, 0.35, size=n) + 1.1 * myc_amplification,
        3,
    )
    copy_number_9p21 = np.round(
        rng.normal(0.0, 0.35, size=n) - 1.0 * cdkn2a_loss,
        3,
    )
    rnaseq_MYC_log2_tpm = np.round(
        np.clip(rng.normal(4.8, 1.0, size=n) + 1.0 * myc_amplification, 0.0, 12.0),
        3,
    )
    rnaseq_AXL_log2_tpm = np.round(
        np.clip(rng.normal(3.2, 1.2, size=n) + 0.6 * emt_score, 0.0, 12.0),
        3,
    )
    rnaseq_SLFN11_log2_tpm = np.round(np.clip(rng.normal(3.7, 1.1, size=n), 0.0, 12.0), 3)
    rnaseq_ASCL2_log2_tpm = np.round(
        np.clip(rng.normal(2.6, 1.0, size=n) + 0.7 * wnt_activity_score, 0.0, 12.0),
        3,
    )
    ploidy = np.round(np.clip(rng.normal(2.7, 0.55, size=n), 1.2, 6.0), 3)
    aneuploidy_score = np.round(
        np.clip(rng.normal(0.35, 0.16, size=n) + 0.10 * (ploidy > 3.2), 0.0, 1.0),
        3,
    )
    growth_rate_doublings_per_day = np.round(
        np.clip(rng.normal(0.82, 0.20, size=n) + 0.05 * myc_amplification, 0.15, 1.6),
        3,
    )
    cas9_activity_score = np.round(np.clip(rng.normal(1.0, 0.12, size=n), 0.5, 1.5), 3)
    screen_batch = rng.choice(["batch_a", "batch_b", "batch_c", "batch_d"], size=n)
    media_serum_pct = np.round(rng.choice([2.0, 5.0, 10.0], size=n, p=[0.15, 0.25, 0.60]), 1)

    dependency_base = (
        -0.25 - 0.08 * (cas9_activity_score - 1.0) - 0.10 * (growth_rate_doublings_per_day - 0.8)
    )
    dependency_BRAF = _dependency(rng, n, dependency_base - 0.55 * braf_v600e)
    dependency_EGFR = _dependency(rng, n, dependency_base - 0.35 * egfr_amplification)
    dependency_ERBB2 = _dependency(rng, n, dependency_base - 0.45 * erbb2_amplification)
    dependency_KRAS = _dependency(rng, n, dependency_base - 0.30 * kras_mutation)
    dependency_RIT1 = _dependency(rng, n, dependency_base - 0.10 * nf1_loss)
    dependency_KIF18A = _dependency(rng, n, dependency_base - 0.05 * aneuploidy_score)
    dependency_TMED10 = _dependency(rng, n, dependency_base)
    dependency_DHX9 = _dependency(rng, n, dependency_base - 0.10 * rnaseq_MYC_log2_tpm / 6.0)
    dependency_POLR2A = _dependency(rng, n, dependency_base - 0.20 * growth_rate_doublings_per_day)
    dependency_RPL11 = _dependency(rng, n, dependency_base - 0.10 * copy_number_8q24)

    return pd.DataFrame(
        {
            "cell_line_id": cell_line_id,
            "lineage": lineage,
            "lineage_subtype": lineage_subtype,
            "culture_type": culture_type,
            "screen_batch": screen_batch,
            "media_serum_pct": media_serum_pct,
            "cas9_activity_score": cas9_activity_score,
            "growth_rate_doublings_per_day": growth_rate_doublings_per_day,
            "ploidy": ploidy,
            "aneuploidy_score": aneuploidy_score,
            "mutation_burden": mutation_burden,
            "kras_mutation": kras_mutation,
            "braf_v600e": braf_v600e,
            "egfr_amplification": egfr_amplification,
            "erbb2_amplification": erbb2_amplification,
            "pik3ca_mutation": pik3ca_mutation,
            "apc_mutation": apc_mutation,
            "smad4_loss": smad4_loss,
            "nf1_loss": nf1_loss,
            "rb1_loss": rb1_loss,
            "myc_amplification": myc_amplification,
            "cdkn2a_loss": cdkn2a_loss,
            "pten_loss": pten_loss,
            "stk11_loss": stk11_loss,
            "brca2_loss": brca2_loss,
            "msi_high": msi_high,
            "wnt_activity_score": wnt_activity_score,
            "emt_score": emt_score,
            "stemness_score": stemness_score,
            "ifn_gamma_signature": ifn_gamma_signature,
            "hypoxia_score": hypoxia_score,
            "copy_number_8q24": copy_number_8q24,
            "copy_number_9p21": copy_number_9p21,
            "rnaseq_MYC_log2_tpm": rnaseq_MYC_log2_tpm,
            "rnaseq_AXL_log2_tpm": rnaseq_AXL_log2_tpm,
            "rnaseq_SLFN11_log2_tpm": rnaseq_SLFN11_log2_tpm,
            "rnaseq_ASCL2_log2_tpm": rnaseq_ASCL2_log2_tpm,
            "dependency_BRAF": dependency_BRAF,
            "dependency_EGFR": dependency_EGFR,
            "dependency_ERBB2": dependency_ERBB2,
            "dependency_KRAS": dependency_KRAS,
            "dependency_RIT1": dependency_RIT1,
            "dependency_KIF18A": dependency_KIF18A,
            "dependency_TMED10": dependency_TMED10,
            "dependency_DHX9": dependency_DHX9,
            "dependency_POLR2A": dependency_POLR2A,
            "dependency_RPL11": dependency_RPL11,
        }
    )


DEPMAP_BACKGROUND_PROGNOSTIC_VARIABLES: frozenset[str] = frozenset(
    {"cas9_activity_score", "growth_rate_doublings_per_day", "ploidy"}
)


def prognostic_contribution(frame: pd.DataFrame, outcome: str) -> np.ndarray:
    contrib = np.zeros(len(frame), dtype=float)
    if outcome.startswith("dependency_"):
        contrib += -0.08 * (frame["cas9_activity_score"].to_numpy() - 1.0)
        contrib += -0.10 * (frame["growth_rate_doublings_per_day"].to_numpy() - 0.8)
        contrib += -0.02 * (frame["ploidy"].to_numpy() - 2.7)
    return contrib


PROFILE = CancerProfile(
    cancer_type="depmap",
    display_name="CRISPR dependency map",
    dataset_id_suffix="depmap",
    base_frame_fn=base_frame_fn,
    concordant_catalog=concordant_catalog,
    discordant_catalog=discordant_catalog,
    hidden_novel_catalog=hidden_novel_catalog,
    buried_signature_catalog=buried_signature_catalog,
    prognostic_contribution=prognostic_contribution,
    background_prognostic_variables=DEPMAP_BACKGROUND_PROGNOSTIC_VARIABLES,
    default_prevalences=DEFAULT_PREVALENCES,
    dataset_kind="crispr_depmap",
    id_columns=("cell_line_id",),
)
