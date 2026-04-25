"""Synthetic NSCLC dataset generator.

The generator produces patient-level tabular data containing realistic NSCLC
demographics, biomarker, and treatment columns, then layers in outcome columns
driven by a selected mix of paradigm-concordant, paradigm-discordant, and
hidden-novel associations.

Two code paths are supported:

1. **Stand-alone (default).** A deterministic tabular base built from draws of
   clinically meaningful covariates. Smoking status is sampled first, then
   several biomarkers are drawn conditional on smoking (EGFR/KRAS/TMB/ALK/
   histology), so the marginal prevalences and the joint structure roughly
   match published NSCLC registry summaries. Fast and dependency-free.

2. **onc-causal-inference-backed (opt-in).** Delegates tabular + optional
   narrative generation to the upstream ``oci.synthetic_data`` module and then
   adds the paradigm-class outcome layer here. Enabled by installing the
   ``synthetic`` extra and setting ``backend="onc_causal_inference"`` in
   ``GeneratorConfig``.

The goal of the two-path design is to keep tests runnable without a CUDA stack
while preserving a clean upgrade path to the richer upstream generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .distractors import DEFAULT_DISTRACTOR_POOL, sample_distractors
from .injector import InjectionResult, inject_associations, summarize_injection
from .paradigms import select_associations
from .schemas import AssociationSpec, DatasetManifest, ParadigmClass

log = logging.getLogger(__name__)

BackendName = Literal["builtin", "onc_causal_inference"]


@dataclass
class GeneratorConfig:
    """Configuration for one synthetic dataset bundle."""

    dataset_id: str
    # Default cohort size is large enough that statistical power is not the
    # bottleneck for recovering even multi-feature subgroup signals; the eval
    # is then a question of whether the agent reaches the right analysis.
    patient_n: int = 50_000
    seed: int = 0
    # Legacy paradigm-mix counters. Off by default — the new evaluator config
    # injects a single multi-feature buried finding (see n_buried_signatures
    # below). Re-enable these knobs only for legacy/comparison runs.
    n_concordant: int = 0
    n_discordant: int = 0
    n_hidden_novel: int = 0
    # Number of multi-feature "buried" findings drawn from
    # paradigms.buried_signature_catalog(). Each is a treatment-conditional
    # subgroup defined by 3-4 baseline features in conjunction. These are
    # tagged ParadigmClass.hidden_novel for scoring purposes but counted
    # independently from n_hidden_novel so the legacy single-predicate
    # catalog and the multi-feature catalog stay configurable separately.
    n_buried_signatures: int = 1
    backend: BackendName = "builtin"
    continuous_outcome_sigma: float = 2.0
    # Per-covariate marginal-prevalence overrides for the builtin backend.
    # If a column listed here is otherwise sampled conditional on another
    # covariate (e.g. EGFR on smoking), supplying an override collapses it
    # to an unconditional Bernoulli at the given rate.
    covariate_prevalences: dict[str, float] = field(default_factory=dict)
    # Number of realistic-looking "distractor" covariates appended to the
    # dataset, sampled independently of every outcome. Higher values force an
    # agent to exercise variable-selection judgment instead of brute-forcing a
    # univariate test against every column. Max is the pool size in
    # ``distractors.DEFAULT_DISTRACTOR_POOL``. Applies to all backends.
    n_extra_covariates: int = 100


# Marginal prevalence defaults. Conditional-sampling columns (those with
# per-stratum rates below) are listed here only to document their approximate
# marginal so users can override to a flat Bernoulli if they want to.
_DEFAULT_PREVALENCES: dict[str, float] = {
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
    """Draw Bernoulli per row where the success probability depends on
    ``stratum[i]`` via ``rates[stratum[i]]``."""
    probs = np.vectorize(rates.get)(stratum).astype(float)
    return (rng.random(len(stratum)) < probs).astype(int)


def _builtin_base_frame(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    overrides = config.covariate_prevalences

    patient_id = [f"P{i:05d}" for i in range(n)]
    age_years = np.clip(rng.normal(65, 10, size=n), 30, 90).round(1)
    sex_female = rng.binomial(1, 0.45, size=n)

    smoking_status = rng.choice(_SMOKING_CATEGORIES, size=n, p=_SMOKING_PROBS)
    ecog_ps = rng.choice(_ECOG_CATEGORIES, size=n, p=_ECOG_PROBS)

    # Histology: squamous enriched in current smokers; otherwise adenocarcinoma.
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

    # ALK rearrangement: enriched in never-smokers and further in age < 55.
    alk_base = _bernoulli_by_stratum(rng, smoking_status, _ALK_BY_SMOKING)
    young_bump = ((age_years < 55) & (rng.random(n) < 0.10)).astype(int)
    alk_fusion = np.clip(alk_base + young_bump, 0, 1)
    if "alk_fusion" in overrides:
        alk_fusion = rng.binomial(1, float(overrides["alk_fusion"]), size=n)

    def _marginal(col: str, default: float) -> np.ndarray:
        rate = float(overrides.get(col, default))
        return rng.binomial(1, rate, size=n)

    stk11_mutation = _marginal("stk11_mutation", _DEFAULT_PREVALENCES["stk11_mutation"])
    brca2_mutation = _marginal("brca2_mutation", _DEFAULT_PREVALENCES["brca2_mutation"])
    has_brain_mets = _marginal("has_brain_mets", _DEFAULT_PREVALENCES["has_brain_mets"])
    stage_iv = _marginal("stage_iv", _DEFAULT_PREVALENCES["stage_iv"])

    # PD-L1 TPS: continuous biomarker in [0, 1], centered at the configured
    # mean (defaults to 0.30).
    pdl1_mean = float(overrides.get("pdl1_tps", _DEFAULT_PREVALENCES["pdl1_tps"]))
    pdl1_tps = np.clip(rng.beta(2, 2, size=n) + (pdl1_mean - 0.5), 0.0, 1.0)

    # Disease-burden / performance-status proxy labs and indices. Sampled
    # independently here (no shared latent factor) but they enter the outcome
    # linear predictors via the background-prognostic layer in injector.py,
    # so they will appear correlated with PFM and OR in the final dataset —
    # matching how a real EHR pull behaves and avoiding the "100 columns,
    # zero signal" tell.
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

    # Treatments are drawn independently of biomarkers (randomized-like).
    t_pembro = _marginal(
        "treatment_pembrolizumab", _DEFAULT_PREVALENCES["treatment_pembrolizumab"]
    )
    t_soto = _marginal(
        "treatment_sotorasib", _DEFAULT_PREVALENCES["treatment_sotorasib"]
    )
    t_olap = _marginal(
        "treatment_olaparib", _DEFAULT_PREVALENCES["treatment_olaparib"]
    )
    t_osim = _marginal(
        "treatment_osimertinib", _DEFAULT_PREVALENCES["treatment_osimertinib"]
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


def _oci_base_frame(config: GeneratorConfig) -> pd.DataFrame:
    """Delegate to the upstream onc-causal-inference synthetic generator.

    This is a thin adapter; the exact upstream API is intentionally imported
    lazily so the package is usable without the heavy ML stack. If/when the
    upstream schema drifts, the mapping below is the single place to update.
    """
    try:
        # The upstream module layout at the time of writing this scaffolding.
        from onc_causal_inference.synthetic_data import (  # type: ignore[import-not-found]
            generator as oci_generator,
        )
    except Exception as exc:  # pragma: no cover - import path depends on upstream
        raise RuntimeError(
            "backend='onc_causal_inference' requires the 'synthetic' extra: "
            "pip install 'onc-co-scientist[synthetic]'"
        ) from exc

    # The upstream generator's public entry point may evolve; the expectation
    # here is that it returns a DataFrame with patient_id, covariates, and one
    # or more treatment columns. We append our paradigm-driven outcomes on top.
    produce = getattr(oci_generator, "generate_tabular", None)
    if produce is None:  # pragma: no cover
        raise RuntimeError(
            "Expected onc_causal_inference.synthetic_data.generator.generate_tabular(...); "
            "upstream API appears to have changed. Update _oci_base_frame() to match."
        )
    frame = produce(n=config.patient_n, seed=config.seed)
    if "patient_id" not in frame.columns:
        frame = frame.assign(patient_id=[f"P{i:05d}" for i in range(len(frame))])
    return frame


def _assert_no_cross_class_contradictions(associations: list[AssociationSpec]) -> None:
    """Reject selections that put concordant and discordant specs on the same
    (outcome, variable-set) key. Such pairs would sum to a non-interpretable
    effect in the injector and are never what the user intends."""
    by_key: dict[tuple[str, frozenset[str]], list[AssociationSpec]] = {}
    for spec in associations:
        key = (spec.outcome, frozenset(spec.variables))
        by_key.setdefault(key, []).append(spec)
    for key, specs in by_key.items():
        classes = {s.paradigm_class for s in specs}
        if ParadigmClass.concordant in classes and ParadigmClass.discordant in classes:
            ids = ", ".join(s.id for s in specs)
            raise ValueError(
                "Cannot inject paradigm-concordant and paradigm-discordant associations "
                f"that share the same outcome and variable set {key!r}. Offenders: {ids}. "
                "Expand the catalog or adjust n_* counts."
            )


def _select_backend(config: GeneratorConfig) -> pd.DataFrame:
    if config.backend == "builtin":
        return _builtin_base_frame(config)
    if config.backend == "onc_causal_inference":
        return _oci_base_frame(config)
    raise ValueError(f"Unknown backend: {config.backend!r}")


# Offset used to spawn an RNG stream for distractor sampling that is
# independent of the base-frame RNG. Chosen so that setting
# n_extra_covariates=0 leaves the base-frame draws byte-identical to the
# pre-distractor implementation, preserving existing test seeds.
_DISTRACTOR_SEED_OFFSET = 10_007


def _append_distractor_covariates(
    base_frame: pd.DataFrame, config: GeneratorConfig
) -> pd.DataFrame:
    """Append `config.n_extra_covariates` distractor columns to `base_frame`.

    Distractors are sampled independently of every existing column and of every
    outcome, using a separate RNG seeded deterministically from `config.seed`.
    Raises `ValueError` if any distractor name would collide with an existing
    column — distractor names must remain disjoint from the paradigm-used set.
    """
    n = config.n_extra_covariates
    if n <= 0:
        return base_frame
    if n > len(DEFAULT_DISTRACTOR_POOL):
        raise ValueError(
            f"n_extra_covariates={n} exceeds DEFAULT_DISTRACTOR_POOL size "
            f"({len(DEFAULT_DISTRACTOR_POOL)}). Either lower the requested "
            f"count or extend the pool in "
            f"src/onc_co_scientist/synthetic/distractors.py."
        )
    rng = np.random.default_rng(config.seed + _DISTRACTOR_SEED_OFFSET)
    columns = sample_distractors(rng, n_patients=len(base_frame), n=n)
    existing = set(base_frame.columns)
    collisions = [name for name in columns if name in existing]
    if collisions:
        raise ValueError(
            f"Distractor names collide with existing base-frame columns: "
            f"{collisions!r}. Rename the offending DistractorSpec entries."
        )
    distractor_frame = pd.DataFrame(columns, index=base_frame.index)
    return pd.concat([base_frame, distractor_frame], axis=1)


def _public_description(
    config: GeneratorConfig, frame: pd.DataFrame, outcomes: list[str]
) -> str:
    """Markdown shown to the agent. Does NOT reveal paradigm class or ground truth."""
    covariates = [c for c in frame.columns if c not in outcomes and c != "patient_id"]
    bullet = "\n".join(f"- `{c}`" for c in covariates)
    outcome_bullet = "\n".join(f"- `{c}`" for c in outcomes)
    return (
        f"# Oncology patient cohort `{config.dataset_id}`\n\n"
        f"This dataset contains {config.patient_n} patient records assembled "
        "from electronic health records aggregated by a commercial healthcare "
        "data vendor. Columns include patient features and clinical outcomes.\n\n"
        "## Columns\n\n"
        "### Identifiers and features\n"
        f"{bullet}\n\n"
        "### Outcomes\n"
        f"{outcome_bullet}\n\n"
        "Each row represents one patient; no missing values are present."
    )


@dataclass
class DatasetBundle:
    config: GeneratorConfig
    frame: pd.DataFrame
    manifest: DatasetManifest
    public_description: str

    @property
    def outcome_columns(self) -> list[str]:
        return list(self.manifest.outcome_columns)


def generate_dataset(config: GeneratorConfig) -> DatasetBundle:
    """Produce a full dataset bundle: frame + ground-truth manifest + public description."""
    associations: list[AssociationSpec] = select_associations(
        n_concordant=config.n_concordant,
        n_discordant=config.n_discordant,
        n_hidden_novel=config.n_hidden_novel,
        n_buried_signatures=config.n_buried_signatures,
    )
    _assert_no_cross_class_contradictions(associations)
    counts = summarize_injection(associations)
    log.info(
        "Selected associations for %s: %s",
        config.dataset_id,
        {k.value: v for k, v in counts.items()},
    )

    base_frame = _select_backend(config)
    base_frame = _append_distractor_covariates(base_frame, config)
    injection: InjectionResult = inject_associations(
        base_frame=base_frame,
        associations=associations,
        seed=config.seed,
        continuous_outcome_sigma=config.continuous_outcome_sigma,
    )
    frame = injection.frame
    outcome_columns = injection.outcome_columns

    treatment_columns = [c for c in frame.columns if c.startswith("treatment_")]
    covariate_columns = [
        c
        for c in frame.columns
        if c != "patient_id" and c not in outcome_columns and c not in treatment_columns
    ]

    manifest = DatasetManifest(
        dataset_id=config.dataset_id,
        seed=config.seed,
        patient_n=config.patient_n,
        columns=list(frame.columns),
        treatment_columns=treatment_columns,
        outcome_columns=outcome_columns,
        covariate_columns=covariate_columns,
        associations=associations,
        notes=(
            f"Paradigm mix: concordant={counts[ParadigmClass.concordant]}, "
            f"discordant={counts[ParadigmClass.discordant]}, "
            f"hidden_novel={counts[ParadigmClass.hidden_novel]}."
        ),
    )
    description = _public_description(config, frame, outcome_columns)
    return DatasetBundle(
        config=config, frame=frame, manifest=manifest, public_description=description
    )
