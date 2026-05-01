"""Synthetic oncology dataset generator with paradigm-class ground truth.

The generator produces patient-level tabular data containing realistic
demographics, biomarker, and treatment columns for a chosen cancer type, then
layers in outcome columns driven by a selected mix of paradigm-concordant,
paradigm-discordant, and hidden-novel associations.

Cancer-type behaviour (base-frame sampler, paradigm catalogs, prognostic
layer, prevalence defaults) is owned by ``CancerProfile`` instances in
``cancer_types/``. The default profile is NSCLC, preserving the pre-multi-
cancer pipeline; CRC, breast, prostate, and AML profiles add support for
those cohorts.

Two backends are supported:

1. **Stand-alone (default).** A deterministic tabular base built from draws of
   clinically meaningful covariates supplied by the active ``CancerProfile``.

2. **onc-causal-inference-backed (opt-in, NSCLC only).** Delegates tabular
   generation to the upstream ``oci.synthetic_data`` module. Currently NSCLC-
   only; selecting it for any other cancer type raises ``NotImplementedError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .cancer_types import CancerProfile, CancerType, get_profile
from .distractors import DEFAULT_DISTRACTOR_POOL, sample_distractors
from .injector import InjectionResult, inject_associations, summarize_injection
from .paradigms import select_associations
from .schemas import AssociationForm, AssociationSpec, DatasetManifest, ParadigmClass

log = logging.getLogger(__name__)

BackendName = Literal["builtin", "onc_causal_inference"]


@dataclass
class GeneratorConfig:
    """Configuration for one synthetic dataset bundle."""

    dataset_id: str
    # Cancer type whose ``CancerProfile`` drives the base-frame sampler and
    # paradigm catalogs. Defaults to NSCLC for backward compatibility with the
    # pre-multi-cancer pipeline.
    cancer_type: str = "nsclc"
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
    # Number of multi-feature "buried" findings drawn from the active
    # profile's ``buried_signature_catalog()``. Each is a treatment-conditional
    # subgroup defined by 3-4 baseline features in conjunction. These are
    # tagged ParadigmClass.hidden_novel for scoring purposes but counted
    # independently from n_hidden_novel so the legacy single-predicate
    # catalog and the multi-feature catalog stay configurable separately.
    n_buried_signatures: int = 1
    # Minimum number of rows that must satisfy both the buried-signature
    # subgroup predicate and the signature's treatment driver. A strict floor
    # keeps selected buried findings statistically recoverable while still
    # leaving the agent to find the right multivariable candidate. Set to 0
    # for tiny smoke-test cohorts.
    min_buried_treated_subgroup_n: int = 1000
    backend: BackendName = "builtin"
    continuous_outcome_sigma: float = 2.0
    # Per-covariate marginal-prevalence overrides for the builtin backend.
    # Keys that don't apply to the active profile (e.g. ``egfr_mutation`` when
    # ``cancer_type='aml'``) are ignored at sampling time, so a single config
    # can drive multiple cancer-type bundles.
    covariate_prevalences: dict[str, float] = field(default_factory=dict)
    # Number of realistic-looking "distractor" covariates appended to the
    # dataset, sampled independently of every outcome. Higher values force an
    # agent to exercise variable-selection judgment instead of brute-forcing a
    # univariate test against every column. Max is the pool size in
    # ``distractors.DEFAULT_DISTRACTOR_POOL``. Applies to all backends.
    n_extra_covariates: int = 50


def _resolve_profile(config: GeneratorConfig) -> CancerProfile:
    return get_profile(config.cancer_type)


def _builtin_base_frame(config: GeneratorConfig) -> pd.DataFrame:
    profile = _resolve_profile(config)
    return profile.base_frame_fn(config)


def _oci_base_frame(config: GeneratorConfig) -> pd.DataFrame:
    """Delegate to the upstream onc-causal-inference synthetic generator.

    Currently NSCLC-only. The upstream generator's column schema is hard-
    coded to NSCLC, so any other ``cancer_type`` value raises
    ``NotImplementedError`` rather than silently delivering NSCLC data under
    the wrong label.
    """
    if CancerType(config.cancer_type) is not CancerType.nsclc:
        raise NotImplementedError(
            "backend='onc_causal_inference' is currently NSCLC-only. "
            f"cancer_type={config.cancer_type!r} requires backend='builtin'."
        )
    try:
        from onc_causal_inference.synthetic_data import (  # type: ignore[import-not-found]
            generator as oci_generator,
        )
    except Exception as exc:  # pragma: no cover - import path depends on upstream
        raise RuntimeError(
            "backend='onc_causal_inference' requires the 'synthetic' extra: "
            "pip install 'onc-co-scientist[synthetic]'"
        ) from exc

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
    for _key, specs in by_key.items():
        classes = {s.paradigm_class for s in specs}
        if ParadigmClass.concordant in classes and ParadigmClass.discordant in classes:
            ids = ", ".join(s.id for s in specs)
            raise ValueError(
                "Cannot inject paradigm-concordant and paradigm-discordant associations "
                f"that share the same outcome and variable set. Offenders: {ids}. "
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
    Distractor specs whose name collides with an existing base-frame column
    for the active profile are filtered out before sampling, and the
    requested count is filled from the next non-colliding pool entries
    instead. This keeps a single config valid across multiple cancer
    profiles whose base-frame columns differ (CRC owns ``cea_ng_ml``, the
    prostate profile owns ``psa_ng_ml``, etc.).
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
    existing = set(base_frame.columns)
    filtered_pool = tuple(
        spec for spec in DEFAULT_DISTRACTOR_POOL if spec.name not in existing
    )
    if n > len(filtered_pool):
        raise ValueError(
            f"n_extra_covariates={n} exceeds the post-collision pool size "
            f"({len(filtered_pool)}) for cancer_type={config.cancer_type!r}. "
            f"Either lower the requested count or extend "
            f"DEFAULT_DISTRACTOR_POOL."
        )
    rng = np.random.default_rng(config.seed + _DISTRACTOR_SEED_OFFSET)
    columns = sample_distractors(
        rng, n_patients=len(base_frame), n=n, pool=filtered_pool
    )
    distractor_frame = pd.DataFrame(columns, index=base_frame.index)
    return pd.concat([base_frame, distractor_frame], axis=1)


def _predicate_mask(frame: pd.DataFrame, predicate: dict[str, object]) -> np.ndarray:
    """Evaluate a subgroup predicate using the injector's predicate semantics."""
    mask = np.ones(len(frame), dtype=bool)
    for col, val in predicate.items():
        if col not in frame.columns:
            raise KeyError(f"Subgroup column {col!r} missing from frame")
        col_vals = frame[col].to_numpy()
        if isinstance(val, dict) and ({"min", "max"} & val.keys()):
            low = val.get("min", -np.inf)
            high = val.get("max", np.inf)
            mask &= (col_vals >= low) & (col_vals <= high)
        else:
            mask &= col_vals == val
    return mask


def _subgroup_driver(spec: AssociationSpec) -> str:
    """Return the treatment/driver column for a subgroup-conditional spec."""
    if spec.form is not AssociationForm.subgroup_conditional or spec.subgroup is None:
        raise ValueError(
            f"Buried-signature {spec.id!r} must be subgroup_conditional."
        )
    active = [c for c in spec.variables if c != spec.outcome]
    predicate_cols = set(spec.subgroup.predicate)
    drivers = [c for c in active if c not in predicate_cols]
    if not drivers:
        raise ValueError(
            f"Buried-signature {spec.id!r} needs a driver variable outside "
            "the subgroup predicate."
        )
    return drivers[0]


def _treated_subgroup_n(frame: pd.DataFrame, spec: AssociationSpec) -> int:
    """Rows satisfying both the subgroup predicate and active treatment driver."""
    if spec.subgroup is None:
        raise ValueError(f"Buried-signature {spec.id!r} requires a subgroup.")
    driver = _subgroup_driver(spec)
    if driver not in frame.columns:
        raise KeyError(f"Driver column {driver!r} missing from frame")
    mask = _predicate_mask(frame, spec.subgroup.predicate)
    treated = frame[driver].to_numpy() == 1
    return int((mask & treated).sum())


def _eligible_buried_pool(
    profile: CancerProfile,
    base_frame: pd.DataFrame,
    config: GeneratorConfig,
) -> list[AssociationSpec]:
    """Filter buried signatures by actual treated subgroup size."""
    threshold = config.min_buried_treated_subgroup_n
    if threshold < 0:
        raise ValueError(
            "min_buried_treated_subgroup_n must be >= 0, "
            f"got {threshold}."
        )

    pool = profile.buried_signature_catalog()
    if threshold == 0:
        return pool

    counts = [(spec, _treated_subgroup_n(base_frame, spec)) for spec in pool]
    eligible = [spec for spec, n in counts if n >= threshold]
    if config.n_buried_signatures > len(eligible):
        detail = ", ".join(f"{spec.id}={n}" for spec, n in counts)
        raise ValueError(
            f"Requested {config.n_buried_signatures} buried-signature "
            "association(s) with "
            f"min_buried_treated_subgroup_n={threshold}, but only "
            f"{len(eligible)} of {len(pool)} candidates are eligible for "
            f"cancer_type={profile.cancer_type!r} at patient_n={config.patient_n}. "
            f"Treated subgroup counts: {detail}."
        )
    return eligible


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
    profile = _resolve_profile(config)
    base_frame = _select_backend(config)
    buried_pool = _eligible_buried_pool(profile, base_frame, config)
    associations: list[AssociationSpec] = select_associations(
        n_concordant=config.n_concordant,
        n_discordant=config.n_discordant,
        n_hidden_novel=config.n_hidden_novel,
        n_buried_signatures=config.n_buried_signatures,
        buried_pool=buried_pool,
        profile=profile,
    )
    _assert_no_cross_class_contradictions(associations)
    counts = summarize_injection(associations)
    log.info(
        "Selected associations for %s (%s): %s",
        config.dataset_id,
        profile.cancer_type,
        {k.value: v for k, v in counts.items()},
    )

    base_frame = _append_distractor_covariates(base_frame, config)
    injection: InjectionResult = inject_associations(
        base_frame=base_frame,
        associations=associations,
        seed=config.seed,
        continuous_outcome_sigma=config.continuous_outcome_sigma,
        profile=profile,
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
        cancer_type=profile.cancer_type,
        columns=list(frame.columns),
        treatment_columns=treatment_columns,
        outcome_columns=outcome_columns,
        covariate_columns=covariate_columns,
        associations=associations,
        notes=(
            f"Cancer type: {profile.cancer_type}. "
            f"Paradigm mix: concordant={counts[ParadigmClass.concordant]}, "
            f"discordant={counts[ParadigmClass.discordant]}, "
            f"hidden_novel={counts[ParadigmClass.hidden_novel]}."
        ),
    )
    description = _public_description(config, frame, outcome_columns)
    return DatasetBundle(
        config=config, frame=frame, manifest=manifest, public_description=description
    )
