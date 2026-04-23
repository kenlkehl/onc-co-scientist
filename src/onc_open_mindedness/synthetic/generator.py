"""Synthetic oncology dataset generator.

The generator produces patient-level tabular data containing biomarker and
treatment columns, then layers in outcome columns driven by a selected mix of
paradigm-concordant, paradigm-discordant, and hidden-novel associations.

Two code paths are supported:

1. **Stand-alone (default).** A deterministic tabular base built from
   independent draws of clinically meaningful covariates. Fast, dependency-free,
   used for tests and the MVP end-to-end smoke.

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

from .injector import InjectionResult, inject_associations, summarize_injection
from .paradigms import select_associations
from .schemas import AssociationSpec, DatasetManifest, ParadigmClass

log = logging.getLogger(__name__)

BackendName = Literal["builtin", "onc_causal_inference"]


@dataclass
class GeneratorConfig:
    """Configuration for one synthetic dataset bundle."""

    dataset_id: str
    patient_n: int = 500
    seed: int = 0
    n_concordant: int = 2
    n_discordant: int = 1
    n_hidden_novel: int = 1
    backend: BackendName = "builtin"
    continuous_outcome_sigma: float = 3.0
    # Per-covariate prevalence overrides for the builtin backend.
    covariate_prevalences: dict[str, float] = field(default_factory=dict)


_DEFAULT_PREVALENCES: dict[str, float] = {
    "egfr_mutation": 0.15,
    "kras_g12c": 0.12,
    "pdl1_tps": 0.30,  # reinterpreted below as a continuous mean, not prevalence
    "tmb_high": 0.25,
    "biomarker_z_high": 0.10,
    "treatment_io": 0.50,
    "treatment_kras_g12c_inhibitor": 0.35,
    "treatment_x": 0.40,
}


def _builtin_base_frame(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.patient_n
    prev = {**_DEFAULT_PREVALENCES, **config.covariate_prevalences}

    frame = pd.DataFrame(
        {
            "patient_id": [f"P{i:05d}" for i in range(n)],
            "age_years": np.clip(rng.normal(65, 10, size=n), 30, 90).round(1),
            "sex_female": rng.binomial(1, 0.45, size=n),
            "egfr_mutation": rng.binomial(1, prev["egfr_mutation"], size=n),
            "kras_g12c": rng.binomial(1, prev["kras_g12c"], size=n),
            # pdl1_tps is modeled as a continuous biomarker between 0 and 1,
            # centered at the configured mean.
            "pdl1_tps": np.clip(
                rng.beta(2, 2, size=n) + (prev["pdl1_tps"] - 0.5), 0.0, 1.0
            ),
            "tmb_high": rng.binomial(1, prev["tmb_high"], size=n),
            "biomarker_z_high": rng.binomial(1, prev["biomarker_z_high"], size=n),
            "treatment_io": rng.binomial(1, prev["treatment_io"], size=n),
            "treatment_kras_g12c_inhibitor": rng.binomial(
                1, prev["treatment_kras_g12c_inhibitor"], size=n
            ),
            "treatment_x": rng.binomial(1, prev["treatment_x"], size=n),
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
            "pip install 'onc-open-mindedness[synthetic]'"
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


def _public_description(
    config: GeneratorConfig, frame: pd.DataFrame, outcomes: list[str]
) -> str:
    """Markdown shown to the agent. Does NOT reveal paradigm class or ground truth."""
    covariates = [c for c in frame.columns if c not in outcomes and c != "patient_id"]
    bullet = "\n".join(f"- `{c}`" for c in covariates)
    outcome_bullet = "\n".join(f"- `{c}`" for c in outcomes)
    return (
        f"# Synthetic oncology dataset `{config.dataset_id}`\n\n"
        f"This dataset contains {config.patient_n} simulated patients with lung-cancer-like "
        "covariates, treatment indicators, and one or more outcomes. It was generated for "
        "use with the Oncology Scientific Open-Mindedness Benchmark.\n\n"
        "The data-generating process embeds an undisclosed mix of associations between "
        "covariates, treatments, and outcomes. Some associations are consistent with "
        "current oncology consensus; others are deliberately inverted; and at least one "
        "is a subgroup-specific signal only recoverable by an analyst willing to probe "
        "heterogeneity. You are not told which is which.\n\n"
        "## Columns\n\n"
        "### Identifiers and covariates\n"
        f"{bullet}\n\n"
        "### Outcomes\n"
        f"{outcome_bullet}\n\n"
        "All rows are independent. Missingness is not simulated in this release."
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
    )
    _assert_no_cross_class_contradictions(associations)
    counts = summarize_injection(associations)
    log.info(
        "Selected associations for %s: %s",
        config.dataset_id,
        {k.value: v for k, v in counts.items()},
    )

    base_frame = _select_backend(config)
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
