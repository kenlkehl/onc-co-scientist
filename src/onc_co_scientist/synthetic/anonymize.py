"""Produce an anonymized twin of a ``DatasetBundle``.

The anonymized variant preserves rows, clinical outcome names, and identifier
columns while renaming predictors to deterministic ``feature_NNN`` identifiers.
DepMap dependency outcomes are also masked, using ``outcome_NNN`` identifiers.
The same generated bundle can therefore be served in two parallel
forms — one with real clinical names, one stripped of any semantic prior — so
the eval can compare an agent's behaviour with and without domain anchoring.

The mapping is built from a seeded shuffle so column-name ordering does not
leak structure (alphabetical ordering would, e.g., group every
``treatment_*`` column together). The mapping is stored in the renamed
manifest so scoring code keeps working unchanged against either variant.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .generator import DatasetBundle, GeneratorConfig
from .schemas import AssociationSpec, DatasetManifest, SubgroupSpec

DEFAULT_ID_COLUMNS: tuple[str, ...] = ("patient_id",)
DEPMAP_MASKING_VERSION = "depmap-features-and-outcomes-v1"


def build_column_mapping(
    columns: list[str],
    outcome_columns: list[str],
    *,
    id_columns: tuple[str, ...] = DEFAULT_ID_COLUMNS,
    seed: int = 0,
    prefix: str = "feature_",
    width: int = 3,
) -> dict[str, str]:
    """Return a deterministic ``{real_name: anonymized_name}`` mapping.

    Outcome columns and id columns are excluded from the mapping (they are
    preserved verbatim in the anonymized bundle). The remaining columns are
    shuffled with ``np.random.default_rng(seed)`` and assigned
    ``feature_001``, ``feature_002``, …
    """
    excluded = set(outcome_columns) | set(id_columns)
    feature_cols = [c for c in columns if c not in excluded]
    rng = np.random.default_rng(seed)
    order = list(feature_cols)
    rng.shuffle(order)
    needed_width = max(width, len(str(len(order))))
    return {original: f"{prefix}{(i + 1):0{needed_width}d}" for i, original in enumerate(order)}


def extend_outcome_mapping(
    mapping: dict[str, str], outcome_columns: list[str], *, seed: int = 0
) -> dict[str, str]:
    """Mask outcome identities without changing existing feature/outcome aliases."""
    outcomes = build_column_mapping(
        outcome_columns, [], id_columns=(), seed=seed, prefix="outcome_"
    )
    extended = {**outcomes, **mapping}
    if len(set(extended.values())) != len(extended):
        raise ValueError("Outcome masking would create duplicate column aliases")
    return extended


def _rename_predicate(predicate: dict[str, object], mapping: dict[str, str]) -> dict[str, object]:
    return {mapping.get(k, k): v for k, v in predicate.items()}


def _rename_association(spec: AssociationSpec, mapping: dict[str, str]) -> AssociationSpec:
    new = spec.model_copy(deep=True)
    new.variables = [mapping.get(v, v) for v in spec.variables]
    new.outcome = mapping.get(spec.outcome, spec.outcome)
    if spec.subgroup is not None:
        new.subgroup = SubgroupSpec(
            name=spec.subgroup.name,
            predicate=_rename_predicate(spec.subgroup.predicate, mapping),
            description=spec.subgroup.description,
        )
    return new


def _rename_manifest(manifest: DatasetManifest, mapping: dict[str, str]) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=manifest.dataset_id,
        seed=manifest.seed,
        patient_n=manifest.patient_n,
        cancer_type=manifest.cancer_type,
        dataset_kind=manifest.dataset_kind,
        id_columns=list(manifest.id_columns),
        columns=[mapping.get(c, c) for c in manifest.columns],
        treatment_columns=[mapping.get(c, c) for c in manifest.treatment_columns],
        outcome_columns=[mapping.get(c, c) for c in manifest.outcome_columns],
        covariate_columns=[mapping.get(c, c) for c in manifest.covariate_columns],
        associations=[_rename_association(a, mapping) for a in manifest.associations],
        generator_version=manifest.generator_version,
        notes=manifest.notes,
    )


def masked_depmap_description(
    dataset_id: str,
    record_n: int,
    columns: list[str],
    outcomes: list[str],
    id_columns: tuple[str, ...],
) -> str:
    """Describe public column roles and score direction without knockout identities."""
    features = [c for c in columns if c not in set(outcomes) | set(id_columns)]
    identifiers = [c for c in columns if c in id_columns]

    def bullet(names: list[str]) -> str:
        return "\n".join(f"- `{name}`" for name in names)

    return (
        f"# CRISPR dependency map `{dataset_id}`\n\n"
        f"This dataset contains {record_n} cancer cell-line records "
        "from a CRISPR knockout dependency screen with CCLE-style molecular "
        "annotations. Cell-line feature names have been replaced with opaque "
        "labels (`feature_001`, `feature_002`, ...); dependency outcomes use "
        "opaque labels (`outcome_001`, `outcome_002`, ...). Each dependency "
        "outcome measures the effect of knocking out a different gene. More negative "
        "dependency scores indicate stronger dependency after knockout.\n\n"
        "## Columns\n\n"
        f"### Identifiers\n{bullet(identifiers)}\n\n"
        f"### Cell-line features\n{bullet(features)}\n\n"
        f"### Dependency outcomes\n{bullet(outcomes)}\n\n"
        "Each row represents one cancer cell line. Inspect the supplied data for "
        "missing values in metadata and screen-QC fields."
    )


def _anonymized_description(
    config: GeneratorConfig,
    columns: list[str],
    outcomes: list[str],
    id_columns: tuple[str, ...],
    dataset_kind: str,
) -> str:
    feature_cols = [c for c in columns if c not in set(outcomes) and c not in set(id_columns)]
    bullet = "\n".join(f"- `{c}`" for c in feature_cols)
    outcome_bullet = "\n".join(f"- `{c}`" for c in outcomes)
    if dataset_kind == "crispr_depmap":
        return masked_depmap_description(
            config.dataset_id, config.patient_n, columns, outcomes, id_columns
        )
    return (
        f"# Oncology patient cohort `{config.dataset_id}`\n\n"
        f"This dataset contains {config.patient_n} patient records assembled "
        "from electronic health records aggregated by a commercial healthcare "
        "data vendor. Patient features have been de-identified to opaque "
        "labels (`feature_001`, `feature_002`, …); clinical outcomes retain "
        "their original names.\n\n"
        "## Columns\n\n"
        "### Identifiers and features\n"
        f"{bullet}\n\n"
        "### Outcomes\n"
        f"{outcome_bullet}\n\n"
        "Each row represents one patient; no missing values are present."
    )


def anonymize_bundle(
    bundle: DatasetBundle,
    *,
    seed: int = 0,
    id_columns: tuple[str, ...] | None = None,
) -> tuple[DatasetBundle, dict[str, str]]:
    """Return an anonymized twin of ``bundle`` plus the rename mapping.

    The original bundle is not mutated. The returned bundle has:

    - the same rows in the same order,
    - the same clinical outcome column names and id columns,
    - DepMap dependency outcomes renamed to ``outcome_NNN``,
    - every predictor column renamed to ``feature_NNN`` per
      ``build_column_mapping(seed=seed)``,
    - a manifest whose ``columns``, ``treatment_columns``,
      ``outcome_columns``, ``covariate_columns`` and per-``AssociationSpec`` ``outcome``,
      ``variables`` /
      ``subgroup.predicate`` keys are remapped consistently.
    """
    active_id_columns = tuple(bundle.manifest.id_columns) if id_columns is None else id_columns
    mapping = build_column_mapping(
        list(bundle.frame.columns),
        list(bundle.manifest.outcome_columns),
        id_columns=active_id_columns,
        seed=seed,
    )
    if bundle.manifest.dataset_kind == "crispr_depmap":
        mapping = extend_outcome_mapping(mapping, bundle.manifest.outcome_columns, seed=seed)
    new_frame = bundle.frame.rename(columns=mapping)
    new_manifest = _rename_manifest(bundle.manifest, mapping)
    new_description = _anonymized_description(
        bundle.config,
        list(new_frame.columns),
        list(new_manifest.outcome_columns),
        active_id_columns,
        new_manifest.dataset_kind,
    )
    new_bundle = DatasetBundle(
        config=deepcopy(bundle.config),
        frame=new_frame,
        manifest=new_manifest,
        public_description=new_description,
    )
    return new_bundle, mapping
