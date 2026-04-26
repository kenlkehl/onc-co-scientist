"""Multi-cancer-type generation helpers.

Wraps the single-cancer ``generate_dataset`` call so the CLI can produce
several cancer-type bundles from one config in one invocation, writing each
bundle under its own subfolder of the output directory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .anonymize import anonymize_bundle
from .cancer_types import CancerType
from .generator import DatasetBundle, GeneratorConfig, generate_dataset
from .io import write_bundle, write_bundle_pair


def _config_for(config: GeneratorConfig, cancer_type: CancerType) -> GeneratorConfig:
    """Clone a base config for one cancer type, auto-suffixing the dataset id.

    The base ``dataset_id`` is treated as a group label; the per-bundle id
    becomes ``f"{base}_{cancer_type.value}"`` so manifests across cancer types
    have distinct identifiers downstream (harness task spec, scoring).
    """
    suffixed_id = f"{config.dataset_id}_{cancer_type.value}"
    return replace(config, dataset_id=suffixed_id, cancer_type=cancer_type.value)


def generate_multi_dataset(
    base_config: GeneratorConfig,
    cancer_types: Iterable[CancerType],
) -> dict[CancerType, DatasetBundle]:
    """Generate one ``DatasetBundle`` per cancer type from a shared config."""
    bundles: dict[CancerType, DatasetBundle] = {}
    for ct in cancer_types:
        bundles[ct] = generate_dataset(_config_for(base_config, ct))
    return bundles


def write_multi_bundle_pair(
    bundles: dict[CancerType, DatasetBundle],
    out_dir: Path | str,
    *,
    anon_seed: int = 0,
) -> dict[CancerType, tuple[Path, Path]]:
    """Write each bundle's named + anonymized twin under ``<out>/<cancer_type>/``.

    Returns a mapping from cancer type to ``(named_dir, anonymized_dir)``.
    """
    out_path = Path(out_dir)
    written: dict[CancerType, tuple[Path, Path]] = {}
    for ct, bundle in bundles.items():
        named_dir, anon_dir = write_bundle_pair(
            bundle, out_path / ct.value, anon_seed=anon_seed
        )
        written[ct] = (named_dir, anon_dir)
    return written


def write_multi_bundle(
    bundles: dict[CancerType, DatasetBundle],
    out_dir: Path | str,
    *,
    anonymize: bool = False,
    anon_seed: int = 0,
) -> dict[CancerType, Path]:
    """Write each bundle as a single variant under ``<out>/<cancer_type>/``.

    When ``anonymize`` is True, the anonymized twin is materialized and
    written; otherwise the named bundle is written. Returns a mapping from
    cancer type to the written bundle root.
    """
    out_path = Path(out_dir)
    written: dict[CancerType, Path] = {}
    for ct, bundle in bundles.items():
        if anonymize:
            anon_bundle, _ = anonymize_bundle(bundle, seed=anon_seed)
            written[ct] = write_bundle(anon_bundle, out_path / ct.value)
        else:
            written[ct] = write_bundle(bundle, out_path / ct.value)
    return written
