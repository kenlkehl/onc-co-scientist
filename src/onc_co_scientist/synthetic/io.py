"""Read/write synthetic dataset bundles.

Bundle layout on disk::

    <bundle_dir>/
    ├── manifest.json          # ground-truth; NEVER shown to the evaluated agent
    └── public/                # agent-safe workspace — okay to run the agent here
        ├── dataset.parquet
        └── dataset_description.md

Paired layout (named + anonymized twin) written by ``write_bundle_pair``::

    <out_dir>/
    ├── named/                 # real-name bundle (above layout)
    └── anonymized/            # feature_NNN-renamed twin
        ├── manifest.json
        ├── column_mapping.json   # original -> anonymized name map
        └── public/
            ├── dataset.parquet
            └── dataset_description.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .anonymize import anonymize_bundle
from .generator import DatasetBundle
from .schemas import DatasetManifest

MANIFEST_FILENAME = "manifest.json"
DATASET_FILENAME = "dataset.parquet"
DESCRIPTION_FILENAME = "dataset_description.md"
COLUMN_MAPPING_FILENAME = "column_mapping.json"
PUBLIC_SUBDIR = "public"
NAMED_SUBDIR = "named"
ANONYMIZED_SUBDIR = "anonymized"


def public_dir(bundle_dir: Path | str) -> Path:
    """Return the agent-safe subdirectory within a dataset bundle."""
    return Path(bundle_dir) / PUBLIC_SUBDIR


def write_bundle(bundle: DatasetBundle, out_dir: Path | str) -> Path:
    out_path = Path(out_dir)
    public = out_path / PUBLIC_SUBDIR
    public.mkdir(parents=True, exist_ok=True)

    (out_path / MANIFEST_FILENAME).write_text(
        bundle.manifest.model_dump_json(indent=2) + "\n"
    )
    bundle.frame.to_parquet(public / DATASET_FILENAME, index=False)
    (public / DESCRIPTION_FILENAME).write_text(bundle.public_description + "\n")
    return out_path


def read_manifest(bundle_dir: Path | str) -> DatasetManifest:
    path = Path(bundle_dir) / MANIFEST_FILENAME
    return DatasetManifest.model_validate_json(path.read_text())


def read_frame(bundle_dir: Path | str) -> pd.DataFrame:
    return pd.read_parquet(public_dir(bundle_dir) / DATASET_FILENAME)


def read_description(bundle_dir: Path | str) -> str:
    return (public_dir(bundle_dir) / DESCRIPTION_FILENAME).read_text()


def write_bundle_pair(
    bundle: DatasetBundle,
    out_dir: Path | str,
    *,
    anon_seed: int = 0,
) -> tuple[Path, Path]:
    """Write both the named bundle and its anonymized twin under ``out_dir``.

    The named bundle is written to ``out_dir / "named"``; the anonymized twin
    is written to ``out_dir / "anonymized"`` along with a ``column_mapping.json``
    file mapping each original column to its anonymized name. Returns
    ``(named_dir, anonymized_dir)``.

    Both bundles share the same generated rows, the same outcome columns, and
    the same buried-finding ground truth — only feature column names differ.
    """
    out_path = Path(out_dir)
    named_dir = out_path / NAMED_SUBDIR
    anonymized_dir = out_path / ANONYMIZED_SUBDIR

    write_bundle(bundle, named_dir)

    anon_bundle, mapping = anonymize_bundle(bundle, seed=anon_seed)
    write_bundle(anon_bundle, anonymized_dir)
    (anonymized_dir / COLUMN_MAPPING_FILENAME).write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    )
    return named_dir, anonymized_dir
