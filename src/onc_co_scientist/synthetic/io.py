"""Read/write synthetic dataset bundles.

Bundle layout on disk::

    <bundle_dir>/
    ├── manifest.json          # ground-truth; NEVER shown to the evaluated agent
    └── public/                # agent-safe workspace — okay to run the agent here
        ├── dataset.parquet
        └── dataset_description.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generator import DatasetBundle
from .schemas import DatasetManifest

MANIFEST_FILENAME = "manifest.json"
DATASET_FILENAME = "dataset.parquet"
DESCRIPTION_FILENAME = "dataset_description.md"
PUBLIC_SUBDIR = "public"


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
