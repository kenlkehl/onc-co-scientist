"""Read/write synthetic dataset bundles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generator import DatasetBundle
from .schemas import DatasetManifest

MANIFEST_FILENAME = "manifest.json"
DATASET_FILENAME = "dataset.parquet"
DESCRIPTION_FILENAME = "dataset_description.md"


def write_bundle(bundle: DatasetBundle, out_dir: Path | str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bundle.frame.to_parquet(out_path / DATASET_FILENAME, index=False)
    (out_path / MANIFEST_FILENAME).write_text(
        bundle.manifest.model_dump_json(indent=2) + "\n"
    )
    (out_path / DESCRIPTION_FILENAME).write_text(bundle.public_description + "\n")
    return out_path


def read_manifest(bundle_dir: Path | str) -> DatasetManifest:
    path = Path(bundle_dir) / MANIFEST_FILENAME
    return DatasetManifest.model_validate_json(path.read_text())


def read_frame(bundle_dir: Path | str) -> pd.DataFrame:
    return pd.read_parquet(Path(bundle_dir) / DATASET_FILENAME)


def read_description(bundle_dir: Path | str) -> str:
    return (Path(bundle_dir) / DESCRIPTION_FILENAME).read_text()
