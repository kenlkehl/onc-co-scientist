#!/usr/bin/env python3
"""Summarize the five generated DepMap profiles for reporting and QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROFILE_LABELS = {
    "nsclc_depmap": "Lung",
    "crc_depmap": "Bowel",
    "breast_depmap": "Breast",
    "prostate_depmap": "Prostate",
    "aml_depmap": "Myeloid",
}


def _resolve_bundle_dir(profile_root: Path) -> Path:
    """Resolve either a single-variant or default named/anonymized layout."""
    for candidate in (profile_root, profile_root / "named"):
        if (candidate / "manifest.json").is_file() and (
            candidate / "public" / "dataset.parquet"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a named DepMap bundle under {profile_root}; expected "
        "manifest.json plus public/dataset.parquet either directly or in named/."
    )


def _numeric(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _counts(series: pd.Series) -> dict[str, dict[str, float | int]]:
    total = len(series)
    values = series.value_counts(dropna=False)
    return {
        str(label): {"n": int(count), "percent": 100.0 * int(count) / total}
        for label, count in values.items()
    }


def summarize(root: Path) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    contexts: dict[str, dict[str, int | float]] = {}
    seeds: set[int] = set()
    for profile, label in PROFILE_LABELS.items():
        bundle = _resolve_bundle_dir(root / profile)
        frame = pd.read_parquet(bundle / "public" / "dataset.parquet")
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if manifest["patient_n"] != len(frame):
            raise ValueError(f"Manifest/frame size mismatch for {profile}")
        frame = frame.assign(task_context=label)
        frames.append(frame)
        contexts[label] = {"n": len(frame), "percent": 0.0}
        seeds.add(int(manifest["seed"]))

    combined = pd.concat(frames, ignore_index=True)
    total = len(combined)
    for context in contexts.values():
        context["percent"] = 100.0 * int(context["n"]) / total
    if total != 10_000:
        raise ValueError(f"Expected 10,000 total DepMap records, found {total}")
    if len(seeds) != 1:
        raise ValueError(f"Expected one generation seed, found {sorted(seeds)}")

    binary_counts = {}
    for column in (
        "has_rna_omics",
        "has_dna_omics",
        "has_matched_rna_dna",
        "has_crispr_qc",
        "has_rna_dna_crispr_qc",
        "crispr_library_avana",
        "crispr_library_humagne_cd",
        "crispr_library_ky",
    ):
        count = int(combined[column].sum())
        binary_counts[column] = {"n": count, "percent": 100.0 * count / total}

    qc_columns = (
        "screen_nnmd",
        "screen_roc_auc",
        "cas9_activity_pct",
        "screen_doubling_time_hours",
    )
    correlations = combined[list(qc_columns)].corr(method="spearman")
    return {
        "calibration_release": "DepMap Public 26Q1",
        "generation_seed": seeds.pop(),
        "population_n": total,
        "task_context": contexts,
        "age_years": _numeric(combined["age_years"]),
        "age_category": _counts(combined["age_category"]),
        "sex": _counts(combined["sex"]),
        "growth_pattern": _counts(combined["growth_pattern"]),
        "default_omics_profile": _counts(combined["default_omics_profile"]),
        "availability_and_library": binary_counts,
        "screen_qc": {column: _numeric(combined[column]) for column in qc_columns},
        "screen_qc_spearman": {
            row: {column: float(correlations.loc[row, column]) for column in qc_columns}
            for row in qc_columns
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Root containing the five DepMap bundles")
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    payload = json.dumps(summarize(args.root), indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
    else:
        args.out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
