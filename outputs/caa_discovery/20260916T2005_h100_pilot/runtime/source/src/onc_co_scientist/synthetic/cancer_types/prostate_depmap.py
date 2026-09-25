"""Prostate cancer CRISPR/DepMap-style cell-line dependency profile."""

from __future__ import annotations

from ._depmap_common import build_depmap_profile, prostate_buried_signature_catalog

PROFILE = build_depmap_profile(
    cancer_type="prostate_depmap",
    display_name="Prostate cancer CRISPR dependency map",
    dataset_id_suffix="prostate_depmap",
    lineage="prostate",
    buried_signature_catalog=prostate_buried_signature_catalog,
)
