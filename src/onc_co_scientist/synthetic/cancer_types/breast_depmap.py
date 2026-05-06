"""Breast cancer CRISPR/DepMap-style cell-line dependency profile."""

from __future__ import annotations

from ._depmap_common import breast_buried_signature_catalog, build_depmap_profile

PROFILE = build_depmap_profile(
    cancer_type="breast_depmap",
    display_name="Breast cancer CRISPR dependency map",
    dataset_id_suffix="breast_depmap",
    lineage="breast",
    buried_signature_catalog=breast_buried_signature_catalog,
)
