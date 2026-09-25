"""NSCLC CRISPR/DepMap-style cell-line dependency profile."""

from __future__ import annotations

from ._depmap_common import build_depmap_profile, nsclc_buried_signature_catalog

PROFILE = build_depmap_profile(
    cancer_type="nsclc_depmap",
    display_name="NSCLC CRISPR dependency map",
    dataset_id_suffix="nsclc_depmap",
    lineage="lung",
    buried_signature_catalog=nsclc_buried_signature_catalog,
)
