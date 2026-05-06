"""Colorectal cancer CRISPR/DepMap-style cell-line dependency profile."""

from __future__ import annotations

from ._depmap_common import build_depmap_profile, crc_buried_signature_catalog

PROFILE = build_depmap_profile(
    cancer_type="crc_depmap",
    display_name="Colorectal cancer CRISPR dependency map",
    dataset_id_suffix="crc_depmap",
    lineage="colorectal",
    buried_signature_catalog=crc_buried_signature_catalog,
)
