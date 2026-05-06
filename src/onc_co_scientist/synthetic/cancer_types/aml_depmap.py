"""AML CRISPR/DepMap-style cell-line dependency profile."""

from __future__ import annotations

from ._depmap_common import aml_buried_signature_catalog, build_depmap_profile

PROFILE = build_depmap_profile(
    cancer_type="aml_depmap",
    display_name="AML CRISPR dependency map",
    dataset_id_suffix="aml_depmap",
    lineage="hematopoietic",
    buried_signature_catalog=aml_buried_signature_catalog,
)
