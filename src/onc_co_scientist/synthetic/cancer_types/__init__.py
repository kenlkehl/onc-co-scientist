"""Per-cancer-type synthetic-generation profiles.

Each supported cancer has a clinical module (``nsclc_clinical.py``,
``crc_clinical.py``, ...) and a CRISPR/DepMap module
(``nsclc_depmap.py``, ``crc_depmap.py``, ...). Each module exposes a
``PROFILE: CancerProfile`` describing its base-frame sampler,
paradigm-association catalogs, prognostic contribution, and prevalence
defaults. The registry below maps each ``CancerType`` enum value to its
profile so the generator can dispatch by cancer type and dataset modality.
"""

from __future__ import annotations

from enum import StrEnum

from .base import CancerProfile

# Profile modules are imported lazily inside ``_load_registry`` to avoid a
# circular import: each profile module imports from this package's siblings,
# and importing them at module top-level would force ``cancer_types/`` to be
# fully constructed before any sub-module's body finishes executing.


class CancerType(StrEnum):
    nsclc_clinical = "nsclc_clinical"
    crc_clinical = "crc_clinical"
    breast_clinical = "breast_clinical"
    prostate_clinical = "prostate_clinical"
    aml_clinical = "aml_clinical"
    nsclc_depmap = "nsclc_depmap"
    crc_depmap = "crc_depmap"
    breast_depmap = "breast_depmap"
    prostate_depmap = "prostate_depmap"
    aml_depmap = "aml_depmap"

    @classmethod
    def _missing_(cls, value: object) -> CancerType | None:
        if not isinstance(value, str):
            return None
        legacy_aliases = {
            "nsclc": cls.nsclc_clinical,
            "crc": cls.crc_clinical,
            "breast": cls.breast_clinical,
            "prostate": cls.prostate_clinical,
            "aml": cls.aml_clinical,
            # The old single DepMap profile's default buried finding was
            # colorectal, so this keeps older smoke configs usable.
            "depmap": cls.crc_depmap,
        }
        return legacy_aliases.get(value.strip().lower())


_REGISTRY: dict[CancerType, CancerProfile] | None = None


def _load_registry() -> dict[CancerType, CancerProfile]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    from . import (
        aml_clinical,
        aml_depmap,
        breast_clinical,
        breast_depmap,
        crc_clinical,
        crc_depmap,
        nsclc_clinical,
        nsclc_depmap,
        prostate_clinical,
        prostate_depmap,
    )

    _REGISTRY = {
        CancerType.nsclc_clinical: nsclc_clinical.PROFILE,
        CancerType.crc_clinical: crc_clinical.PROFILE,
        CancerType.breast_clinical: breast_clinical.PROFILE,
        CancerType.prostate_clinical: prostate_clinical.PROFILE,
        CancerType.aml_clinical: aml_clinical.PROFILE,
        CancerType.nsclc_depmap: nsclc_depmap.PROFILE,
        CancerType.crc_depmap: crc_depmap.PROFILE,
        CancerType.breast_depmap: breast_depmap.PROFILE,
        CancerType.prostate_depmap: prostate_depmap.PROFILE,
        CancerType.aml_depmap: aml_depmap.PROFILE,
    }
    return _REGISTRY


def get_profile(cancer_type: CancerType | str) -> CancerProfile:
    """Look up a registered ``CancerProfile`` by cancer-type enum or string."""
    return _load_registry()[CancerType(cancer_type)]


def all_cancer_types() -> list[CancerType]:
    """Return every registered cancer type, in declaration order."""
    return list(CancerType)


__all__ = [
    "CancerProfile",
    "CancerType",
    "all_cancer_types",
    "get_profile",
]
