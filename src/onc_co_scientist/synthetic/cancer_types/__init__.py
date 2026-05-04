"""Per-cancer-type synthetic-generation profiles.

Each cancer type lives in its own module (``nsclc.py``, ``crc.py``, ...) and
exposes a ``PROFILE: CancerProfile`` describing its base-frame sampler,
paradigm-association catalogs, prognostic contribution, and prevalence
defaults. The ``REGISTRY`` below maps each ``CancerType`` enum value to its
profile so the generator can dispatch by cancer type.
"""

from __future__ import annotations

from enum import StrEnum

from .base import CancerProfile

# Profile modules are imported lazily inside ``_load_registry`` to avoid a
# circular import: each profile module imports from this package's siblings,
# and importing them at module top-level would force ``cancer_types/`` to be
# fully constructed before any sub-module's body finishes executing.


class CancerType(StrEnum):
    nsclc = "nsclc"
    crc = "crc"
    breast = "breast"
    prostate = "prostate"
    aml = "aml"
    depmap = "depmap"


_REGISTRY: dict[CancerType, CancerProfile] | None = None


def _load_registry() -> dict[CancerType, CancerProfile]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    from . import aml, breast, crc, depmap, nsclc, prostate

    _REGISTRY = {
        CancerType.nsclc: nsclc.PROFILE,
        CancerType.crc: crc.PROFILE,
        CancerType.breast: breast.PROFILE,
        CancerType.prostate: prostate.PROFILE,
        CancerType.aml: aml.PROFILE,
        CancerType.depmap: depmap.PROFILE,
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
