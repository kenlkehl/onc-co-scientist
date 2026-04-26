"""Paradigm-class association selection across cancer-type profiles.

The four catalogs (concordant, discordant, hidden_novel, buried) are owned by
each ``CancerProfile`` in ``cancer_types/``. This module is the cancer-
agnostic selector: given a profile and per-class counts, it returns the first
N entries from each catalog. The deterministic seed-path used by the initial
pipeline (and tests) takes the first N entries from each catalog, so ordering
inside each profile's catalog functions is part of the public contract.

Backward-compatible re-exports point at the NSCLC profile so legacy callers
that imported ``DEFAULT_POOL``, ``concordant_catalog``, etc. from this module
keep working unchanged. The legacy ``select_associations`` signature (no
profile argument) defaults to the NSCLC profile for the same reason.
"""

from __future__ import annotations

from .cancer_types import CancerProfile, get_profile
from .cancer_types.nsclc import (
    buried_signature_catalog as _nsclc_buried_signature_catalog,
)
from .cancer_types.nsclc import (
    concordant_catalog as _nsclc_concordant_catalog,
)
from .cancer_types.nsclc import (
    discordant_catalog as _nsclc_discordant_catalog,
)
from .cancer_types.nsclc import (
    hidden_novel_catalog as _nsclc_hidden_novel_catalog,
)
from .schemas import AssociationSpec, ParadigmClass

# Backward-compatible re-exports (NSCLC). External callers that imported these
# names directly from ``paradigms`` keep working under the new dispatch.
concordant_catalog = _nsclc_concordant_catalog
discordant_catalog = _nsclc_discordant_catalog
hidden_novel_catalog = _nsclc_hidden_novel_catalog
buried_signature_catalog = _nsclc_buried_signature_catalog


# Sentinel pool key for the buried multi-feature catalog. Distinct from
# ``ParadigmClass`` keys so callers can request buried signatures
# independently from the legacy paradigm-mix counters even though buried
# specs are tagged ``hidden_novel`` for scoring continuity.
BURIED_POOL_KEY: str = "buried"


def _profile_pool(
    profile: CancerProfile,
) -> dict[ParadigmClass, list[AssociationSpec]]:
    return {
        ParadigmClass.concordant: profile.concordant_catalog(),
        ParadigmClass.discordant: profile.discordant_catalog(),
        ParadigmClass.hidden_novel: profile.hidden_novel_catalog(),
    }


# Backward-compatible NSCLC pool. Constructed once at import time.
DEFAULT_POOL: dict[ParadigmClass, list[AssociationSpec]] = _profile_pool(
    get_profile("nsclc")
)


def select_associations(
    n_concordant: int,
    n_discordant: int,
    n_hidden_novel: int,
    n_buried_signatures: int = 0,
    pool: dict[ParadigmClass, list[AssociationSpec]] | None = None,
    buried_pool: list[AssociationSpec] | None = None,
    profile: CancerProfile | None = None,
) -> list[AssociationSpec]:
    """Pick the first N from each catalog of the given profile.

    Resolution order for the catalog source:

    1. If ``pool`` and ``buried_pool`` are both supplied, they are used as-is
       (legacy direct-injection path used by tests and the LLM-driven
       expansion).
    2. Otherwise, if ``profile`` is supplied, its catalogs are used.
    3. Otherwise, the NSCLC profile is used (backward compatible default).

    ``n_buried_signatures`` selects from the multi-feature buried catalog;
    these are tagged ``hidden_novel`` for scoring purposes but are counted
    independently from ``n_hidden_novel`` so the legacy single-predicate
    catalog and the multi-feature buried catalog can be configured
    separately.
    """
    if pool is None or buried_pool is None:
        active_profile = profile if profile is not None else get_profile("nsclc")
        if pool is None:
            pool = _profile_pool(active_profile)
        if buried_pool is None:
            buried_pool = active_profile.buried_signature_catalog()

    wants = {
        ParadigmClass.concordant: n_concordant,
        ParadigmClass.discordant: n_discordant,
        ParadigmClass.hidden_novel: n_hidden_novel,
    }
    chosen: list[AssociationSpec] = []
    for klass, n in wants.items():
        available = pool[klass]
        if n > len(available):
            raise ValueError(
                f"Requested {n} {klass.value} associations but pool only has "
                f"{len(available)}."
            )
        chosen.extend(available[:n])
    if n_buried_signatures < 0:
        raise ValueError(
            f"n_buried_signatures must be >= 0, got {n_buried_signatures}."
        )
    if n_buried_signatures > len(buried_pool):
        raise ValueError(
            f"Requested {n_buried_signatures} buried-signature associations "
            f"but pool only has {len(buried_pool)}."
        )
    chosen.extend(buried_pool[:n_buried_signatures])
    return chosen
