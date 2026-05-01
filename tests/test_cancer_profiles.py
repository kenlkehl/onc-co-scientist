"""Per-cancer-type smoke + invariant tests.

For each registered ``CancerProfile``, generate a small bundle and check the
shared properties any profile must satisfy:

- the bundle generates without error,
- the manifest's ``cancer_type`` matches the requested type,
- the buried-signature predicate columns all exist in the generated frame,
- the disjointness invariant holds: the union of shared and profile-specific
  background-prognostic variables shares no column with the variables /
  subgroup predicates of the profile's catalogs.
"""

from __future__ import annotations

import numpy as np
import pytest

from onc_co_scientist.synthetic.cancer_types import (
    CancerType,
    all_cancer_types,
    get_profile,
)
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.injector import background_prognostic_variables


@pytest.fixture(params=all_cancer_types(), ids=lambda ct: ct.value)
def cancer_type(request: pytest.FixtureRequest) -> CancerType:
    return request.param


def test_profile_generates_a_bundle(cancer_type: CancerType) -> None:
    cfg = GeneratorConfig(
        dataset_id=f"smoke_{cancer_type.value}",
        cancer_type=cancer_type.value,
        patient_n=300,
        seed=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    assert len(bundle.frame) == 300
    assert bundle.manifest.patient_n == 300
    assert bundle.manifest.cancer_type == cancer_type.value
    assert set(bundle.manifest.outcome_columns) <= set(bundle.frame.columns)
    # The buried-signature predicate columns must exist in the frame.
    assert len(bundle.manifest.associations) == 1
    spec = bundle.manifest.associations[0]
    assert spec.subgroup is not None
    for col in spec.subgroup.predicate:
        assert col in bundle.frame.columns, (
            f"Buried predicate column {col!r} missing from {cancer_type.value} frame"
        )


def test_profile_disjointness_invariant(cancer_type: CancerType) -> None:
    """Background-prognostic vars must not overlap any catalog's variables.

    The unscored prognostic layer must be disjoint from every paradigm-tagged
    variable (concordant, discordant, hidden_novel single-predicate, and
    multi-feature buried) so the buried effect is cleanly attributable rather
    than entangled with the prognostic layer.
    """
    profile = get_profile(cancer_type)
    bg = background_prognostic_variables(profile)
    paradigm_vars: set[str] = set()
    for catalog_fn in (
        profile.concordant_catalog,
        profile.discordant_catalog,
        profile.hidden_novel_catalog,
        profile.buried_signature_catalog,
    ):
        for spec in catalog_fn():
            paradigm_vars.update(spec.variables)
            if spec.subgroup is not None:
                paradigm_vars.update(spec.subgroup.predicate.keys())
    overlap = bg & paradigm_vars
    assert overlap == set(), (
        f"{cancer_type.value} background-prognostic variables overlap "
        f"paradigm variables: {overlap!r}. Move overlapping variables out "
        "of one or the other."
    )


def test_profile_buried_signal_recoverable(cancer_type: CancerType) -> None:
    """Power sanity check: the buried signature must shift its driver outcome
    in the configured direction at the modest cohort size of 5,000.
    """
    cfg = GeneratorConfig(
        dataset_id=f"power_{cancer_type.value}",
        cancer_type=cancer_type.value,
        patient_n=5_000,
        seed=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    spec = bundle.manifest.associations[0]
    df = bundle.frame
    predicate_cols = set(spec.subgroup.predicate)
    drivers = [
        v for v in spec.variables if v != spec.outcome and v not in predicate_cols
    ]
    driver = drivers[0]
    mask = np.ones(len(df), dtype=bool)
    for col, val in spec.subgroup.predicate.items():
        col_vals = df[col].to_numpy()
        if isinstance(val, dict) and ({"min", "max"} & val.keys()):
            low = val.get("min", -np.inf)
            high = val.get("max", np.inf)
            mask &= (col_vals >= low) & (col_vals <= high)
        else:
            mask &= col_vals == val
    treated = df[driver].to_numpy() == 1
    in_grp = treated & mask
    out_grp = treated & ~mask
    if in_grp.sum() < 30 or out_grp.sum() < 30:
        pytest.skip(
            f"Subgroup too small for power check at n=5000: "
            f"in={int(in_grp.sum())}, out={int(out_grp.sum())}"
        )
    in_mean = float(df.loc[in_grp, spec.outcome].mean())
    out_mean = float(df.loc[out_grp, spec.outcome].mean())
    diff = in_mean - out_mean
    assert np.sign(diff) == np.sign(spec.effect_size), (
        f"{cancer_type.value} buried signature shifts outcome with the wrong "
        f"sign: in_mean={in_mean:.3f}, out_mean={out_mean:.3f}, "
        f"expected sign(effect)={np.sign(spec.effect_size)}"
    )


def test_profile_full_paradigm_mix_smoke(cancer_type: CancerType) -> None:
    """Each profile must support requesting one entry from every catalog."""
    cfg = GeneratorConfig(
        dataset_id=f"mix_{cancer_type.value}",
        cancer_type=cancer_type.value,
        patient_n=200,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    assert len(bundle.manifest.associations) == 4
    # Outcomes referenced by associations should be materialized.
    referenced = {a.outcome for a in bundle.manifest.associations}
    for outcome in referenced:
        assert outcome in bundle.frame.columns
