import numpy as np
import pytest

from onc_co_scientist.synthetic.distractors import (
    DEFAULT_DISTRACTOR_POOL,
    sample_distractors,
)
from onc_co_scientist.synthetic.paradigms import DEFAULT_POOL


def _all_paradigm_column_names() -> set[str]:
    names: set[str] = set()
    for specs in DEFAULT_POOL.values():
        for spec in specs:
            names.update(spec.variables)
            if spec.subgroup is not None:
                names.update(spec.subgroup.predicate.keys())
    return names


def test_pool_has_at_least_one_hundred_entries():
    assert len(DEFAULT_DISTRACTOR_POOL) >= 100


def test_pool_names_are_unique():
    names = [spec.name for spec in DEFAULT_DISTRACTOR_POOL]
    assert len(names) == len(set(names))


def test_pool_disjoint_from_paradigm_and_base_columns():
    # Columns used by the paradigm catalogs (treatments, biomarkers, outcomes).
    paradigm_cols = _all_paradigm_column_names()
    # Columns always emitted by the builtin base frame.
    base_cols = {
        "patient_id",
        "age_years",
        "sex_female",
        "smoking_status",
        "ecog_ps",
        "histology",
        "stage_iv",
        "has_brain_mets",
        "egfr_mutation",
        "kras_g12c",
        "alk_fusion",
        "stk11_mutation",
        "brca2_mutation",
        "pdl1_tps",
        "tmb_high",
        "treatment_pembrolizumab",
        "treatment_sotorasib",
        "treatment_olaparib",
        "treatment_osimertinib",
    }
    distractor_names = {spec.name for spec in DEFAULT_DISTRACTOR_POOL}
    assert distractor_names.isdisjoint(paradigm_cols)
    assert distractor_names.isdisjoint(base_cols)


def test_sample_distractors_produces_requested_shape():
    rng = np.random.default_rng(0)
    cols = sample_distractors(rng, n_patients=200, n=50)
    assert len(cols) == 50
    for arr in cols.values():
        assert len(arr) == 200


def test_sample_distractors_is_deterministic():
    a = sample_distractors(np.random.default_rng(42), n_patients=100, n=30)
    b = sample_distractors(np.random.default_rng(42), n_patients=100, n=30)
    assert list(a.keys()) == list(b.keys())
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_sample_distractors_rejects_overflow():
    with pytest.raises(ValueError, match="pool only has"):
        sample_distractors(
            np.random.default_rng(0),
            n_patients=10,
            n=len(DEFAULT_DISTRACTOR_POOL) + 1,
        )


def test_sample_distractors_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        sample_distractors(np.random.default_rng(0), n_patients=10, n=-1)


def test_sample_distractors_takes_pool_prefix():
    cols = sample_distractors(np.random.default_rng(0), n_patients=10, n=5)
    expected = [spec.name for spec in DEFAULT_DISTRACTOR_POOL[:5]]
    assert list(cols.keys()) == expected
