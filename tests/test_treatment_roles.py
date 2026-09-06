import pytest

from onc_co_scientist.harness.treatment_roles import (
    render_treatment_roles,
    visible_treatment_columns,
)


def test_missing_mask_mapping_fails_instead_of_disclosing_named_fallback():
    with pytest.raises(KeyError):
        visible_treatment_columns(["treatment_example"], ["feature_123"], {})


def test_role_must_exist_in_public_schema():
    with pytest.raises(ValueError, match="absent from public dataset"):
        visible_treatment_columns(["treatment_example"], ["feature_123"])


def test_mapping_cannot_collapse_multiple_treatments():
    with pytest.raises(ValueError, match="unique"):
        visible_treatment_columns(
            ["treatment_a", "treatment_b"], ["feature_123"],
            {"treatment_a": "feature_123", "treatment_b": "feature_123"},
        )


def test_no_treatments_means_no_treatment_instructions():
    assert render_treatment_roles([]) == ""
