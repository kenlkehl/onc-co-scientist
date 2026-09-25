import copy

import numpy as np
import pandas as pd
import pytest

from onc_co_scientist.expected_surprising.ablation import collapse_outcomes
from onc_co_scientist.expected_surprising.generation import conditional_means, sample
from onc_co_scientist.expected_surprising.masking import MASKING_VERSION, SemanticMask
from onc_co_scientist.expected_surprising.review_policy import require_current_pair, spec_digest
from onc_co_scientist.expected_surprising.schemas import PairSpec
from onc_co_scientist.synthetic.anonymize import build_column_mapping, extend_outcome_mapping
from tests.test_expected_surprising import pair


def test_projection_preserves_original_evidence_covariates_and_six_signal_components():
    source = pair("nsclc_depmap", "subgroup")
    before = source.model_dump()
    derived = collapse_outcomes(source)
    assert source.model_dump() == before
    assert derived.evidence == source.evidence
    assert derived.outcome_projection.source_spec_sha256 == spec_digest(source)
    assert derived.dgp_review is None
    with pytest.raises(ValueError, match="full-DGP"):
        require_current_pair(derived)
    for version in ("expected", "surprising"):
        original_frame = sample(source, version, seed=source.seed)
        actual = sample(derived, version, seed=source.seed)
        pd.testing.assert_frame_equal(
            original_frame.drop(columns=[o.name for o in source.outcomes]),
            actual.drop(columns=[derived.outcomes[0].name]),
        )
        means = conditional_means(source, actual, version)
        expected = source.outcomes[0].intercept + sum(
            means[name] - source.outcomes[0].intercept
            for name in derived.outcome_projection.source_outcomes.values()
        )
        np.testing.assert_allclose(
            conditional_means(derived, actual, version)[derived.outcomes[0].name], expected
        )
        for prior, after in zip(source.discoveries, derived.discoveries, strict=True):
            restored = after.model_dump()
            restored["hypothesis"]["outcome"] = prior.hypothesis.outcome
            assert restored == prior.model_dump()


def test_projection_rejects_incomplete_or_changed_source_endpoint_maps():
    derived = collapse_outcomes(pair("nsclc_depmap"))
    data = derived.model_dump()
    changed = copy.deepcopy(data)
    changed["outcome_projection"]["source_outcomes"].pop(derived.focal_id)
    with pytest.raises(ValueError, match="all six"):
        PairSpec.model_validate(changed)
    changed = copy.deepcopy(data)
    changed["outcome_projection"]["source_outcomes"][derived.focal_id] = "unreviewed_endpoint"
    with pytest.raises(ValueError, match="reviewed source endpoint"):
        PairSpec.model_validate(changed)


def test_projected_pair_survives_masked_schema_validation():
    derived = collapse_outcomes(pair("nsclc_depmap"))
    frame = sample(derived, "expected", seed=123)
    names = [o.name for o in derived.outcomes]
    columns = extend_outcome_mapping(build_column_mapping(list(frame), names), names)
    masked = SemanticMask(dict(version=MASKING_VERSION, columns=columns, values={})).spec(derived)
    restored = PairSpec.model_validate(masked.model_dump())
    assert restored.outcome_projection.outcome == restored.outcomes[0].name
