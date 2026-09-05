import copy
import json

import pytest

from experiments.aim1_recovery.score import same_transcript_records
from onc_co_scientist.harness.transcript import Transcript


@pytest.mark.parametrize("changed_value", [None, 0.0, 0.01, float("inf")])
def test_failed_analysis_nan_preserves_integrity_without_hiding_changes(changed_value):
    payload = {
        "dataset_id": "d",
        "model_id": "m",
        "harness_id": "h",
        "max_iterations": 1,
        "iterations": [
            {
                "index": 1,
                "proposed_hypotheses": [],
                "analyses": [
                    {
                        "hypothesis_ids": [],
                        "result_summary": "Failed test",
                        "p_value": float("nan"),
                        "effect_estimate": float("nan"),
                    }
                ],
            }
        ],
    }
    saved = Transcript.model_validate_json(json.dumps(payload))
    assembled = Transcript.model_validate_json(json.dumps(payload))
    assert saved.model_dump() != assembled.model_dump()
    assert same_transcript_records(saved, assembled)
    altered = copy.deepcopy(payload)
    altered["iterations"][0]["analyses"][0]["p_value"] = changed_value
    assert not same_transcript_records(saved, Transcript.model_validate(altered))
