import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from onc_co_scientist.harness.transcript import Transcript

EXAMPLE = (
    Path(__file__).parent.parent / "src/onc_co_scientist/harness/templates/transcript_example.json"
)


def test_example_transcript_validates():
    text = EXAMPLE.read_text()
    transcript = Transcript.model_validate_json(text)
    assert transcript.dataset_id == "ds001"
    assert len(transcript.iterations) == 2
    # Round-trip JSON must match after normalization.
    round_tripped = json.loads(transcript.model_dump_json())
    original = json.loads(text)
    assert round_tripped["iterations"][0]["index"] == original["iterations"][0]["index"]


def test_transcript_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        Transcript.model_validate(
            {
                "dataset_id": "ds001",
                # model_id intentionally missing
                "harness_id": "test",
                "max_iterations": 3,
                "iterations": [],
            }
        )


def test_transcript_rejects_zero_iteration_index():
    with pytest.raises(ValidationError):
        Transcript.model_validate(
            {
                "dataset_id": "ds001",
                "model_id": "m",
                "harness_id": "h",
                "max_iterations": 1,
                "iterations": [{"index": 0, "proposed_hypotheses": []}],
            }
        )


def test_transcript_allows_extra_fields():
    transcript = Transcript.model_validate(
        {
            "dataset_id": "ds001",
            "model_id": "m",
            "harness_id": "h",
            "max_iterations": 1,
            "iterations": [
                {
                    "index": 1,
                    "proposed_hypotheses": [
                        {"id": "h1", "text": "x", "kind": "novel", "harness_trace": 42}
                    ],
                    "analyses": [],
                }
            ],
            "wall_clock_seconds": 12.3,
        }
    )
    assert transcript.iterations[0].proposed_hypotheses[0].id == "h1"
