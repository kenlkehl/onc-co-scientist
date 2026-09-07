"""Regression check for progress read from the actual nested stage-event format."""

import importlib
import json
from pathlib import Path


def test_progress_reads_nested_records_once_and_waits_for_complete_lines(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "scripts/expected_surprising")
    )
    runner = importlib.import_module("run_replicates")
    path = tmp_path / "transcript.jsonl"
    stage = json.dumps({"kind": "stage", "record": {"iteration": 7, "stage": "synthesize"}})
    path.write_text(stage)
    assert runner.progress(path)["successful_stages"] == 0
    with path.open("a") as handle:
        handle.write(
            "\n" + json.dumps({"kind": "attempt_error", "error": "malformed record"}) + "\n"
        )
    state = runner.progress(path)
    assert state == {
        "successful_stages": 1,
        "attempt_errors": 1,
        "last_iteration": 7,
        "last_stage": "synthesize",
    }
    assert runner.progress(path) == state
