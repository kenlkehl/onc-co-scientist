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


def test_terminal_failure_retains_denominator_without_reading_corrupt_claims(tmp_path, monkeypatch):
    import hashlib

    from experiments.aim1_recovery import score

    workspace = tmp_path / "job"
    workspace.mkdir()
    for name in ("dataset.parquet", "agent_instructions.md"):
        (workspace / name).write_text("immutable input")
    # A failed run may have an invalid assembled transcript. It must never be scored.
    (workspace / "transcript.json").write_text("{corrupt transcript")
    digest = hashlib.sha256(b"immutable input").hexdigest()
    job = dict(
        job_id="job_0001",
        workspace=str(workspace),
        dataset_id="d",
        family="clinical",
        task="nsclc",
        variant="named",
        replicate=1,
        max_iterations=25,
        data_sha256=digest,
        instructions_sha256=digest,
    )
    protocol = dict(scorer_version=score.SCORER_VERSION, model_id="m", backend="work")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(protocol=protocol, sources=[], jobs=[job])))
    failure = dict(
        job_id="job_0001",
        reason="receipt integrity failure",
        retain_in_denominator=True,
        recovery_credit=False,
    )
    ledger = tmp_path / "terminal_failures.json"
    ledger.write_text(json.dumps({"job_0001": failure}))

    def no_scoring(*args, **kwargs):
        pytest.fail("Failed transcript must not be parsed or evaluated")

    monkeypatch.setattr(score, "finalize_workspace", no_scoring)
    monkeypatch.setattr(score, "score_transcript", no_scoring)
    monkeypatch.setattr(score, "plot", lambda *args, **kwargs: None)
    summary = score.generate(plan, tmp_path / "out")
    assert summary["complete"]
    assert summary["n_completed"] == 0
    assert summary["n_terminal"] == 1
    group = next(
        g
        for g in summary["groups"]
        if g["task"] == "all" and g["family"] == "clinical" and g["variant"] == "named"
    )
    assert group["n"] == group["intended_n"] == 1
    assert group["primary_n"] == 0
    assert group["primary_rate"] == 0
    assert (
        tmp_path / "out/transcripts/job_0001/transcript.json"
    ).read_text() == "{corrupt transcript"
    ledger.write_text(json.dumps({"unknown_job": failure}))
    with pytest.raises(ValueError, match="unknown jobs"):
        score.generate(plan, tmp_path / "out2")


def test_checkpoint_guard_preserves_existing_archive_while_workers_are_active(tmp_path):
    from experiments.aim1_recovery.archive import pack

    root = tmp_path / "experiment"
    root.mkdir()
    (root / "coordinator_state.json").write_text(
        json.dumps({"jobs": {"job_0001": {"status": "running"}}})
    )
    archive = tmp_path / "checkpoint.tar.gz"
    archive.write_bytes(b"previous verified checkpoint")
    with pytest.raises(ValueError, match="workers are active"):
        pack(root, archive, require_idle=True)
    assert archive.read_bytes() == b"previous verified checkpoint"
