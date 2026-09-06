"""Checks for the local CLI boundary; no real model requests."""

import json
import sys
from pathlib import Path

import pytest

from experiments.aim1_recovery.local_cli import (
    MODEL,
    cli_command,
    launch,
    prepare_local,
    session_id,
    validate_job,
)
from experiments.aim1_recovery.prepare import prepare
from experiments.aim1_recovery.score import failed_run_score
from onc_co_scientist.harness.structured_runner import StructuredRunner, finalize_workspace

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan(tmp_path):
    out = tmp_path / "setup"
    prepare_local(REPO, out, Path(sys.executable), 1)
    return json.loads((out / "plan.json").read_text())


def test_only_nsclc_and_paired_split(plan):
    assert len(plan["jobs"]) == 2
    assert {j["variant"] for j in plan["jobs"]} == {"named", "anonymized"}
    assert {s["task"] for s in plan["sources"]} == {"nsclc"}
    source = plan["sources"][0]
    assert (source["source_n"], source["discovery_n"], source["evaluation_n"]) == (
        50000, 40000, 10000,
    )
    for job in plan["jobs"]:
        validate_job(plan, job)
        assert not (Path(job["workspace"]) / "iterations").exists()


def test_reject_unknown_task_before_writing(tmp_path):
    out = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Invalid task"):
        prepare(REPO, out, Path(sys.executable), tasks=("misspelled",))
    assert not out.exists()


def test_reject_changed_input(plan):
    job = plan["jobs"][0]
    path = Path(job["workspace"]) / "agent_instructions.md"
    path.write_text(path.read_text() + "\nchanged")
    with pytest.raises(ValueError, match="Frozen input changed"):
        validate_job(plan, job)


def test_persistent_commands_do_not_bypass_rules():
    job = {"workspace": "/tmp/a task"}
    fresh = cli_command("codex", job)
    resumed = cli_command("codex", job, "thread-123")
    for command in (fresh, resumed):
        assert MODEL in command
        assert 'model_reasoning_effort="medium"' in command
        assert "workspace-write" in command
        assert "--ephemeral" not in command
        assert "--ignore-rules" not in command
        assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert "resume" not in fresh
    assert resumed[resumed.index("resume") + 2] == "thread-123"


def test_failed_cli_attempt_resumes_same_thread_and_preserves_log(plan, tmp_path):
    fake = tmp_path / "fake-codex"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        "if '--version' in sys.argv:\n"
        " print('fake-codex test-only'); sys.exit(0)\n"
        "print(json.dumps({'type':'thread.started','thread_id':'test-thread'}))\n"
        "print(json.dumps({'type':'turn.failed','error':{'message':'fixture failure'}}))\n"
        "sys.exit(1)\n"
    )
    fake.chmod(0o700)
    job = plan["jobs"][0]
    logdir = Path(job["workspace"]) / "cli_logs"
    result = launch(plan, job, str(fake), False, 10)
    assert result["status"] == "incomplete"
    original = (logdir / "attempt_001.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Existing attempt"):
        launch(plan, job, str(fake), False, 10)
    resumed = launch(plan, job, str(fake), True, 10)
    assert resumed["resume_thread"] == "test-thread"
    assert resumed["status"] == "incomplete"
    assert (logdir / "attempt_001.jsonl").read_bytes() == original
    assert not (logdir / "running.lock").exists()


def test_conflicting_thread_ids_rejected(tmp_path):
    for i, thread in enumerate(("first", "second")):
        (tmp_path / f"attempt_{i:03d}.jsonl").write_text(
            json.dumps({"type": "thread.started", "thread_id": thread}) + "\n"
        )
    with pytest.raises(ValueError, match="Multiple CLI thread IDs"):
        session_id(tmp_path)


def test_terminal_cli_failure_retains_harness_and_denominator(plan):
    job = plan["jobs"][0]
    failure = {"job_id": job["job_id"], "reason": "irretrievable test interruption",
               "retain_in_denominator": True, "recovery_credit": False}
    result = failed_run_score(job, plan["protocol"], failure)
    assert result["harness_id"] == "codex-cli-structured-v2"
    assert result["primary_recovered"] is False


def test_loose_prompt_preserves_data_and_allows_early_finalization(plan, tmp_path):
    from experiments.aim1_recovery.loose_prompt import HARNESS, STYLE

    out = tmp_path / "loose"
    prepare_local(REPO, out, Path(sys.executable), 1, STYLE)
    loose = json.loads((out / "plan.json").read_text())
    assert loose["sources"] == plan["sources"]
    assert loose["protocol"]["scorer_version"] == plan["protocol"]["scorer_version"]
    for old, job in zip(plan["jobs"], loose["jobs"], strict=True):
        assert old["variant"] == job["variant"]
        assert old["data_sha256"] == job["data_sha256"]
        for name in ("dataset_description.md", "transcript_schema.json"):
            assert old["public_input_sha256"][name] == job["public_input_sha256"][name]
        validate_job(loose, job)
        ws = Path(job["workspace"])
        brief = (ws / "agent_instructions.md").read_text()
        archived = (
            REPO / "example_data_clinical_all_claude/ds001/tasks/nsclc"
            / job["variant"] / "agent_instructions.md"
        ).read_text().replace("**Patients:** 50000", "**Patients:** 40000")
        assert brief.startswith(archived)
        assert "research_step" not in brief
        assert "Fixed exploration budget" not in brief
        assert "finish the remaining budget" not in cli_command("codex", job, "thread")[-1]
        example = json.loads((ws / "transcript_example.json").read_text())
        record = example["iterations"][0]
        runner = StructuredRunner(ws, base_url="http://unused", model=MODEL)
        submitted = []
        result = runner._submit(record, submitted, 25)
        assert len(submitted) == 1, result
        transcript = finalize_workspace(ws)
        assert len(transcript.iterations) == 1
        assert transcript.harness_id == HARNESS
        assert transcript.max_iterations == 25
        # Early stopping does not disable ordered, immutable record receipts.
        path = ws / "iterations/001.json"
        path.write_text(path.read_text() + "\n")
        with pytest.raises(ValueError, match="integrity"):
            finalize_workspace(ws)
