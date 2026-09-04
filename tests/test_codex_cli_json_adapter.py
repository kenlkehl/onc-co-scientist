from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from onc_co_scientist.harness.experiment import (
    ModelSpec,
    ResourceBudget,
    load_experiment_spec,
)
from onc_co_scientist.harness.runtime import AgentArtifact, AgentRequest, CliJsonRuntime
from scripts.codex_cli_json_adapter import (
    ARTIFACT_SCHEMA,
    _analysis_guard_root,
    _artifact_from_text,
    _codex_runtime_root,
    _resource_retry_reason,
    _validate_artifact_contract,
    artifact_schema,
    extract_codex_thread_id,
    parse_codex_events,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts" / "codex_cli_json_adapter.py"


def test_nsclc_adapter_timeout_precedes_harness_deadline() -> None:
    experiment_dir = ROOT / "experiments" / "nsclc_coordination"

    for config_name in ("nsclc_smoke.yaml", "nsclc_pilot.yaml"):
        spec = load_experiment_spec(experiment_dir / config_name)
        extra_args = spec.models[0].extra_args
        timeout_index = extra_args.index("--timeout-seconds")
        adapter_timeout = int(extra_args[timeout_index + 1])
        harness_timeout = spec.budget.max_runtime_seconds_per_call

        assert adapter_timeout == 1180
        assert harness_timeout == 1200
        assert adapter_timeout < harness_timeout


def test_artifact_schema_uses_closed_subgroup_predicate_objects() -> None:
    claim = ARTIFACT_SCHEMA["properties"]["claims"]["items"]
    subgroup = claim["properties"]["subgroup"]
    predicate = subgroup["items"]

    assert subgroup["type"] == "array"
    assert predicate["type"] == "object"
    assert predicate["additionalProperties"] is False
    assert predicate["required"] == ["variable", "operator", "value"]
    assert predicate["properties"]["operator"]["enum"] == [
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "in",
        "not_in",
    ]


def test_artifact_schema_locks_final_answer_to_the_stage_contract() -> None:
    intermediate = artifact_schema(require_final_answer=False)
    synthesis = artifact_schema(require_final_answer=True)

    assert intermediate["properties"]["final_answer"] == {"type": "null"}
    answer = synthesis["properties"]["final_answer"]
    assert answer["type"] == "object"
    assert answer["additionalProperties"] is False
    assert answer["properties"]["conclusion"]["minLength"] == 1
    assert "uniqueItems" not in answer["properties"]["supported_claim_indices"]
    assert "uniqueItems" not in json.dumps(synthesis)
    assert answer["required"] == ["conclusion", "supported_claim_indices"]


def test_post_generation_contract_rejects_duplicate_supported_claim_indices() -> None:
    artifact = AgentArtifact(
        summary="synthesis",
        handoff="handoff",
        final_answer={
            "conclusion": "conclusion",
            "supported_claim_indices": [0, 0],
        },
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        _validate_artifact_contract(artifact, require_final_answer=True)


def test_artifact_parser_rejects_fields_outside_closed_stage_schema() -> None:
    payload = {
        "summary": "result",
        "handoff": "handoff",
        "hypotheses": [],
        "analyses": [],
        "claims": [],
        "evidence": [],
        "concerns": [],
        "minority_report": "",
        "final_answer": None,
        "unexpected": "must fail",
    }

    with pytest.raises(ValueError, match="violated the stage schema"):
        _artifact_from_text(
            json.dumps(payload),
            schema=artifact_schema(require_final_answer=False),
        )


def _write_fake_codex(path: Path) -> None:
    path.write_text(
        r"""#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys

argv = sys.argv[1:]
is_resume = argv[:2] == ["exec", "resume"]
last_message = pathlib.Path(argv[argv.index("--output-last-message") + 1])
schema_path = pathlib.Path(argv[argv.index("--output-schema") + 1])
if not schema_path.is_file():
    raise SystemExit("schema missing")
schema = json.loads(schema_path.read_text())

prompt = sys.stdin.read()
entry = {
    "argv": argv,
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    "schema_additional_properties": schema["additionalProperties"],
    "final_answer_schema": schema["properties"]["final_answer"],
    "tmpdir": os.environ.get("TMPDIR"),
    "pythonhome": os.environ.get("PYTHONHOME"),
    "pythonpath": os.environ.get("PYTHONPATH"),
}
log_path = pathlib.Path(os.environ["FAKE_CODEX_LOG"])
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(entry) + "\n")

artifact = {
    "summary": "resumed result" if is_resume else "initial result",
    "handoff": "self-contained handoff",
    "hypotheses": ["H1"],
    "analyses": [{"method": "test", "result": "supported"}],
    "claims": [],
    "evidence": ["dataset.parquet"],
    "concerns": [],
    "minority_report": "",
    "final_answer": (
        {"conclusion": "synthesis result", "supported_claim_indices": []}
        if schema["properties"]["final_answer"].get("type") == "object"
        else None
    ),
}
last_message.write_text(json.dumps(artifact), encoding="utf-8")

if not is_resume:
    print(json.dumps({"type": "thread.started", "thread_id": "thread-test-001"}))
item = {"id": "command-1", "type": "command_execution"}
print(json.dumps({"type": "item.started", "item": item}))
print(json.dumps({"type": "item.completed", "item": item}))
print(json.dumps({"type": "item.completed", "item": {"id": "answer-1", "type": "agent_message"}}))
usage = {
    "input_tokens": 80 if is_resume else 120,
    "output_tokens": 20 if is_resume else 30,
}
print(json.dumps({"type": "turn.completed", "usage": usage}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_contract_repair_fake_codex(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

argv = sys.argv[1:]
is_resume = argv[:2] == ["exec", "resume"]
last_message = pathlib.Path(argv[argv.index("--output-last-message") + 1])
schema_path = pathlib.Path(argv[argv.index("--output-schema") + 1])
schema = json.loads(schema_path.read_text())
is_index_repair = schema.get("title") == "Supported claim index correction"
prompt = sys.stdin.read()
with pathlib.Path(os.environ["FAKE_CODEX_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "argv": argv,
        "is_resume": is_resume,
        "is_index_repair": is_index_repair,
        "index_item_schema": (
            schema["properties"]["supported_claim_indices"]["items"]
            if is_index_repair else None
        ),
        "has_exact_error": (
            "supported_claim_indices references absent claim index 2" in prompt
        ),
        "asks_to_avoid_new_analysis": "do not rerun tools" in prompt.lower(),
    }) + "\n")

claim = {
    "exposure": "treatment",
    "outcome": "pfs_months",
    "direction": "positive",
    "subgroup": [],
    "comparator": "unexposed",
    "effect_estimate": 1.0,
    "effect_unit": "months",
    "p_value": 0.01,
    "subgroup_n": 100,
    "exposed_n": 50,
    "comparator_n": 50,
    "supported": False,
    "confidence": 0.8,
    "evidence": ["analysis"],
}
supported_claim = dict(claim)
supported_claim["supported"] = True
if is_index_repair:
    last_message.write_text(
        json.dumps({"supported_claim_indices": [1]}), encoding="utf-8"
    )
else:
    artifact = {
        "summary": "invalid cross-reference",
        "handoff": "handoff",
        "hypotheses": [],
        "analyses": [],
        "claims": [claim, supported_claim],
        "evidence": [],
        "concerns": [],
        "minority_report": "",
        "final_answer": {
            "conclusion": "result",
            "supported_claim_indices": [2],
        },
    }
    last_message.write_text(json.dumps(artifact), encoding="utf-8")
if not is_resume:
    print(json.dumps({"type": "thread.started", "thread_id": "thread-repair-001"}))
print(json.dumps({
    "type": "item.completed",
    "item": {"id": "answer", "type": "agent_message"},
}))
print(json.dumps({
    "type": "turn.completed",
    "usage": {
        "input_tokens": 80 if is_resume else 120,
        "output_tokens": 20 if is_resume else 30,
    },
}))
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_failing_fake_codex(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

with pathlib.Path(os.environ["FAKE_CODEX_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:]}) + "\n")
print("simulated provider failure", file=sys.stderr)
raise SystemExit(7)
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_transient_then_success_fake_codex(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

argv = sys.argv[1:]
is_resume = argv[:2] == ["exec", "resume"]
last_message = pathlib.Path(argv[argv.index("--output-last-message") + 1])
prompt = sys.stdin.read()
log_path = pathlib.Path(os.environ["FAKE_CODEX_LOG"])
prior = log_path.read_text().splitlines() if log_path.exists() else []
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "argv": argv,
        "is_resume": is_resume,
        "is_transport_retry": "transient controller transport failure" in prompt,
    }) + "\n")

if not prior:
    print(json.dumps({"type": "thread.started", "thread_id": "thread-transient-001"}))
    print(
        "unexpected status 404 Not Found from backend-api/codex/responses",
        file=sys.stderr,
    )
    raise SystemExit(7)

artifact = {
    "summary": "recovered",
    "handoff": "handoff",
    "hypotheses": [],
    "analyses": [],
    "claims": [],
    "evidence": [],
    "concerns": [],
    "minority_report": "",
    "final_answer": None,
}
last_message.write_text(json.dumps(artifact), encoding="utf-8")
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 12, "output_tokens": 3},
}))
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_resource_failure_then_success_fake_codex(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

argv = sys.argv[1:]
is_resume = argv[:2] == ["exec", "resume"]
last_message = pathlib.Path(argv[argv.index("--output-last-message") + 1])
prompt = sys.stdin.read()
log_path = pathlib.Path(os.environ["FAKE_CODEX_LOG"])
prior = log_path.read_text().splitlines() if log_path.exists() else []
with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({
        "argv": argv,
        "is_resume": is_resume,
        "is_resource_retry": "controlled analysis-resource failure" in prompt,
        "checks_shapes": "broadcasting operations" in prompt,
    }) + "\n")

if not prior:
    print(json.dumps({"type": "thread.started", "thread_id": "thread-resource-001"}))
    print(
        "OCS_ANALYSIS_RESOURCE_LIMIT: memory ceiling 4096 MiB exceeded",
        file=sys.stderr,
    )
    raise SystemExit(7)

artifact = {
    "summary": "recovered after fixing a broadcast",
    "handoff": "handoff",
    "hypotheses": [],
    "analyses": [],
    "claims": [],
    "evidence": [],
    "concerns": [],
    "minority_report": "",
    "final_answer": None,
}
last_message.write_text(json.dumps(artifact), encoding="utf-8")
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 12, "output_tokens": 3},
}))
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _request(
    tmp_path: Path,
    *,
    index: int,
    prompt: str,
    session_id: str = "shared-persistent-session",
    require_final_answer: bool = False,
    reasoning_effort: str = "low",
) -> AgentRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "dataset.parquet").touch()
    return AgentRequest(
        request_id=f"request-{index}",
        experiment_id="experiment",
        run_id="run",
        task_id="nsclc",
        workflow_id="persistent",
        model_profile="luna-low",
        model_id="gpt-5.6-luna",
        reasoning_effort=reasoning_effort,
        stage_id=f"stage-{index}",
        require_final_answer=require_final_answer,
        role="scientist",
        agent_id=f"agent-{index}",
        session_id=session_id,
        prompt=prompt,
        workspace=workspace,
        scratch_dir=tmp_path / "run" / "scratch" / f"agent-{index}",
        call_dir=tmp_path / "run" / "calls" / f"call_{index:04d}",
    )


def test_codex_adapter_starts_then_resumes_and_accounts_usage(tmp_path: Path, monkeypatch) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))

    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-low",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "low",
                "--timeout-seconds",
                "10",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )
    # Python startup can be slow on the shared execution filesystem; this is
    # only an outer test-process guard, not the experiment's locked timeout.
    budget = ResourceBudget(max_runtime_seconds_per_call=60)

    first_prompt = "FIRST_SECRET_PROMPT"
    first = runtime.run(_request(tmp_path, index=1, prompt=first_prompt), budget)
    second_prompt = "SECOND_SECRET_PROMPT"
    second = runtime.run(
        _request(
            tmp_path,
            index=2,
            prompt=second_prompt,
            require_final_answer=True,
        ),
        budget,
    )

    assert first.artifact.summary == "initial result"
    assert first.usage.input_tokens == 120
    assert first.usage.output_tokens == 30
    assert first.usage.tool_calls == 1
    assert first.runtime_metadata["session_action"] == "started"
    assert second.artifact.summary == "resumed result"
    assert second.usage.input_tokens == 80
    assert second.usage.output_tokens == 20
    assert second.usage.tool_calls == 1
    assert second.runtime_metadata["session_action"] == "resumed"
    assert second.runtime_metadata["codex_thread_id"] == "thread-test-001"
    assert second.artifact.final_answer == {
        "conclusion": "synthesis result",
        "supported_claim_indices": [],
    }

    invocations = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert len(invocations) == 2
    first_argv = invocations[0]["argv"]
    second_argv = invocations[1]["argv"]
    assert first_argv[0] == "exec"
    assert second_argv[:2] == ["exec", "resume"]
    assert "thread-test-001" in second_argv
    assert "--ephemeral" not in first_argv + second_argv
    assert first_argv[first_argv.index("--model") + 1] == "gpt-5.6-luna"
    assert "model_reasoning_effort=low" in first_argv
    assert "--sandbox" not in first_argv + second_argv
    assert 'default_permissions="nsclc_eval"' in first_argv
    assert 'default_permissions="nsclc_eval"' in second_argv
    assert "allow_login_shell=false" in first_argv
    first_profile = next(
        value for value in first_argv if value.startswith("permissions.nsclc_eval=")
    )
    second_profile = next(
        value for value in second_argv if value.startswith("permissions.nsclc_eval=")
    )
    assert '":minimal"="read"' in first_profile
    assert f'{json.dumps(str((tmp_path / "workspace").resolve()))}="write"' in first_profile
    assert (
        f'{json.dumps(str((tmp_path / "run" / "scratch" / "agent-1").resolve()))}="write"'
        in first_profile
    )
    assert (
        f'{json.dumps(str((tmp_path / "run" / "scratch" / "agent-2").resolve()))}="write"'
        in second_profile
    )
    assert "network={enabled=false}" in first_profile
    assert "-C" in first_argv
    assert first_argv[first_argv.index("--add-dir") + 1] == str(
        (tmp_path / "run" / "scratch" / "agent-1").resolve()
    )
    assert "-C" not in second_argv
    assert "--add-dir" not in second_argv
    assert invocations[0]["prompt_sha256"] == hashlib.sha256(first_prompt.encode()).hexdigest()
    assert invocations[1]["prompt_sha256"] == hashlib.sha256(second_prompt.encode()).hexdigest()
    assert invocations[0]["schema_additional_properties"] is False
    assert invocations[0]["final_answer_schema"] == {"type": "null"}
    assert invocations[1]["final_answer_schema"]["type"] == "object"
    assert invocations[0]["tmpdir"] == str((tmp_path / "run" / "scratch" / "agent-1").resolve())
    assert invocations[1]["tmpdir"] == str((tmp_path / "run" / "scratch" / "agent-2").resolve())
    schema = json.loads(
        (tmp_path / "run" / "calls" / "call_0001" / "codex_agent_artifact.schema.json").read_text()
    )
    assert "claims" in schema["properties"]
    assert "claims" in schema["required"]

    session_files = list((tmp_path / "run" / "scratch" / ".codex_sessions").glob("*.json"))
    assert len(session_files) == 1
    session_record = json.loads(session_files[0].read_text())
    assert session_record["codex_thread_id"] == "thread-test-001"
    assert session_record["model_id"] == "gpt-5.6-luna"
    assert session_record["reasoning_effort"] == "low"

    for index, secret in ((1, first_prompt), (2, second_prompt)):
        call_dir = tmp_path / "run" / "calls" / f"call_{index:04d}"
        assert (call_dir / "codex_events.jsonl").is_file()
        audit_text = (call_dir / "codex_call.json").read_text()
        assert secret not in audit_text
        audit = json.loads(audit_text)
        assert audit["stdin"] == "<PROMPT_ON_STDIN>"
        assert audit["tool_calls"] == 1
        assert audit["require_final_answer"] is (index == 2)
        assert len(audit["output_schema_sha256"]) == 64


def test_codex_adapter_repairs_semantic_contract_in_same_session_and_preserves_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_contract_repair_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))
    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-medium-fast",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "medium",
                "--service-tier",
                "fast",
                "--max-contract-repairs",
                "2",
                "--timeout-seconds",
                "10",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    response = runtime.run(
        _request(
            tmp_path,
            index=1,
            prompt="ORIGINAL_SECRET_PROMPT",
            require_final_answer=True,
            reasoning_effort="medium",
        ),
        ResourceBudget(max_runtime_seconds_per_call=60),
    )

    assert response.artifact.summary == "invalid cross-reference"
    assert response.artifact.final_answer["supported_claim_indices"] == [1]
    assert response.usage.input_tokens == 200
    assert response.usage.output_tokens == 50
    assert response.runtime_metadata["attempt_count"] == 2
    assert response.runtime_metadata["contract_repair_count"] == 1
    assert response.runtime_metadata["service_tier"] == "fast"

    invocations = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert len(invocations) == 2
    assert invocations[0]["is_resume"] is False
    assert invocations[1]["is_resume"] is True
    assert invocations[0]["is_index_repair"] is False
    assert invocations[1]["is_index_repair"] is True
    assert invocations[1]["index_item_schema"]["enum"] == [1]
    assert invocations[0]["has_exact_error"] is False
    assert invocations[1]["has_exact_error"] is True
    assert invocations[1]["asks_to_avoid_new_analysis"] is True
    assert "thread-repair-001" in invocations[1]["argv"]
    for invocation in invocations:
        assert "model_reasoning_effort=medium" in invocation["argv"]
        assert 'service_tier="fast"' in invocation["argv"]

    call_dir = tmp_path / "run" / "calls" / "call_0001"
    first_dir = call_dir / "attempts" / "attempt_0001"
    second_dir = call_dir / "attempts" / "attempt_0002"
    for attempt_dir in (first_dir, second_dir):
        assert (attempt_dir / "codex_last_message.json").is_file()
        assert (attempt_dir / "codex_events.jsonl").is_file()
        assert (attempt_dir / "codex_stderr.log").is_file()
        assert (attempt_dir / "codex_call.json").is_file()
        assert (attempt_dir / "codex_agent_artifact.schema.json").is_file()
    first_artifact = json.loads((first_dir / "codex_last_message.json").read_text())
    second_artifact = json.loads((second_dir / "codex_last_message.json").read_text())
    assert first_artifact["final_answer"]["supported_claim_indices"] == [2]
    assert second_artifact == {"supported_claim_indices": [1]}
    merged_artifact = json.loads(
        (second_dir / "codex_merged_artifact.json").read_text()
    )
    assert merged_artifact["summary"] == first_artifact["summary"]
    assert merged_artifact["final_answer"]["supported_claim_indices"] == [1]
    canonical_artifact = json.loads((call_dir / "codex_last_message.json").read_text())
    assert canonical_artifact == merged_artifact

    root_audit_text = (call_dir / "codex_call.json").read_text()
    assert "ORIGINAL_SECRET_PROMPT" not in root_audit_text
    root_audit = json.loads(root_audit_text)
    assert root_audit["status"] == "accepted"
    assert root_audit["attempt_count"] == 2
    assert root_audit["contract_repair_count"] == 1
    assert root_audit["max_contract_repairs"] == 2
    assert root_audit["service_tier"] == "fast"
    assert [attempt["status"] for attempt in root_audit["attempts"]] == [
        "contract_rejected",
        "accepted",
    ]
    assert [attempt["task_kind"] for attempt in root_audit["attempts"]] == [
        "original",
        "supported_claim_indices_repair",
    ]
    first_audit = json.loads((first_dir / "codex_call.json").read_text())
    second_audit = json.loads((second_dir / "codex_call.json").read_text())
    assert first_audit["session_action"] == "started"
    assert "absent claim index 2" in first_audit["validation_error"]
    assert second_audit["session_action"] == "resumed"
    assert "absent claim index 2" in second_audit["repair_for_error"]

    session_file = next((tmp_path / "run" / "scratch" / ".codex_sessions").glob("*.json"))
    session = json.loads(session_file.read_text())
    assert session["codex_thread_id"] == "thread-repair-001"
    assert session["service_tier"] == "fast"


def test_codex_adapter_retries_transient_exit_and_persists_failed_turn_thread(
    tmp_path: Path, monkeypatch
) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_transient_then_success_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))
    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-medium-fast",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "medium",
                "--service-tier",
                "fast",
                "--max-runtime-retries",
                "2",
                "--runtime-retry-initial-seconds",
                "0",
                "--runtime-retry-max-seconds",
                "0",
                "--transport-circuit-failure-threshold",
                "1",
                "--transport-circuit-cooldown-seconds",
                "0",
                "--timeout-seconds",
                "10",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    response = runtime.run(
        _request(
            tmp_path,
            index=1,
            prompt="ORIGINAL_PENDING_TASK",
            reasoning_effort="medium",
        ),
        ResourceBudget(max_runtime_seconds_per_call=60),
    )

    assert response.artifact.summary == "recovered"
    assert response.runtime_metadata["attempt_count"] == 2
    assert response.runtime_metadata["runtime_retry_count"] == 1
    assert response.runtime_metadata["contract_repair_count"] == 0
    invocations = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert len(invocations) == 2
    assert invocations[0]["is_resume"] is False
    assert invocations[1]["is_resume"] is True
    assert invocations[1]["is_transport_retry"] is True
    assert "thread-transient-001" in invocations[1]["argv"]

    call_dir = tmp_path / "run" / "calls" / "call_0001"
    audit = json.loads((call_dir / "codex_call.json").read_text())
    assert audit["status"] == "accepted"
    assert audit["runtime_retry_count"] == 1
    assert [item["status"] for item in audit["attempts"]] == [
        "transport_retry_pending",
        "accepted",
    ]
    assert audit["attempts"][0]["transport_retry_reason"] == "codex_backend_404"
    first_audit = json.loads(
        (call_dir / "attempts" / "attempt_0001" / "codex_call.json").read_text()
    )
    assert first_audit["codex_thread_id"] == "thread-transient-001"
    session_file = next((tmp_path / "run" / "scratch" / ".codex_sessions").glob("*.json"))
    assert json.loads(session_file.read_text())["codex_thread_id"] == "thread-transient-001"


def test_codex_adapter_retries_analysis_resource_failure_with_corrective_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_resource_failure_then_success_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))
    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-medium-fast",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "medium",
                "--max-runtime-retries",
                "2",
                "--runtime-retry-initial-seconds",
                "0",
                "--runtime-retry-max-seconds",
                "0",
                "--timeout-seconds",
                "10",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    response = runtime.run(
        _request(
            tmp_path,
            index=1,
            prompt="ORIGINAL_PENDING_TASK",
            reasoning_effort="medium",
        ),
        ResourceBudget(max_runtime_seconds_per_call=60),
    )

    assert response.artifact.summary == "recovered after fixing a broadcast"
    assert response.runtime_metadata["runtime_retry_count"] == 1
    assert response.runtime_metadata["transport_retry_count"] == 0
    assert response.runtime_metadata["resource_retry_count"] == 1
    invocations = [json.loads(line) for line in invocation_log.read_text().splitlines()]
    assert invocations[1]["is_resume"] is True
    assert invocations[1]["is_resource_retry"] is True
    assert invocations[1]["checks_shapes"] is True

    call_dir = tmp_path / "run" / "calls" / "call_0001"
    audit = json.loads((call_dir / "codex_call.json").read_text())
    assert audit["status"] == "accepted"
    assert audit["transport_retry_count"] == 0
    assert audit["resource_retry_count"] == 1
    assert [item["status"] for item in audit["attempts"]] == [
        "resource_retry_pending",
        "accepted",
    ]
    assert audit["attempts"][0]["resource_retry_reason"] == "analysis_memory_limit"
    assert not list(tmp_path.rglob(".codex_transport_circuit.json"))


def test_codex_adapter_does_not_retry_nontransport_runtime_exit(
    tmp_path: Path, monkeypatch
) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_failing_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))
    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-medium-fast",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "medium",
                "--service-tier",
                "fast",
                "--max-contract-repairs",
                "2",
                "--max-runtime-retries",
                "2",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    with pytest.raises(RuntimeError, match="Agent command exited"):
        runtime.run(
            _request(
                tmp_path,
                index=1,
                prompt="probe",
                reasoning_effort="medium",
            ),
            ResourceBudget(max_runtime_seconds_per_call=60),
        )

    assert len(invocation_log.read_text().splitlines()) == 1
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    assert not (call_dir / "attempts" / "attempt_0002").exists()
    audit = json.loads((call_dir / "codex_call.json").read_text())
    assert audit["status"] == "runtime_error"
    assert audit["attempt_count"] == 1
    assert audit["contract_repair_count"] == 0
    assert audit["returncode"] == 7


def test_codex_adapter_rejects_reasoning_effort_mismatch_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))
    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-medium",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--reasoning-effort",
                "medium",
            ],
        )
    )

    with pytest.raises(RuntimeError, match="reasoning-effort mismatch"):
        runtime.run(
            _request(
                tmp_path,
                index=1,
                prompt="must not launch",
                reasoning_effort="low",
            ),
            ResourceBudget(max_runtime_seconds_per_call=60),
        )

    assert not invocation_log.exists()


def test_codex_adapter_adds_only_explicit_read_roots(tmp_path: Path, monkeypatch) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    read_root = tmp_path / "safe-runtime"
    read_root.mkdir()
    _write_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))

    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-low",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--read-root",
                str(read_root),
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    runtime.run(_request(tmp_path, index=1, prompt="probe"), ResourceBudget())

    argv = json.loads(invocation_log.read_text().splitlines()[0])["argv"]
    profile = next(value for value in argv if value.startswith("permissions.nsclc_eval="))
    assert f'{json.dumps(str(read_root.resolve()))}="read"' in profile
    assert f'{json.dumps(str(fake_codex.parent.resolve()))}="read"' in profile
    assert '"/"="read"' not in profile
    assert ".codex/auth.json" not in profile


def test_codex_runtime_root_selects_package_not_user_state(tmp_path: Path) -> None:
    package = tmp_path / ".local" / "lib" / "node_modules" / "@openai" / "codex"
    launcher = package / "bin" / "codex.js"
    launcher.parent.mkdir(parents=True)
    launcher.touch()

    assert _codex_runtime_root(str(launcher)) == package.resolve()
    assert _codex_runtime_root(str(launcher)) != tmp_path / ".codex"


def test_codex_adapter_configures_isolated_python_runtime(tmp_path: Path, monkeypatch) -> None:
    fake_codex = tmp_path / "fake-codex"
    invocation_log = tmp_path / "fake_invocations.jsonl"
    _write_fake_codex(fake_codex)
    monkeypatch.setenv("FAKE_CODEX_LOG", str(invocation_log))

    runtime = CliJsonRuntime(
        ModelSpec(
            id="luna-low",
            model_id="gpt-5.6-luna",
            adapter="cli-json",
            command=[sys.executable, str(ADAPTER)],
            extra_args=[
                "--codex",
                str(fake_codex),
                "--analysis-python",
                sys.executable,
                "--analysis-memory-limit-mb",
                "4096",
            ],
            env_passthrough=["FAKE_CODEX_LOG"],
        )
    )

    runtime.run(_request(tmp_path, index=1, prompt="probe"), ResourceBudget())

    invocation = json.loads(invocation_log.read_text().splitlines()[0])
    profile = next(
        value for value in invocation["argv"] if value.startswith("permissions.nsclc_eval=")
    )
    environment = next(
        value for value in invocation["argv"] if value.startswith("shell_environment_policy=")
    )
    assert '":root"="deny"' in profile
    assert '":slash_tmp"="deny"' in profile
    assert 'inherit="none"' in environment
    assert '"PYTHONPATH"=' in environment
    assert '"PYTHONNOUSERSITE"="1"' in environment
    assert '"OCS_ANALYSIS_MEMORY_LIMIT_MB"="4096"' in environment
    assert '"OPENBLAS_NUM_THREADS"="1"' in environment
    assert "python3." in environment
    assert invocation["pythonhome"] is None
    assert invocation["pythonpath"] is None
    assert '="read"' in profile
    assert f'{json.dumps(str(_analysis_guard_root()))}="read"' in profile


def test_analysis_python_guard_turns_large_allocation_into_marked_memory_error() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_analysis_guard_root())
    environment["OCS_ANALYSIS_MEMORY_LIMIT_MB"] = "256"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import resource; "
                "print(resource.getrlimit(resource.RLIMIT_AS)); "
                "bytearray(512 * 1024 * 1024)"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "(268435456, 268435456)" in completed.stdout
    assert "OCS_ANALYSIS_RESOURCE_LIMIT" in completed.stderr
    assert "MemoryError" in completed.stderr


@pytest.mark.parametrize(
    ("diagnostic", "returncode", "reason"),
    [
        ("MemoryError", 1, "python_memory_error"),
        ("Unable to allocate 18.6 GiB for an array", 1, "allocation_failure"),
        ("", -9, "codex_process_sigkill"),
        ('{"exit_code":137}', 1, "analysis_process_sigkill"),
        ("segmentation fault", 1, "analysis_process_sigsegv"),
    ],
)
def test_resource_retry_classifier(
    diagnostic: str, returncode: int, reason: str
) -> None:
    assert _resource_retry_reason("", diagnostic, returncode) == reason


def test_codex_event_parser_deduplicates_tools_and_accepts_nested_usage() -> None:
    raw = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-42"}),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"id": "cmd-1", "type": "command_execution"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "cmd-1", "type": "command_execution"},
                }
            ),
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "usage": {"inputTokens": 321, "outputTokens": 45},
                        "item": {"call_id": "mcp-1", "type": "mcp_tool_call"},
                    },
                }
            ),
            "not-json",
        ]
    )

    stats = parse_codex_events(raw)

    assert extract_codex_thread_id(raw) == "thread-42"
    assert stats.input_tokens == 321
    assert stats.output_tokens == 45
    assert stats.tool_calls == 2
    assert stats.item_types == ("command_execution", "mcp_tool_call")
