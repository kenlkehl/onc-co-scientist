from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from onc_co_scientist.harness.experiment import (
    ModelSpec,
    ResourceBudget,
    load_experiment_spec,
)
from onc_co_scientist.harness.runtime import AgentRequest, CliJsonRuntime
from scripts.codex_cli_json_adapter import (
    ARTIFACT_SCHEMA,
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

prompt = sys.stdin.read()
entry = {
    "argv": argv,
    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    "schema_additional_properties": json.loads(schema_path.read_text())["additionalProperties"],
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
    "final_answer": None,
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


def _request(
    tmp_path: Path,
    *,
    index: int,
    prompt: str,
    session_id: str = "shared-persistent-session",
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
        stage_id=f"stage-{index}",
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
    second = runtime.run(_request(tmp_path, index=2, prompt=second_prompt), budget)

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
    assert '"/"="read"' not in profile


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
    assert "python3." in environment
    assert invocation["pythonhome"] is None
    assert invocation["pythonpath"] is None
    assert '="read"' in profile


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
