"""Checks for CLI transport accounting, isolation, and transient usage limits."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onc_co_scientist.providers.base import ChatMessage
from onc_co_scientist.providers.codex_cli import CodexCLIConfig, CodexCLIProvider


def fake_cli(monkeypatch, outcomes):
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            self.command, self.kwargs = command, kwargs
            self.returncode = 0
            calls.append(self)

        def communicate(self, prompt, timeout):
            self.prompt = prompt
            outcome = outcomes.pop(0)
            if outcome == "quota":
                self.returncode = 1
                self.kwargs["stdout"].write(
                    json.dumps({"type": "error", "message": "usage_limit_reached"}) + "\n"
                )
                return
            events = [
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 10,
                    },
                }
            ]
            if outcome == "tool":
                events.insert(
                    0, {"type": "item.completed", "item": {"type": "mcp_tool_call", "id": "tool-1"}}
                )
            if isinstance(outcome, dict):
                self.returncode = outcome.get("returncode", 0)
                events = outcome["events"]
            self.kwargs["stdout"].write("\n".join(map(json.dumps, events)) + "\n")
            if isinstance(outcome, dict) and not outcome.get("write_final", True):
                return
            output = Path(self.command[self.command.index("--output-last-message") + 1])
            output.write_text('{"ready": true}')

    monkeypatch.setattr("onc_co_scientist.providers.codex_cli.subprocess.Popen", Process)
    monkeypatch.setattr(CodexCLIProvider, "_quota_until", 0)
    return calls


def test_cli_preserves_prompt_records_limits_and_uses_fresh_sessions(tmp_path, monkeypatch):
    calls = fake_cli(monkeypatch, ["ok", "ok"])
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-use-api-billing")
    provider = CodexCLIProvider(
        CodexCLIConfig(
            model_id="gpt-5.6-luna",
            audit_dir=str(tmp_path / "audit"),
            reasoning_effort="medium",
            service_tier="priority",
        )
    )
    prompt = 'Public research history {"stage": "appraise"}'
    for _ in range(2):
        response = provider.chat([ChatMessage("user", prompt)], max_tokens=125000)
        assert json.loads(response.text) == {"ready": True}
        assert response.raw["metrics"]["cli_usage"]["reasoning_output_tokens"] == 10
        assert response.raw["metrics"]["temperature_control"] == "CLI default"
        assert response.raw["metrics"]["service_tier_requested"] == "priority"
    assert len(calls) == 2
    for call in calls:
        assert call.prompt == prompt
        assert "--ephemeral" in call.command and "resume" not in call.command
        assert 'model_reasoning_effort="medium"' in call.command
        assert 'service_tier="priority"' in call.command
        assert 'forced_login_method="chatgpt"' in call.command
        assert any('":root"="deny"' in arg for arg in call.command)
        assert any("limit_tokens=125000" in arg for arg in call.command)
        assert "OPENAI_API_KEY" not in call.kwargs["env"]
    assert (tmp_path / "audit/call-0001/prompt.txt").exists()
    assert (tmp_path / "audit/call-0002/prompt.txt").exists()


def test_usage_limit_retries_identical_stage_without_becoming_protocol_failure(
    tmp_path, monkeypatch
):
    calls = fake_cli(monkeypatch, ["quota", "ok"])
    provider = CodexCLIProvider(
        CodexCLIConfig(model_id="gpt-6-astra", audit_dir=str(tmp_path / "audit"), usage_retry_s=0)
    )
    result = provider.chat([ChatMessage("user", "unchanged public history")])
    assert result.raw["metrics"]["infrastructure_attempts"] == 2
    assert calls[0].prompt == calls[1].prompt
    assert provider.calls == 1


def test_unexpected_tools_do_not_silently_enter_controller_only_evaluation(tmp_path, monkeypatch):
    fake_cli(monkeypatch, ["tool"])
    provider = CodexCLIProvider(
        CodexCLIConfig(model_id="gpt-5.6-sol", audit_dir=str(tmp_path / "audit"))
    )
    with pytest.raises(RuntimeError, match="Unexpected tool use"):
        provider.chat([ChatMessage("user", "public history")])


def test_resume_appends_audit_and_preserves_interrupted_calls(tmp_path, monkeypatch):
    fake_cli(monkeypatch, ["ok", "ok"])
    config = dict(model_id="gpt-5.6-terra", audit_dir=str(tmp_path / "audit"))
    first = CodexCLIProvider(CodexCLIConfig(**config))
    first.chat([ChatMessage("user", "first")])
    original = (first.root / "call-0001/prompt.txt").read_bytes()
    (first.root / "call-0002").mkdir()  # Interrupted before a response was journaled.
    with pytest.raises(FileExistsError):
        CodexCLIProvider(CodexCLIConfig(**config))
    resumed = CodexCLIProvider(CodexCLIConfig(**config, resume_audit=True))
    resumed.chat([ChatMessage("user", "next")])
    assert resumed.calls == 3
    assert (first.root / "call-0001/prompt.txt").read_bytes() == original
    assert (first.root / "call-0003/prompt.txt").read_text() == "next"
    resumed.instructions.write_text("changed")
    with pytest.raises(ValueError, match="changed instructions"):
        CodexCLIProvider(CodexCLIConfig(**config, resume_audit=True))


def test_reconnect_warnings_followed_by_completion_preserve_response_and_usage(
    tmp_path, monkeypatch
):
    events = [
        {"type": "turn.started"},
        {"type": "error", "message": "Reconnecting... 2/5 (request timed out)"},
        {"type": "error", "message": "Reconnecting... 3/5 (request timed out)"},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 123, "output_tokens": 456, "reasoning_output_tokens": 200},
        },
    ]
    calls = fake_cli(monkeypatch, [{"events": events}])
    provider = CodexCLIProvider(
        CodexCLIConfig(model_id="gpt-6-astra", audit_dir=str(tmp_path / "audit"))
    )
    response = provider.chat([ChatMessage("user", "public ledger")])
    assert json.loads(response.text) == {"ready": True}
    assert len(calls) == 1
    assert response.raw["metrics"]["usage"]["completion_tokens"] == 456
    assert response.raw["metrics"]["recovered_error_count"] == 2
    assert response.raw["metrics"]["cli_usage"]["reasoning_output_tokens"] == 200


@pytest.mark.parametrize(
    "events,returncode,write_final,match",
    [
        (
            [{"type": "error", "message": "at capacity"}, {"type": "turn.failed"}],
            0,
            True,
            "turn failed",
        ),
        ([{"type": "turn.completed"}, {"type": "turn.failed"}], 0, True, "turn failed"),
        ([{"type": "turn.completed"}, {"type": "turn.started"}], 0, True, "turn failed"),
        (
            [{"type": "turn.completed"}, {"type": "error", "message": "fatal"}],
            0,
            True,
            "turn failed",
        ),
        ([{"type": "turn.started"}], 0, True, "turn failed"),
        ([{"type": "turn.completed"}], 1, True, "exited 1"),
        ([{"type": "turn.completed"}], 0, False, "no final message"),
        (
            [
                {"type": "error", "message": "reconnecting"},
                {"type": "item.completed", "item": {"type": "mcp_tool_call"}},
                {"type": "turn.completed"},
            ],
            0,
            True,
            "Unexpected tool use",
        ),
    ],
)
def test_completion_does_not_mask_terminal_failure_or_missing_output(
    tmp_path, monkeypatch, events, returncode, write_final, match
):
    fake_cli(
        monkeypatch, [{"events": events, "returncode": returncode, "write_final": write_final}]
    )
    provider = CodexCLIProvider(
        CodexCLIConfig(model_id="gpt-6-astra", audit_dir=str(tmp_path / "audit"))
    )
    with pytest.raises(RuntimeError, match=match):
        provider.chat([ChatMessage("user", "public ledger")])


def test_completed_turn_with_missing_usage_remains_unknown(tmp_path, monkeypatch):
    fake_cli(monkeypatch, [{"events": [{"type": "turn.completed"}]}])
    provider = CodexCLIProvider(
        CodexCLIConfig(model_id="gpt-6-astra", audit_dir=str(tmp_path / "audit"))
    )
    metrics = provider.chat([ChatMessage("user", "public ledger")]).raw["metrics"]
    assert metrics["usage"]["completion_tokens"] is None
    assert metrics["usage"]["prompt_tokens"] is None


def test_azure_refreshes_every_attempt_without_personal_auth_or_logged_credentials(
    tmp_path, monkeypatch
):
    calls = fake_cli(monkeypatch, ["ok", "ok"])
    tokens = iter(["fresh-token-one", "fresh-token-two"])
    refreshes = []

    def refresh(command, **kwargs):
        refreshes.append(command)
        return SimpleNamespace(returncode=0, stdout=next(tokens), stderr="")

    monkeypatch.setattr("onc_co_scientist.providers.codex_cli.subprocess.run", refresh)
    monkeypatch.setenv("OCS_AZURE_ACCESS_TOKEN", "stale")
    monkeypatch.setenv("OPENAI_API_KEY", "personal-key")
    p = CodexCLIProvider(
        CodexCLIConfig(
            model_id="gpt-5.6-sol",
            backend="azure",
            azure_endpoint="https://example.openai.azure.com/openai/v1",
            audit_dir=str(tmp_path / "azure"),
        )
    )
    for _ in range(2):
        assert p.chat([ChatMessage("user", "test")]).raw["metrics"]["backend"] == "azure"
    assert len(refreshes) == 2
    for i, call in enumerate(calls):
        assert 'model_provider="ocs_azure"' in call.command
        assert "model_providers.ocs_azure.requires_openai_auth=false" in call.command
        assert not any("forced_login_method" in arg for arg in call.command)
        assert "OPENAI_API_KEY" not in call.kwargs["env"]
        assert (
            call.kwargs["env"]["OCS_AZURE_ACCESS_TOKEN"]
            == ["fresh-token-one", "fresh-token-two"][i]
        )
    for path in p.root.rglob("*"):
        if path.is_file():
            assert "fresh-token" not in path.read_text()


def test_azure_refresh_failure_never_starts_codex(tmp_path, monkeypatch):
    calls = fake_cli(monkeypatch, [])
    monkeypatch.setattr(
        "onc_co_scientist.providers.codex_cli.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="secret", stderr="secret"),
    )
    p = CodexCLIProvider(
        CodexCLIConfig(
            model_id="gpt-5.6-sol",
            backend="azure",
            azure_endpoint="https://example.openai.azure.com/openai/v1",
            audit_dir=str(tmp_path / "azure"),
        )
    )
    with pytest.raises(RuntimeError, match="no personal-account fallback") as error:
        p.chat([ChatMessage("user", "test")])
    assert "secret" not in str(error.value) and not calls
