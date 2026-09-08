"""Checks for CLI transport accounting, isolation, and transient usage limits."""

import json
from pathlib import Path

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
            self.kwargs["stdout"].write("\n".join(map(json.dumps, events)) + "\n")
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
