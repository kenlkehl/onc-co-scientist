"""Shared deployment admission and infrastructure-only throttle retries."""

import json

import pytest

from onc_co_scientist.providers import codex_cli as cli
from onc_co_scientist.providers.base import ChatMessage
from tests.test_codex_cli_provider import fake_cli


def configuration(tmp_path, **kwargs):
    return cli.CodexCLIConfig(
        model_id="test",
        backend="azure",
        azure_endpoint="https://example.openai.azure.com/openai/v1",
        audit_dir=str(tmp_path / "audit"),
        azure_pacing_dir=str(tmp_path / "pacing"),
        **kwargs,
    )


def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cli.time, "time", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(cli.random, "uniform", lambda *a: 0)
    return now


def test_shared_budget_and_cooldown_survive_new_instances(tmp_path, monkeypatch):
    now = clock(monkeypatch)
    config = configuration(tmp_path, azure_tokens_per_minute=1000)
    path = tmp_path / "state.json"
    first = cli.AzureRequestLease(config, path, tmp_path)
    first.admit(600)
    second = cli.AzureRequestLease(config, path, tmp_path)
    second.admit(600)
    assert now[0] == 1060
    second.throttled([{"type": "error", "message": "429 retry-after-ms: 120000"}], "")
    third = cli.AzureRequestLease(config, path, tmp_path)
    third.admit(10)
    assert now[0] == 1180
    third.throttled([{"type": "error", "message": "429"}], "")
    assert third.state["consecutive_rate_limits"] == 2
    assert third.state["cooldown_until"] == 1300
    third.succeeded()
    assert json.loads(path.read_text())["consecutive_rate_limits"] == 0


def test_throttle_retries_same_call_and_refreshes_token(tmp_path, monkeypatch):
    slot = cli.CodexCLIProvider.request_slot
    now = clock(monkeypatch)
    calls = fake_cli(
        monkeypatch,
        [
            dict(
                returncode=1,
                write_final=False,
                events=[{"type": "error", "message": "429 rate limit retry after 90"}],
            ),
            "ok",
        ],
    )
    monkeypatch.setattr(cli.CodexCLIProvider, "request_slot", slot)
    refreshes = []

    def environment(self):
        refreshes.append(len(refreshes))
        return {"OCS_AZURE_ACCESS_TOKEN": str(len(refreshes))}

    monkeypatch.setattr(cli.CodexCLIProvider, "environment", environment)
    provider = cli.CodexCLIProvider(configuration(tmp_path))
    result = provider.chat([ChatMessage("user", "identical history")], max_tokens=125000)
    assert provider.calls == 1 and len(calls) == 2
    assert calls[0].prompt == calls[1].prompt
    assert calls[0].kwargs["env"] != calls[1].kwargs["env"]
    assert now[0] == 1090
    assert result.raw["metrics"]["azure_rate_limit_retries"] == 1
    retry = json.loads((provider.root / "call-0001/attempt-0001/rate_limit_retry.json").read_text())
    assert not retry["scientific_stage_retry_consumed"]


def test_billing_and_success_content_do_not_trigger_transient_retry():
    assert not cli.CodexCLIProvider.azure_rate_limit_failure(
        [{"type": "error", "message": "429 insufficient_quota"}], ""
    )
    assert not cli.CodexCLIProvider.azure_rate_limit_failure(
        [{"type": "item.completed", "item": {"text": "rate limit 429"}}], ""
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), 1.5, True])
def test_invalid_quota(tmp_path, value):
    with pytest.raises(ValueError):
        configuration(tmp_path, azure_tokens_per_minute=value)


def test_deployment_slot_is_exclusive_across_processes(tmp_path):
    import subprocess
    import sys
    import time

    script = """
import sys
from pathlib import Path
from onc_co_scientist.providers.codex_cli import CodexCLIConfig, CodexCLIProvider
root=Path(sys.argv[1]); name=sys.argv[2]
p=CodexCLIProvider(CodexCLIConfig(model_id="test", backend="azure",
 azure_endpoint="https://example.openai.azure.com/openai/v1",
 audit_dir=str(root/name), azure_pacing_dir=str(root/"pacing"), azure_request_interval_s=0))
with p.request_slot(p.root, "prompt", 1):
 print("admitted", flush=True)
 sys.stdin.readline()
"""

    def launch(name):
        return subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    first = launch("first")
    second = None
    try:
        assert first.stdout.readline().strip() == "admitted"
        second = launch("second")
        time.sleep(0.2)
        assert not (tmp_path / "second/pacing.json").exists()
        first.communicate("release\n", timeout=5)
        assert second.stdout.readline().strip() == "admitted"
        second.communicate("release\n", timeout=5)
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()


def test_adaptive_reserve_learns_large_outputs_and_keeps_scientific_ceiling(tmp_path, monkeypatch):
    now = clock(monkeypatch)
    p = cli.CodexCLIProvider(configuration(tmp_path))
    for name in ("one", "two"):
        d = tmp_path / name
        d.mkdir()
        with p.request_slot(d, "x" * 120000, 125000) as lease:
            assert lease.output_reservation(125000) == 8192
            lease.succeeded({"output_tokens": 2000})
    # Both 56,192-token reservations fit in the minute, instead of one 173,000.
    assert now[0] == 1010
    receipt = json.loads((tmp_path / "two/reservation_estimate.json").read_text())
    assert receipt["scientific_output_ceiling"] == 125000
    assert receipt["output_reserve_tokens"] == 8192
    with p.request_slot(tmp_path, "short", 125000) as lease:
        lease.succeeded({"output_tokens": 20000})
    other = cli.CodexCLIProvider(configuration(tmp_path / "other"))
    # Share deployment admission state, even with a separate provider audit root.
    from dataclasses import replace

    other.config = replace(other.config, azure_pacing_dir=p.config.azure_pacing_dir)
    with other.request_slot(tmp_path, "short", 125000) as lease:
        assert lease.output_reservation(125000) == 31024
        assert lease.output_reservation(500) == 500
    assert any("limit_tokens=125000" in arg for arg in p.command(tmp_path / "out", 125000))


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_output_reserve(tmp_path, value):
    with pytest.raises(ValueError):
        configuration(tmp_path, azure_output_reserve_tokens=value)
