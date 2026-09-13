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
 audit_dir=str(root/name), azure_pacing_dir=str(root/"pacing"),
 azure_request_interval_s=0, azure_max_inflight=1))
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


def test_concurrent_lease_updates_do_not_erase_quota_or_cooldown(tmp_path, monkeypatch):
    now = clock(monkeypatch)
    cfg = configuration(tmp_path, azure_request_interval_s=0)
    path = tmp_path / "state.json"
    first = cli.AzureRequestLease(cfg, path, tmp_path)
    first.admit(100)
    now[0] += 1
    second = cli.AzureRequestLease(cfg, path, tmp_path)
    second.admit(200)
    second.throttled([{"type": "error", "message": "429"}], "")
    first.succeeded({"output_tokens": 77})
    saved = json.loads(path.read_text())
    assert len(saved["reservations"]) == 2
    assert saved["consecutive_rate_limits"] == 1
    assert saved["cooldown_until"] > now[0]
    assert saved["recent_output_tokens"] == [77]


def test_one_hung_request_does_not_block_other_slots_and_dead_owner_releases(tmp_path):
    import select
    import subprocess
    import sys
    import time

    script = """
import sys
from pathlib import Path
from onc_co_scientist.providers.codex_cli import CodexCLIConfig, CodexCLIProvider
root=Path(sys.argv[1]); name=sys.argv[2]
p=CodexCLIProvider(CodexCLIConfig(model_id="test", backend="azure",
 azure_endpoint="https://example.openai.azure.com/openai/v1", azure_max_inflight=3,
 audit_dir=str(root/name), azure_pacing_dir=str(root/"pacing"), azure_request_interval_s=0))
with p.request_slot(p.root, "prompt", 1):
 print("admitted", flush=True)
 sys.stdin.readline()
"""
    processes = []
    try:
        for name in ("hung", "second", "third", "fourth"):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(tmp_path), name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
            if name != "fourth":
                assert select.select([processes[-1].stdout], [], [], 10)[0]
                assert processes[-1].stdout.readline().strip() == "admitted"
        time.sleep(0.2)
        assert not (tmp_path / "fourth/pacing.json").exists()
        processes[0].kill()
        processes[0].wait(timeout=5)
        assert select.select([processes[-1].stdout], [], [], 10)[0]
        assert processes[-1].stdout.readline().strip() == "admitted"
        state = next((tmp_path / "pacing").glob("*.json"))
        assert len(json.loads(state.read_text())["reservations"]) == 4
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_server_failure_retries_identical_prompt_with_fresh_auth(tmp_path, monkeypatch):
    calls = fake_cli(
        monkeypatch,
        [
            dict(
                returncode=1,
                write_final=False,
                events=[
                    {
                        "type": "turn.failed",
                        "error": {"message": "The server had an error processing your request"},
                    }
                ],
            ),
            "ok",
        ],
    )
    refreshes = []

    def environment(self):
        refreshes.append(1)
        return {"OCS_AZURE_ACCESS_TOKEN": str(len(refreshes))}

    monkeypatch.setattr(cli.CodexCLIProvider, "environment", environment)
    p = cli.CodexCLIProvider(configuration(tmp_path, azure_transport_retry_s=0))
    result = p.chat([ChatMessage("user", "unchanged")], max_tokens=125000)
    assert len(calls) == len(refreshes) == 2
    assert calls[0].prompt == calls[1].prompt == "unchanged"
    assert result.raw["metrics"]["azure_transport_retries"] == 1
    assert all(any("limit_tokens=125000" in arg for arg in c.command) for c in calls)
    assert "model_providers.ocs_azure.stream_max_retries=0" in calls[0].command
    receipt = json.loads((p.root / "call-0001/attempt-0001/transport_retry.json").read_text())
    assert receipt["scientific_stage_retry_consumed"] is False


def test_transport_retry_is_bounded_and_does_not_retry_task_text(tmp_path, monkeypatch):
    event = {"type": "turn.failed", "error": {"message": "503 Service unavailable"}}
    calls = fake_cli(monkeypatch, [dict(returncode=1, events=[event], write_final=False)] * 3)
    monkeypatch.setattr(cli.CodexCLIProvider, "environment", lambda self: {})
    p = cli.CodexCLIProvider(
        configuration(tmp_path, azure_transport_retries=2, azure_transport_retry_s=0)
    )
    with pytest.raises(RuntimeError, match="after 2 retries"):
        p.chat([ChatMessage("user", "same")])
    assert len(calls) == 3
    assert not cli.CodexCLIProvider.azure_transport_failure(
        [{"type": "item.completed", "item": {"text": "503 stream disconnected"}}], ""
    )
    assert not cli.CodexCLIProvider.azure_transport_failure([], "models_manager: 503")


@pytest.mark.parametrize("progress", [False, True])
def test_watchdog_ignores_diagnostics_but_allows_model_progress(tmp_path, monkeypatch, progress):
    import sys
    import time

    script = tmp_path / "codex-fixture"
    script.write_text(f"""#!{sys.executable}
import json,sys,time
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output-last-message')+1])
prompt=sys.stdin.read()
(output.parent/'received.txt').write_text(prompt)
print(json.dumps({{"type":"turn.started"}}),flush=True)
if output.parent.name=='attempt-0001':
 for i in range(8):
  item=({{"type":"reasoning","text":"working"}} if {progress!r}
   else {{"type":"error","message":"heartbeat"}})
  print(json.dumps({{"type":"item.completed","item":item}}),flush=True)
  time.sleep(.08)
output.write_text('{{"ok":true}}')
print(json.dumps({{"type":"turn.completed","usage":{{"input_tokens":20,"output_tokens":5}}}}),flush=True)
""")
    script.chmod(0o700)
    monkeypatch.setattr(cli.CodexCLIProvider, "environment", lambda self: {})
    p = cli.CodexCLIProvider(
        configuration(
            tmp_path,
            executable=str(script),
            azure_response_idle_s=0.25,
            azure_transport_retry_s=0,
            azure_request_interval_s=0,
        )
    )
    start = time.monotonic()
    result = p.chat([ChatMessage("user", "exact original prompt")])
    assert time.monotonic() - start < 5
    assert result.raw["metrics"]["azure_transport_retries"] == (0 if progress else 1)
    if not progress:
        base = p.root / "call-0001"
        assert (base / "attempt-0001/received.txt").read_text() == (
            base / "attempt-0002/received.txt"
        ).read_text()
        receipt = json.loads((base / "attempt-0001/transport_retry.json").read_text())
        assert receipt["reason"] == "response_watchdog"
        assert receipt["interrupted_usage_unknown"]
