"""Launcher contracts without starting processes or using GPUs."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher():
    path = Path(__file__).parents[1] / "scripts/expected_surprising/run_local_caa_pilot.py"
    spec = importlib.util.spec_from_file_location("run_local_caa_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_separates_queued_active_and_terminal(tmp_path):
    pilot = launcher()
    jobs = [{"view": "masked", "run_id": key, "replicate": 1}
            for key in ("queued", "active", "done")]
    (tmp_path / "schedule.json").write_text(json.dumps(jobs))
    runs = tmp_path / "masked/runs"
    (runs / "active/calls").mkdir(parents=True)
    (runs / "active/calls/one.json").write_text("{}")
    (runs / "done").mkdir()
    (runs / "done/run.json").write_text(json.dumps({
        "status": "completed", "iterations_completed": 6, "agent_calls": 24,
    }))
    rows = pilot.snapshot(tmp_path)
    assert [r["status"] for r in rows] == ["queued", "running", "completed"]
    assert rows[1]["saved_calls"] == 1
    assert rows[2]["iterations_completed"] == 6
    assert pilot.snapshot(tmp_path, active=False)[1]["status"] == "interrupted"


def test_launch_freezes_source_and_detaches_only_new_supervisor(tmp_path, monkeypatch):
    pilot = launcher()
    source = tmp_path / "repository"
    package = source / "src/onc_co_scientist"
    package.mkdir(parents=True)
    (package / "example.py").write_text("original")
    scripts = source / "scripts/expected_surprising"
    scripts.mkdir(parents=True)
    for name in ("run_local_caa_pilot.py", "caa_discovery.py", "build_presentation_results.py"):
        (scripts / name).write_text("script")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("vectors.npz", "vectors.json", "aliases.json"):
        (artifacts / name).write_text("artifact")
    calls = []

    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(pilot, "ROOT", source)
    monkeypatch.setattr(pilot.subprocess, "Popen", spawn)
    args = SimpleNamespace(runtime=tmp_path / "runtime", campaign=tmp_path / "campaign",
                           model=tmp_path / "model", smoke_artifacts=artifacts,
                           iterations=6, max_hours=12, max_tokens=100000,
                           request_timeout=21600, run_limit=1, gpus="0,1,2,3",
                           inter_gpu_transfer="cpu", prefill_chunk_size=2048,
                           max_context_tokens=131072, port=18765)
    pilot.launch(args)
    (package / "example.py").write_text("changed")
    assert (args.runtime / "source/src/onc_co_scientist/example.py").read_text() == "original"
    assert calls[0][1]["start_new_session"] is True
    assert "--worker" in calls[0][0][0]
    config = json.loads((args.runtime / "launch.json").read_text())
    assert config["iterations"] == 6 and config["max_hours"] == 12
    assert config["max_tokens"] == 100000 and config["request_timeout"] == 21600
    assert config["run_limit"] == 1 and config["gpus"] == "0,1,2,3"
    assert config["max_context_tokens"] == 131072
    assert config["port"] == 18765
    assert not args.campaign.exists()


def test_existing_campaign_not_overwritten(tmp_path):
    pilot = launcher()
    campaign = tmp_path / "existing"
    campaign.mkdir()
    args = SimpleNamespace(runtime=tmp_path / "unused", campaign=campaign)
    with pytest.raises(FileExistsError):
        pilot.launch(args)
    assert not args.runtime.exists()


@pytest.mark.parametrize("exit_code,expected_status", [(0, "gate_passed"), (1, "failed")])
@pytest.mark.parametrize("resume,manifest_match", [(False, True), (True, True), (True, False)])
def test_supervisor_forwards_capacity_options_and_cleans_up(
    tmp_path, monkeypatch, exit_code, expected_status, resume, manifest_match,
):
    pilot = launcher()
    runtime, campaign = tmp_path / "runtime", tmp_path / "campaign"
    runtime.mkdir()
    config = dict(campaign=str(campaign), model="fixture", iterations=6, max_hours=12,
                  max_tokens=100000, request_timeout=21600, run_limit=2 if resume else 1, gpus="2,3",
                  inter_gpu_transfer="cpu", prefill_chunk_size=1024,
                  max_context_tokens=131072, named="named", masked="masked", base_config="base",
                  port=18765)
    pilot.write(runtime / "launch.json", config)
    jobs = [{"view": "masked", "run_id": f"cell-{i}", "replicate": 1} for i in range(8)]
    commands, stopped = [], []
    server = SimpleNamespace(pid=111, poll=lambda: None)
    controller = SimpleNamespace(pid=222, returncode=exit_code, poll=lambda: exit_code)
    manifest_data = {"resolved_device_map": {"layer0": "0", "layer1": "1"},
                     "fingerprint": "a" * 64}
    if resume:
        folder = campaign / "masked/runs/cell-0"
        folder.mkdir(parents=True)
        pilot.write(folder / "run.json", {"status": "completed"})
        pilot.write(campaign / "schedule.json", jobs)
        pilot.write(campaign / "server_manifest.json", {
            **manifest_data, "fingerprint": "a" * 64 if manifest_match else "b" * 64,
        })

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            assert address == ("127.0.0.1", 18765)
            return 1

    def spawn(command, **kwargs):
        commands.append((command, kwargs))
        if "serve" in command:
            return server
        if "prepare" in command:
            campaign.mkdir()
            pilot.write(campaign / "schedule.json", jobs)
            return SimpleNamespace(wait=lambda **k: 0)
        if "verify" in command:
            return SimpleNamespace(wait=lambda **k: 0)
        folder = campaign / f"masked/runs/cell-{1 if resume else 0}"
        folder.mkdir(parents=True)
        pilot.write(folder / "run.json", {"status": "completed" if exit_code == 0 else "failed"})
        return controller

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if "prepare" in command:
            campaign.mkdir()
            pilot.write(campaign / "schedule.json", jobs)

    monkeypatch.setattr("socket.socket", Socket)
    monkeypatch.setattr(pilot.signal, "signal", lambda *a: None)
    monkeypatch.setattr(pilot.subprocess, "Popen", spawn)
    monkeypatch.setattr(pilot.subprocess, "run", run)
    monkeypatch.setattr(pilot, "stop_owned", stopped.append)
    manifest = json.dumps(manifest_data).encode()
    monkeypatch.setattr(pilot.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(manifest))
    pilot.supervise(runtime, resume=resume)
    if not manifest_match:
        assert stopped == [None, server]
        result = json.loads((runtime / "status.json").read_text())
        assert result["status"] == "failed"
        assert "differs from the frozen serving manifest" in result["error"]
        assert len(commands) == 2  # Verify and serve only; no controller or prepare.
        assert json.loads((campaign / "server_manifest.json").read_text())["fingerprint"] == "b" * 64
        return
    assert stopped == [controller, server]
    result = json.loads((runtime / "status.json").read_text())
    assert result["status"] == expected_status and result["server_stopped"] is True
    if resume:
        verify, serve, execute, summarize = [c for c, _ in commands]
        assert "verify" in verify
        assert not any("prepare" in c for c, _ in commands)
    else:
        serve, prepare, execute, summarize = [c for c, _ in commands]
        assert prepare[prepare.index("--base-url") + 1] == "http://127.0.0.1:18765/v1"
        assert prepare[prepare.index("--max-tokens") + 1] == "100000"
        assert prepare[prepare.index("--request-timeout") + 1] == "21600"
        assert "--smoke" in prepare
    assert serve[serve.index("--default-max-new-tokens") + 1] == "100000"
    assert serve[serve.index("--max-context-tokens") + 1] == "131072"
    assert serve[serve.index("--prefill-chunk-size") + 1] == "1024"
    assert serve[serve.index("--port") + 1] == "18765"
    assert execute[-2:] == ["--limit", "1"] and "summarize" in summarize
    assert all(kw["env"]["CUDA_VISIBLE_DEVICES"] == "2,3" for _, kw in commands)
    assert sum(r["status"] == "queued" for r in result["runs"]) == (6 if resume else 7)


def test_pause_requires_timezone_and_uses_eastern_offset():
    pilot = launcher()
    assert pilot.pause_time("2026-09-17T08:55:00-04:00").isoformat() == "2026-09-17T12:55:00+00:00"
    with pytest.raises(ValueError, match="timezone"):
        pilot.pause_time("2026-09-17T08:55:00")


def test_deadline_interrupts_blocking_work_and_releases_owned_process():
    """Exercise the real alarm and process-group cleanup without a model/GPU."""
    import datetime as dt
    import signal
    import subprocess
    import sys
    import time

    pilot = launcher()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    original = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    try:
        pilot.arm_pause((dt.datetime.now(dt.UTC) + dt.timedelta(seconds=0.2)).isoformat())
        with pytest.raises(pilot.ScheduledPause):
            time.sleep(10)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
        pilot.stop_owned(process, immediate=True)
    assert process.poll() is not None
    assert time.monotonic() - started < 5


def test_pause_preserves_campaign_and_is_not_failure(tmp_path, monkeypatch):
    pilot = launcher()
    campaign = tmp_path / "campaign"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    call_dir = campaign / "masked/runs/partial/calls"
    call_dir.mkdir(parents=True)
    (call_dir / "saved.json").write_text('{"saved": true}')
    pilot.write(campaign / "schedule.json", [{"view": "masked", "run_id": "partial"}])
    pilot.write(campaign / "freeze.json", {"frozen": True})
    pilot.write(runtime / "launch.json", dict(
        campaign=str(campaign), gpus="0,1", max_tokens=100000, run_limit=8,
        max_hours=48, pause_at="2026-09-17T08:55:00-04:00",
    ))
    stopped = []
    monkeypatch.setattr(pilot.signal, "signal", lambda *a: None)
    monkeypatch.setattr(pilot.signal, "setitimer", lambda *a: None)
    monkeypatch.setattr(pilot, "stop_owned", lambda p, **k: stopped.append((p, k)))

    def pause(value):
        raise pilot.ScheduledPause("scheduled cutoff")

    monkeypatch.setattr(pilot, "arm_pause", pause)
    pilot.supervise(runtime)
    state = json.loads((runtime / "status.json").read_text())
    assert state["status"] == "paused" and state["resume_available"]
    assert state["runs"][0]["status"] == "paused"
    assert state["runs"][0]["saved_calls"] == 1
    assert all(k == {"immediate": True} for _, k in stopped)
    assert not (campaign / "halt.json").exists()
    assert not (call_dir.parent / "run.json").exists()
    assert json.loads((call_dir / "saved.json").read_text()) == {"saved": True}


def test_resume_uses_frozen_launcher_and_new_cutoff(tmp_path, monkeypatch):
    pilot = launcher()
    pilot.write(tmp_path / "launch.json", {"campaign": "original-campaign"})
    commands = []
    monkeypatch.setattr(pilot.subprocess, "Popen",
                        lambda command, **kw: commands.append(command) or SimpleNamespace(pid=123))
    pilot.spawn_worker(tmp_path, resume=True, pause_at="2026-09-18T08:55:00-04:00")
    command = commands[0]
    assert command[1] == str(tmp_path / "source/scripts/expected_surprising/run_local_caa_pilot.py")
    assert "--resume-existing" in command
    assert command[-2:] == ["--pause-at", "2026-09-18T08:55:00-04:00"]


def test_live_supervisor_cutoff_stops_children_and_keeps_saved_calls(tmp_path, monkeypatch):
    """Real timer and real child processes; only model/HTTP/science are substituted."""
    import datetime as dt
    import signal
    import subprocess
    import sys

    pilot = launcher()
    runtime, campaign = tmp_path / "runtime", tmp_path / "campaign"
    (runtime / "source").mkdir(parents=True)
    pilot.write(runtime / "launch.json", dict(
        campaign=str(campaign), model="fixture", iterations=25, max_hours=48,
        max_tokens=100000, request_timeout=21600, run_limit=8, gpus="0,1", port=18765,
        inter_gpu_transfer="cpu", prefill_chunk_size=2048, max_context_tokens=None,
        named="named", masked="masked", base_config="base",
        pause_at=(dt.datetime.now(dt.UTC) + dt.timedelta(seconds=0.5)).isoformat(),
    ))
    original_popen = subprocess.Popen
    children = []
    originals = {s: signal.getsignal(s) for s in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]}

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    def spawn(command, **kwargs):
        if "prepare" in command:
            calls = campaign / "masked/runs/partial/calls"
            calls.mkdir(parents=True)
            (calls / "saved.json").write_text('{"durable": true}')
            pilot.write(campaign / "freeze.json", {"frozen": True})
            pilot.write(campaign / "schedule.json", [{"view": "masked", "run_id": "partial"}])
            return SimpleNamespace(wait=lambda **kw: 0)
        child = original_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("socket.socket", Socket)
    monkeypatch.setattr(pilot.subprocess, "Popen", spawn)
    manifest = json.dumps({"resolved_device_map": {"layer0": "0"}, "fingerprint": "a" * 64}).encode()
    monkeypatch.setattr(pilot.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(manifest))
    try:
        pilot.supervise(runtime)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in originals.items():
            signal.signal(signum, handler)
        for child in children:
            pilot.stop_owned(child, immediate=True)
    state = json.loads((runtime / "status.json").read_text())
    assert len(children) == 2 and all(child.poll() is not None for child in children)
    assert state["status"] == "paused" and state["server_stopped"]
    assert state["resume_available"] and not state["pause_timer_armed"]
    assert state["runs"][0]["saved_calls"] == 1
    assert not (campaign / "halt.json").exists()
