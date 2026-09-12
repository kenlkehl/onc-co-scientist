"""Deadline timing, process isolation, and exact replay across transport changes."""

import fcntl
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from scripts.expected_surprising import deadline_azure_federation as cutover


def test_deadline_check_does_nothing_early(tmp_path, monkeypatch):
    source = tmp_path / "source"
    cutover.write(source / "control/sol/execution.json", {"active": [{}]})
    plan = tmp_path / "control/plan.json"
    cutover.write(
        plan,
        dict(
            source_root=str(source),
            azure_root=str(tmp_path / "azure"),
            models=["sol"],
            deadline=(datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("An early deadline check must not stop processes or make requests")

    monkeypatch.setattr(cutover, "stop_personal", forbidden)
    monkeypatch.setattr(cutover.subprocess, "Popen", forbidden)
    assert cutover.check(plan)["status"] == "waiting_for_deadline"
    with pytest.raises(RuntimeError, match="Resume held"):
        cutover.worker(plan, "sol")


def test_invisible_but_locked_driver_is_not_assumed_stopped(tmp_path, monkeypatch):
    lock_path = tmp_path / "driver.lock"
    monkeypatch.setattr(cutover, "process_info", lambda pid: None)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="not visible but still locked"):
            cutover.stop_personal(dict(drivers=[dict(pid=123, lock_path=str(lock_path))]), tmp_path)
    assert not (tmp_path / "personal_stopped.json").exists()


def test_reused_pid_is_never_signaled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cutover,
        "process_info",
        lambda pid: dict(
            pid=pid,
            state="S",
            start_ticks="new",
            command=["unrelated"],
            ppid=1,
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reused PID must not be signaled")

    monkeypatch.setattr(cutover.os, "kill", forbidden)
    with pytest.raises(RuntimeError, match="reused PID"):
        cutover.stop_personal(
            dict(drivers=[dict(pid=123, start_ticks="old", command=["worker"])]), tmp_path
        )


def test_stop_only_driver_and_descendants_including_separate_sessions(tmp_path):
    child_file = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
        "start_new_session=True); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", code, str(child_file)], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for _ in range(100):
            if child_file.exists():
                break
            time.sleep(0.02)
        child_pid = int(child_file.read_text())
        info = cutover.process_info(parent.pid)
        cutover.stop_personal(dict(drivers=[info]), tmp_path / "control")
        assert parent.wait(timeout=5) < 0
        assert not cutover.live(cutover.process_info(child_pid))
        assert unrelated.poll() is None
        stopped = cutover.read(tmp_path / "control/personal_stopped.json")["processes"]
        assert {p["pid"] for p in stopped} == {parent.pid, child_pid}
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        unrelated.kill()
        unrelated.wait()


def test_federation_resume_with_azure_override_preserves_provenance_and_cached_calls(
    tmp_path, monkeypatch
):
    from onc_co_scientist.expected_surprising import experiment
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans
    from tests.test_expected_surprising_experiment import config, providers
    from tests.test_expected_surprising_federation import FederatedScientist

    _, pairs, raw, path = config(tmp_path)
    raw["federation"] = {"site_counts": [2]}
    raw["workflows"] = [raw["workflows"][0]]
    raw["budget"]["max_agent_calls"] = 120
    raw["models"][0]["provider_config"].update(kind="codex_cli", timeout_s=120)
    path.write_text(yaml.safe_dump(raw))
    grid = load_experiment_spec(path)
    plan = build_run_plans(grid)[0]

    class Interrupted(FederatedScientist):
        def chat(self, messages, **kwargs):
            if len(self.messages) == 8:
                raise KeyboardInterrupt("deadline during an uncommitted request")
            return super().chat(messages, **kwargs)

    providers(monkeypatch, pairs[0], Interrupted)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=False)
    root = grid.output_root / "runs" / plan.run_id
    provenance = (root / "provenance.json").read_bytes()
    cached = {
        p: p.read_bytes() for p in (root / "calls").rglob("*.json") if "requests" not in p.parts
    }
    factory = cutover.install_transport(
        experiment,
        Path("src/onc_co_scientist/providers/codex_cli.py").resolve(),
        "https://example.openai.azure.com/openai/v1",
        dict(backend="azure", reason="test deadline"),
    )
    delegate = FederatedScientist(pairs[0])
    module = sys.modules["onc_co_scientist.providers.deadline_azure_transport"]

    def azure_only(self, messages, **kwargs):
        assert self.config.backend == "azure"
        return delegate.chat(messages, **kwargs)

    monkeypatch.setattr(module.CodexCLIProvider, "chat", azure_only)
    result = experiment.run_cell(grid, plan, grid.output_root, grid.fingerprint(), resume=True)
    assert result["status"] == "completed", result.get("error")
    assert (root / "provenance.json").read_bytes() == provenance
    assert all(p.read_bytes() == original for p, original in cached.items())
    assert result["agent_calls"] == len(cached) + len(delegate.messages)
    assert len(cached) == 8
    marker = cutover.read(root / "provider_audit/azure_transport_cutover.json")
    assert marker["backend"] == "azure"
    with pytest.raises(ValueError, match="only the selected Codex"):
        factory({"kind": "vllm_openai"})
