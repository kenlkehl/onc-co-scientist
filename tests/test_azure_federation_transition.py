"""A reserved identity cannot spend personal quota; Azure waits for every active run."""

import json
from types import SimpleNamespace

import pytest

from onc_co_scientist.expected_surprising.experiment import run_cell
from scripts.expected_surprising.launch_azure_federation import launch
from scripts.expected_surprising.run_federated_grid import verify_predecessor


def test_legacy_reservation_blocks_before_provider_initialization(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "queued").write_text('{"status":"reserved_for_azure"}')
    # No spec is supplied: admission must fail before any spec/provider access.
    with pytest.raises(NotADirectoryError):
        run_cell(None, SimpleNamespace(run_id="queued"), tmp_path, "unused", resume=False)
    assert not (tmp_path / "archive").exists()
    assert list(runs.iterdir()) == [runs / "queued"]


def test_azure_gate_requires_all_drivers_and_every_original_active_identity(tmp_path):
    transition = dict(
        source_root=str(tmp_path),
        models=["sol", "terra"],
        active=[dict(condition="named", run_id="active")],
        queued=[dict(condition="named", run_id="queued")],
    )
    (tmp_path / "named/runs").mkdir(parents=True)
    reservation = tmp_path / "named/runs/queued"
    reservation.write_text('{"status":"reserved_for_azure"}')

    def status(model, **kwargs):
        p = tmp_path / "control" / model / "execution.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(kwargs))

    status("sol", active=[], queued_runs=0, completed_at="done", finished=[])
    status("terra", active=[{}], queued_runs=0)
    with pytest.raises(RuntimeError, match="not drained"):
        verify_predecessor(transition)
    status("terra", active=[], queued_runs=0, completed_at="done", finished=[])
    with pytest.raises(RuntimeError, match="lack terminal records"):
        verify_predecessor(transition)
    status("sol", active=[], queued_runs=0, completed_at="done", finished=transition["active"])
    verify_predecessor(transition)
    reservation.unlink()
    reservation.mkdir()
    with pytest.raises(RuntimeError, match="reservation changed"):
        verify_predecessor(transition)


def test_launcher_does_not_authenticate_or_spawn_before_drain(tmp_path, monkeypatch):
    source = tmp_path / "original"
    (source / "control/sol").mkdir(parents=True)
    (source / "control/sol/execution.json").write_text('{"active":[{}]}')
    root = tmp_path / "azure"
    root.mkdir()
    (root / "frozen_manifest.json").write_text(
        json.dumps(
            dict(
                azure_transition=dict(
                    source_root=str(source), models=["sol"], active=[], queued=[]
                ),
                hashes={},
            )
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("No auth or process launch is allowed before drain")

    monkeypatch.setattr(
        "scripts.expected_surprising.launch_azure_federation.subprocess.Popen", forbidden
    )
    monkeypatch.setattr(
        "scripts.expected_surprising.launch_azure_federation.CodexCLIProvider", forbidden
    )
    assert launch(root)["status"] == "waiting_for_original_runs"


def test_launcher_dispatches_once_and_preserves_scope(tmp_path, monkeypatch):
    from scripts.expected_surprising import launch_azure_federation as module

    source = tmp_path / "original"
    (source / "control/sol").mkdir(parents=True)
    (source / "control/sol/execution.json").write_text(
        json.dumps(
            dict(
                completed_at="done",
                active=[],
                queued_runs=0,
                finished=[],
            )
        )
    )
    (source / "azure_transition.json").write_text('{"status":"draining"}')
    root = tmp_path / "azure"
    root.mkdir()
    (root / "frozen_manifest.json").write_text(
        json.dumps(
            dict(
                azure_transition=dict(
                    source_root=str(source), models=["sol"], active=[], queued=[]
                ),
                hashes={},
            )
        )
    )
    (root / "release_policy.json").write_text('{"released_models":["sol"]}')
    (root / "azure_preflight.json").write_text(
        json.dumps(
            dict(
                checks=[
                    dict(
                        metrics=dict(
                            azure_endpoint="https://example.openai.azure.com/openai/v1",
                        )
                    )
                ]
            )
        )
    )
    monkeypatch.setattr(module.CodexCLIProvider, "environment", lambda self: {})
    calls = []

    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    assert launch(root)["status"] == "azure_dispatched"
    assert launch(root)["models"][0]["status"] == "already_dispatched"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-4:] == ["--model", "sol", "--workers", "10"]
    assert kwargs["env"]["PYTHONPATH"] == str(root / "source/src")
    assert "OPENAI_API_KEY" not in kwargs["env"]
    (root / "release_policy.json").write_text('{"released_models":["sol","astra"]}')
    with pytest.raises(ValueError, match="release scope changed"):
        launch(root)
    assert len(calls) == 1
