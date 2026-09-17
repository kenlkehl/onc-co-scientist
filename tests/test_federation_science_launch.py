"""Admission accounting and completion of the explicitly bounded federation batch."""

import importlib.util
from pathlib import Path

import pytest

from onc_co_scientist.providers.azure_budget import atomic_json

MODULE = Path(__file__).parents[1] / "scripts/expected_surprising/launch_federation_science_v2.py"
spec = importlib.util.spec_from_file_location("science_launch", MODULE)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_remaining_allocation_does_not_reset_previous_costs():
    policy = {"enabled": False, "additional_budget_usd": 984.189870}
    state = {"spent_micro_usd": 87425906, "attempts": {"paid": {"status": "settled"}}}
    assert launch.remaining_micro(policy, state) == 896763964
    assert 896763964 + 92604911 + 10631125 == 1000000000


@pytest.mark.parametrize("change", ["enabled", "pending", "unknown", "reserved", "spent"])
def test_preparation_refuses_conflicting_or_unfunded_predecessor(change):
    policy = {"enabled": False, "additional_budget_usd": 100}
    state = {"spent_micro_usd": 10, "attempts": {}}
    if change == "enabled":
        policy["enabled"] = True
    elif change == "pending":
        policy["diagnostic_pending_allocation_usd"] = 30
    elif change == "spent":
        state["spent_micro_usd"] = 100000000
    else:
        state["attempts"]["attempt"] = {"status": change}
    with pytest.raises(ValueError):
        launch.remaining_micro(policy, state)


def test_observer_closes_release_and_spending_at_initial30_completion(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    (old / "LIVE_PROGRESS.md").write_text("old results")
    root = tmp_path / "new"
    (root / "control").mkdir(parents=True)
    atomic_json(
        root / "frozen_manifest.json",
        {
            "predecessor": str(old),
            "allocated_micro_usd": 896763964,
            "previous_recorded_additional_micro_usd": 92604911,
            "previous_unknown_micro_usd": 10631125,
        },
    )
    atomic_json(root / "control/spend_policy.json", {"enabled": True})
    atomic_json(root / "control/spend_state.json", {"spent_micro_usd": 0, "attempts": {}})
    processes = []
    grid = []
    for model in launch.MODELS:
        finished = []
        for i in range(10):
            rid = f"{model}-{i}"
            finished.append({"condition": "named", "run_id": rid, "status": "completed"})
            grid.append(
                {
                    "condition": "named",
                    "run_id": rid,
                    "model_profile": model,
                    "workflow_id": "persistent",
                    "site_count": 2,
                    "semantic_condition": "expected",
                }
            )
        atomic_json(
            root / "control" / model / "execution.json",
            {"status": "completed", "active": [], "finished": finished},
        )
        processes.append({"model": model, "pid": 0, "start_ticks": "0"})
    atomic_json(root / "control/processes.json", {"launch_complete": True, "drivers": processes})
    atomic_json(root / "grid.json", grid)
    monkeypatch.setattr(launch, "alive", lambda p: False)
    launch.observe(root)
    assert not launch.read(root / "control/spend_policy.json")["enabled"]
    assert launch.read(root / "release_policy.json")["released_models"] == []
    assert launch.read(root / "control/review_ready.json")["totals"] == {"completed": 30}
    assert "Completed 30" in (old / "LIVE_PROGRESS.md").read_text()
    assert (root / "PREVIOUS_LIVE_PROGRESS.md").read_text() == "old results"
