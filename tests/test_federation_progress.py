"""The observer counts nested central/site receipts while preserving held arms."""

import json

from scripts.expected_surprising.monitor_federated_grid import refresh


def test_nested_usage_and_held_native_cells(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    write("release_policy.json", {"released_models": ["sol_medium"]})
    write(
        "grid.json",
        [
            dict(model_profile="sol_medium", condition="named", site_count=2, run_id="sol"),
            dict(model_profile="astra_medium", condition="masked", site_count=4, run_id="astra"),
        ],
    )
    write("named/runs/sol/calls/central/one.json", {"result": {"usage": {"output_tokens": 7}}})
    write("named/runs/sol/calls/site_1/two.json", {"result": {"usage": {}, "error": "error"}})
    write("biomni/plan.json", [dict(condition="masked", sites=4, run_path="masked-n4/runs/test")])
    refresh(tmp_path)
    result = json.loads((tmp_path / "live_progress.json").read_text())
    assert result["totals"]["running"] == 1
    assert result["totals"]["held"] == 2
    assert result["totals"]["tokens"] == 7
    assert result["totals"]["calls"] == 2 and result["totals"]["unknown"] == 1
    assert not (tmp_path / "biomni/masked-n4").exists()
    assert "biomni_native" in (tmp_path / "LIVE_PROGRESS.md").read_text()
    # Status can refresh independently using the last completed usage scan.
    refresh(
        tmp_path,
        audits={("sol_medium", "named", 2): {"calls": 9, "tokens": 30}},
        audit_stamp="previous-scan",
    )
    result = json.loads((tmp_path / "live_progress.json").read_text())
    assert result["totals"]["held"] == 2 and result["totals"]["calls"] == 9
    assert result["usage_updated_at"] == "previous-scan"


def test_azure_reservations_are_queued_and_transferred_results_count_once(tmp_path):
    azure = tmp_path / "azure"
    (tmp_path / "named/runs").mkdir(parents=True)
    (tmp_path / "named/runs/sol").write_text('{"status":"reserved_for_azure"}')
    (tmp_path / "grid.json").write_text(
        json.dumps(
            [dict(model_profile="sol_medium", condition="named", site_count=2, run_id="sol")]
        )
    )
    (tmp_path / "release_policy.json").write_text('{"released_models":["sol_medium"]}')
    (tmp_path / "azure_transition.json").write_text(
        json.dumps(
            dict(
                status="draining",
                target_root=str(azure),
                active=[],
                queued=[dict(condition="named", run_id="sol")],
            )
        )
    )
    refresh(tmp_path)
    result = json.loads((tmp_path / "live_progress.json").read_text())
    assert result["totals"]["queued"] == 1
    assert result["totals"].get("running", 0) == 0
    run = azure / "named/runs/sol"
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"status":"completed"}')
    refresh(tmp_path)
    result = json.loads((tmp_path / "live_progress.json").read_text())
    assert result["totals"]["completed"] == 1
    assert result["totals"].get("queued", 0) == 0
