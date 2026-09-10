import json

import pytest

from onc_co_scientist.expected_surprising.packaging import sha256
from onc_co_scientist.external.followup import publish_report, terminal_runs
from onc_co_scientist.external.transport import write_json


def campaign(root, view):
    plans = []
    for version in ("expected", "surprising"):
        for replicate in range(1, 11):
            plans.append({"task_id": version, "replicate": replicate})
            run_id = f"{version}__biomni_native__r{replicate:03d}"
            folder = root / "runs" / run_id
            failed = replicate == 1
            report = {
                "run_id": run_id,
                "pair_id": "pair",
                "version": version,
                "dataset_sha256": f"{view}-{version}",
                "source_dataset_sha256": f"named-{version}",
                "dataset_view": view,
                "model": "local",
                "harness": "external-native",
                "versions": {"scoring": "profile-3.0.0"},
                "protocol_errors": ["early stop"] if failed else [],
                "scores": {
                    "D": 0 if failed else 50,
                    "E": 20,
                    "B": None,
                    "discovery": {"exact": {"R": 0.5, "Q": None}},
                },
                "usage": {"input_tokens": 100, "output_tokens": 20, "cost_usd": None},
                "completed_rounds": 11 if failed else 25,
                "responsiveness": {"accuracy": None, "events": []},
            }
            write_json(folder / "report.json", report)
            write_json(
                folder / "run.json",
                {
                    "run_id": run_id,
                    "status": "failed" if failed else "completed",
                    "report_sha256": sha256(folder / "report.json"),
                    "error": "early stop" if failed else None,
                },
            )
    write_json(root / "plan.json", plans)


def test_complete_named_masked_report_preserves_nulls_and_failures(tmp_path):
    named, masked, out = [tmp_path / x for x in ("named", "masked", "report")]
    campaign(named, "named")
    campaign(masked, "masked")
    assert terminal_runs(named)[0]
    result = publish_report(named, masked, out)
    assert len(result["runs"]) == 40
    assert len(result["conditions"]) == 4
    for group in result["comparison"]["conditions"]:
        assert group["runs"] == 10 and group["failures"] == 1
        assert group["P"] == {"mean": None, "available_n": 0, "missing_n": 10}
        assert group["F1_star"]["mean"] == 45
    for group in result["conditions"]:
        assert group["metrics"]["usage.cost_usd"]["mean"] is None
    assert "early stop" in (out / "biomni_report.md").read_text()
    assert json.loads((out / "biomni_report.json").read_text())["runs"] == result["runs"]
    path = next(masked.glob("runs/*/report.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash changed"):
        publish_report(named, masked, out)


def test_unfinished_campaign_cannot_publish(tmp_path):
    campaign(tmp_path / "named", "named")
    campaign(tmp_path / "masked", "masked")
    next((tmp_path / "masked").glob("runs/*/run.json")).unlink()
    assert not terminal_runs(tmp_path / "masked")[0]
    with pytest.raises(ValueError, match="not terminal"):
        publish_report(tmp_path / "named", tmp_path / "masked", tmp_path / "out")
