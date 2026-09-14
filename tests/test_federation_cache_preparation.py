"""Preparation is held, immutable, and has no credential/inference side effects."""

import hashlib
import json

import pytest

from onc_co_scientist.providers.azure_budget import atomic_json
from scripts.expected_surprising.federation_cache_fix import MODELS, prepare, retail_rates


def test_prepare_held_and_never_overwrite_spending_ledger(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    atomic_json(root / "release_policy.json", {"released_models": [], "held_models": list(MODELS)})
    (root / "science.txt").write_text("unchanged science")
    original = (root / "science.txt").read_bytes()
    atomic_json(
        root / "frozen_manifest.json",
        {
            "hashes": {
                "science.txt": hashlib.sha256(original).hexdigest(),
            }
        },
    )
    for m in MODELS:
        (root / "control" / m).mkdir(parents=True)
        atomic_json(
            root / "selections" / f"{m}.json",
            [{"condition": "named", "run_id": f"{m}-{i}"} for i in range(10)],
        )
    items = []
    for model in ("sol", "terra", "luna"):
        for band in ("ShortCo", "LongCo"):
            for meter in ("Inp", "Cd Inp", "Cd Wr", "Opt"):
                items.append(
                    {
                        "meterName": f"5.6 {model} {band} {meter} Std Gl 1M Tokens",
                        "armRegionName": "eastus2",
                        "type": "Consumption",
                        "currencyCode": "USD",
                        "unitOfMeasure": "1M",
                        "retailPrice": 1,
                    }
                )
    prices = tmp_path / "prices.json"
    atomic_json(prices, {"Items": items})
    target = prepare(root, prices)
    policy = json.loads((target / "spend_policy.json").read_text())
    assert policy["enabled"] is False and policy["additional_budget_usd"] == 0
    assert policy["rates"] == retail_rates(prices)
    state = (target / "spend_state.json").read_bytes()
    assert json.loads(state)["spent_micro_usd"] == 0
    assert json.loads(state)["attempts"] == {}
    assert (root / "science.txt").read_bytes() == original
    manifest = json.loads((target / "manifest.json").read_text())
    for name, value in manifest["source_hashes"].items():
        assert hashlib.sha256((target / "source" / name).read_bytes()).hexdigest() == value
    with pytest.raises(FileExistsError):
        prepare(root, prices)
    assert (target / "spend_state.json").read_bytes() == state
    atomic_json(root / "release_policy.json", {"released_models": ["sol_medium"]})
    with pytest.raises(ValueError, match="Pause"):
        prepare(root, prices)
    items.pop()
    atomic_json(prices, {"Items": items})
    with pytest.raises(ValueError, match="retail meter"):
        retail_rates(prices)


def test_driver_distinguishes_pending_active_and_budget_paused():
    from onc_co_scientist.providers.azure_budget import ExperimentPaused
    from scripts.expected_surprising.federation_cache_fix import drive

    selection = [{"run_id": str(n), "condition": "named"} for n in range(4)]
    snapshots = []

    def run(row):
        if row["run_id"] == "1":
            raise ExperimentPaused("cap")
        return {"status": "completed"}

    result = drive(
        selection,
        1,
        run,
        lambda active, queued, finished: snapshots.append((active, queued, finished)),
        ExperimentPaused,
    )
    assert all(len(active) <= 1 for active, _, _ in snapshots)
    assert snapshots[0][1] == 3 and snapshots[-1][:2] == ([], 0)
    assert [r["status"] for r in result] == ["completed", "paused", "completed", "completed"]
