"""Freeze a masked twin of the six-model clinical workflow benchmark."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from onc_co_scientist.expected_surprising.masking import mask_package
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-for-vllm-control", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source_grid.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    repo = Path(__file__).resolve().parents[2]
    original_frozen = json.loads((source / "frozen_manifest.json").read_text())
    for name, expected in {
        **original_frozen["input_hashes"],
        "config.yaml": original_frozen["config_sha256"],
    }.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    raw = yaml.safe_load((source / "config.yaml").read_text())
    out.mkdir(parents=True)
    audit = mask_package(Path(raw["expected_surprising"]["root"]), out / "input_data")
    raw.update(
        experiment_id="clinical_10pct_masked_six_models_20260910",
        description=(
            "Masked twin: same rows, model/workflow grid and budgets; "
            "repaired harness for all models."
        ),
        output_root=str(out),
    )
    raw["expected_surprising"].update(
        root=str(out / "input_data"),
        peer_failure_policy="chair_with_available",
        stage_failure_policy="retain_scientific_scores",
    )
    (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    policy = dict(
        wait_for_vllm_control=str(args.wait_for_vllm_control.resolve()),
        policy=(
            "Codex cells may start immediately; vLLM cells wait for the unmasked "
            "continuation to complete. No other run is stopped or modified."
        ),
        max_parallel=raw["max_parallel"],
    )
    (out / "launch_policy.json").write_text(json.dumps(policy, indent=2))
    shutil.copytree(
        repo / "src", out / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("run_masked_grid.py", "monitor_workflow_grid.py", "prepare_masked_grid.py"):
        shutil.copyfile(repo / "scripts/expected_surprising" / name, out / "source" / name)
    run_experiment(load_experiment_spec(out / "config.yaml"), dry_run=True)
    plans = json.loads((out / "plan.json").read_text())
    from collections import Counter

    cells = Counter((p["model_profile"], p["workflow_id"], p["semantic_condition"]) for p in plans)
    assert len(plans) == 360 and len(cells) == 36 and set(cells.values()) == {10}

    def hashes(folder):
        return {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }

    def digest(name):
        return hashlib.sha256((out / name).read_bytes()).hexdigest()

    frozen = dict(
        frozen_at=datetime.now(UTC).isoformat(),
        original_grid=str(source),
        source_hashes=hashes(out / "source"),
        input_hashes=hashes(out / "input_data"),
        config_sha256=digest("config.yaml"),
        launch_policy_sha256=digest("launch_policy.json"),
        plan_sha256=digest("plan.json"),
        masking=audit,
        requested_reasoning="medium",
        requested_service_tier="default",
    )
    (out / "frozen_manifest.json").write_text(json.dumps(frozen, indent=2))
    print(
        json.dumps(
            {"root": str(out), "runs": len(plans), "cells": len(cells), "masking": audit}, indent=2
        )
    )


if __name__ == "__main__":
    main()
