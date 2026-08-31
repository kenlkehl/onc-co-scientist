from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "nsclc_coordination"
sys.path.insert(0, str(EXPERIMENT))

from prepare_experiment import (  # noqa: E402
    EXPECTED_HASHES,
    EXPECTED_SHAPE,
    MASKED_FORBIDDEN_TERMS,
    _prepare_public_workspaces,
    _sha256,
    _source_paths,
)


def test_pinned_nsclc_parity_and_public_gold_isolation(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    paths = _source_paths(repo)
    assert {label: _sha256(path) for label, path in paths.items()} == EXPECTED_HASHES
    mapping = json.loads(paths["column_mapping"].read_text(encoding="utf-8"))

    workspaces = _prepare_public_workspaces(
        paths=paths,
        public_root=tmp_path / "public",
        mapping=mapping,
    )

    for workspace in workspaces.values():
        assert {item.name for item in workspace.iterdir()} == {
            "dataset.parquet",
            "dataset_description.md",
        }
    masked_text = (workspaces["masked"] / "dataset_description.md").read_text(
        encoding="utf-8"
    ).lower()
    assert not any(term in masked_text for term in MASKED_FORBIDDEN_TERMS)
    assert "manifest.json" not in masked_text
    assert "column_mapping.json" not in masked_text

    named_text = (workspaces["named"] / "dataset_description.md").read_text(
        encoding="utf-8"
    )
    restored_masked_text = (
        workspaces["masked"] / "dataset_description.md"
    ).read_text(encoding="utf-8")
    for named, masked in sorted(mapping.items(), key=lambda item: len(item[1]), reverse=True):
        restored_masked_text = restored_masked_text.replace(masked, named)
    assert restored_masked_text == named_text
    assert EXPECTED_SHAPE == (50_000, 35)


def test_tracked_grid_template_locks_calls_timeouts_and_parallelism() -> None:
    template = yaml.safe_load(
        (EXPERIMENT / "nsclc_semantic_workflow_grid.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert template["iteration_policy"] == {"iterations": 20, "completion_mode": "fixed"}
    assert template["replicates"] == 5
    assert template["max_parallel"] == 2
    assert template["budget"] == {
        "max_agent_calls": 240,
        "max_runtime_seconds_per_call": 3600,
    }
    args = template["models"][0]["extra_args"]
    assert args[args.index("--timeout-seconds") + 1] == "3580"
    assert template["models"][0]["model_id"] == "gpt-5.6-luna"
    assert template["models"][0]["reasoning_effort"] == "low"
