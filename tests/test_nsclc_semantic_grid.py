from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "nsclc_coordination"
sys.path.insert(0, str(EXPERIMENT))

import freeze_provenance as freeze_module  # noqa: E402
import prepare_experiment as prepare_module  # noqa: E402
from prepare_experiment import (  # noqa: E402
    EXPECTED_HASHES,
    EXPECTED_SHAPE,
    MASKED_FORBIDDEN_TERMS,
    _locked_model_manifest,
    _prepare_public_workspaces,
    _sha256,
    _source_paths,
)


def test_active_python_preserves_virtualenv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_python = tmp_path / ".venv" / "bin" / "python"
    base_python = tmp_path / "base" / "bin" / "python3.13"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    monkeypatch.setattr(prepare_module.sys, "executable", str(venv_python))

    selected = prepare_module._active_python_executable()

    assert selected == venv_python
    assert selected.resolve() == base_python


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


def test_medium_v3_template_locks_fast_repair_reasoning_and_four_hour_ceiling() -> None:
    template = yaml.safe_load(
        (EXPERIMENT / "nsclc_semantic_workflow_grid_medium.template.yaml").read_text(
            encoding="utf-8"
        )
    )

    model = _locked_model_manifest(template)

    assert template["experiment_id"] == (
        "nsclc-semantic-workflow-grid-luna-medium-fast-repair-v3-20x5"
    )
    assert model == {
        "profile": "codex-luna-medium-fast-repair",
        "id": "gpt-5.6-luna",
        "adapter": "cli-json",
        "reasoning_effort": "medium",
        "service_tier": "fast",
        "max_contract_repairs": 2,
        "adapter_timeout_seconds": 14_380,
        "harness_timeout_seconds": 14_400,
    }
    assert template["iteration_policy"] == {"iterations": 20, "completion_mode": "fixed"}
    assert template["replicates"] == 5
    assert template["max_parallel"] == 6

    mismatched = json.loads(json.dumps(template))
    mismatched["models"][0]["reasoning_effort"] = "low"
    with pytest.raises(ValueError, match="does not match"):
        _locked_model_manifest(mismatched)

    invalid_tier = json.loads(json.dumps(template))
    args = invalid_tier["models"][0]["extra_args"]
    args[args.index("--service-tier") + 1] = "slow"
    with pytest.raises(ValueError, match="service tier"):
        _locked_model_manifest(invalid_tier)

    invalid_repairs = json.loads(json.dumps(template))
    args = invalid_repairs["models"][0]["extra_args"]
    args[args.index("--max-contract-repairs") + 1] = "-1"
    with pytest.raises(ValueError, match="repair limit"):
        _locked_model_manifest(invalid_repairs)


def test_provenance_freeze_is_hash_locked_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results"
    output.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "freeze-test",
                "output_root": str(output),
                "schedule_seed": 7,
            }
        ),
        encoding="utf-8",
    )
    machine = tmp_path / "machine.yaml"
    machine.write_text(
        yaml.safe_dump({"git": {"clean": True, "commit": "abc"}}), encoding="utf-8"
    )
    preparation = tmp_path / "preparation.json"
    preparation.write_text("{}\n", encoding="utf-8")
    template = tmp_path / "medium.template.yaml"
    template.write_text("model: medium\n", encoding="utf-8")
    protocol = tmp_path / "protocol.medium.md"
    protocol.write_text("# Medium protocol\n", encoding="utf-8")
    for name, payload in {
        "schedule.json": {"spec_fingerprint": "fingerprint"},
        "plan.json": [],
        "resolved_spec.json": {},
        "private_evaluation_index.json": {},
    }.items():
        (output / name).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(freeze_module, "_git_status", lambda repo: "")
    monkeypatch.setattr(freeze_module, "_git_commit", lambda repo: "abc")

    first = freeze_module.freeze(
        repo=tmp_path,
        config=config,
        output_root=output,
        machine_manifest=machine,
        preparation_manifest=preparation,
        template=template,
        protocol=protocol,
    )
    first_payload = first.read_text(encoding="utf-8")
    second = freeze_module.freeze(
        repo=tmp_path,
        config=config,
        output_root=output,
        machine_manifest=machine,
        preparation_manifest=preparation,
        template=template,
        protocol=protocol,
    )

    assert second == first
    assert second.read_text(encoding="utf-8") == first_payload
    assert (output / "provenance" / "scorer.py").is_file()
    assert (output / "provenance" / "template.yaml").read_text() == "model: medium\n"
    assert (output / "provenance" / "protocol.md").read_text() == "# Medium protocol\n"
    (output / "schedule.json").write_text(
        json.dumps({"spec_fingerprint": "changed"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Frozen provenance differs"):
        freeze_module.freeze(
            repo=tmp_path,
            config=config,
            output_root=output,
            machine_manifest=machine,
            preparation_manifest=preparation,
            template=template,
            protocol=protocol,
        )
