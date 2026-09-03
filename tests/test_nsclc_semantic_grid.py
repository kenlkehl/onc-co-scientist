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
import prepare_vllm_experiment as prepare_vllm_module  # noqa: E402
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


def test_medium_v6_template_locks_resilient_fast_reasoning_and_eight_hour_ceiling() -> None:
    template = yaml.safe_load(
        (EXPERIMENT / "nsclc_semantic_workflow_grid_medium.template.yaml").read_text(
            encoding="utf-8"
        )
    )

    model = _locked_model_manifest(template)

    assert template["experiment_id"] == (
        "nsclc-semantic-workflow-grid-luna-medium-fast-resilient-localenv-v6-20x5"
    )
    assert model == {
        "profile": "codex-luna-medium-fast-resilient-localenv-v6",
        "id": "gpt-5.6-luna",
        "adapter": "cli-json",
        "reasoning_effort": "medium",
        "service_tier": "fast",
        "max_contract_repairs": 2,
        "max_runtime_retries": 24,
        "runtime_retry_initial_seconds": 30.0,
        "runtime_retry_max_seconds": 900.0,
        "runtime_retry_budget_seconds": 21_600.0,
        "transport_circuit_failure_threshold": 2,
        "transport_circuit_cooldown_seconds": 300.0,
        "adapter_timeout_seconds": 28_780,
        "harness_timeout_seconds": 28_800,
    }
    assert template["iteration_policy"] == {"iterations": 20, "completion_mode": "fixed"}
    assert template["replicates"] == 5
    assert template["max_parallel"] == 12

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

    invalid_runtime_retries = json.loads(json.dumps(template))
    args = invalid_runtime_retries["models"][0]["extra_args"]
    args[args.index("--max-runtime-retries") + 1] = "-1"
    with pytest.raises(ValueError, match="runtime-retry limit"):
        _locked_model_manifest(invalid_runtime_retries)


def test_prepare_smoke_parallelism_is_explicit() -> None:
    assert prepare_module.parse_args([]).smoke_max_parallel == 2
    assert (
        prepare_module.parse_args(["--smoke-max-parallel", "6"]).smoke_max_parallel
        == 6
    )


def test_qwen38_vllm_v2_template_locks_controller_sampling_and_isolation() -> None:
    template = yaml.safe_load(
        (
            EXPERIMENT
            / "nsclc_semantic_workflow_grid_qwen38_27b_vllm_v2.template.yaml"
        ).read_text(encoding="utf-8")
    )
    python = Path("/home/klkehl/thisenv/bin/python")
    adapter = Path("/repo/scripts/vllm_cli_json_adapter.py")
    bwrap = Path("/usr/bin/bwrap")
    base_url = "http://camus.dfci.harvard.edu:8060/v1"
    resolved = prepare_module._replace_tokens(
        template,
        {
            "__MAIN_OUTPUT_ROOT__": "/results/main",
            "__NAMED_PUBLIC_WORKSPACE__": "/public/named",
            "__MASKED_PUBLIC_WORKSPACE__": "/public/masked",
            "__NAMED_PRIVATE_MANIFEST__": "/private/named.json",
            "__MASKED_PRIVATE_MANIFEST__": "/private/masked.json",
            "__COLUMN_MAPPING__": "/private/mapping.json",
            "__LOCAL_PYTHON__": str(python),
            "__VLLM_ADAPTER__": str(adapter),
            "__VLLM_BASE_URL__": base_url,
            "__BWRAP__": str(bwrap),
        },
    )

    model = prepare_vllm_module._locked_model_manifest(
        resolved,
        python=python,
        adapter=adapter,
        bwrap=bwrap,
        base_url=base_url,
    )

    assert resolved["experiment_id"] == (
        "nsclc-semantic-workflow-grid-qwen38-27b-vllm-v2-20x5"
    )
    assert resolved["replicates"] == 5
    assert resolved["max_parallel"] == 6
    assert model["id"] == "Qwen/Qwen3.8-27B"
    assert model["adapter"] == "vllm-cli-json"
    assert model["sampling"] == {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 100_000,
        "thinking_mode": "server-default",
        "seed_policy": "sha256(request_id,prompt_hash,turn,schema,attempt)",
    }
    assert model["limits"]["max_tool_calls"] == 32
    assert model["limits"]["max_contract_repairs"] == 2
    assert model["limits"]["api_timeout_seconds"] == 7_200
    assert model["limits"]["adapter_timeout_seconds"] == 14_380
    assert model["limits"]["harness_timeout_seconds"] == 14_400

    changed = json.loads(json.dumps(resolved))
    extra_args = changed["models"][0]["extra_args"]
    extra_args[extra_args.index("--max-tool-calls") + 1] = "33"
    with pytest.raises(ValueError, match="runtime lock mismatch"):
        prepare_vllm_module._locked_model_manifest(
            changed,
            python=python,
            adapter=adapter,
            bwrap=bwrap,
            base_url=base_url,
        )


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_gemma4_vllm_template_locks_thinking_and_controller_runtime(
    version: str,
) -> None:
    template = yaml.safe_load(
        (
            EXPERIMENT
            / f"nsclc_semantic_workflow_grid_gemma4_31b_vllm_{version}.template.yaml"
        ).read_text(encoding="utf-8")
    )
    python = Path("/home/klkehl/thisenv/bin/python")
    adapter = Path("/repo/scripts/vllm_cli_json_adapter.py")
    bwrap = Path("/usr/bin/bwrap")
    base_url = "http://172.24.216.113:1234/v1"
    resolved = prepare_module._replace_tokens(
        template,
        {
            "__MAIN_OUTPUT_ROOT__": "/results/main",
            "__NAMED_PUBLIC_WORKSPACE__": "/public/named",
            "__MASKED_PUBLIC_WORKSPACE__": "/public/masked",
            "__NAMED_PRIVATE_MANIFEST__": "/private/named.json",
            "__MASKED_PRIVATE_MANIFEST__": "/private/masked.json",
            "__COLUMN_MAPPING__": "/private/mapping.json",
            "__LOCAL_PYTHON__": str(python),
            "__VLLM_ADAPTER__": str(adapter),
            "__VLLM_BASE_URL__": base_url,
            "__BWRAP__": str(bwrap),
        },
    )

    model = prepare_vllm_module._locked_model_manifest(
        resolved,
        python=python,
        adapter=adapter,
        bwrap=bwrap,
        base_url=base_url,
        expected_model_id="gemma4-31b",
        expected_thinking_mode="enabled",
        expected_interaction_mode="native-tools",
    )

    assert resolved["experiment_id"] == (
        f"nsclc-semantic-workflow-grid-gemma4-31b-vllm-{version}-20x5"
    )
    assert resolved["models"][0]["id"] == f"gemma4-31b-vllm-controller-{version}"
    assert resolved["replicates"] == 5
    assert resolved["max_parallel"] == 6
    assert model["id"] == "gemma4-31b"
    assert model["interaction_mode"] == "native-tools"
    assert model["sampling"]["thinking_mode"] == "enabled"
    assert model["sampling"]["max_tokens"] == 100_000
    assert model["limits"]["api_timeout_seconds"] == 7_200

    changed = json.loads(json.dumps(resolved))
    extra_args = changed["models"][0]["extra_args"]
    extra_args[extra_args.index("--thinking-mode") + 1] = "disabled"
    with pytest.raises(ValueError, match="thinking-mode lock mismatch"):
        prepare_vllm_module._locked_model_manifest(
            changed,
            python=python,
            adapter=adapter,
            bwrap=bwrap,
            base_url=base_url,
            expected_model_id="gemma4-31b",
            expected_thinking_mode="enabled",
            expected_interaction_mode="native-tools",
        )


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_gemma4_vllm_hardened_templates_lock_split_ceilings_and_sampling(
    version: str,
) -> None:
    template = yaml.safe_load(
        (
            EXPERIMENT
            / f"nsclc_semantic_workflow_grid_gemma4_31b_vllm_{version}.template.yaml"
        ).read_text(encoding="utf-8")
    )
    python = Path("/home/klkehl/thisenv/bin/python")
    adapter = Path("/repo/scripts/vllm_cli_json_adapter.py")
    bwrap = Path("/usr/bin/bwrap")
    base_url = "http://172.24.216.113:1234/v1"
    resolved = prepare_module._replace_tokens(
        template,
        {
            "__MAIN_OUTPUT_ROOT__": "/results/main",
            "__NAMED_PUBLIC_WORKSPACE__": "/public/named",
            "__MASKED_PUBLIC_WORKSPACE__": "/public/masked",
            "__NAMED_PRIVATE_MANIFEST__": "/private/named.json",
            "__MASKED_PRIVATE_MANIFEST__": "/private/masked.json",
            "__COLUMN_MAPPING__": "/private/mapping.json",
            "__LOCAL_PYTHON__": str(python),
            "__VLLM_ADAPTER__": str(adapter),
            "__VLLM_BASE_URL__": base_url,
            "__BWRAP__": str(bwrap),
        },
    )

    model = prepare_vllm_module._locked_model_manifest(
        resolved,
        python=python,
        adapter=adapter,
        bwrap=bwrap,
        base_url=base_url,
        expected_model_id="gemma4-31b",
        expected_thinking_mode="enabled",
        expected_interaction_mode="native-tools",
        expected_temperature=1.0,
        expected_top_p=0.95,
        expected_top_k=64,
        expected_repetition_penalty=1.1,
        expected_max_decision_tokens=16_000,
    )

    assert resolved["experiment_id"] == (
        f"nsclc-semantic-workflow-grid-gemma4-31b-vllm-{version}-20x5"
    )
    assert resolved["models"][0]["id"] == f"gemma4-31b-vllm-controller-{version}"
    assert model["sampling"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 100_000,
        "thinking_mode": "enabled",
        "seed_policy": "sha256(request_id,prompt_hash,turn,schema,attempt)",
        "top_k": 64,
        "repetition_penalty": 1.1,
    }
    assert model["limits"]["max_decision_tokens"] == 16_000
    assert resolved["max_parallel"] == 6

    changed = json.loads(json.dumps(resolved))
    extra_args = changed["models"][0]["extra_args"]
    extra_args[extra_args.index("--top-k") + 1] = "63"
    with pytest.raises(ValueError, match="runtime lock mismatch"):
        prepare_vllm_module._locked_model_manifest(
            changed,
            python=python,
            adapter=adapter,
            bwrap=bwrap,
            base_url=base_url,
            expected_model_id="gemma4-31b",
            expected_thinking_mode="enabled",
            expected_interaction_mode="native-tools",
            expected_temperature=1.0,
            expected_top_p=0.95,
            expected_top_k=64,
            expected_repetition_penalty=1.1,
            expected_max_decision_tokens=16_000,
        )


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
