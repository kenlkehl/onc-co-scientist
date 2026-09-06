#!/usr/bin/env python3
"""Verify, isolate, and resolve an OpenAI-compatible vLLM NSCLC grid."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from experiments.nsclc_coordination import prepare_experiment as common
except ModuleNotFoundError:  # Direct execution puts this directory on sys.path.
    import prepare_experiment as common  # type: ignore[no-redef]


def _fetch_models(base_url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("vLLM /models response does not have the expected object shape.")
    return payload


def _flag(extra_args: list[str], name: str) -> str:
    return common._one_flag_value(extra_args, name)


def _set_model_identity(
    config: dict[str, Any], *, model_id: str, profile_id: str | None
) -> None:
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("The vLLM grid requires exactly one model profile.")
    models[0]["model_id"] = model_id
    if profile_id is not None:
        models[0]["id"] = profile_id


def _locked_model_manifest(
    config: dict[str, Any],
    *,
    python: Path,
    adapter: Path,
    bwrap: Path,
    base_url: str,
    expected_model_id: str = "Qwen/Qwen3.8-27B",
    expected_thinking_mode: str = "server-default",
    expected_interaction_mode: str = "json-schema",
    expected_temperature: float = 0.2,
    expected_top_p: float = 0.95,
    expected_top_k: int | None = None,
    expected_repetition_penalty: float | None = None,
    expected_max_decision_tokens: int | None = None,
    expected_python_memory_limit_mb: int | None = None,
) -> dict[str, Any]:
    if expected_thinking_mode not in {"server-default", "enabled", "disabled"}:
        raise ValueError(f"Unsupported thinking mode: {expected_thinking_mode}")
    if expected_interaction_mode not in {"json-schema", "native-tools"}:
        raise ValueError(f"Unsupported interaction mode: {expected_interaction_mode}")
    models = config.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError("The vLLM grid requires exactly one model profile.")
    model = models[0]
    if model.get("adapter") != "cli-json":
        raise ValueError("The vLLM grid requires the generic cli-json runtime boundary.")
    if model.get("model_id") != expected_model_id:
        raise ValueError(f"The vLLM grid is locked to {expected_model_id}.")
    if model.get("reasoning_effort") is not None:
        raise ValueError("A local vLLM profile must not claim a Codex reasoning effort.")
    command = model.get("command")
    if command != [str(python), str(adapter)]:
        raise ValueError("The vLLM adapter command does not match the local runtime lock.")
    extra_args = model.get("extra_args")
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise ValueError("The vLLM model profile must provide string extra_args.")
    thinking_positions = [
        index for index, item in enumerate(extra_args) if item == "--thinking-mode"
    ]
    if len(thinking_positions) > 1:
        raise ValueError("The vLLM model profile repeats --thinking-mode.")
    observed_thinking_mode = "server-default"
    if thinking_positions:
        position = thinking_positions[0]
        if position + 1 >= len(extra_args):
            raise ValueError("The vLLM model profile omits the thinking-mode value.")
        observed_thinking_mode = extra_args[position + 1]
    if observed_thinking_mode != expected_thinking_mode:
        raise ValueError(
            "vLLM thinking-mode lock mismatch: "
            f"expected={expected_thinking_mode}, observed={observed_thinking_mode}"
        )
    interaction_positions = [
        index for index, item in enumerate(extra_args) if item == "--interaction-mode"
    ]
    if len(interaction_positions) > 1:
        raise ValueError("The vLLM model profile repeats --interaction-mode.")
    observed_interaction_mode = "json-schema"
    if interaction_positions:
        position = interaction_positions[0]
        if position + 1 >= len(extra_args):
            raise ValueError("The vLLM model profile omits the interaction-mode value.")
        observed_interaction_mode = extra_args[position + 1]
    if observed_interaction_mode != expected_interaction_mode:
        raise ValueError(
            "vLLM interaction-mode lock mismatch: "
            f"expected={expected_interaction_mode}, observed={observed_interaction_mode}"
        )
    expected = {
        "--base-url": base_url,
        "--api-key": "EMPTY",
        "--analysis-python": str(python),
        "--bwrap": str(bwrap),
        "--timeout-seconds": "14380",
        "--api-timeout-seconds": "7200",
        "--python-timeout-seconds": "300",
        "--max-api-retries": "2",
        "--max-contract-repairs": "2",
        "--max-tool-calls": "32",
        "--min-tool-calls": "1",
        "--max-controller-decisions": "40",
        "--max-tool-output-chars": "40000",
        "--max-history-chars": "180000",
        "--max-tokens": "100000",
        "--temperature": str(expected_temperature),
        "--top-p": str(expected_top_p),
    }
    if expected_top_k is not None:
        expected["--top-k"] = str(expected_top_k)
    if expected_repetition_penalty is not None:
        expected["--repetition-penalty"] = str(expected_repetition_penalty)
    if expected_max_decision_tokens is not None:
        expected["--max-decision-tokens"] = str(expected_max_decision_tokens)
    if expected_python_memory_limit_mb is not None:
        expected["--python-memory-limit-mb"] = str(
            expected_python_memory_limit_mb
        )
    if expected_thinking_mode != "server-default":
        expected["--thinking-mode"] = expected_thinking_mode
    if expected_interaction_mode != "json-schema":
        expected["--interaction-mode"] = expected_interaction_mode
    observed = {flag: _flag(extra_args, flag) for flag in expected}
    if observed != expected:
        raise ValueError(f"vLLM runtime lock mismatch: expected={expected}, observed={observed}")
    harness_timeout = int(config["budget"]["max_runtime_seconds_per_call"])
    if int(observed["--timeout-seconds"]) >= harness_timeout:
        raise ValueError("The adapter timeout must precede the harness timeout.")
    sampling: dict[str, Any] = {
        "temperature": float(observed["--temperature"]),
        "top_p": float(observed["--top-p"]),
        "max_tokens": int(observed["--max-tokens"]),
        "thinking_mode": expected_thinking_mode,
        "seed_policy": "sha256(request_id,prompt_hash,turn,schema,attempt)",
    }
    if expected_top_k is not None:
        sampling["top_k"] = int(observed["--top-k"])
    if expected_repetition_penalty is not None:
        sampling["repetition_penalty"] = float(
            observed["--repetition-penalty"]
        )
    limits: dict[str, Any] = {
        "max_api_retries": int(observed["--max-api-retries"]),
        "max_contract_repairs": int(observed["--max-contract-repairs"]),
        "min_tool_calls": int(observed["--min-tool-calls"]),
        "max_tool_calls": int(observed["--max-tool-calls"]),
        "max_controller_decisions": int(observed["--max-controller-decisions"]),
        "max_tool_output_chars": int(observed["--max-tool-output-chars"]),
        "max_history_chars": int(observed["--max-history-chars"]),
        "python_timeout_seconds": int(observed["--python-timeout-seconds"]),
        "api_timeout_seconds": float(observed["--api-timeout-seconds"]),
        "adapter_timeout_seconds": int(observed["--timeout-seconds"]),
        "harness_timeout_seconds": harness_timeout,
    }
    if expected_max_decision_tokens is not None:
        limits["max_decision_tokens"] = int(observed["--max-decision-tokens"])
    if expected_python_memory_limit_mb is not None:
        limits["python_memory_limit_mb"] = int(
            observed["--python-memory-limit-mb"]
        )
    return {
        "profile": str(model.get("id", "")),
        "id": str(model["model_id"]),
        "adapter": "vllm-cli-json",
        "base_url": base_url,
        "interaction_mode": expected_interaction_mode,
        "sampling": sampling,
        "limits": limits,
    }


def _machine_manifest(
    *,
    repo: Path,
    configs: dict[str, Path],
    source_paths: dict[str, Path],
    schedule_seed: int,
    model: dict[str, Any],
    endpoint_models: dict[str, Any],
    adapter: Path,
    bwrap: Path,
) -> dict[str, Any]:
    git_status = common._run(
        ["git", "-c", f"safe.directory={repo}", "status", "--short"], cwd=repo
    )
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "schema_version": "1",
        "utc_start_time": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": {
            "commit": common._run(
                ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"], cwd=repo
            ),
            "pinned_source_commit": "4a8fd25f104869d9209ec010bac504b8a91a4964",
            "status": git_status,
            "clean": not bool(git_status),
        },
        "host": {
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "python": sys.version,
            "python_executable": str(common._active_python_executable()),
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        },
        "runtime": {
            "adapter_path": str(adapter),
            "adapter_sha256": common._sha256(adapter),
            "bubblewrap_path": str(bwrap),
            "bubblewrap_version": common._run([str(bwrap), "--version"], cwd=repo),
            "bubblewrap_sha256": common._sha256(bwrap),
        },
        "endpoint_models": endpoint_models,
        "model": model,
        "schedule_seed": schedule_seed,
        "config_sha256": {label: common._sha256(path) for label, path in configs.items()},
        "substrate_sha256": {
            label: common._sha256(path) for label, path in source_paths.items()
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.python_memory_limit_mb is not None and args.python_memory_limit_mb < 0:
        raise ValueError("python_memory_limit_mb must be non-negative.")
    repo = args.repo.resolve(strict=True)
    source_paths = common._source_paths(repo)
    observed = {label: common._sha256(path) for label, path in source_paths.items()}
    if observed != common.EXPECTED_HASHES:
        raise ValueError(f"Pinned source hash mismatch: {observed}")
    mapping = json.loads(source_paths["column_mapping"].read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("column_mapping.json must contain one object.")
    workspaces = common._prepare_public_workspaces(
        paths=source_paths,
        public_root=args.public_root.resolve(),
        mapping={str(key): str(value) for key, value in mapping.items()},
    )

    python = common._active_python_executable()
    adapter = (repo / "scripts" / "vllm_cli_json_adapter.py").resolve(strict=True)
    bwrap_raw = shutil.which(args.bwrap)
    if bwrap_raw is None:
        raise ValueError(f"bubblewrap executable not found: {args.bwrap}")
    bwrap = Path(bwrap_raw).resolve(strict=True)
    endpoint_models = _fetch_models(args.base_url)
    served_ids = {
        model_id
        for item in endpoint_models["data"]
        if isinstance(item, dict) and isinstance((model_id := item.get("id")), str)
    }
    if args.model_id not in served_ids:
        raise ValueError(f"Expected model {args.model_id!r}; endpoint serves {sorted(served_ids)}")

    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    replacements = {
        "__MAIN_OUTPUT_ROOT__": str(args.main_output_root.resolve()),
        "__NAMED_PUBLIC_WORKSPACE__": str(workspaces["named"].resolve()),
        "__MASKED_PUBLIC_WORKSPACE__": str(workspaces["masked"].resolve()),
        "__NAMED_PRIVATE_MANIFEST__": str(source_paths["named_manifest"].resolve()),
        "__MASKED_PRIVATE_MANIFEST__": str(source_paths["masked_manifest"].resolve()),
        "__COLUMN_MAPPING__": str(source_paths["column_mapping"].resolve()),
        "__LOCAL_PYTHON__": str(python),
        "__VLLM_ADAPTER__": str(adapter),
        "__VLLM_BASE_URL__": args.base_url.rstrip("/"),
        "__BWRAP__": str(bwrap),
    }
    main = common._replace_tokens(template, replacements)
    common._set_treatment_columns(main, source_paths, mapping)
    if args.main_experiment_id is not None:
        main["experiment_id"] = args.main_experiment_id
    _set_model_identity(
        main,
        model_id=args.model_id,
        profile_id=args.model_profile_id,
    )
    if args.python_memory_limit_mb is not None:
        extra_args = main["models"][0]["extra_args"]
        flag = "--python-memory-limit-mb"
        if flag in extra_args:
            extra_args[extra_args.index(flag) + 1] = str(
                args.python_memory_limit_mb
            )
        else:
            extra_args.extend([flag, str(args.python_memory_limit_mb)])
    unresolved = re.findall(r"__[A-Z0-9_]+__", json.dumps(main))
    if unresolved:
        raise ValueError(f"Unresolved template tokens: {sorted(set(unresolved))}")
    model_manifest = _locked_model_manifest(
        main,
        python=python,
        adapter=adapter,
        bwrap=bwrap,
        base_url=args.base_url.rstrip("/"),
        expected_model_id=args.model_id,
        expected_thinking_mode=args.thinking_mode,
        expected_interaction_mode=args.interaction_mode,
        expected_temperature=args.temperature,
        expected_top_p=args.top_p,
        expected_top_k=args.top_k,
        expected_repetition_penalty=args.repetition_penalty,
        expected_max_decision_tokens=args.max_decision_tokens,
        expected_python_memory_limit_mb=args.python_memory_limit_mb,
    )
    smoke = json.loads(json.dumps(main))
    smoke["experiment_id"] = args.smoke_experiment_id
    smoke["output_root"] = str(args.smoke_output_root.resolve())
    smoke["iteration_policy"]["iterations"] = 1
    smoke["replicates"] = 1
    smoke["max_parallel"] = 2
    smoke["budget"]["max_agent_calls"] = 12
    stub = json.loads(json.dumps(main))
    stub["experiment_id"] = args.stub_experiment_id
    stub["output_root"] = str(args.stub_output_root.resolve())
    stub["iteration_policy"]["iterations"] = 2
    stub["replicates"] = 1
    stub["max_parallel"] = 2
    stub["budget"]["max_agent_calls"] = 24
    stub["budget"]["max_runtime_seconds_per_call"] = 60
    stub["models"] = [{"id": "stub", "model_id": "stub", "adapter": "stub"}]
    configs = {
        "main": args.main_config.resolve(),
        "smoke": args.smoke_config.resolve(),
        "stub": args.stub_config.resolve(),
    }
    common._write_yaml(configs["main"], main)
    common._write_yaml(configs["smoke"], smoke)
    common._write_yaml(configs["stub"], stub)
    manifest = _machine_manifest(
        repo=repo,
        configs=configs,
        source_paths=source_paths,
        schedule_seed=int(main["schedule_seed"]),
        model=model_manifest,
        endpoint_models=endpoint_models,
        adapter=adapter,
        bwrap=bwrap,
    )
    common._write_yaml(args.machine_manifest.resolve(), manifest)
    preparation = {
        "schema_version": "1",
        "source_hashes": observed,
        "shape": list(common.EXPECTED_SHAPE),
        "row_value_dtype_parity": True,
        "workspaces": {
            condition: {
                "path": str(path.resolve()),
                "files": sorted(item.name for item in path.iterdir()),
                "dataset_sha256": common._sha256(path / "dataset.parquet"),
                "description_sha256": common._sha256(path / "dataset_description.md"),
            }
            for condition, path in workspaces.items()
        },
        "configs": {label: str(path) for label, path in configs.items()},
        "template": {
            "path": str(args.template.resolve()),
            "sha256": common._sha256(args.template.resolve()),
        },
        "machine_manifest": str(args.machine_manifest.resolve()),
        "endpoint_model_ids": sorted(served_ids),
        "identity_overrides": {
            "main_experiment_id": args.main_experiment_id,
            "model_profile_id": args.model_profile_id,
            "model_id": args.model_id,
            "python_memory_limit_mb": args.python_memory_limit_mb,
        },
    }
    preparation_path = args.public_root.resolve() / "preparation_manifest.json"
    preparation_path.write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    return preparation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    local = experiment / "local"
    results = experiment / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--template",
        type=Path,
        default=experiment / "nsclc_semantic_workflow_grid_qwen38_27b_vllm_v2.template.yaml",
    )
    parser.add_argument(
        "--public-root", type=Path, default=local / "public-qwen38-27b-vllm-v2"
    )
    parser.add_argument(
        "--main-config",
        type=Path,
        default=experiment / "nsclc_semantic_workflow_grid_qwen38_27b_vllm_v2.local.yaml",
    )
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=experiment / "nsclc_semantic_workflow_grid_qwen38_27b_vllm_v2.smoke.local.yaml",
    )
    parser.add_argument(
        "--stub-config",
        type=Path,
        default=experiment / "nsclc_semantic_workflow_grid_qwen38_27b_vllm_v2.stub.local.yaml",
    )
    parser.add_argument(
        "--machine-manifest",
        type=Path,
        default=local / "machine_manifest.qwen38_27b_vllm_v2.yaml",
    )
    parser.add_argument(
        "--smoke-experiment-id",
        default="nsclc-semantic-workflow-grid-qwen38-27b-vllm-v2-live-smoke",
    )
    parser.add_argument(
        "--stub-experiment-id",
        default="nsclc-semantic-workflow-grid-qwen38-27b-vllm-v2-stub-gate",
    )
    parser.add_argument("--main-experiment-id")
    parser.add_argument("--model-profile-id")
    parser.add_argument(
        "--main-output-root",
        type=Path,
        default=results / "semantic-workflow-grid-qwen38-27b-vllm-v2-main",
    )
    parser.add_argument(
        "--smoke-output-root",
        type=Path,
        default=results / "semantic-workflow-grid-qwen38-27b-vllm-v2-smoke",
    )
    parser.add_argument(
        "--stub-output-root",
        type=Path,
        default=results / "semantic-workflow-grid-qwen38-27b-vllm-v2-stub",
    )
    parser.add_argument("--base-url", default="http://camus.dfci.harvard.edu:8060/v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--thinking-mode",
        choices=("server-default", "enabled", "disabled"),
        default="server-default",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=("json-schema", "native-tools"),
        default="json-schema",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-decision-tokens", type=int)
    parser.add_argument("--python-memory-limit-mb", type=int)
    parser.add_argument("--bwrap", default="bwrap")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = prepare(parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
