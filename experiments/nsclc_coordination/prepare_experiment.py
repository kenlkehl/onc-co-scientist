#!/usr/bin/env python3
"""Verify, isolate, and resolve the NSCLC semantic-workflow grid locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

EXPECTED_HASHES = {
    "named_dataset": "c93065845b99676904f8ec902b0c1c24fb0ab98579e084c706bcdf804d025fbb",
    "masked_dataset": "a84474a29fd5efda1fdf109163fdbff3b374f676434ccc24fc569ac42e67ba61",
    "named_manifest": "a844619fceb456a5ef4d9b5ba3dff5e7f07363eb83554226601f845ed22ce064",
    "masked_manifest": "e78bb273b53647fdc41af4d0e7b34ffbb0d0dec86b0e09b78e689be764ca0409",
    "column_mapping": "6d291a3d653803b12ad150654635201b079c6dd37e09fd7037fd0bcbf08d9cd7",
}
EXPECTED_SHAPE = (50_000, 35)
NAMED_CANDIDATES = [
    "treatment_pembrolizumab",
    "treatment_sotorasib",
    "treatment_olaparib",
    "treatment_osimertinib",
]
MASKED_CANDIDATES = ["feature_012", "feature_018", "feature_020", "feature_027"]
PRIVATE_FILENAMES = {"manifest.json", "column_mapping.json"}
MASKED_FORBIDDEN_TERMS = (
    "pembrolizumab",
    "sotorasib",
    "olaparib",
    "osimertinib",
    "kras",
    "brca",
    "alk_fusion",
    "sex_female",
    "male",
    "female",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def _source_paths(repo: Path) -> dict[str, Path]:
    root = repo / "example_data_clinical_all_claude" / "ds001" / "nsclc"
    return {
        "named_dataset": root / "named" / "public" / "dataset.parquet",
        "masked_dataset": root / "anonymized" / "public" / "dataset.parquet",
        "named_manifest": root / "named" / "manifest.json",
        "masked_manifest": root / "anonymized" / "manifest.json",
        "column_mapping": root / "anonymized" / "column_mapping.json",
    }


def _description(candidates: list[str], modifiers: list[str]) -> str:
    lines = [
        "# Oncology patient cohort `ds001_nsclc`",
        "",
        "This dataset contains 50,000 patient records. Each row is one patient and no values "
        "are missing.",
        "",
        "## Identifier",
        "",
        "- `patient_id`",
        "",
        "## Outcome",
        "",
        "- `pfs_months`",
        "",
        "## Candidate treatment/exposure indicators",
        "",
        *[f"- `{name}`" for name in candidates],
        "",
        "## Possible modifiers/covariates",
        "",
        *[f"- `{name}`" for name in modifiers],
        "",
        "The four candidate indicators are anonymous experimental exposures. Search all four "
        "systematically; no target exposure, modifier set, or number of signals is disclosed.",
        "",
    ]
    return "\n".join(lines)


def _prepare_public_workspaces(
    *, paths: dict[str, Path], public_root: Path, mapping: dict[str, str]
) -> dict[str, Path]:
    named = pd.read_parquet(paths["named_dataset"])
    masked = pd.read_parquet(paths["masked_dataset"])
    if named.shape != EXPECTED_SHAPE or masked.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected both Parquet files to be {EXPECTED_SHAPE}; "
            f"got named={named.shape}, masked={masked.shape}."
        )
    inverse = {masked_name: named_name for named_name, masked_name in mapping.items()}
    restored = masked.rename(columns=inverse)
    if set(restored.columns) != set(named.columns):
        raise ValueError("Named and inverse-renamed masked columns do not match.")
    pd.testing.assert_frame_equal(
        named,
        restored.loc[:, named.columns],
        check_dtype=True,
        check_exact=True,
        check_names=True,
    )

    named_modifiers = [
        column
        for column in named.columns
        if column not in {"patient_id", "pfs_months", *NAMED_CANDIDATES}
    ]
    masked_modifiers = [
        column
        for column in masked.columns
        if column not in {"patient_id", "pfs_months", *MASKED_CANDIDATES}
    ]
    workspaces = {"named": public_root / "named", "masked": public_root / "masked"}
    descriptions = {
        "named": _description(NAMED_CANDIDATES, named_modifiers),
        "masked": _description(MASKED_CANDIDATES, masked_modifiers),
    }
    source_datasets = {"named": paths["named_dataset"], "masked": paths["masked_dataset"]}
    for condition, workspace in workspaces.items():
        workspace.mkdir(parents=True, exist_ok=True)
        unexpected = {
            item.name for item in workspace.iterdir() if item.name not in {
                "dataset.parquet",
                "dataset_description.md",
            }
        }
        if unexpected:
            raise ValueError(f"Unexpected file(s) in {workspace}: {sorted(unexpected)}")
        shutil.copy2(source_datasets[condition], workspace / "dataset.parquet")
        (workspace / "dataset_description.md").write_text(
            descriptions[condition], encoding="utf-8"
        )
        if {item.name for item in workspace.iterdir()} != {
            "dataset.parquet",
            "dataset_description.md",
        }:
            raise ValueError(f"Public workspace is not minimal: {workspace}")

    masked_schema = {str(column).lower() for column in masked.columns}
    masked_text = (workspaces["masked"] / "dataset_description.md").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in MASKED_FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(forbidden)}\b", masked_text) or forbidden in masked_schema:
            raise ValueError(f"Masked public workspace leaks forbidden term: {forbidden}")
    for workspace in workspaces.values():
        if any(item.name in PRIVATE_FILENAMES for item in workspace.rglob("*")):
            raise ValueError(f"Private evaluator filename leaked into {workspace}")
    return workspaces


def _replace_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tokens(item, replacements) for key, item in value.items()}
    return value


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _active_python_executable() -> Path:
    """Return the invoked Python path without dereferencing a virtualenv shim.

    Resolving ``sys.executable`` crosses the virtualenv symlink and silently
    drops its site-packages when the adapter is launched from an isolated run
    workspace.  An absolute, non-resolved path preserves the active environment.
    """

    executable = Path(sys.executable).expanduser().absolute()
    if not executable.is_file():
        raise ValueError(f"Active Python executable does not exist: {executable}")
    return executable


def _machine_manifest(
    *,
    repo: Path,
    codex: Path,
    configs: dict[str, Path],
    source_paths: dict[str, Path],
    schedule_seed: int,
) -> dict[str, Any]:
    git_status = _run(
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
            "commit": _run(
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
            "python_executable": str(_active_python_executable()),
            "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        },
        "codex": {
            "path": str(codex),
            "version": _run([str(codex), "--version"], cwd=repo),
            "sha256": _sha256(codex),
        },
        "model": {"id": "gpt-5.6-luna", "reasoning_effort": "low"},
        "schedule_seed": schedule_seed,
        "config_sha256": {label: _sha256(path) for label, path in configs.items()},
        "substrate_sha256": {
            label: _sha256(path) for label, path in source_paths.items()
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(strict=True)
    source_paths = _source_paths(repo)
    observed = {label: _sha256(path) for label, path in source_paths.items()}
    if observed != EXPECTED_HASHES:
        raise ValueError(f"Pinned source hash mismatch: {observed}")
    mapping = json.loads(source_paths["column_mapping"].read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("column_mapping.json must contain one object.")
    workspaces = _prepare_public_workspaces(
        paths=source_paths,
        public_root=args.public_root.resolve(),
        mapping={str(key): str(value) for key, value in mapping.items()},
    )

    codex_raw = shutil.which(args.codex)
    if codex_raw is None:
        raise ValueError(f"Codex executable not found: {args.codex}")
    codex = Path(codex_raw).resolve(strict=True)
    python = _active_python_executable()
    adapter = (repo / "scripts" / "codex_cli_json_adapter.py").resolve(strict=True)
    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    replacements = {
        "__MAIN_OUTPUT_ROOT__": str(args.main_output_root.resolve()),
        "__NAMED_PUBLIC_WORKSPACE__": str(workspaces["named"].resolve()),
        "__MASKED_PUBLIC_WORKSPACE__": str(workspaces["masked"].resolve()),
        "__NAMED_PRIVATE_MANIFEST__": str(source_paths["named_manifest"].resolve()),
        "__MASKED_PRIVATE_MANIFEST__": str(source_paths["masked_manifest"].resolve()),
        "__COLUMN_MAPPING__": str(source_paths["column_mapping"].resolve()),
        "__VENV_PYTHON__": str(python),
        "__CODEX_ADAPTER__": str(adapter),
        "__CODEX_BINARY__": str(codex),
    }
    main = _replace_tokens(template, replacements)
    unresolved = re.findall(r"__[A-Z0-9_]+__", json.dumps(main))
    if unresolved:
        raise ValueError(f"Unresolved template tokens: {sorted(set(unresolved))}")
    smoke = json.loads(json.dumps(main))
    smoke["experiment_id"] = "nsclc-semantic-workflow-grid-live-smoke"
    smoke["output_root"] = str(args.smoke_output_root.resolve())
    smoke["iteration_policy"]["iterations"] = 1
    smoke["replicates"] = 1
    smoke["budget"]["max_agent_calls"] = 12
    stub = json.loads(json.dumps(main))
    stub["experiment_id"] = "nsclc-semantic-workflow-grid-stub-gate"
    stub["output_root"] = str(args.stub_output_root.resolve())
    stub["iteration_policy"]["iterations"] = 2
    stub["replicates"] = 1
    stub["budget"]["max_agent_calls"] = 24
    stub["budget"]["max_runtime_seconds_per_call"] = 60
    stub["models"] = [{"id": "stub", "model_id": "stub", "adapter": "stub"}]
    configs = {
        "main": args.main_config.resolve(),
        "smoke": args.smoke_config.resolve(),
        "stub": args.stub_config.resolve(),
    }
    _write_yaml(configs["main"], main)
    _write_yaml(configs["smoke"], smoke)
    _write_yaml(configs["stub"], stub)
    manifest = _machine_manifest(
        repo=repo,
        codex=codex,
        configs=configs,
        source_paths=source_paths,
        schedule_seed=int(main["schedule_seed"]),
    )
    _write_yaml(args.machine_manifest.resolve(), manifest)
    preparation = {
        "schema_version": "1",
        "source_hashes": observed,
        "shape": list(EXPECTED_SHAPE),
        "row_value_dtype_parity": True,
        "workspaces": {
            condition: {
                "path": str(path.resolve()),
                "files": sorted(item.name for item in path.iterdir()),
                "dataset_sha256": _sha256(path / "dataset.parquet"),
                "description_sha256": _sha256(path / "dataset_description.md"),
            }
            for condition, path in workspaces.items()
        },
        "configs": {label: str(path) for label, path in configs.items()},
        "machine_manifest": str(args.machine_manifest.resolve()),
    }
    preparation_path = args.public_root.resolve() / "preparation_manifest.json"
    preparation_path.write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    return preparation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent / "local"
    results = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("nsclc_semantic_workflow_grid.template.yaml"),
    )
    parser.add_argument("--public-root", type=Path, default=local / "public")
    parser.add_argument(
        "--main-config",
        type=Path,
        default=Path(__file__).with_name("nsclc_semantic_workflow_grid.local.yaml"),
    )
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=Path(__file__).with_name("nsclc_semantic_workflow_grid.smoke.local.yaml"),
    )
    parser.add_argument(
        "--stub-config",
        type=Path,
        default=Path(__file__).with_name("nsclc_semantic_workflow_grid.stub.local.yaml"),
    )
    parser.add_argument("--machine-manifest", type=Path, default=local / "machine_manifest.yaml")
    parser.add_argument(
        "--main-output-root", type=Path, default=results / "semantic-workflow-grid-main"
    )
    parser.add_argument(
        "--smoke-output-root", type=Path, default=results / "semantic-workflow-grid-smoke"
    )
    parser.add_argument(
        "--stub-output-root", type=Path, default=results / "semantic-workflow-grid-stub"
    )
    parser.add_argument("--codex", default="codex")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = prepare(parse_args(argv))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
