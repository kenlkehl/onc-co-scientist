#!/usr/bin/env python3
"""Freeze the exact protocol, runtime, schedule, and scorer beside NSCLC runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "status", "--short"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze(
    *,
    repo: Path,
    config: Path,
    output_root: Path,
    machine_manifest: Path,
    preparation_manifest: Path,
    template: Path | None = None,
    protocol: Path | None = None,
) -> Path:
    repo = repo.resolve(strict=True)
    config = config.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    machine_manifest = machine_manifest.resolve(strict=True)
    preparation_manifest = preparation_manifest.resolve(strict=True)
    experiment_dir = Path(__file__).resolve().parent
    template = (
        template.resolve(strict=True)
        if template is not None
        else experiment_dir / "nsclc_semantic_workflow_grid.template.yaml"
    )
    protocol = (
        protocol.resolve(strict=True)
        if protocol is not None
        else experiment_dir / "protocol.md"
    )
    configured = yaml.safe_load(config.read_text(encoding="utf-8"))
    if Path(str(configured.get("output_root", ""))).resolve() != output_root:
        raise ValueError("Config output_root does not match the root being frozen.")
    manifest = yaml.safe_load(machine_manifest.read_text(encoding="utf-8"))
    if _git_status(repo):
        raise ValueError("Current implementation worktree is not clean.")
    if not manifest.get("git", {}).get("clean"):
        raise ValueError("Machine manifest is not tied to a clean implementation commit.")
    current_commit = _git_commit(repo)
    if manifest.get("git", {}).get("commit") != current_commit:
        raise ValueError("Machine manifest commit does not match the current implementation.")

    sources = {
        "experiment_config.yaml": config,
        "machine_manifest.yaml": machine_manifest,
        "preparation_manifest.json": preparation_manifest,
        "protocol.md": protocol,
        "template.yaml": template,
        "scorer.py": experiment_dir / "score_experiment.py",
        "schedule.json": output_root / "schedule.json",
        "plan.json": output_root / "plan.json",
        "resolved_spec.json": output_root / "resolved_spec.json",
        "private_evaluation_index.json": output_root / "private_evaluation_index.json",
    }
    for label, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot freeze missing {label}: {path}")
    scorer_text = sources["scorer.py"].read_text(encoding="utf-8")
    match = re.search(r'^SCORER_VERSION\s*=\s*"([^"]+)"', scorer_text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("Could not identify SCORER_VERSION in the scorer source.")

    provenance = output_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    desired_hashes = {label: _sha256(path) for label, path in sources.items()}
    freeze_path = provenance / "freeze_manifest.json"
    if freeze_path.exists():
        prior = json.loads(freeze_path.read_text(encoding="utf-8"))
        if prior.get("files") != desired_hashes or prior.get("git_commit") != current_commit:
            raise RuntimeError(
                "Frozen provenance differs from current inputs; use a new result root."
            )
        return freeze_path

    for label, source in sources.items():
        destination = provenance / label
        if destination.exists() and _sha256(destination) != desired_hashes[label]:
            raise RuntimeError(f"Refusing to overwrite mismatched provenance file: {destination}")
        shutil.copy2(source, destination)
    payload = {
        "schema_version": "1",
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": current_commit,
        "pinned_source_commit": "4a8fd25f104869d9209ec010bac504b8a91a4964",
        "experiment_id": configured["experiment_id"],
        "spec_fingerprint": json.loads(
            sources["schedule.json"].read_text(encoding="utf-8")
        )["spec_fingerprint"],
        "schedule_seed": configured["schedule_seed"],
        "scorer_version": match.group(1),
        "files": desired_hashes,
    }
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parent / "local"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--machine-manifest", type=Path, default=local / "machine_manifest.yaml"
    )
    parser.add_argument(
        "--preparation-manifest",
        type=Path,
        default=local / "public" / "preparation_manifest.json",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--protocol", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = freeze(
        repo=args.repo,
        config=args.config,
        output_root=args.output_root,
        machine_manifest=args.machine_manifest,
        preparation_manifest=args.preparation_manifest,
        template=args.template,
        protocol=args.protocol,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
