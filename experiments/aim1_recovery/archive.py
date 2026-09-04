#!/usr/bin/env python3
"""Pack research records once per input copy; restore them for deterministic rescoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def pack(root: Path, archive_path: Path) -> None:
    """Preserve private inputs and all research artifacts, omitting duplicate datasets."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for source, prefix in [
            (root, "experiment"),
            (root.with_name(root.name + "_setup"), "excluded_setup"),
        ]:
            if not source.exists():
                continue
            for path in sorted(source.rglob("*")):
                rel = path.relative_to(source)
                if not path.is_file() or path.is_symlink() or "__pycache__" in rel.parts:
                    continue
                if rel.parts[0] == "public" and path.name == "dataset.parquet":
                    continue
                archive.add(path, arcname=str(Path(prefix) / rel), recursive=False)


def restore(archive_path: Path, out: Path) -> Path:
    """Restore a portable scoring workspace without modifying retained submissions."""
    if out.exists() and any(out.iterdir()):
        raise ValueError("Restore destination must be empty")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(out, filter="data")
    root = out / "experiment"
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text())
    for job in plan["jobs"]:
        workspace = root / "public" / job["job_id"]
        evaluator = root / "private" / job["task"]
        workspace.mkdir(parents=True, exist_ok=True)
        source = evaluator / f"discovery_{job['variant']}.parquet"
        shutil.copyfile(source, workspace / "dataset.parquet")
        digest = hashlib.sha256((workspace / "dataset.parquet").read_bytes()).hexdigest()
        if digest != job["data_sha256"]:
            raise ValueError(f"Reconstructed input differs for {job['job_id']}")
        job["workspace"] = str(workspace.resolve())
        job["evaluator"] = str(evaluator.resolve())
    shutil.copyfile(plan_path, root / "plan_original_paths.json")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("pack")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--archive", type=Path, required=True)
    unpack = sub.add_parser("restore")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.root, args.archive)
        print(args.archive)
    else:
        print(restore(args.archive, args.out))


if __name__ == "__main__":
    main()
