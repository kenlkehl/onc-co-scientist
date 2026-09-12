"""Launch the prepared Azure continuation only after the predecessor fully drains."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.harness.durable_io import atomic_write_json
from onc_co_scientist.providers.codex_cli import CodexCLIConfig, CodexCLIProvider
from scripts.expected_surprising.run_federated_grid import verify_predecessor


def launch(root):
    root = root.resolve()
    control = root / "control"
    control.mkdir(exist_ok=True)
    with (control / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = json.loads((root / "frozen_manifest.json").read_text())
        transition = frozen["azure_transition"]
        try:
            verify_predecessor(transition)
        except RuntimeError as exc:
            return dict(status="waiting_for_original_runs", reason=str(exc))
        for name, expected in frozen["hashes"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen file changed: {name}")
        policy = json.loads((root / "release_policy.json").read_text())
        if set(policy["released_models"]) != set(transition["models"]):
            raise ValueError("Azure continuation release scope changed")
        checks = json.loads((root / "azure_preflight.json").read_text())
        endpoint = checks["checks"][0]["metrics"]["azure_endpoint"]
        # Check fresh credentials once before admitting any scientific run.
        checker = CodexCLIProvider(
            CodexCLIConfig(
                model_id="auth-check-only",
                backend="azure",
                azure_endpoint=endpoint,
                audit_dir=str(control / "azure_auth_check"),
                resume_audit=True,
            )
        )
        checker.environment()
        records = []
        for model in transition["models"]:
            folder = control / model
            folder.mkdir(exist_ok=True)
            receipt = folder / "launch.json"
            execution = folder / "execution.json"
            if receipt.exists() or execution.exists():
                # Never automatically replay a partial/failed launch: an operator
                # must inspect its durable state before choosing --resume.
                records.append(dict(model=model, status="already_dispatched"))
                continue
            command = [
                sys.executable,
                str(root / "source/run_federated_grid.py"),
                "--root",
                str(root),
                "--model",
                model,
                "--workers",
                "10",
            ]
            env = dict(os.environ, PYTHONPATH=str(root / "source/src"))
            for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OCS_AZURE_ACCESS_TOKEN"):
                env.pop(key, None)
            # Write intent before spawning so a crashed launcher cannot silently
            # dispatch the same model twice on its next heartbeat.
            record = dict(
                model=model,
                status="dispatching",
                command=command,
                started_at=datetime.now(UTC).isoformat(),
            )
            atomic_write_json(receipt, record)
            with (folder / "driver.log").open("a") as log:
                process = subprocess.Popen(
                    command, env=env, stdout=log, stderr=log, start_new_session=True, cwd=root
                )
            record.update(status="dispatched", pid=process.pid)
            atomic_write_json(receipt, record)
            records.append(record)
        source = Path(transition["source_root"])
        state = json.loads((source / "azure_transition.json").read_text())
        state.update(status="azure_dispatched", dispatched_at=datetime.now(UTC).isoformat())
        atomic_write_json(source / "azure_transition.json", state)
        return dict(status="azure_dispatched", models=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.root), indent=2))
