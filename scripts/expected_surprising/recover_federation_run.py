"""Fork one failed federated run at its first failed iteration, with exact prefix replay.

The parent bundle is read-only. A separate allocation and frozen source govern new
calls. Later parent responses are never imported after the scientific branch point.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import yaml


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def prepare(parent, root, condition, run_id, cap):
    if not 0 < cap <= 500:
        raise ValueError("This recovery is authorized for at most $500")
    original = parent / condition / "runs" / run_id
    result, report = read(original / "run.json"), read(original / "report.json")
    if result["status"] != "failed" or result["model_profile"] != "terra_medium":
        raise ValueError("Only the failed Terra run is authorized")
    if result["scientific_report_sha256"] != sha(original / "report.json"):
        raise ValueError("Original report checksum changed")
    start = min(e["iteration"] for e in report["protocol_errors"])
    root.mkdir(parents=True)  # Never overwrite a prepared recovery.
    control = root / "control"
    control.mkdir()
    shutil.copytree(
        parent / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__")
    )
    package = root / "source/src/onc_co_scientist"
    shutil.copy2(package / "expected_surprising/federation.py", control / "original_federation.py")
    repo = Path(__file__).resolve().parents[2]
    shutil.copy2(
        repo / "src/onc_co_scientist/expected_surprising/federation.py",
        package / "expected_surprising/federation.py",
    )
    for receipt_name, filename in (
        ("transport_override", "azure_federation.py"),
        ("budget_override", "azure_budget.py"),
    ):
        receipt = read(parent / "control" / f"{receipt_name}.json")
        source = Path(receipt["override_path"])
        if sha(source) != receipt["override_sha256"]:
            raise ValueError("Runtime override checksum changed")
        shutil.copy2(source, package / "providers" / filename)
        write(control / f"parent_{receipt_name}.json", receipt)
    shutil.copy2(__file__, root / "driver.py")
    config = yaml.safe_load((parent / condition / "config.yaml").read_text())
    config["output_root"] = str(root / condition)
    config["experiment_id"] = root.name
    config["models"] = [m for m in config["models"] if m["id"] == "terra_medium"]
    options = config["models"][0]["provider_config"]
    if options["backend"] != "azure" or options["model_id"] != "gpt-5.6-terra":
        raise ValueError("Recovery must use the original Azure Terra model")
    options["budget_policy_path"] = str(control / "spend_policy.json")
    (root / condition).mkdir()
    (root / condition / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    policy = read(parent / "control/spend_policy.json")
    policy = {
        k: policy[k]
        for k in (
            "rates_valid_until",
            "rates",
            "model_profiles",
            "max_unknown_attempts",
            "max_reusable_prefix_misses",
            "cache_miss_pause_enabled",
            "unknown_attempt_limit_scope",
        )
    }
    policy.update(
        enabled=True,
        additional_budget_usd=cap,
        release_policy_path=str(control / "release_policy.json"),
        scope="Separate user-authorized $500 maximum for one Terra recovery; parent unchanged",
    )
    write(control / "spend_policy.json", policy)
    write(control / "spend_state.json", {"spent_micro_usd": 0, "attempts": {}})
    write(control / "release_policy.json", {"released_models": ["terra_medium"]})
    prefix = {}
    for scope in ("central", "site_1", "site_2", "site_3", "site_4"):
        for path in sorted((original / "calls" / scope).glob("*.json")):
            record = read(path)
            if record["request"]["iteration"] >= start:
                continue
            relative = path.relative_to(original)
            expected = report["coordination"]["participant_artifact_sha256"][str(relative)]
            if sha(path) != expected:
                raise ValueError(f"Original participant artifact changed: {relative}")
            target = root / condition / "runs" / run_id / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            prefix[str(target.relative_to(root))] = expected
    files = [p for p in (root / "source").rglob("*") if p.is_file()]
    files += [
        control / "original_federation.py",
        root / "driver.py",
        root / condition / "config.yaml",
    ]
    write(
        root / "recovery.json",
        dict(
            created_at=datetime.now(UTC).isoformat(),
            parent=str(parent),
            original=str(original),
            condition=condition,
            run_id=run_id,
            start_iteration=start,
            max_new_usd=cap,
            original_run_sha256=sha(original / "run.json"),
            original_report_sha256=sha(original / "report.json"),
            prefix=prefix,
            hashes={str(p.relative_to(root)): sha(p) for p in files},
            policy="Replay the exact pre-failure iterations; regenerate every subsequent stage. "
            "A separate recovery version, excluded from original-batch completed counts.",
        ),
    )
    write(control / "execution.json", {"status": "prepared"})
    print(root)


class ReplayVerified(Exception):
    """The exact saved prefix reconstructed without making a new model call."""


def install_recovery_coordinator(root, start, check_only=False):
    from onc_co_scientist.expected_surprising import coordination, federation
    from onc_co_scientist.expected_surprising.workflow import WorkflowInfrastructureError

    spec = importlib.util.spec_from_file_location(
        "onc_co_scientist.expected_surprising._original_federation",
        root / "control/original_federation.py",
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    current = federation.FederatedCoordinator

    class RecoveryCoordinator(current):
        def respond(self, prompt, *, iteration, stage, attempt):
            if check_only and iteration >= start:
                # Infrastructure exceptions escape the scientific repair loop.
                raise WorkflowInfrastructureError("PREFIX_REPLAY_VERIFIED") from ReplayVerified()
            respond = (
                original.FederatedCoordinator.respond if iteration < start else current.respond
            )
            return respond(self, prompt, iteration=iteration, stage=stage, attempt=attempt)

    old_call = coordination.StageCoordinator._call

    def guarded_call(self, slot, messages, **kwargs):
        if kwargs["iteration"] < start and not (self.calls_dir / f"{slot}.json").exists():
            raise WorkflowInfrastructureError(
                "Missing prefix artifact; refusing paid prefix replay"
            )
        return old_call(self, slot, messages, **kwargs)

    coordination.StageCoordinator._call = guarded_call
    federation.FederatedCoordinator = RecoveryCoordinator


def progress(root, meta):
    state = read(root / "control/execution.json")
    budget = read(root / "control/spend_state.json")
    run = root / meta["condition"] / "runs" / meta["run_id"]
    calls = list((run / "calls").glob("*/*.json"))
    requests = list((run / "calls/requests").glob("*/*.json"))
    latest = max(requests, key=lambda p: p.stat().st_mtime, default=None)
    reserved = (
        sum(
            a["reserved_micro_usd"]
            for a in budget["attempts"].values()
            if a["status"] in {"reserved", "unknown"}
        )
        / 1e6
    )
    latest_label = latest.stem if latest else "reconstructing saved iterations 1–2"
    text = (
        "# Terra recovery — expected / unmasked, sequential, four sites\n\n"
        f"Updated: {datetime.now(UTC).isoformat()}\n\n"
        f"Status: **{state['status']}**\n\n"
        f"Completed participant calls: {len(calls)}; "
        f"{len(meta['prefix'])} inherited calls replayed locally.\n\n"
        f"Latest requested step: {latest_label}.\n\n"
        f"New Azure spending: **${budget['spent_micro_usd'] / 1e6:.2f} "
        f"/ ${meta['max_new_usd']:.2f}**. "
        f"Outstanding/unknown reservations: ${reserved:.2f}.\n\n"
        f"Resume point: iteration {meta['start_iteration']}. Later stages use clarified refinement "
        "instructions and fresh responses. The original failed run remains unchanged.\n\n"
        f"Original record: [{meta['run_id']}]({meta['original']}/report.json).\n"
    )
    if state.get("error"):
        text += f"\nError: {state['error']}\n"
    if state.get("result"):
        text += f"\nFull iterations completed: {state['result']['iterations_completed']}/25.\n"
    path = root / "LIVE_PROGRESS.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run(root, check_only=False):
    from onc_co_scientist.expected_surprising.experiment import _provenance, run_cell
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans

    meta = read(root / "recovery.json")
    for name, expected in {**meta["hashes"], **meta["prefix"]}.items():
        if sha(root / name) != expected:
            raise ValueError(f"Recovery artifact changed: {name}")
    original = Path(meta["original"])
    for name in ("run", "report"):
        if sha(original / f"{name}.json") != meta[f"original_{name}_sha256"]:
            raise ValueError("Original run changed")
    spec = load_experiment_spec(root / meta["condition"] / "config.yaml")
    plan = next(p for p in build_run_plans(spec) if p.run_id == meta["run_id"])
    if plan.model.id != "terra_medium" or plan.federation.sites != 4:
        raise ValueError("Unexpected recovery identity")
    # All frozen scientific code except the documented federation prompt must match.
    lineage = read(original / "provenance.json")
    new_provenance = _provenance(spec, plan, spec.fingerprint())
    for section in ("public", "private"):
        if lineage[section] != new_provenance[section]:
            raise ValueError("Scientific input changed")
    allowed = {
        "expected_surprising/federation.py",
        "providers/azure_federation.py",
        "providers/azure_budget.py",
    }
    if {
        k
        for k, v in new_provenance["implementation"].items()
        if lineage["implementation"].get(k) != v
    } - allowed:
        raise ValueError("Unexpected scientific implementation change")
    run_dir = root / meta["condition"] / "runs" / meta["run_id"]
    if not (run_dir / "provenance.json").exists():
        write(run_dir / "provenance.json", new_provenance)
    install_recovery_coordinator(root, meta["start_iteration"], check_only)
    progress(root, meta)
    if check_only:
        try:
            run_cell(spec, plan, root / meta["condition"], spec.fingerprint(), resume=True)
        except Exception as exc:
            if not isinstance(exc.__cause__, ReplayVerified):
                raise
        else:
            raise ValueError("Replay check did not reach the recovery boundary")
        print(json.dumps({"status": "verified", "prefix_calls": len(meta["prefix"])}))
        return
    with (root / "control/driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "control/execution.json", {"status": "active", "pid": os.getpid()})
        stop = threading.Event()

        def watch():
            while not stop.wait(20):
                progress(root, meta)

        observer = threading.Thread(target=watch, daemon=True)
        observer.start()
        try:
            result = run_cell(spec, plan, root / meta["condition"], spec.fingerprint(), resume=True)
            state = {"status": result["status"], "result": result}
        except Exception as exc:
            state = {"status": "paused", "error": repr(exc)}
        finally:
            stop.set()
            observer.join()
        write(root / "control/execution.json", state)
        progress(root, meta)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--condition", default="named")
    parser.add_argument("--run-id")
    parser.add_argument("--cap", type=float, default=500)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.parent:
        prepare(args.parent.resolve(), args.root.resolve(), args.condition, args.run_id, args.cap)
    else:
        run(args.root.resolve(), args.check)
