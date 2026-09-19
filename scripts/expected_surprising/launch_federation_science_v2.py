"""Frozen, bounded initial-30 federation run with uncached Azure inference."""

import argparse
import ast
import copy
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from onc_co_scientist.providers.azure_budget import AzureBudget, ExperimentPaused, atomic_json

MODELS = ("sol_medium", "terra_medium", "luna_medium")


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remaining_micro(policy, state):
    """The predecessor policy already subtracts every diagnostic/unknown reservation."""
    if policy["enabled"] or policy.get("diagnostic_pending_allocation_usd", 0):
        raise ValueError("Predecessor must be held without an active diagnostic allocation")
    if any(a["status"] in {"reserved", "unknown"} for a in state["attempts"].values()):
        raise ValueError("Settle or separately reserve predecessor scientific attempts first")
    left = round(policy["additional_budget_usd"] * 1_000_000) - state["spent_micro_usd"]
    if left <= 0:
        raise ValueError("No authorized spending remains")
    return left


def prepare(root, predecessor):
    import yaml

    old_gate = predecessor / "control/cache_fix_v1/spend_policy.json"
    policy, state = read(old_gate), read(old_gate.with_name("spend_state.json"))
    available = remaining_micro(policy, state)
    if read(predecessor / "release_policy.json")["released_models"]:
        raise ValueError("Predecessor model release gate is open")
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    root.mkdir(parents=True)  # Refuse overwrite of a versioned bundle.
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            "--format=tar",
            commit,
            "src",
            "scripts/expected_surprising/launch_federation_science_v2.py",
            "pyproject.toml",
        ],
        cwd=repo,
    )
    (root / "source.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(root / "source", filter="data")
    selected = {m: read(predecessor / "selections" / f"{m}.json") for m in MODELS}
    if any(len(rows) != 10 for rows in selected.values()):
        raise ValueError("Expected exactly ten original identities per model")
    for model, rows in selected.items():
        atomic_json(root / "selections" / f"{model}.json", rows)
    for condition in ("named", "masked"):
        folder = root / condition
        folder.mkdir()
        shutil.copytree(predecessor / condition / "input_data", folder / "input_data")
        config = yaml.safe_load((predecessor / condition / "config.yaml").read_text())
        config.update(experiment_id=f"{root.name}_{condition}", output_root=str(folder))
        config["expected_surprising"]["root"] = str(folder / "input_data")
        config["federation"]["context_policy"] = "federated_v2"
        for model in config["models"]:
            if model["id"] not in MODELS:
                raise ValueError("Only Sol, Terra and Luna may be prepared")
            options = model["provider_config"]
            if options.get("kind") != "codex_cli" or options.get("backend") != "azure":
                raise ValueError("Only Azure federation providers may be prepared")
            options.update(
                budget_policy_path=str(root / "control/spend_policy.json"), prompt_caching=False
            )
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(
        root / "release_policy.json",
        {
            "released_models": [],
            "held_models": list(MODELS),
            "instruction": "Initial 30 only; pause for review at completion or spending limit.",
        },
    )
    new_policy = copy.deepcopy(policy)
    new_policy.update(
        enabled=False,
        additional_budget_usd=available / 1e6,
        cache_miss_pause_enabled=False,
        release_policy_path=str(root / "release_policy.json"),
        scope=(
            "Fresh v2 initial 30. No cache gate or cache writes. "
            "Remaining allocation within user-authorized $1000 additional allowance."
        ),
    )
    new_policy["diagnostic_pending_allocation_usd"] = 0
    atomic_json(root / "control/spend_policy.json", new_policy)
    atomic_json(root / "control/spend_state.json", {"spent_micro_usd": 0, "attempts": {}})
    hashes = {}
    for folder in [
        root / "source",
        root / "named/input_data",
        root / "masked/input_data",
        root / "selections",
    ]:
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes[str(path.relative_to(root))] = sha(path)
    for condition in ("named", "masked"):
        p = root / condition / "config.yaml"
        hashes[str(p.relative_to(root))] = sha(p)
    prior_spent = state["spent_micro_usd"] + round(policy["diagnostic_spend_usd"] * 1e6)
    prior_unknown = round(policy.get("diagnostic_unknown_reservation_usd", 0) * 1e6)
    manifest = dict(
        created_at=now(),
        commit=commit,
        hashes=hashes,
        predecessor=str(predecessor),
        selection=selected,
        allocated_micro_usd=available,
        previous_recorded_additional_micro_usd=prior_spent,
        previous_unknown_micro_usd=prior_unknown,
        total_authorized_micro_usd=round(policy["total_authorized_additional_budget_usd"] * 1e6),
        instruction=(
            "User authorized moving on from cache gate. New version of the same "
            "30 identities; do not mix or overwrite old results. Cache writes disabled. "
            "Normal single-site code/defaults unchanged."
        ),
    )
    if available + prior_spent + prior_unknown != manifest["total_authorized_micro_usd"]:
        raise ValueError("Allocation does not reconcile to total user authorization")
    atomic_json(root / "frozen_manifest.json", manifest)
    atomic_json(
        root / "control/allocation.json", {"status": "prepared_not_transferred", **manifest}
    )
    # Validate using frozen code, never the possibly unrelated working-tree changes.
    command = [
        sys.executable,
        str(root / "source/scripts/expected_surprising/launch_federation_science_v2.py"),
        "--root",
        str(root),
        "--check",
    ]
    subprocess.run(command, env={**os.environ, "PYTHONPATH": str(root / "source/src")}, check=True)
    print(root, flush=True)


def verify(root):
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans

    manifest = read(root / "frozen_manifest.json")
    with ThreadPoolExecutor(max_workers=12) as pool:
        actual = list(pool.map(lambda name: sha(root / name), manifest["hashes"]))
    if actual != list(manifest["hashes"].values()):
        raise ValueError("Frozen source, data, selection or config changed")
    specs = {c: load_experiment_spec(root / c / "config.yaml") for c in ("named", "masked")}
    plans = {c: {p.run_id: p for p in build_run_plans(spec)} for c, spec in specs.items()}
    grid = []
    for model, rows in manifest["selection"].items():
        for row in rows:
            p = plans[row["condition"]][row["run_id"]]
            if (
                p.model.id != model
                or p.federation.context_policy != "federated_v2"
                or p.federation.sites not in (2, 4)
            ):
                raise ValueError("Selected scientific identity changed")
            if p.model.provider_config.get("prompt_caching") is not False:
                raise ValueError("Cache writes must be disabled")
            grid.append({"condition": row["condition"], **p.public_dict()})
    if len(grid) != 30 or len({(r["condition"], r["run_id"]) for r in grid}) != 30:
        raise ValueError("Exactly 30 unique identities required")
    atomic_json(root / "grid.json", grid)
    return manifest, specs, plans


def run_model(root, model, workers):
    from onc_co_scientist.expected_surprising.experiment import run_cell

    manifest, specs, plans = verify(root)
    folder = root / "control" / model
    folder.mkdir(exist_ok=True)
    with (folder / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(
            status="running",
            pid=os.getpid(),
            started_at=now(),
            workers=workers,
            active=[],
            queued_runs=10,
            finished=[],
        )

        def save():
            status["updated_at"] = now()
            atomic_json(folder / "execution.json", status)

        save()
        pending = list(manifest["selection"][model])
        active = {}

        def work(row):
            c, rid = row["condition"], row["run_id"]
            return run_cell(specs[c], plans[c][rid], root / c, specs[c].fingerprint(), resume=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or active:
                while pending and len(active) < workers:
                    row = pending.pop(0)
                    try:
                        AzureBudget(root / "control/spend_policy.json").check(
                            "gpt-5.6-" + model.split("_")[0]
                        )
                    except ExperimentPaused as exc:
                        status["finished"] += [
                            {**x, "status": "paused", "reason": str(exc)} for x in [row, *pending]
                        ]
                        pending.clear()
                        break
                    active[pool.submit(work, row)] = row
                status.update(active=list(active.values()), queued_runs=len(pending))
                save()
                if not active:
                    break
                done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    row = active.pop(future)
                    try:
                        result = future.result()
                        status["finished"].append(
                            {
                                **row,
                                "status": result["status"],
                                "iterations_completed": result.get("iterations_completed"),
                                "error": result.get("error"),
                            }
                        )
                    except ExperimentPaused as exc:
                        status["finished"].append({**row, "status": "paused", "reason": str(exc)})
                    except Exception as exc:
                        status["finished"].append({**row, "status": "failed", "reason": repr(exc)})
            status.update(
                status="completed"
                if all(r["status"] == "completed" for r in status["finished"])
                else "paused",
                active=[],
                queued_runs=0,
                ended_at=now(),
            )
            save()


def install_method_override(root, path, provider, class_name, method_names, record_name, scope):
    """Allow only named infrastructure methods; retain immutable override receipts."""
    baseline = Path(provider.__file__)
    trees = [ast.parse(p.read_text()) for p in (baseline, path)]
    methods = []
    for tree in trees:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        selected = [
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names
        ]
        if {n.name for n in selected} != set(method_names):
            raise ValueError("Missing required override methods")
        cls.body = [n for n in cls.body if n not in selected]
        methods.append(selected)
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError("Override changes more than the allowed infrastructure methods")
    receipt = {
        "baseline_sha256": sha(baseline),
        "override_path": str(path.resolve()),
        "override_sha256": sha(path),
        "launcher_sha256": sha(Path(__file__)),
        "scope": scope,
    }
    record = root / "control" / record_name
    if record.exists():
        prior = read(record)
        if any(prior[k] != receipt[k] for k in receipt if k != "launcher_sha256"):
            raise ValueError("Previously recorded infrastructure override changed")
    else:
        atomic_json(record, receipt)
    audit = (
        root
        / "control/override_receipts"
        / (record.stem + "-" + receipt["launcher_sha256"] + ".json")
    )
    if not audit.exists():
        atomic_json(audit, receipt)
    namespace = dict(vars(provider))
    module = ast.Module(body=methods[1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    for name in method_names:
        setattr(getattr(provider, class_name), name, namespace[name])


def install_transport_override(root, path):
    from onc_co_scientist.providers import azure_federation as provider

    install_method_override(
        root,
        path,
        provider,
        "AzureFederationProvider",
        ["_send"],
        "transport_override.json",
        "AzureFederationProvider._send only; scientific code and requests unchanged",
    )


def install_budget_override(root, path):
    from onc_co_scientist.providers import azure_budget as provider

    install_method_override(
        root,
        path,
        provider,
        "AzureBudget",
        ["_check", "unknown", "settle"],
        "budget_override.json",
        "Consecutive missing-usage guard; reservations and scientific code retained",
    )


def prepare_resume(root):
    previous = read(root / "control/processes.json")
    if any(alive(p) for p in previous["drivers"]):
        raise ValueError("Drain existing drivers before resuming")
    with (root / "control/observer.lock").open("a") as observer_lock:
        fcntl.flock(observer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if read(root / "control/spend_policy.json")["enabled"]:
        raise ValueError("Resume requires a held spending gate")
    if read(root / "control/allocation.json")["status"] != "transferred":
        raise ValueError("Resume must retain an existing budget allocation")
    archive = root / "control" / ("before_resume_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    archive.mkdir()
    for path in (root / "control").glob("*.json"):
        shutil.copy2(path, archive / path.name)
    for model in MODELS:
        path = root / "control" / model / "execution.json"
        shutil.copy2(path, archive / f"{model}.json")
        atomic_json(path, {"status": "starting", "active": [], "queued_runs": 10, "finished": []})


def launch(root, workers, *, resume=False, transport_override=None, budget_override=None):
    manifest, _, _ = verify(root)
    if (root / "control/transport_override.json").exists() and transport_override is None:
        raise ValueError("Recorded transport override is required for this continuation")
    if (root / "control/budget_override.json").exists() and budget_override is None:
        raise ValueError("Recorded budget override is required for this continuation")
    with (root / "control/launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "control/processes.json").exists() and not resume:
            raise ValueError("Already dispatched; inspect before a deliberate resume")
        if resume:
            prepare_resume(root)
        else:
            old_path = Path(manifest["predecessor"]) / "control/cache_fix_v1/spend_policy.json"
            gate = AzureBudget(old_path)
            with gate.locked():
                if remaining_micro(gate.policy, gate.state) != manifest["allocated_micro_usd"]:
                    raise ValueError("Predecessor spending changed since preparation")
                prior = gate.policy
                prior.update(
                    additional_budget_usd=gate.state["spent_micro_usd"] / 1e6,
                    transferred_allocation_usd=manifest["allocated_micro_usd"] / 1e6,
                    transferred_to=str(root),
                )
                atomic_json(old_path, prior)
            atomic_json(root / "control/allocation.json", {"status": "transferred", **manifest})
        policy = read(root / "control/spend_policy.json")
        policy["enabled"] = True
        policy.pop("maintenance_reason", None)
        atomic_json(root / "control/spend_policy.json", policy)
        atomic_json(
            root / "release_policy.json",
            {
                "released_models": list(MODELS),
                "held_models": [
                    "astra_medium",
                    "claude_opus5_medium",
                    "biomni_native",
                    "gemma_4_31b",
                    "qwen_3_8_27b",
                ],
                "instruction": "Initial 30 only; no next batch.",
            },
        )
        processes = []
        atomic_json(
            root / "control/processes.json", {"launch_complete": False, "drivers": processes}
        )
        env = {**os.environ, "PYTHONPATH": str(root / "source/src")}
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OCS_AZURE_ACCESS_TOKEN"):
            env.pop(key, None)
        runner = str(Path(__file__).resolve())
        override_args = (
            ["--transport-override", str(transport_override.resolve())]
            if transport_override
            else []
        )
        if budget_override:
            override_args += ["--budget-override", str(budget_override.resolve())]
        for model in MODELS:
            folder = root / "control" / model
            folder.mkdir(exist_ok=True)
            with (folder / "driver.log").open("a") as log:
                p = subprocess.Popen(
                    [
                        sys.executable,
                        runner,
                        "--root",
                        str(root),
                        "--run-model",
                        model,
                        "--workers",
                        str(workers),
                        *override_args,
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                    cwd=root,
                    start_new_session=True,
                )
            stat = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
            processes.append({"model": model, "pid": p.pid, "start_ticks": stat[19]})
            atomic_json(
                root / "control/processes.json", {"launch_complete": False, "drivers": processes}
            )
        atomic_json(
            root / "control/processes.json", {"launch_complete": True, "drivers": processes}
        )
        with (root / "control/observer.log").open("a") as log:
            observer = subprocess.Popen(
                [sys.executable, runner, "--root", str(root), "--observe"],
                env=env,
                stdout=log,
                stderr=log,
                cwd=root,
                start_new_session=True,
            )
        atomic_json(root / "control/observer.json", {"pid": observer.pid})
        print(
            json.dumps({"status": "launched", "root": str(root), "drivers": processes}), flush=True
        )


def alive(process):
    try:
        stat = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] == process["start_ticks"] and stat[0] != "Z"
    except OSError:
        return False


def observe(root):
    manifest = read(root / "frozen_manifest.json")
    predecessor = Path(manifest["predecessor"])
    old_page = predecessor / "LIVE_PROGRESS.md"
    archive = root / "PREVIOUS_LIVE_PROGRESS.md"
    if not archive.exists():
        shutil.copy2(old_page, archive)
    grid = read(root / "grid.json")
    with (root / "control/observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            procs = read(root / "control/processes.json")
            states, done = {}, bool(procs["launch_complete"])
            for process in procs["drivers"]:
                model = process["model"]
                path = root / "control" / model / "execution.json"
                state = (
                    read(path)
                    if path.exists()
                    else {"status": "starting", "active": [], "queued_runs": 10, "finished": []}
                )
                living = alive(process)
                if not living and state["status"] not in ("completed", "paused"):
                    state.update(
                        status="paused",
                        active=[],
                        queued_runs=0,
                        reason="Driver exited; review required",
                    )
                    atomic_json(path, state)
                states[model] = state
                done = done and not living
            budget = read(root / "control/spend_state.json")
            per_run = defaultdict(Counter)
            for path, attempt in budget["attempts"].items():
                parts = Path(path).relative_to(root).parts
                count = per_run[(parts[0], parts[2])]
                count["attempts"] += 1
                count[attempt["status"]] += 1
                if attempt["status"] == "settled":
                    count["cost_micro_usd"] += attempt["cost_micro_usd"]
                    count.update(attempt["usage"])
            rows, totals = [], Counter()
            for row in grid:
                state = states.get(row["model_profile"], {})
                key = (row["condition"], row["run_id"])

                def matches(x, selected=key):
                    return (x["condition"], x["run_id"]) == selected

                result = next((x for x in state.get("finished", []) if matches(x)), None)
                status = (
                    result["status"]
                    if result
                    else "active"
                    if any(matches(x) for x in state.get("active", []))
                    else "paused"
                    if state.get("status") == "paused"
                    else "queued"
                )
                totals[status] += 1
                rows.append(
                    {**row, "status": status, "usage": dict(per_run[key]), "result": result}
                )
            reserved = sum(
                a["reserved_micro_usd"]
                for a in budget["attempts"].values()
                if a["status"] in ("reserved", "unknown")
            )
            policy = read(root / "control/spend_policy.json")
            allocation = round(
                policy.get("additional_budget_usd", manifest["allocated_micro_usd"] / 1e6) * 1e6
            )
            authorized = policy.get("total_authorized_additional_budget_usd", 1000)
            remaining = (allocation - budget["spent_micro_usd"] - reserved) / 1e6
            if done:
                gate = AzureBudget(root / "control/spend_policy.json")
                with gate.locked():
                    p = gate.policy
                    p["enabled"] = False
                    atomic_json(root / "control/spend_policy.json", p)
                atomic_json(
                    root / "release_policy.json",
                    {
                        "released_models": [],
                        "held_models": list(MODELS),
                        "instruction": (
                            "Initial 30 completed or paused. Await user review; no next batch."
                        ),
                    },
                )
                atomic_json(
                    root / "control/review_ready.json",
                    {"at": now(), "totals": dict(totals), "reason": budget.get("hold_reason")},
                )
            summary = dict(
                updated_at=now(),
                root=str(root),
                totals=dict(totals),
                authorized_total_usd=authorized,
                new_spend_usd=budget["spent_micro_usd"] / 1e6,
                new_reserved_usd=reserved / 1e6,
                remaining_usd=remaining,
                prior_recorded_usd=manifest["previous_recorded_additional_micro_usd"] / 1e6,
                prior_unknown_usd=manifest["previous_unknown_micro_usd"] / 1e6,
                hold_reason=budget.get("hold_reason"),
                finished=done,
                runs=rows,
            )
            atomic_json(root / "live_progress.json", summary)
            page = (
                f"# Federated science — fixed memory, uncached Azure\n"
                f"\n"
                f"Updated **{summary['updated_at']}**\n"
                f"\n"
                f"**Active {totals['active']} · Queued {totals['queued']} · Completed {totals['completed']} · Failed {totals['failed']} · Paused {totals['paused']}**\n"  # noqa: E501
                f"\n"
                f"These are scientific runs, not canary calls. This is a fresh version of the original 30 identities with federated v2 memory. Earlier results remain in the [previous snapshot]({archive}). Results across versions are kept separate. Only Sol/Terra/Luna are released; the initial 30 pause for review, with no next batch.\n"  # noqa: E501
                f"\n"
                f"Cache writes are disabled; cache misses never pause this batch. The ${authorized:,.0f} additional allowance includes prior recorded spending **${summary['prior_recorded_usd']:.2f}** and prior unknown reservations **${summary['prior_unknown_usd']:.2f}**. New scientific spending **${summary['new_spend_usd']:.2f}**, in-flight/unknown reservations **${summary['new_reserved_usd']:.2f}**, conservatively remaining **${remaining:.2f}**.\n"  # noqa: E501
                f"\n"
                f"Status: {'paused for review' if done else 'running'}. Budget hold: {budget.get('hold_reason') or 'none'}.\n"  # noqa: E501
                f"\n"
                f"| Model | View | Condition | Workflow | Sites | Status | Completed calls | Input tokens | Output tokens | Cost |\n"  # noqa: E501
                f"|---|---|---|---|---:|---|---:|---:|---:|---:|\n"
            )
            for row in rows:
                u = row["usage"]
                page += f"| {row['model_profile']} | {row['condition']} | {row['semantic_condition']} | {row['workflow_id']} | {row['site_count']} | {row['status']} | {u.get('settled', 0)} | {u.get('input_tokens', 0):,} | {u.get('output_tokens', 0):,} | ${u.get('cost_micro_usd', 0) / 1e6:.4f} |\n"  # noqa: E501
            page += (
                f"\n"
                f"[Versioned experiment]({root}) · [Spending policy]({root / 'control/spend_policy.json'}) · [Frozen provenance]({root / 'frozen_manifest.json'})\n"  # noqa: E501
            )
            if (root / "control/transport_override.json").exists():
                page += (
                    "\nTransport retry fix active; frozen science and completed calls retained.\n"
                )
            (root / "LIVE_PROGRESS.md").write_text(page)
            old_page.write_text(page)
            atomic_json(predecessor / "live_progress.json", summary)
            if done:
                return
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run-model", choices=MODELS)
    modes.add_argument("--observe", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--transport-override", type=Path)
    parser.add_argument("--budget-override", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.transport_override:
        install_transport_override(root, args.transport_override.resolve())
    if args.budget_override:
        install_budget_override(root, args.budget_override.resolve())
    if args.prepare:
        if not args.predecessor:
            parser.error("--predecessor required")
        prepare(root, args.predecessor.resolve())
    elif args.check:
        verify(root)
        print(json.dumps({"verified": True, "model_calls": 0, "selected_runs": 30}))
    elif args.launch:
        launch(
            root,
            args.workers,
            resume=args.resume,
            transport_override=args.transport_override,
            budget_override=args.budget_override,
        )
    elif args.run_model:
        run_model(root, args.run_model, args.workers)
    else:
        observe(root)


if __name__ == "__main__":
    main()
