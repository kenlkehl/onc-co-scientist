"""Prepare a held cache/budget override; resume only through explicit release and budget gates."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path

MODELS = ("sol_medium", "terra_medium", "luna_medium")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_original(root):
    for path, expected in read(root / "frozen_manifest.json")["hashes"].items():
        if sha(root / path) != expected:
            raise ValueError(f"Original frozen file changed: {path}")


def retail_rates(path):
    data = read(path)
    if data.get("NextPageLink"):
        raise ValueError("Retail price snapshot is incomplete")
    result = {}
    for name in ("sol", "terra", "luna"):
        model = f"gpt-5.6-{name}"
        result[model] = {"short_context_tokens": 272000, "context_window": 1050000}
        for band, label in (("short", "ShortCo"), ("long", "LongCo")):
            result[model][band] = {}
            for key, meter in (
                ("input", "Inp"),
                ("cached", "Cd Inp"),
                ("write", "Cd Wr"),
                ("output", "Opt"),
            ):
                rows = [
                    x
                    for x in data["Items"]
                    if (
                        x["meterName"] == f"5.6 {name} {label} {meter} Std Gl 1M Tokens"
                        and x["armRegionName"] == "eastus2"
                        and x["type"] == "Consumption"
                        and x["currencyCode"] == "USD"
                        and x["unitOfMeasure"] == "1M"
                    )
                ]
                if len(rows) != 1:
                    raise ValueError(f"Missing or ambiguous retail meter: {model}/{band}/{key}")
                result[model][band][key] = rows[0]["retailPrice"]
    return result


def prepare(root, prices):
    from onc_co_scientist.providers.azure_budget import atomic_json

    verify_original(root)
    if read(root / "release_policy.json").get("released_models"):
        raise ValueError("Pause and hold the experiment before preparing this override")
    locks = []
    try:
        for model in MODELS:
            lock = (root / "control" / model / "driver.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        rates = retail_rates(prices)
        target = root / "control/cache_fix_v1"
        target.mkdir()  # Never overwrite an existing spending ledger or override.
        source = target / "source"
        source.mkdir()
        repo = Path(__file__).resolve().parents[2]
        files = [
            repo / "src/onc_co_scientist/providers" / f"{name}.py"
            for name in ("azure_budget", "azure_federation")
        ]
        files += [Path(__file__), repo / "scripts/expected_surprising/monitor_federated_grid.py"]
        for path in files:
            shutil.copy2(path, source / path.name)
        shutil.copy2(prices, source / "retail_rates.json")
        selected = {m: read(root / "selections" / f"{m}.json") for m in MODELS}
        if sum(map(len, selected.values())) != 30:
            raise ValueError("This continuation is restricted to the initial 30 identities")
        atomic_json(
            target / "spend_policy.json",
            {
                "enabled": False,
                "additional_budget_usd": 0,
                "release_policy_path": str(root / "release_policy.json"),
                "rates_valid_until": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
                "rates": rates,
                "model_profiles": {f"gpt-5.6-{m.split('_')[0]}": m for m in MODELS},
                "max_unknown_attempts": 3,
                "max_reusable_prefix_misses": 6,
                "scope": "Additional usage after cache transport override; prior spend excluded",
            },
        )
        atomic_json(
            target / "spend_state.json",
            {
                "spent_micro_usd": 0,
                "attempts": {},
                "cache_prefixes": {},
            },
        )
        atomic_json(
            target / "manifest.json",
            {
                "version": "azure-federation-cache-v1",
                "prepared_at": datetime.now(UTC).isoformat(),
                "root": str(root),
                "original_manifest_sha256": sha(root / "frozen_manifest.json"),
                "source_hashes": {p.name: sha(p) for p in source.iterdir()},
                "selection": selected,
                "scientific_source": "original frozen source unchanged",
                "prompt_change": "Responses message boundaries, explicit cache prefixes; iteration "
                "header moved to end; no evidence omitted",
            },
        )
        atomic_json(
            root / "control/required_transport.json",
            {
                "status": "prepared_held",
                "manifest": str(target / "manifest.json"),
                "instruction": "Resume only with cache_fix_v1/source/federation_cache_fix.py, "
                "after explicit user release and additional spending authorization.",
            },
        )
        return target
    finally:
        for lock in locks:
            lock.close()


def install(root):
    """Load only two audited transport modules into the original scientific package."""
    target = root / "control/cache_fix_v1"
    manifest = read(target / "manifest.json")
    verify_original(root)
    if sha(root / "frozen_manifest.json") != manifest["original_manifest_sha256"]:
        raise ValueError("Original manifest changed")
    for name, expected in manifest["source_hashes"].items():
        if sha(target / "source" / name) != expected:
            raise ValueError(f"Override source changed: {name}")
    if read(target / "spend_policy.json")["rates"] != retail_rates(
        target / "source/retail_rates.json"
    ):
        raise ValueError("Spending rates differ from the verified retail snapshot")
    from onc_co_scientist.expected_surprising import experiment

    if not Path(experiment.__file__).resolve().is_relative_to(root / "source/src"):
        raise ValueError("Set PYTHONPATH to the original frozen source/src")
    for name in ("azure_budget", "azure_federation"):
        fullname = "onc_co_scientist.providers." + name
        spec = importlib.util.spec_from_file_location(fullname, target / "source" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    provider = sys.modules["onc_co_scientist.providers.azure_federation"]
    budget = sys.modules["onc_co_scientist.providers.azure_budget"]

    def factory(config):
        if config.get("kind") != "codex_cli" or config.get("backend") != "azure":
            raise ValueError("Cache override is restricted to the federated Azure Codex arms")
        options = {k: v for k, v in config.items() if k != "kind"}
        options.update(
            budget_policy_path=str(target / "spend_policy.json"),
            cache_namespace=str(Path(options["audit_dir"]).resolve()),
        )
        return provider.AzureFederationProvider(provider.AzureFederationConfig(**options))

    experiment.get_provider = factory
    return experiment, budget, manifest


def drive(selection, workers, run, status, paused_type):
    """Admit only worker slots; queued scientific runs are not reported as active."""
    pending, active, finished = list(selection), {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            while pending and len(active) < workers:
                row = pending.pop(0)
                active[pool.submit(run, row)] = row
            status(list(active.values()), len(pending), list(finished))
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = active.pop(future)
                try:
                    finished.append({**row, "status": future.result()["status"]})
                except paused_type as exc:
                    finished.append({**row, "status": "paused", "reason": str(exc)})
                except Exception as exc:
                    finished.append({**row, "status": "worker_error", "error": type(exc).__name__})
    status([], 0, finished)
    return finished


def resume(root, model, workers):
    experiment, budget, manifest = install(root)
    from onc_co_scientist.harness.experiment import load_experiment_spec
    from onc_co_scientist.harness.orchestrator import build_run_plans

    gate = budget.AzureBudget(root / "control/cache_fix_v1/spend_policy.json")
    gate.check(f"gpt-5.6-{model.split('_')[0]}")
    control = root / "control" / model
    with (control / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = {c: load_experiment_spec(root / c / "config.yaml") for c in ("named", "masked")}
        plans = {c: {p.run_id: p for p in build_run_plans(s)} for c, s in specs.items()}
        selection = manifest["selection"][model]
        if read(root / "selections" / f"{model}.json") != selection:
            raise ValueError("Initial-30 selection changed")
        for row in selection:
            plan = plans[row["condition"]][row["run_id"]]
            if plan.model.id != model or plan.federation is None or plan.federation.sites < 2:
                raise ValueError("Override cannot run single-site or other model identities")
        prior = control / "execution.json"
        if prior.exists():
            shutil.copy2(prior, control / f"execution-before-cache-{os.getpid()}.json")
        state = {
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "transport": manifest["version"],
            "selected_runs": len(selection),
            "active": [],
            "queued_runs": len(selection),
            "finished": [],
            "workers": workers,
        }
        budget.atomic_json(prior, state)

        def status(active, queued, finished):
            state.update(
                active=active,
                queued_runs=queued,
                finished=finished,
                updated_at=datetime.now(UTC).isoformat(),
            )
            budget.atomic_json(prior, state)

        def run(row):
            c, rid = row["condition"], row["run_id"]
            return experiment.run_cell(
                specs[c], plans[c][rid], root / c, specs[c].fingerprint(), resume=True
            )

        results = drive(selection, workers, run, status, budget.ExperimentPaused)
        state["status"] = (
            "completed" if all(r["status"] == "completed" for r in results) else "paused"
        )
        state["updated_at"] = datetime.now(UTC).isoformat()
        if state["status"] == "completed":
            state["completed_at"] = state["updated_at"]
        budget.atomic_json(prior, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument(
        "--check", action="store_true", help="Verify provenance and gates, no inference"
    )
    modes.add_argument("--resume-model", choices=MODELS)
    parser.add_argument("--retail-rates", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        if args.retail_rates is None:
            parser.error("--prepare requires --retail-rates")
        print(prepare(root, args.retail_rates))
    elif args.check:
        _, budget, manifest = install(root)
        gates = {}
        for model in MODELS:
            try:
                budget.AzureBudget(root / "control/cache_fix_v1/spend_policy.json").check(
                    f"gpt-5.6-{model.split('_')[0]}"
                )
                gates[model] = "released"
            except budget.ExperimentPaused as exc:
                gates[model] = str(exc)
        print(json.dumps({"verified": manifest["version"], "gates": gates, "model_calls": 0}))
    else:
        resume(root, args.resume_model, args.workers)


if __name__ == "__main__":
    main()
