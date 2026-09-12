"""Freeze the reserved, unopened identities in a separate Azure experiment bundle."""

import argparse
import copy
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import build_run_plans, run_experiment
from onc_co_scientist.providers.codex_cli import CodexCLIConfig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(source, endpoint, preflight):
    source = source.resolve()
    repo = Path(__file__).resolve().parents[2]
    transition = json.loads((source / "azure_transition.json").read_text())
    out = Path(transition["target_root"])
    CodexCLIConfig(model_id="validation", backend="azure", azure_endpoint=endpoint)
    original = json.loads((source / "frozen_manifest.json").read_text())
    for name, expected in original["hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Original frozen file changed: {name}")
    queued = {(r["condition"], r["run_id"]) for r in transition["queued"]}
    active = {(r["condition"], r["run_id"]) for r in transition["active"]}
    if len(queued) != len(transition["queued"]) or queued & active or not queued:
        raise ValueError("Queued identities must be unique, unopened and disjoint from active runs")
    for condition, rid in queued:
        path = source / condition / "runs" / rid
        if not path.is_file() or json.loads(path.read_text()).get("status") != "reserved_for_azure":
            raise ValueError(f"Missing admission reservation: {rid}")
    checks = json.loads(preflight.read_text())
    if not checks.get("personal_auth_unchanged") or not checks.get("excluded_from_experiment"):
        raise ValueError("Azure preflight must preserve personal auth and exclude smoke calls")
    expected_models = set()
    for condition in ("named", "masked"):
        raw = yaml.safe_load((source / condition / "config.yaml").read_text())
        expected_models.update(
            m["model_id"] for m in raw["models"] if m["id"] in transition["models"]
        )
    passed = {
        c["model"]
        for c in checks["checks"]
        if c["status"] == "passed"
        and c["metrics"]["backend"] == "azure"
        and c["metrics"]["azure_endpoint"] == endpoint
    }
    if expected_models != passed:
        raise ValueError(
            "Azure preflight must pass for exactly the transferred models and endpoint"
        )
    out.mkdir(parents=True, exist_ok=False)
    # Preserve all scientific implementation bytes; replace transport and admission only.
    shutil.copytree(
        source / "source/src",
        out / "source/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        repo / "src/onc_co_scientist/providers/codex_cli.py",
        out / "source/src/onc_co_scientist/providers/codex_cli.py",
    )
    shutil.copy2(
        repo / "scripts/expected_surprising/run_federated_grid.py",
        out / "source/run_federated_grid.py",
    )
    shutil.copy2(Path(__file__), out / "source/prepare_azure_federation.py")
    selected_plans = []
    for condition in ("named", "masked"):
        folder = out / condition
        folder.mkdir()
        shutil.copytree(source / condition / "input_data", folder / "input_data")
        config = yaml.safe_load((source / condition / "config.yaml").read_text())
        config.update(experiment_id=f"{out.name}_{condition}", output_root=str(folder))
        config["expected_surprising"]["root"] = str(folder / "input_data")
        for model in config["models"]:
            if model["provider_config"]["kind"] == "codex_cli":
                model["provider_config"].update(backend="azure", azure_endpoint=endpoint)
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        spec = load_experiment_spec(folder / "config.yaml")
        run_experiment(spec, dry_run=True)
        plans = build_run_plans(spec)
        selected_plans += [
            dict(condition=condition, **p.public_dict())
            for p in plans
            if (condition, p.run_id) in queued
        ]
    if {(r["condition"], r["run_id"]) for r in selected_plans} != queued:
        raise ValueError("Replacement plans changed scientific identities")
    for model in transition["models"]:
        rows = [
            {k: r[k] for k in ("condition", "run_id")}
            for r in transition["queued"]
            if r["model"] == model
        ]
        write(out / "selections" / f"{model}.json", rows)
    write(out / "grid.json", selected_plans)
    shutil.copy2(source / "partition_audit.json", out / "partition_audit.json")
    write(out / "azure_preflight.json", checks)
    policy = json.loads((source / "release_policy.json").read_text())
    policy["instruction"] = (
        "Only reserved Sol/Terra/Luna identities; predecessor drain gate is mandatory. "
        "All other models remain held."
    )
    write(out / "release_policy.json", policy)
    frozen_transition = {
        k: copy.deepcopy(transition[k]) for k in ("source_root", "models", "active", "queued")
    }
    (out / "README.md").write_text(
        "# Azure continuation of the federated grid\n\n"
        f"{len(queued)} unopened Sol/Terra/Luna identities, 10 workers per model. "
        f"The {len(active)} original runs finish under their original transport. "
        "Source datasets, partitions, model names, reasoning, workflow and budgets are preserved. "
        "The provider and bundle provenance change explicitly.\n\n"
        "Every CLI attempt refreshes its Azure Entra token. No credentials are frozen, "
        "and there is no personal-account fallback. Normal Codex sessions are unchanged. "
        "Astra, Claude, Biomni, Gemma and Qwen remain held.\n\n"
        "The runner refuses to start until all original drivers have drained. "
        "Reservation-file worker errors in their execution records mean admission was blocked "
        "before a provider was created; they are not scientific failures.\n\n"
        f"[Combined live progress]({source / 'LIVE_PROGRESS.md'})\n"
    )
    hashes = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "release_policy.json"
    }
    write(
        out / "frozen_manifest.json",
        dict(
            frozen_at=datetime.now(UTC).isoformat(),
            hashes=hashes,
            source_manifest_sha256=digest(source / "frozen_manifest.json"),
            azure_transition=frozen_transition,
            runs=len(queued),
            released_runs=len(queued),
        ),
    )
    transition.update(
        status="draining",
        azure_bundle_ready=True,
        endpoint=endpoint,
        preflight="passed",
        prepared_at=datetime.now(UTC).isoformat(),
    )
    write(source / "azure_transition.json", transition)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.endpoint, args.preflight))
