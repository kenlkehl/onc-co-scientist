"""Preparation, release gate and execution for the native Biomni federation arm."""

from __future__ import annotations

import fcntl
import json
import random
import secrets
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from ..expected_surprising.federation_spec import FederationCell
from ..expected_surprising.packaging import sha256
from ..expected_surprising.schemas import WorkflowVersions
from ..expected_surprising.site_statistics import partition_frame
from ..expected_surprising.workflow import finalize_workflow
from .exchanges import NativeController
from .federation import NativeFederation, NativeFederationSpec, NativeSite
from .runner import preflight, provenance, task_masking, validate_inputs
from .transport import RequestBudget, RunBroker, fingerprint, usage_summary, write_json


def load_config(path):
    return NativeFederationSpec.model_validate(yaml.safe_load(Path(path).read_text()))


def prepare(grid):
    """Freeze all 80 native cells; this function performs no provider/network requests."""
    grid = Path(grid).resolve()
    root = grid / "biomni"
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    shutil.copytree(
        repo / "src", root / "source/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(
        repo / "scripts/expected_surprising/biomni_federation.py",
        root / "source/biomni_federation.py",
    )
    rows = []
    for condition in ("named", "masked"):
        source = grid / condition / "input_data"
        shutil.copytree(source, root / "input_data" / condition)
        original = yaml.safe_load((grid / f"biomni_{condition}_reference.yaml").read_text())
        for sites in (2, 4):
            config = {
                **original,
                "schema_version": "external-native-federation-1",
                "execution_mode": "external-native-federation",
                "sites": sites,
                "partition_seed": 20260908,
                "max_requests": 4200,
                "experiment_id": f"biomni_federation_{condition}_n{sites}",
                "input_root": str(root / "input_data" / condition),
                "output_root": str(root / f"{condition}-n{sites}"),
            }
            spec = NativeFederationSpec.model_validate(config)
            # Full offline scientific validation; no native installation is needed for planning.
            tasks, _, _ = validate_inputs(spec)
            path = root / "configs" / f"{condition}-n{sites}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False))
            for task in tasks:
                for repeat in range(1, spec.replicates + 1):
                    run_id = f"{task.id}__biomni_native__r{repeat:03d}__n{sites}-random"
                    rows.append(
                        dict(
                            condition=condition,
                            sites=sites,
                            partition="random",
                            semantic_condition=task.semantic_condition,
                            replicate=repeat,
                            task_id=task.id,
                            run_id=run_id,
                            model_profile="biomni_native",
                            workflow_id="external-orchestrator-biomni-native",
                            config=str(path.relative_to(root)),
                            run_path=f"{condition}-n{sites}/runs/{run_id}",
                        )
                    )
    random.Random(20260908).shuffle(rows)
    assert len(rows) == 80
    write_json(root / "plan.json", rows)
    write_json(
        root / "execution.json",
        dict(
            status="held",
            planned_runs=len(rows),
            reason="Implementation only; native and central model calls remain held.",
        ),
    )
    (root / "README.md").write_text(
        "# Native Biomni federation\n\n"
        "**Implemented and held. No model calls or live pilots have started.**\n\n"
        "80 cells: named/masked × expected/surprising × 2/4 random sites × 10 repeats. "
        "25 reporting rounds. External Qwen/xhigh orchestration and native Qwen/xhigh Biomni "
        "at each site, preserving code, conversation and namespace across "
        "research/review handoffs.\n\n"
        "One common claim ledger, 12 canonical analyses per round and the existing shared "
        "validation policy. Canonical estimates combine trusted site sufficient statistics; "
        "small cells are suppressed. Central context contains aggregates and proposals, not rows. "
        "The 4,200-request allowance includes the orchestrator, all sites, helpers and failures. "
        "Site-local exploratory code remains native and is outside the "
        "canonical-analysis count.\n\n"
        "Launch remains blocked by the parent release_policy.json. A later explicit release "
        "must be followed by full-length excluded pilots for all four configurations and the "
        "runtime/data-lake/endpoint preflight. The configured Biomni installation is on its "
        "original worker host; validate those paths there before launching.\n\n"
        "Run the frozen source/biomni_federation.py with its source/src on PYTHONPATH. "
        "Subcommands: pilot, run (optional --resume). Both require explicit release. "
        "run additionally requires matching passed pilot fingerprints.\n\n"
        "Completed dispatches replay their checked handoffs without calling models. Interrupted "
        "native dispatches resume only from a matching conversation/namespace checkpoint; "
        "unsafe checkpoints fail without replaying arbitrary code. Terminal failed experiments "
        "are retained. Fresh formal output roots are required after source/config changes.\n\n"
        "[Live progress](../LIVE_PROGRESS.md) · [Plan](plan.json)\n"
    )
    hashes = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "execution.json"
    }
    write_json(
        root / "frozen_manifest.json",
        dict(
            frozen_at=datetime.now(UTC).isoformat(), hashes=hashes, runs=80, original_grid=str(grid)
        ),
    )
    return dict(root=str(root), runs=80, status="held")


def check_release(root):
    root = Path(root).resolve()
    policy = json.loads((root.parent / "release_policy.json").read_text())
    if "biomni_native" not in policy.get("released_models", []):
        raise ValueError("Biomni is held; explicit user release is required before any live call")


def verify_frozen(root):
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    for name, expected in frozen["hashes"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Frozen Biomni file changed: {name}")
    if not Path(__file__).is_relative_to(root / "source"):
        raise ValueError("Run with this bundle's frozen source/src on PYTHONPATH")


def run_cell(
    spec, task, pair, policy, replicate, digest, *, resume=False, runner=None, central_factory=None
):
    run_id = f"{task.id}__biomni_native__r{replicate:03d}__n{spec.sites}-random"
    root = spec.output_root / "runs" / run_id
    path = root / "run.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if not resume or prior["fingerprint"] != digest:
            raise ValueError("Existing native federation run; resume requires identical provenance")
        if prior["status"] in {"completed", "failed"}:
            for name, expected in prior["artifact_sha256"].items():
                if sha256(root / name) != expected:
                    raise ValueError("Saved native federation artifact changed")
            return prior
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Partial run has no durable run identity")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(path, dict(run_id=run_id, fingerprint=digest, status="running", started_at=started))
    frame = pd.read_parquet(task.public_workspace / "dataset.parquet")
    controller = NativeController(
        pair, task.semantic_condition, frame, policy, f"replicate-{replicate:03d}"
    )
    masking = task_masking(task)
    if masking:
        from ..expected_surprising.masking import MaskedValidationService

        controller.spec = masking.spec(pair)
        controller.outcomes = {o.name: o for o in controller.spec.outcomes}
        controller.service = MaskedValidationService(
            pair, task.semantic_condition, f"replicate-{replicate:03d}", policy, masking
        )
    frames, _ = partition_frame(
        frame,
        FederationCell(sites=spec.sites, seed=spec.partition_seed),
        pair.pair_id,
        f"replicate-{replicate:03d}",
    )
    used = sum(1 for _ in root.rglob("llm/*.json"))
    budget = RequestBudget(spec.max_requests, used)
    sites = {}
    for site, local in frames.items():
        folder = root / "sites" / site
        public = folder / "public"
        public.mkdir(parents=True, exist_ok=True)
        data = public / "dataset.parquet"
        if data.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(data), local.reset_index(drop=True))
        else:
            local.reset_index(drop=True).to_parquet(data, index=False)
        for name in ("task.json", "data_dictionary.json"):
            obj = json.loads((task.public_workspace / name).read_text())
            if name == "task.json":
                obj.update(n=len(local), site=site)
            target = public / name
            if target.exists() and json.loads(target.read_text()) != obj:
                raise ValueError("Native site public metadata changed")
            write_json(target, obj)
        sites[site] = NativeSite(spec, site, public, folder, budget, runner=runner, resume=resume)
    central = (
        central_factory(spec, root / "central", budget)
        if central_factory
        else RunBroker(spec, None, root / "central", secrets.token_hex(32), shared_budget=budget)
    )
    coordinator = NativeFederation(spec, controller, root, central, sites)
    failure = None
    try:
        coordinator.run()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    errors = [{"error": failure}] if failure else []
    report = finalize_workflow(
        controller,
        task.public_workspace,
        root,
        model_id=spec.model,
        run_id=run_id,
        iterations=spec.rounds,
        versions=WorkflowVersions().model_dump(),
        stage_failure_policy=spec.stage_failure_policy,
        errors=errors,
        exhausted=errors,
    )
    scoped = {}
    for scope, folder in {
        "central": root / "central",
        **{s: root / "sites" / s for s in sites},
    }.items():
        records = [json.loads(p.read_text()) for p in sorted(folder.rglob("llm/*.json"))]
        scoped[scope] = usage_summary(records)
    records = [json.loads(p.read_text()) for p in sorted(root.rglob("llm/*.json"))]
    report.update(
        harness="external-native-federation",
        workflow_id="external-orchestrator-biomni-native",
        clock="reporting_round",
        completed_rounds=coordinator.gateway.state["completed_rounds"],
        initial_expectations_blinded=False,
        federation=coordinator.partition,
        usage=usage_summary(records),
        scoped_usage=scoped,
        dataset_view="masked" if masking else "named",
        source_dataset_sha256=sha256(task.public_workspace / "dataset.parquet"),
        native_resources=spec.resource_policy,
        central_model=spec.model,
        site_model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        error=failure,
    )
    report.pop("successful_stages", None)
    report.pop("expected_stages", None)
    report["completion_rate"] = report["completed_rounds"] / spec.rounds
    report["termination"] = failure or "reporting_rounds_complete"
    report["retry_policy"] = dict(
        atomic_exchange=True,
        central_retries=spec.central_retries,
        native_automatic_retries=0,
        stage_failure_policy=spec.stage_failure_policy,
    )
    assignment = json.loads((task.private_evaluation_path.parent / "assignment.json").read_text())
    origin = assignment[task.semantic_condition]
    report["source_dataset_sha256"] = origin.get("source_sha256", origin["sha256"])
    write_json(root / "report.json", report)
    result = dict(
        run_id=run_id,
        fingerprint=digest,
        status="failed" if failure else "completed",
        error=failure,
        completed_rounds=report["completed_rounds"],
        usage=report["usage"],
        scoped_usage=scoped,
        duration_seconds=time.time() - started,
        artifact_sha256={
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink() and p != path and p.name != "worker.log"
        },
    )
    write_json(path, result)
    return result


def execute(root, *, pilot=False, resume=False):
    root = Path(root).resolve()
    # Before preflight, provider initialization, pilot generation or any live request.
    check_release(root)
    verify_frozen(root)
    lock = (root / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = json.loads((root / "plan.json").read_text())
    specs = {row["config"]: load_config(root / row["config"]) for row in rows}
    metadata = {}
    for name, spec in specs.items():
        tasks, pair, policy = validate_inputs(spec)
        preflight(spec)
        proof = provenance(spec, tasks)
        digest = fingerprint(proof)
        metadata[name] = (tasks, pair, policy, digest)
        if not pilot:
            gate = json.loads((root / "pilots" / Path(name).stem / "gate.json").read_text())
            if gate.get("fingerprint") != digest or gate.get("status") != "passed":
                raise ValueError("Full-length excluded native federation pilot missing or stale")
        write_json(root / "runtime_provenance" / f"{Path(name).stem}.json", proof)
    results = []
    selected = rows
    if pilot:
        selected = [r for r in rows if r["replicate"] == 1]
    write_json(
        root / "execution.json",
        dict(status="running_pilots" if pilot else "running", selected_runs=len(selected)),
    )
    for row in selected:
        check_release(root)
        original = specs[row["config"]]
        tasks, pair, policy, digest = metadata[row["config"]]
        if fingerprint(provenance(original, tasks)) != digest:
            raise ValueError("Native runtime provenance changed during the campaign")
        spec = original.model_copy(deep=True)
        task = next(t for t in tasks if t.id == row["task_id"])
        if pilot:
            # Fresh datasets, both versions, all 25 rounds; exclude from formal scores.
            from ..expected_surprising.generation import sample

            base = root / "pilots" / Path(row["config"]).stem
            spec.output_root = base / "results"
            public = base / "public" / task.id
            public.mkdir(parents=True, exist_ok=True)
            frame = sample(pair, task.semantic_condition, seed=990012)
            masking = task_masking(task)
            if masking:
                frame = masking.frame(frame)
            target = public / "dataset.parquet"
            if not target.exists():
                frame.to_parquet(target, index=False)
            for name in ("task.json", "data_dictionary.json"):
                shutil.copy2(task.public_workspace / name, public / name)
            task = task.model_copy(update={"public_workspace": public})
        result = run_cell(spec, task, pair, policy, row["replicate"], digest, resume=resume)
        results.append({**row, **result})
        write_json(root / ("pilot_summary.json" if pilot else "summary.json"), results)
    if pilot:
        for name, (_, _, _, digest) in metadata.items():
            subset = [r for r in results if r["config"] == name]
            passed = len(subset) == 2 and all(
                r["status"] == "completed" and r["completed_rounds"] == 25 for r in subset
            )
            write_json(
                root / "pilots" / Path(name).stem / "gate.json",
                dict(status="passed" if passed else "failed", fingerprint=digest, rounds=25),
            )
    write_json(
        root / "execution.json",
        dict(
            status="pilots_finished" if pilot else "completed",
            finished_runs=len(results),
            failed_runs=sum(r["status"] == "failed" for r in results),
        ),
    )
    return results
