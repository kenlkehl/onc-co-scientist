"""Aim 2 matrix integration for the Aim 1 expected/surprising protocol."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

from ..harness.durable_io import atomic_write_json, durable_read_json, durable_replace
from ..providers.registry import get_provider
from .coordination import COORDINATION_VERSION, StageCoordinator, digest
from .packaging import sha256
from .schemas import PairSpec, ValidationPolicy, WorkflowVersions
from .workflow import STAGES, run_workflow

PROTOCOL = "expected_surprising"


def validate_experiment(spec):
    from ..harness.experiment import SafeguardSpec, default_stages

    if spec.clinical_benchmark is not None:
        raise ValueError("expected_surprising cannot be mixed with clinical_benchmark imports")
    if spec.stages != default_stages():
        raise ValueError("expected_surprising requires the standard four scientific stages")
    if any(t.metadata.get("analysis_protocol") != PROTOCOL for t in spec.tasks):
        raise ValueError("expected_surprising tasks must come from the frozen pair import")
    for workflow in spec.workflows:
        if workflow.federated or workflow.safeguards != SafeguardSpec():
            raise ValueError(
                "expected_surprising supports the three core workflows with the shared evidence "
                "ledger; federation and additional safeguard interventions need separate protocols"
            )
    for model in spec.models:
        if model.adapter != "provider":
            raise ValueError("expected_surprising requires adapter: provider")
        config = model.provider_config
        if model.reasoning_effort and config.get("reasoning_effort") != model.reasoning_effort:
            raise ValueError("reasoning_effort must match provider_config.reasoning_effort")
        if model.site_model_ids or model.central_model_id or model.command or model.extra_args:
            raise ValueError("provider profiles do not use native-runtime or federated options")
        if config.get("kind") not in {
            "vllm_openai",
            "codex_cli",
            "gemini_vertex",
            "anthropic_vertex",
        }:
            raise ValueError("Unknown expected_surprising provider kind")
        if config.get("api_key") not in (None, "EMPTY"):
            raise ValueError("Do not put credentials in experiment manifests")
        # The timeout belongs to the deployment, but cannot silently exceed the matrix limit.
        if config.get("timeout_s", 120 if config["kind"] != "codex_cli" else 1800) > (
            spec.budget.max_runtime_seconds_per_call
        ):
            raise ValueError("provider timeout_s exceeds max_runtime_seconds_per_call")
    if any(
        getattr(spec.budget, name) is not None
        for name in ("max_input_tokens", "max_output_tokens", "max_tool_calls", "max_cost_usd")
    ):
        raise ValueError(
            "expected_surprising uses max_agent_calls and max_tokens_per_call; aggregate token, "
            "tool and cost caps are not supported across all providers"
        )


def import_tasks(source, iterations):
    from ..harness.experiment import TaskSpec

    tasks, found_pairs, found_profiles = [], set(), set()
    for path in sorted((source.root / "private").glob("*/pair.json")):
        pair = PairSpec.model_validate_json(path.read_text())
        if source.pair_ids and pair.pair_id not in source.pair_ids:
            continue
        if source.profiles and pair.profile not in source.profiles:
            continue
        assignment = json.loads((path.parent / "assignment.json").read_text())
        for version in ("expected", "surprising"):
            item = assignment[version]
            public = source.root / "public" / item["task_id"]
            task = json.loads((public / "task.json").read_text())
            if task["versions"] != WorkflowVersions().model_dump():
                raise ValueError(f"Use current ledger packages: {public}")
            if not 1 <= iterations <= task["iterations"]:
                raise ValueError(f"Iterations exceed packaged budget: {public}")
            ValidationPolicy.default(pair.profile, iterations)
            if sha256(public / "dataset.parquet") != item["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {public}")
            tasks.append(
                TaskSpec(
                    id=task["task_id"],
                    semantic_condition=version,
                    prompt="Investigate the supplied dataset using the shared scientific ledger.",
                    public_workspace=public.resolve(),
                    private_evaluation_path=path.resolve(),
                    metadata={"analysis_protocol": PROTOCOL, "pair_id": pair.pair_id},
                )
            )
        found_pairs.add(pair.pair_id)
        found_profiles.add(pair.profile)
    if not tasks or set(source.pair_ids) - found_pairs or set(source.profiles) - found_profiles:
        raise ValueError("No pairs found, or requested pair IDs/profiles are missing")
    return tasks


def implementation_hashes():
    package = Path(__file__).parent.parent
    paths = [
        *Path(__file__).parent.glob("*.py"),
        *(package / "providers").glob("*.py"),
        *(package / "synthetic").rglob("*.py"),
    ]
    paths.extend(
        package / "harness" / name for name in ("experiment.py", "orchestrator.py", "durable_io.py")
    )
    return {str(p.relative_to(package)): sha256(p) for p in sorted(paths)}


def _provenance(spec, plan, fingerprint):
    private = plan.task.private_evaluation_path.parent
    return {
        "spec_fingerprint": fingerprint,
        "coordination_version": COORDINATION_VERSION,
        "implementation": implementation_hashes(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "pyarrow", "pydantic")
        },
        "public": {
            p.name: sha256(p) for p in sorted(plan.task.public_workspace.iterdir()) if p.is_file()
        },
        "private": {
            name: sha256(private / name)
            for name in ("pair.json", "assignment.json", "workflow.json")
        },
    }


def run_cell(spec, plan, root, fingerprint, *, resume):
    """Replay cached participant calls to reconstruct state after an interruption.

    Replay is exact: scientific seeds, code, task bytes, config and every prompt
    must match. No restored result is resampled or billed as another agent call.
    An interrupted in-flight request without a durable response may need reissuing.
    """
    run_dir = root / "runs" / plan.run_id
    run_path = run_dir / "run.json"
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        archive = root / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        durable_replace(run_dir, archive / f"{plan.run_id}__{datetime.now(UTC):%Y%m%dT%H%M%S%f}")
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(spec, plan, fingerprint)
    manifest = run_dir / "provenance.json"
    if manifest.exists():
        if durable_read_json(manifest) != provenance:
            raise RuntimeError(
                "Cannot resume expected/surprising run: inputs or implementation changed"
            )
    elif resume and any(run_dir.iterdir()):
        raise RuntimeError("Cannot resume without expected/surprising provenance")
    else:
        atomic_write_json(manifest, provenance)
    if resume and run_path.exists():
        prior = durable_read_json(run_path)
        if prior.get("scientific_report_sha256"):
            if sha256(run_dir / "report.json") != prior["scientific_report_sha256"]:
                raise RuntimeError("Cannot resume: scientific report changed")
            report = durable_read_json(run_dir / "report.json")
            for path, expected in report["coordination"]["participant_artifact_sha256"].items():
                if sha256(run_dir / path) != expected:
                    raise RuntimeError("Cannot resume: participant artifact changed")
            return {**prior, "resumed": True}
    pair = PairSpec.model_validate_json(plan.task.private_evaluation_path.read_text())
    assignment = durable_read_json(plan.task.private_evaluation_path.parent / "assignment.json")
    version = plan.task.semantic_condition
    if (
        assignment[version]["task_id"] != plan.task.public_workspace.name
        or assignment[version]["sha256"] != provenance["public"]["dataset.parquet"]
    ):
        raise ValueError("Private paired assignment does not match public task")
    task = durable_read_json(plan.task.public_workspace / "task.json")
    packaged = durable_read_json(plan.task.private_evaluation_path.parent / "workflow.json")
    if packaged["versions"] != task["versions"]:
        raise ValueError("Private workflow policy and public task versions differ")
    iterations = spec.iteration_policy.iterations
    policy = (
        ValidationPolicy.model_validate(packaged["policy"])
        if iterations == task["iterations"]
        else ValidationPolicy.default(pair.profile, iterations)
    )
    config = dict(plan.model.provider_config)
    if config["kind"] == "codex_cli":
        config["audit_dir"] = str(run_dir / "provider_audit")
    try:
        provider = get_provider(config)
    except Exception as exc:
        result = {
            **plan.public_dict(),
            "status": "failed",
            "spec_fingerprint": fingerprint,
            "ended_at": datetime.now(UTC).isoformat(),
            "agent_calls": 0,
            "call_attempts": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "primary_recovery": 0,
            "iterations_completed": 0,
            "terminal_iteration": None,
            "stop_reason": "provider_initialization_failed",
            "resumed": resume,
            "iteration_policy": spec.iteration_policy.model_dump(),
        }
        atomic_write_json(run_path, result)
        return result
    coordinator = StageCoordinator(
        provider,
        plan.workflow,
        spec.stages,
        spec.expected_surprising,
        spec.budget,
        run_dir / "calls",
    )
    # Preserve interrupted transcripts; rebuild into a new directory using the call journal.
    sequence = 1
    while (run_dir / f"science-{sequence:04d}").exists():
        sequence += 1
    public = plan.task.public_workspace
    if spec.workspace_strategy == "copy":
        public = run_dir / "public_workspace"
        if not public.exists():
            shutil.copytree(plan.task.public_workspace, public)
        if {p.name: sha256(p) for p in public.iterdir() if p.is_file()} != provenance["public"]:
            raise RuntimeError("Copied public workspace changed")
    report = run_workflow(
        pair,
        version,
        public,
        run_dir / f"science-{sequence:04d}",
        provider,
        run_id=plan.run_id,
        replicate_id=f"replicate-{plan.replicate:03d}",
        iterations=iterations,
        policy=policy.model_dump(),
        stage_executor=coordinator,
        max_tokens_per_call=spec.expected_surprising.max_tokens_per_call,
        max_retries_per_stage=spec.expected_surprising.max_retries_per_stage,
    )
    audit = coordinator.audit()
    if audit["draft_errors"]:
        report["confirmation"]["first_attempt_primary_recovery"] = 0
        for scope in ("exact", "exact_or_near"):
            report["scores"]["discovery"][scope]["first_attempt_D"] = 0
    report["retry_policy"].update(
        max_retries_per_peer_draft=spec.expected_surprising.max_retries_per_stage,
        reuse_peer_drafts_on_chair_repair=True,
    )
    report.update(
        harness="expected_surprising_multiagent",
        model_profile=plan.model.id,
        workflow_id=plan.workflow.id,
        workflow_mode=plan.workflow.mode,
        coordination=audit,
        provenance_sha256=digest(provenance),
    )
    atomic_write_json(run_dir / "report.json", report)
    errors = report["protocol_errors"]
    result = {
        **plan.public_dict(),
        "status": "failed" if errors else "completed",
        "spec_fingerprint": fingerprint,
        "ended_at": datetime.now(UTC).isoformat(),
        "agent_calls": audit["agent_calls"],
        "call_attempts": audit["agent_calls"],
        "usage": audit["usage"],
        "call_failures": report["attempt_errors"] + audit["draft_errors"],
        "timeout_count": sum("Timeout" in e["error"] for e in report["attempt_errors"]),
        "iteration_policy": spec.iteration_policy.model_dump(),
        "iterations_completed": sum(
            all(f"i{i:03d}-{s}" in audit["committed_stages"] for s in STAGES)
            for i in range(1, iterations + 1)
        ),
        "terminal_iteration": iterations,
        "stop_reason": "unrecovered_stage_error" if errors else "fixed_iterations_complete",
        "scientific_report": str(run_dir / "report.json"),
        "scientific_report_sha256": sha256(run_dir / "report.json"),
        "primary_recovery": report["confirmation"]["primary_recovery"],
        "resumed": resume and sequence > 1,
    }
    atomic_write_json(run_path, result)
    return result
