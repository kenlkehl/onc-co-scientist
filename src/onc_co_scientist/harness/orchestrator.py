"""Experiment-matrix orchestration for controlled co-scientist workflows."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .experiment import (
    ExperimentSpec,
    ModelSpec,
    ResourceBudget,
    StageSpec,
    TaskSpec,
    WorkflowSpec,
)
from .runtime import (
    AgentArtifact,
    AgentRequest,
    AgentResponse,
    AgentRuntime,
    AgentUsage,
    artifact_output_instructions,
    create_runtime,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "-" for char in value)
    cleaned = cleaned.strip("-_.") or "item"
    if len(cleaned) <= 80:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:68]}-{digest}"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    task: TaskSpec
    workflow: WorkflowSpec
    model: ModelSpec
    replicate: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task.id,
            "workflow_id": self.workflow.id,
            "workflow_mode": self.workflow.mode,
            "federated": self.workflow.federated,
            "model_profile": self.model.id,
            "model_id": self.model.model_id,
            "site_model_ids": self.model.site_model_ids,
            "central_model_id": self.model.central_model_id,
            "replicate": self.replicate,
        }


def build_run_plans(spec: ExperimentSpec) -> list[RunPlan]:
    plans: list[RunPlan] = []
    for task in spec.tasks:
        for workflow in spec.workflows:
            if workflow.federated and not task.site_workspaces:
                raise ValueError(
                    f"Workflow {workflow.id!r} is federated but task {task.id!r} "
                    "has no site_workspaces."
                )
            for model in spec.models:
                for replicate in range(1, spec.replicates + 1):
                    run_id = "__".join(
                        (
                            _slug(task.id),
                            _slug(workflow.id),
                            _slug(model.id),
                            f"r{replicate:03d}",
                        )
                    )
                    plans.append(
                        RunPlan(
                            run_id=run_id,
                            task=task,
                            workflow=workflow,
                            model=model,
                            replicate=replicate,
                        )
                    )
    return plans


class EventRecorder:
    """Append-only provenance ledger for a single run."""

    def __init__(self, path: Path, run_id: str, attempt: int = 1):
        self.path = path
        self.run_id = run_id
        self.attempt = attempt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "attempt": self.attempt,
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(event, sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.agent_calls = 0
        self.usage = AgentUsage()

    def before_call(self) -> None:
        if self.agent_calls >= self.budget.max_agent_calls:
            raise BudgetExceeded(
                f"max_agent_calls={self.budget.max_agent_calls} exhausted before next call."
            )

    def add(self, usage: AgentUsage) -> None:
        self.agent_calls += 1
        self.usage = self.usage + usage
        checks = (
            ("input tokens", self.usage.input_tokens, self.budget.max_input_tokens),
            ("output tokens", self.usage.output_tokens, self.budget.max_output_tokens),
            ("tool calls", self.usage.tool_calls, self.budget.max_tool_calls),
            ("cost USD", self.usage.cost_usd, self.budget.max_cost_usd),
        )
        for label, observed, limit in checks:
            if limit is not None and observed > limit:
                raise BudgetExceeded(f"Run exceeded {label} budget: {observed} > {limit}.")


class RunController:
    """Execute one task/workflow/model/replicate cell."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        plan: RunPlan,
        run_dir: Path,
        runtime: AgentRuntime,
        fingerprint: str,
        scoped_runtimes: dict[str, AgentRuntime] | None = None,
        attempt: int = 1,
    ):
        self.spec = spec
        self.plan = plan
        self.run_dir = run_dir
        self.runtime = runtime
        self.scoped_runtimes = scoped_runtimes or {}
        self.fingerprint = fingerprint
        self.attempt = attempt
        self.recorder = EventRecorder(run_dir / "events.jsonl", plan.run_id, attempt)
        self.output_recorder = EventRecorder(
            run_dir / "agent_outputs.jsonl", plan.run_id, attempt
        )
        self.ledger = BudgetLedger(spec.budget)
        existing_call_indices = [
            int(path.name.removeprefix("call_"))
            for path in (run_dir / "calls").glob("call_*")
            if path.name.removeprefix("call_").isdigit()
        ]
        self.call_index = max(existing_call_indices, default=0)
        self.artifacts: list[dict[str, Any]] = []

    def _runtime_scope(self, site_id: str | None) -> str | None:
        if site_id is not None:
            return site_id
        if self.plan.workflow.federated:
            return "central"
        return None

    def _runtime_for(self, site_id: str | None) -> AgentRuntime:
        scope = self._runtime_scope(site_id)
        return self.scoped_runtimes.get(scope or "", self.runtime)

    def _workspace(self, source: Path, label: str) -> Path:
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(
                f"Public workspace does not exist or is not a directory: {source}"
            )
        if self.spec.workspace_strategy == "reference":
            return source
        target = self.run_dir / "workspaces" / _slug(label)
        if not target.exists():
            shutil.copytree(source, target)
        return target

    def _central_workspace(self) -> Path:
        path = self.run_dir / "workspaces" / "central"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _base_prompt(
        self,
        *,
        stage: StageSpec,
        workspace: Path,
        context: str = "",
        peer_context: str = "",
        central: bool = False,
    ) -> str:
        safeguards = self.plan.workflow.safeguards
        rules = [
            "Use only the task information in this prompt and the designated public workspace.",
            "Do not search parent directories or attempt to locate scoring keys or "
            "evaluator files.",
            "Preserve null and negative findings; do not manufacture agreement or significance.",
            f"Write temporary analysis files only under {self.run_dir / 'scratch'}.",
        ]
        if central:
            rules.append(
                "You are a central reviewer. You may use only the structured site handoffs below; "
                "do not request or infer access to site-level data or private site transcripts."
            )
        if safeguards.preliminary_commitment:
            rules.append(
                "Before considering later evidence or peer views, explicitly record your "
                "preliminary conclusion in the artifact."
            )
        if safeguards.evidence_ledger:
            rules.append(
                "Populate the evidence list with the data artifact, statistic, or observation "
                "supporting each material claim."
            )
        if safeguards.minority_report:
            rules.append(
                "Preserve the strongest material disagreement in minority_report, even if the "
                "final synthesis reaches a majority conclusion."
            )
        lines = [
            f"You are the {stage.role} in a controlled oncology co-scientist experiment.",
            "",
            "TASK",
            self.plan.task.prompt,
            "",
            f"STAGE: {stage.id}",
            stage.instructions,
            "",
            "EXPERIMENTAL CONTROLS",
            *[f"- {rule}" for rule in rules],
            "",
            f"PUBLIC WORKSPACE: {workspace}",
        ]
        if context:
            lines.extend(["", "AUTHORIZED WRITTEN HANDOFF", context])
        if peer_context:
            lines.extend(["", "AUTHORIZED PEER MATERIAL", peer_context])
        lines.append(artifact_output_instructions())
        return "\n".join(lines)

    def _call(
        self,
        *,
        stage_id: str,
        role: str,
        agent_id: str,
        session_id: str,
        prompt: str,
        workspace: Path,
        site_id: str | None = None,
        round_index: int | None = None,
    ) -> AgentArtifact:
        self.ledger.before_call()
        self.call_index += 1
        call_dir = self.run_dir / "calls" / f"call_{self.call_index:04d}"
        request = AgentRequest(
            request_id=f"{self.plan.run_id}:c{self.call_index:04d}",
            experiment_id=self.spec.experiment_id,
            run_id=self.plan.run_id,
            task_id=self.plan.task.id,
            workflow_id=self.plan.workflow.id,
            model_profile=self.plan.model.id,
            model_id=self.plan.model.for_scope(self._runtime_scope(site_id)).model_id,
            stage_id=stage_id,
            role=role,
            agent_id=agent_id,
            session_id=session_id,
            prompt=prompt,
            workspace=workspace,
            scratch_dir=self.run_dir / "scratch" / _slug(agent_id),
            call_dir=call_dir,
            metadata={
                "replicate": self.plan.replicate,
                "site_id": site_id,
                "round": round_index,
                "workflow_mode": self.plan.workflow.mode,
                "federated": self.plan.workflow.federated,
                "task_metadata": self.plan.task.metadata,
            },
        )
        request_payload = request.model_dump(mode="json", exclude={"prompt"})
        request_payload["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        _atomic_json(
            call_dir / "request.json",
            request.model_dump(mode="json", exclude={"call_dir"}),
        )
        self.recorder.emit("agent_request", request_payload)
        started = time.monotonic()
        try:
            response: AgentResponse = self._runtime_for(site_id).run(request, self.spec.budget)
        except Exception as exc:
            self.recorder.emit(
                "agent_error",
                {
                    "request_id": request.request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "duration_seconds": time.monotonic() - started,
                },
            )
            raise
        budget_error: BudgetExceeded | None = None
        try:
            self.ledger.add(response.usage)
        except BudgetExceeded as exc:
            budget_error = exc
        raw_text = response.raw_text
        raw_bytes = raw_text.encode("utf-8")
        raw_response_path = call_dir / "raw_response.txt"
        _atomic_text(raw_response_path, raw_text)
        record = {
            "request_id": request.request_id,
            "stage_id": stage_id,
            "role": role,
            "agent_id": agent_id,
            "session_id": session_id,
            "site_id": site_id,
            "round": round_index,
            "artifact": response.artifact.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json"),
            "runtime_metadata": response.runtime_metadata,
            "raw_response": {
                "path": str(raw_response_path.relative_to(self.run_dir)),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "bytes": len(raw_bytes),
            },
        }
        self.artifacts.append(record)
        self.recorder.emit("agent_response", record)
        self.output_recorder.emit(
            "agent_output",
            {
                **record,
                "raw_text": raw_text,
            },
        )
        _atomic_json(call_dir / "normalized_response.json", record)
        _atomic_json(self.run_dir / "artifacts.json", self.artifacts)
        if budget_error is not None:
            raise budget_error
        return response.artifact

    def _run_linear(
        self,
        *,
        workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        previous: AgentArtifact | None = None
        persistent_session = f"{self.plan.run_id}:{scope_id}:persistent"
        for stage in self.spec.stages:
            if self.plan.workflow.mode == "persistent":
                session_id = persistent_session
                context = ""
            else:
                session_id = f"{self.plan.run_id}:{scope_id}:{stage.id}"
                context = previous.handoff if previous is not None else ""
            prompt = self._base_prompt(
                stage=stage,
                workspace=workspace,
                context=context,
            )
            previous = self._call(
                stage_id=stage.id,
                role=stage.role,
                agent_id=f"{scope_id}:{stage.id}",
                session_id=session_id,
                prompt=prompt,
                workspace=workspace,
                site_id=site_id,
                round_index=1,
            )
        assert previous is not None
        return previous

    def _run_deliberative(
        self,
        *,
        workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        prior_handoff = ""
        consensus: AgentArtifact | None = None
        workflow = self.plan.workflow
        for stage in self.spec.stages:
            current: list[AgentArtifact] = []
            session_ids = [
                f"{self.plan.run_id}:{scope_id}:{stage.id}:peer{index:02d}"
                for index in range(1, workflow.agents_per_stage + 1)
            ]
            for index, session_id in enumerate(session_ids, start=1):
                prompt = self._base_prompt(
                    stage=stage,
                    workspace=workspace,
                    context=prior_handoff,
                )
                current.append(
                    self._call(
                        stage_id=stage.id,
                        role=stage.role,
                        agent_id=f"{scope_id}:{stage.id}:peer{index:02d}",
                        session_id=session_id,
                        prompt=prompt,
                        workspace=workspace,
                        site_id=site_id,
                        round_index=1,
                    )
                )

            for round_index in range(2, workflow.deliberation_rounds + 1):
                revised: list[AgentArtifact] = []
                for index, session_id in enumerate(session_ids, start=1):
                    peer_payload = [
                        {
                            "peer": peer_index,
                            "handoff": artifact.handoff,
                            "evidence": artifact.evidence,
                            "concerns": artifact.concerns,
                        }
                        for peer_index, artifact in enumerate(current, start=1)
                        if peer_index != index
                    ]
                    prompt = self._base_prompt(
                        stage=stage,
                        workspace=workspace,
                        context=prior_handoff,
                        peer_context=(
                            "Review the other scientists' structured artifacts below, then revise "
                            "your own conclusion while retaining justified disagreement.\n"
                            + json.dumps(peer_payload, indent=2)
                        ),
                    )
                    revised.append(
                        self._call(
                            stage_id=stage.id,
                            role=stage.role,
                            agent_id=f"{scope_id}:{stage.id}:peer{index:02d}",
                            session_id=session_id,
                            prompt=prompt,
                            workspace=workspace,
                            site_id=site_id,
                            round_index=round_index,
                        )
                    )
                current = revised

            peer_payload = [
                {
                    "peer": index,
                    "handoff": artifact.handoff,
                    "evidence": artifact.evidence,
                    "concerns": artifact.concerns,
                    "minority_report": artifact.minority_report,
                }
                for index, artifact in enumerate(current, start=1)
            ]
            consensus_stage = StageSpec(
                id=f"{stage.id}_consensus",
                role=f"{stage.role} consensus chair",
                instructions=(
                    "Synthesize the peer artifacts into one stage result. Weight evidence rather "
                    "than votes, identify unsupported convergence, and preserve material dissent."
                ),
            )
            prompt = self._base_prompt(
                stage=consensus_stage,
                workspace=workspace,
                context=prior_handoff,
                peer_context=json.dumps(peer_payload, indent=2),
            )
            consensus = self._call(
                stage_id=consensus_stage.id,
                role=consensus_stage.role,
                agent_id=f"{scope_id}:{stage.id}:chair",
                session_id=f"{self.plan.run_id}:{scope_id}:{stage.id}:chair",
                prompt=prompt,
                workspace=workspace,
                site_id=site_id,
                round_index=workflow.deliberation_rounds + 1,
            )
            prior_handoff = consensus.handoff
        assert consensus is not None
        return consensus

    def _run_site(
        self,
        *,
        workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        if self.plan.workflow.mode == "deliberative":
            return self._run_deliberative(
                workspace=workspace,
                scope_id=scope_id,
                site_id=site_id,
            )
        return self._run_linear(
            workspace=workspace,
            scope_id=scope_id,
            site_id=site_id,
        )

    def _independent_verification(
        self,
        *,
        final_artifact: AgentArtifact,
        workspace: Path,
        peer_context: str = "",
    ) -> AgentArtifact | None:
        if not self.plan.workflow.safeguards.independent_rerun:
            return None
        stage = StageSpec(
            id="independent_verification",
            role="independent verification scientist",
            instructions=(
                "Independently check the proposed final result. Attempt to reproduce the key "
                "analysis or evidence synthesis, and clearly state confirmations and failures."
            ),
        )
        prompt = self._base_prompt(
            stage=stage,
            workspace=workspace,
            context=final_artifact.handoff,
            peer_context=peer_context,
        )
        return self._call(
            stage_id=stage.id,
            role=stage.role,
            agent_id="independent-verifier",
            session_id=f"{self.plan.run_id}:independent-verifier",
            prompt=prompt,
            workspace=workspace,
            round_index=1,
        )

    def execute(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.run_dir / "artifacts.json", self.artifacts)
        self.recorder.emit("run_started", self.plan.public_dict())
        if self.plan.workflow.federated:
            site_reports: dict[str, AgentArtifact] = {}
            for site_id, source in sorted(self.plan.task.site_workspaces.items()):
                workspace = self._workspace(source, f"site-{site_id}")
                site_reports[site_id] = self._run_site(
                    workspace=workspace,
                    scope_id=f"site-{site_id}",
                    site_id=site_id,
                )
            authorized_reports = {
                site_id: {"handoff": artifact.handoff} for site_id, artifact in site_reports.items()
            }
            central_workspace = self._central_workspace()
            central_stage = StageSpec(
                id="federated_synthesis",
                role="federated synthesis scientist",
                instructions=(
                    "Synthesize the site reports without access to raw site data or internal site "
                    "messages. Check heterogeneity, incompatible estimands, duplicated evidence "
                    "lineage, and unresolved disagreement."
                ),
            )
            central_prompt = self._base_prompt(
                stage=central_stage,
                workspace=central_workspace,
                peer_context=json.dumps(authorized_reports, indent=2),
                central=True,
            )
            final_artifact = self._call(
                stage_id=central_stage.id,
                role=central_stage.role,
                agent_id="central-reviewer",
                session_id=f"{self.plan.run_id}:central-reviewer",
                prompt=central_prompt,
                workspace=central_workspace,
                round_index=1,
            )
            verification = self._independent_verification(
                final_artifact=final_artifact,
                workspace=central_workspace,
                peer_context=json.dumps(authorized_reports, indent=2),
            )
            site_handoffs = {
                site_id: artifact.handoff for site_id, artifact in site_reports.items()
            }
        else:
            workspace = self._workspace(self.plan.task.public_workspace, "single-site")
            final_artifact = self._run_site(
                workspace=workspace,
                scope_id="single-site",
                site_id=None,
            )
            verification = self._independent_verification(
                final_artifact=final_artifact,
                workspace=workspace,
            )
            site_handoffs = {}

        result = {
            **self.plan.public_dict(),
            "status": "completed",
            "spec_fingerprint": self.fingerprint,
            "attempt": self.attempt,
            "started_at": None,
            "ended_at": _utc_now(),
            "agent_calls": self.ledger.agent_calls,
            "usage": self.ledger.usage.model_dump(mode="json"),
            "final_artifact": final_artifact.model_dump(mode="json"),
            "verification_artifact": (
                verification.model_dump(mode="json") if verification is not None else None
            ),
            "site_handoffs": site_handoffs,
            "artifact_count": len(self.artifacts),
        }
        _atomic_json(self.run_dir / "artifacts.json", self.artifacts)
        self.recorder.emit(
            "run_completed",
            {
                "agent_calls": self.ledger.agent_calls,
                "usage": self.ledger.usage.model_dump(mode="json"),
            },
        )
        return result


def _run_one(
    spec: ExperimentSpec,
    plan: RunPlan,
    output_root: Path,
    fingerprint: str,
    *,
    resume: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / plan.run_id
    run_path = run_dir / "run.json"
    if resume and run_path.exists():
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed" and prior.get("spec_fingerprint") == fingerprint:
            return {**prior, "resumed": True}

    started_at = _utc_now()
    events_path = run_dir / "events.jsonl"
    attempt = 1
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "run_started":
                attempt += 1
    runtime = create_runtime(plan.model.for_scope(None))
    scoped_runtimes: dict[str, AgentRuntime] = {}
    if plan.workflow.federated:
        if plan.model.central_model_id:
            scoped_runtimes["central"] = create_runtime(plan.model.for_scope("central"))
        for site_id in plan.task.site_workspaces:
            if site_id in plan.model.site_model_ids:
                scoped_runtimes[site_id] = create_runtime(plan.model.for_scope(site_id))
    controller = RunController(
        spec=spec,
        plan=plan,
        run_dir=run_dir,
        runtime=runtime,
        fingerprint=fingerprint,
        scoped_runtimes=scoped_runtimes,
        attempt=attempt,
    )
    try:
        result = controller.execute()
        result["started_at"] = started_at
        result["resumed"] = False
    except Exception as exc:
        result = {
            **plan.public_dict(),
            "status": "failed",
            "spec_fingerprint": fingerprint,
            "attempt": attempt,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "agent_calls": controller.ledger.agent_calls,
            "usage": controller.ledger.usage.model_dump(mode="json"),
            "resumed": False,
        }
        controller.recorder.emit(
            "run_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "agent_calls": controller.ledger.agent_calls,
                "usage": controller.ledger.usage.model_dump(mode="json"),
            },
        )
    finally:
        for scoped_runtime in scoped_runtimes.values():
            scoped_runtime.close()
        runtime.close()
    _atomic_json(run_path, result)
    return result


def run_experiment(
    spec: ExperimentSpec,
    *,
    output_root: Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    max_parallel: int | None = None,
) -> dict[str, Any]:
    """Expand and execute a full matched experiment matrix."""

    root = (output_root or spec.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = spec.fingerprint()
    plans = build_run_plans(spec)
    resolved_payload = spec.model_dump(mode="json", exclude_none=True)
    for task in resolved_payload.get("tasks", []):
        task.pop("private_evaluation_path", None)
    _atomic_json(root / "resolved_spec.json", resolved_payload)
    _atomic_json(root / "plan.json", [plan.public_dict() for plan in plans])
    if dry_run:
        summary = {
            "experiment_id": spec.experiment_id,
            "status": "planned",
            "spec_fingerprint": fingerprint,
            "n_runs": len(plans),
            "output_root": str(root),
        }
        _atomic_json(root / "summary.json", summary)
        return summary

    workers = max_parallel or spec.max_parallel
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for plan in plans:
            results.append(_run_one(spec, plan, root, fingerprint, resume=resume))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one, spec, plan, root, fingerprint, resume=resume): plan
                for plan in plans
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["run_id"])
    counts = {
        status: sum(1 for result in results if result["status"] == status)
        for status in ("completed", "failed")
    }
    summary = {
        "experiment_id": spec.experiment_id,
        "status": "completed" if counts["failed"] == 0 else "completed_with_failures",
        "spec_fingerprint": fingerprint,
        "n_runs": len(results),
        "n_completed": counts["completed"],
        "n_failed": counts["failed"],
        "n_resumed": sum(1 for result in results if result.get("resumed")),
        "output_root": str(root),
        "ended_at": _utc_now(),
        "runs": results,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
