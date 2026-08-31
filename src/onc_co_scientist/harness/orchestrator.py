"""Experiment-matrix orchestration for controlled co-scientist workflows."""

from __future__ import annotations

import hashlib
import json
import random
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
    required_agent_calls,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_hashes(path: Path) -> dict[str, str]:
    """Hash one public substrate without following links outside its root."""

    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Public workspace is not a directory: {root}")
    hashes: dict[str, str] = {}
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Public workspaces may not contain symbolic links: {item}")
        if item.is_file():
            hashes[item.relative_to(root).as_posix()] = _sha256_file(item)
    if not hashes:
        raise ValueError(f"Public workspace contains no files: {root}")
    return hashes


def _substrate_hashes(plan: RunPlan) -> dict[str, dict[str, str]]:
    payload = {"public": _workspace_hashes(plan.task.public_workspace)}
    payload.update(
        {
            f"site:{site_id}": _workspace_hashes(path)
            for site_id, path in sorted(plan.task.site_workspaces.items())
        }
    )
    return payload


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
            "semantic_condition": self.task.semantic_condition,
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
        for replicate in range(1, spec.replicates + 1):
            for workflow in spec.workflows:
                if workflow.federated and not task.site_workspaces:
                    raise ValueError(
                        f"Workflow {workflow.id!r} is federated but task {task.id!r} "
                        "has no site_workspaces."
                    )
                for model in spec.models:
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

    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
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

    def restore(self, *, agent_calls: int, usage: dict[str, Any]) -> None:
        self.agent_calls = agent_calls
        self.usage = AgentUsage.model_validate(usage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_calls": self.agent_calls,
            "usage": self.usage.model_dump(mode="json"),
        }


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
        resume: bool = False,
    ):
        self.spec = spec
        self.plan = plan
        self.run_dir = run_dir
        self.runtime = runtime
        self.scoped_runtimes = scoped_runtimes or {}
        self.fingerprint = fingerprint
        self.recorder = EventRecorder(run_dir / "events.jsonl", plan.run_id)
        self.ledger = BudgetLedger(spec.budget)
        self.state_path = run_dir / "run_state.json"
        self.substrate_hashes = _substrate_hashes(plan)
        self.implementation_sha256 = _sha256_file(Path(__file__).resolve())
        self.call_index = self._existing_call_index()
        self.artifacts: list[dict[str, Any]] = []
        self.completed_slots: dict[str, dict[str, Any]] = {}
        self.previous_authoritative_handoff = ""
        self.iterations_completed = 0
        self.terminal_iteration: int | None = None
        self.last_position: dict[str, Any] | None = None
        self.call_failures: list[dict[str, Any]] = []
        self.resumed_from_state = False
        if resume:
            self._restore_state()

    def _existing_call_index(self) -> int:
        maximum = 0
        for path in (self.run_dir / "calls").glob("call_[0-9]*"):
            try:
                maximum = max(maximum, int(path.name.removeprefix("call_")))
            except ValueError:
                continue
        return maximum

    def _restore_state(self) -> None:
        if not self.state_path.exists():
            if self.call_index:
                raise RuntimeError(
                    "Cannot resume: call directories exist but run_state.json is missing. "
                    "Archive the attempt and start fresh."
                )
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        expected = {
            "run_id": self.plan.run_id,
            "spec_fingerprint": self.fingerprint,
            "substrate_hashes": self.substrate_hashes,
            "implementation_sha256": self.implementation_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"Cannot resume {self.plan.run_id}: {key} changed. "
                    "Use a new output root or archive this attempt before starting fresh."
                )
        raw_artifacts = state.get("artifacts")
        raw_slots = state.get("completed_slots")
        raw_ledger = state.get("usage_ledger")
        if not isinstance(raw_artifacts, list) or not isinstance(raw_slots, dict):
            raise RuntimeError("Cannot resume: run_state.json has ambiguous artifact state.")
        if not isinstance(raw_ledger, dict):
            raise RuntimeError("Cannot resume: run_state.json has no valid usage ledger.")
        self.artifacts = [dict(item) for item in raw_artifacts if isinstance(item, dict)]
        self.completed_slots = {
            str(slot): dict(record)
            for slot, record in raw_slots.items()
            if isinstance(record, dict)
        }
        if len(self.artifacts) != len(self.completed_slots):
            raise RuntimeError(
                "Cannot resume: artifact and completed-call-slot counts disagree."
            )
        self.ledger.restore(
            agent_calls=int(raw_ledger.get("agent_calls", -1)),
            usage=dict(raw_ledger.get("usage", {})),
        )
        if self.ledger.agent_calls != len(self.completed_slots):
            raise RuntimeError(
                "Cannot resume: usage ledger would double-count or omit completed calls."
            )
        self.call_index = max(self.call_index, int(state.get("call_index", 0)))
        self.previous_authoritative_handoff = str(
            state.get("previous_authoritative_handoff", "")
        )
        self.iterations_completed = int(state.get("iterations_completed", 0))
        terminal = state.get("terminal_iteration")
        self.terminal_iteration = int(terminal) if terminal is not None else None
        self.last_position = state.get("position")
        raw_failures = state.get("call_failures", [])
        if not isinstance(raw_failures, list):
            raise RuntimeError("Cannot resume: run_state.json has malformed call failures.")
        self.call_failures = [dict(item) for item in raw_failures if isinstance(item, dict)]
        self.resumed_from_state = True

    def _session_records(self) -> list[dict[str, str]]:
        root = self.run_dir / "scratch" / ".codex_sessions"
        if not root.is_dir():
            return []
        return [
            {
                "path": path.relative_to(self.run_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in sorted(root.glob("*.json"))
        ]

    def _workspace_state(self) -> list[str]:
        root = self.run_dir / "workspaces"
        if not root.is_dir():
            return []
        return [
            path.relative_to(self.run_dir).as_posix()
            for path in sorted(root.iterdir())
            if path.is_dir()
        ]

    def _partial_peer_results(self) -> list[dict[str, Any]]:
        chairs = {
            (item.get("iteration_index"), item.get("canonical_stage"), item.get("site_id"))
            for item in self.artifacts
            if item.get("position_kind") == "chair"
        }
        return [
            {
                "call_slot": item.get("call_slot"),
                "iteration_index": item.get("iteration_index"),
                "canonical_stage": item.get("canonical_stage"),
                "peer_index": item.get("peer_index"),
                "round": item.get("round"),
            }
            for item in self.artifacts
            if item.get("position_kind") == "peer"
            and (
                item.get("iteration_index"),
                item.get("canonical_stage"),
                item.get("site_id"),
            )
            not in chairs
        ]

    def _write_state(self, *, status: str = "running", stop_reason: str | None = None) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": "1",
                "run_id": self.plan.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "spec_fingerprint": self.fingerprint,
                "substrate_hashes": self.substrate_hashes,
                "implementation_sha256": self.implementation_sha256,
                "call_slot_cursor": (
                    self.last_position.get("call_slot") if self.last_position else None
                ),
                "position": self.last_position,
                "call_index": self.call_index,
                "usage_ledger": self.ledger.as_dict(),
                "completed_slots": self.completed_slots,
                "artifacts": self.artifacts,
                "previous_authoritative_handoff": self.previous_authoritative_handoff,
                "partial_deliberative_peer_results": self._partial_peer_results(),
                "session_records": self._session_records(),
                "workspace_state": self._workspace_state(),
                "call_failures": self.call_failures,
                "iterations_completed": self.iterations_completed,
                "terminal_iteration": self.terminal_iteration,
                "updated_at": _utc_now(),
            },
        )

    def _recover_runtime_success(
        self,
        *,
        call_slot: str,
        prompt: str,
        stage_id: str,
        iteration_index: int,
        stage_index: int,
        position_kind: str,
        peer_index: int | None,
        round_index: int | None,
    ) -> tuple[int, Path, AgentResponse] | None:
        """Adopt a completed runtime call that predates its controller checkpoint."""

        candidates: list[tuple[int, Path, AgentResponse]] = []
        for call_dir in sorted((self.run_dir / "calls").glob("call_[0-9]*")):
            if (call_dir / "controller_rejected.json").exists():
                continue
            request_path = call_dir / "request.json"
            if not request_path.is_file():
                continue
            request = json.loads(request_path.read_text(encoding="utf-8"))
            metadata = request.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("call_slot") != call_slot:
                continue
            expected = {
                "run_id": self.plan.run_id,
                "task_id": self.plan.task.id,
                "workflow_id": self.plan.workflow.id,
                "stage_id": stage_id,
                "iteration_index": iteration_index,
                "stage_index": stage_index,
            }
            for key, value in expected.items():
                if request.get(key) != value:
                    raise RuntimeError(
                        f"Cannot adopt runtime success for {call_slot!r}: {key} mismatch."
                    )
            metadata_expected = {
                "position_kind": position_kind,
                "peer_index": peer_index,
                "round": round_index,
            }
            for key, value in metadata_expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(
                        f"Cannot adopt runtime success for {call_slot!r}: metadata {key} "
                        "mismatch."
                    )
            if request.get("prompt") != prompt:
                raise RuntimeError(
                    f"Cannot adopt runtime success for {call_slot!r}: prompt changed."
                )

            response_payload: Any | None = None
            marker_path = call_dir / "runtime_success.json"
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("prompt_sha256") != hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest():
                    raise RuntimeError(
                        f"Cannot adopt runtime success for {call_slot!r}: prompt hash mismatch."
                    )
                response_payload = marker.get("response")
            else:
                # The NSCLC Codex adapter writes a successful audit before its
                # outer CLI wrapper can persist the generic marker.
                output_path = call_dir / "response.json"
                audit_path = call_dir / "codex_call.json"
                if output_path.is_file() and audit_path.is_file():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    if audit.get("returncode") == 0 and audit.get("error") is None:
                        response_payload = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(response_payload, dict):
                continue
            if "request_id" in response_payload and "artifact" in response_payload:
                response = AgentResponse.model_validate(response_payload)
            else:
                response = AgentResponse(
                    request_id=str(request["request_id"]),
                    artifact=AgentArtifact.model_validate(response_payload),
                )
            if response.request_id != request.get("request_id"):
                raise RuntimeError(
                    f"Cannot adopt runtime success for {call_slot!r}: request ID mismatch."
                )
            try:
                call_index = int(call_dir.name.removeprefix("call_"))
            except ValueError as exc:
                raise RuntimeError(f"Malformed call directory: {call_dir}") from exc
            candidates.append((call_index, call_dir, response))
        if len(candidates) > 1:
            raise RuntimeError(
                f"Cannot resume: multiple uncheckpointed successes exist for {call_slot!r}."
            )
        return candidates[0] if candidates else None

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

    def _session_workspace(self, source: Path, session_id: str) -> Path:
        """Return the workspace visible to one conversational session.

        Reference mode intentionally preserves the configured shared source.
        Copy mode creates one snapshot per session, so persistent turns share a
        snapshot while sequential stages and deliberative participants do not.
        """

        return self._workspace(source, f"session-{session_id}")

    def _central_workspace(self, session_id: str) -> Path:
        path = self.run_dir / "workspaces" / _slug(f"session-{session_id}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _scratch_dir(self, session_id: str) -> Path:
        return self.run_dir / "scratch" / _slug(session_id)

    def _base_prompt(
        self,
        *,
        stage: StageSpec,
        iteration_index: int,
        stage_index: int,
        workspace: Path,
        scratch_dir: Path,
        context: str = "",
        peer_context: str = "",
        central: bool = False,
        require_final_answer: bool = False,
        terminal: bool = False,
    ) -> str:
        safeguards = self.plan.workflow.safeguards
        rules = [
            "Use only the task information in this prompt and the designated public workspace.",
            "Do not search parent directories or attempt to locate scoring keys or "
            "evaluator files.",
            "Preserve null and negative findings; do not manufacture agreement or significance.",
            f"Write temporary analysis files only under {scratch_dir}.",
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
            (
                f"ITERATION: {iteration_index} of "
                f"{self.spec.iteration_policy.iterations} (one-based)"
            ),
            f"STAGE POSITION: {stage_index} zero-based / {stage_index + 1} one-based",
            f"STAGE: {stage.id}",
            f"TERMINAL CHECKPOINT: {'yes' if terminal else 'no'}",
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
        lines.append(artifact_output_instructions(require_final_answer=require_final_answer))
        return "\n".join(lines)

    def _call(
        self,
        *,
        call_slot: str,
        stage_id: str,
        canonical_stage: str,
        role: str,
        agent_id: str,
        session_id: str,
        prompt: str,
        workspace: Path,
        iteration_index: int,
        stage_index: int,
        position_kind: str,
        site_id: str | None = None,
        round_index: int | None = None,
        peer_index: int | None = None,
        authoritative: bool = False,
        require_final_answer: bool = False,
        terminal: bool = False,
    ) -> AgentArtifact:
        completed = self.completed_slots.get(call_slot)
        if completed is not None:
            expected = {
                "stage_id": stage_id,
                "canonical_stage": canonical_stage,
                "iteration_index": iteration_index,
                "stage_index": stage_index,
                "position_kind": position_kind,
                "site_id": site_id,
                "round": round_index,
                "peer_index": peer_index,
            }
            for key, value in expected.items():
                if completed.get(key) != value:
                    raise RuntimeError(
                        f"Cannot resume call slot {call_slot!r}: {key} does not match "
                        "the frozen execution graph."
                    )
            artifact = AgentArtifact.model_validate(completed.get("artifact", {}))
            if authoritative:
                self.previous_authoritative_handoff = artifact.handoff
            return artifact

        recovered = self._recover_runtime_success(
            call_slot=call_slot,
            prompt=prompt,
            stage_id=stage_id,
            iteration_index=iteration_index,
            stage_index=stage_index,
            position_kind=position_kind,
            peer_index=peer_index,
            round_index=round_index,
        )
        self.ledger.before_call()
        if recovered is not None:
            successful_call_index, call_dir, response = recovered
            request_id = response.request_id
            self.recorder.emit(
                "runtime_success_adopted",
                {
                    "request_id": request_id,
                    "call_slot": call_slot,
                    "call_index": successful_call_index,
                },
            )
        else:
            self.call_index += 1
            successful_call_index = self.call_index
            call_dir = self.run_dir / "calls" / f"call_{self.call_index:04d}"
            if call_dir.exists():
                raise RuntimeError(f"Refusing to overwrite existing call directory: {call_dir}")
            request = AgentRequest(
                request_id=f"{self.plan.run_id}:c{self.call_index:04d}",
                experiment_id=self.spec.experiment_id,
                run_id=self.plan.run_id,
                task_id=self.plan.task.id,
                workflow_id=self.plan.workflow.id,
                model_profile=self.plan.model.id,
                model_id=self.plan.model.for_scope(self._runtime_scope(site_id)).model_id,
                stage_id=stage_id,
                iteration_index=iteration_index,
                max_iterations=self.spec.iteration_policy.iterations,
                stage_index=stage_index,
                stage_position=stage_index + 1,
                terminal=terminal,
                role=role,
                agent_id=agent_id,
                session_id=session_id,
                prompt=prompt,
                workspace=workspace,
                scratch_dir=self._scratch_dir(session_id),
                call_dir=call_dir,
                metadata={
                    "replicate": self.plan.replicate,
                    "semantic_condition": self.plan.task.semantic_condition,
                    "site_id": site_id,
                    "round": round_index,
                    "peer_index": peer_index,
                    "position_kind": position_kind,
                    "call_slot": call_slot,
                    "workflow_mode": self.plan.workflow.mode,
                    "federated": self.plan.workflow.federated,
                },
            )
            request_id = request.request_id
            request_payload = request.model_dump(mode="json", exclude={"prompt"})
            request_payload["prompt_sha256"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            self.recorder.emit("agent_request", request_payload)
            started = time.monotonic()
            try:
                response = self._runtime_for(site_id).run(request, self.spec.budget)
            except Exception as exc:
                duration = time.monotonic() - started
                self.call_failures.append(
                    {
                        "call_index": self.call_index,
                        "call_slot": call_slot,
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_seconds": duration,
                        "timestamp": _utc_now(),
                    }
                )
                self.last_position = {
                    "call_slot": call_slot,
                    "iteration_index": iteration_index,
                    "stage_index": stage_index,
                    "stage_position": stage_index + 1,
                    "stage_id": stage_id,
                    "position_kind": position_kind,
                    "peer_index": peer_index,
                    "round": round_index,
                    "terminal": terminal,
                    "completed": False,
                }
                self.recorder.emit(
                    "agent_error",
                    {
                        "request_id": request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "duration_seconds": duration,
                    },
                )
                self._write_state(status="interrupted", stop_reason=type(exc).__name__)
                raise
        artifact = response.artifact
        if require_final_answer and artifact.final_answer is None:
            exc = ValueError(
                f"Synthesis call slot {call_slot!r} returned a null final_answer."
            )
            self.recorder.emit(
                "artifact_contract_error",
                {"request_id": request_id, "error": str(exc)},
            )
            _atomic_json(call_dir / "controller_rejected.json", {"error": str(exc)})
            self.call_failures.append(
                {
                    "call_index": successful_call_index,
                    "call_slot": call_slot,
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timestamp": _utc_now(),
                }
            )
            self.last_position = {
                "call_slot": call_slot,
                "iteration_index": iteration_index,
                "stage_index": stage_index,
                "stage_position": stage_index + 1,
                "stage_id": stage_id,
                "position_kind": position_kind,
                "peer_index": peer_index,
                "round": round_index,
                "terminal": terminal,
                "completed": False,
            }
            self._write_state(status="interrupted", stop_reason="artifact_contract_error")
            raise exc
        if not require_final_answer and artifact.final_answer is not None:
            exc = ValueError(
                f"Non-synthesis call slot {call_slot!r} returned a non-null final_answer."
            )
            self.recorder.emit(
                "artifact_contract_error",
                {"request_id": request_id, "error": str(exc)},
            )
            _atomic_json(call_dir / "controller_rejected.json", {"error": str(exc)})
            self.call_failures.append(
                {
                    "call_index": successful_call_index,
                    "call_slot": call_slot,
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timestamp": _utc_now(),
                }
            )
            self.last_position = {
                "call_slot": call_slot,
                "iteration_index": iteration_index,
                "stage_index": stage_index,
                "stage_position": stage_index + 1,
                "stage_id": stage_id,
                "position_kind": position_kind,
                "peer_index": peer_index,
                "round": round_index,
                "terminal": terminal,
                "completed": False,
            }
            self._write_state(status="interrupted", stop_reason="artifact_contract_error")
            raise exc
        budget_error: BudgetExceeded | None = None
        try:
            self.ledger.add(response.usage)
        except BudgetExceeded as exc:
            budget_error = exc
        record = {
            "request_id": request_id,
            "call_index": successful_call_index,
            "call_slot": call_slot,
            "stage_id": stage_id,
            "canonical_stage": canonical_stage,
            "iteration_index": iteration_index,
            "max_iterations": self.spec.iteration_policy.iterations,
            "stage_index": stage_index,
            "stage_position": stage_index + 1,
            "position_kind": position_kind,
            "peer_index": peer_index,
            "authoritative": authoritative,
            "terminal": terminal,
            "role": role,
            "agent_id": agent_id,
            "session_id": session_id,
            "site_id": site_id,
            "round": round_index,
            "artifact": response.artifact.model_dump(mode="json"),
            "usage": response.usage.model_dump(mode="json"),
            "runtime_metadata": response.runtime_metadata,
            "artifact_valid": True,
        }
        self.artifacts.append(record)
        self.completed_slots[call_slot] = record
        if authoritative:
            self.previous_authoritative_handoff = artifact.handoff
            if canonical_stage == "synthesis":
                self.iterations_completed = max(self.iterations_completed, iteration_index)
                if terminal:
                    self.terminal_iteration = iteration_index
        self.last_position = {
            "call_slot": call_slot,
            "iteration_index": iteration_index,
            "stage_index": stage_index,
            "stage_position": stage_index + 1,
            "stage_id": stage_id,
            "position_kind": position_kind,
            "peer_index": peer_index,
            "round": round_index,
            "terminal": terminal,
            "completed": True,
        }
        self.recorder.emit("agent_response", record)
        _atomic_json(call_dir / "normalized_response.json", record)
        _atomic_json(self.run_dir / "artifacts.json", self.artifacts)
        self._write_state()
        if budget_error is not None:
            raise budget_error
        return artifact

    def _run_linear(
        self,
        *,
        source_workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        previous: AgentArtifact | None = None
        persistent_session = f"{self.plan.run_id}:{scope_id}:persistent"
        max_iterations = self.spec.iteration_policy.iterations
        for iteration_index in range(1, max_iterations + 1):
            for stage_index, stage in enumerate(self.spec.stages):
                synthesis = stage_index == len(self.spec.stages) - 1
                terminal = synthesis and iteration_index == max_iterations
                if self.plan.workflow.mode == "persistent":
                    session_id = persistent_session
                    context = ""
                else:
                    session_id = (
                        f"{self.plan.run_id}:{scope_id}:i{iteration_index:03d}:{stage.id}"
                    )
                    context = previous.handoff if previous is not None else ""
                workspace = self._session_workspace(source_workspace, session_id)
                prompt = self._base_prompt(
                    stage=stage,
                    iteration_index=iteration_index,
                    stage_index=stage_index,
                    workspace=workspace,
                    scratch_dir=self._scratch_dir(session_id),
                    context=context,
                    require_final_answer=synthesis,
                    terminal=terminal,
                )
                call_slot = (
                    f"{scope_id}:i{iteration_index:03d}:s{stage_index:02d}:linear"
                )
                previous = self._call(
                    call_slot=call_slot,
                    stage_id=stage.id,
                    canonical_stage=stage.id,
                    role=stage.role,
                    agent_id=f"{scope_id}:i{iteration_index:03d}:{stage.id}",
                    session_id=session_id,
                    prompt=prompt,
                    workspace=workspace,
                    iteration_index=iteration_index,
                    stage_index=stage_index,
                    position_kind="linear",
                    site_id=site_id,
                    round_index=1,
                    authoritative=True,
                    require_final_answer=synthesis,
                    terminal=terminal,
                )
        assert previous is not None
        return previous

    def _run_deliberative(
        self,
        *,
        source_workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        prior_handoff = ""
        consensus: AgentArtifact | None = None
        workflow = self.plan.workflow
        max_iterations = self.spec.iteration_policy.iterations
        for iteration_index in range(1, max_iterations + 1):
            for stage_index, stage in enumerate(self.spec.stages):
                require_final_answer = stage_index == len(self.spec.stages) - 1
                terminal = require_final_answer and iteration_index == max_iterations
                current: list[AgentArtifact] = []
                session_ids = [
                    (
                        f"{self.plan.run_id}:{scope_id}:i{iteration_index:03d}:"
                        f"{stage.id}:peer{index:02d}"
                    )
                    for index in range(1, workflow.agents_per_stage + 1)
                ]
                for round_index in range(1, workflow.deliberation_rounds + 1):
                    previous_round = current
                    current = []
                    for index, session_id in enumerate(session_ids, start=1):
                        workspace = self._session_workspace(source_workspace, session_id)
                        peer_context = ""
                        if round_index > 1:
                            peer_payload = [
                                {
                                    "peer": peer_index,
                                    "handoff": artifact.handoff,
                                    "claims": [
                                        claim.model_dump(mode="json")
                                        for claim in artifact.claims
                                    ],
                                    "evidence": artifact.evidence,
                                    "concerns": artifact.concerns,
                                }
                                for peer_index, artifact in enumerate(previous_round, start=1)
                                if peer_index != index
                            ]
                            peer_context = (
                                "Review the other scientists' structured artifacts below, then "
                                "revise your own conclusion while retaining justified "
                                "disagreement.\n"
                                + json.dumps(peer_payload, indent=2)
                            )
                        prompt = self._base_prompt(
                            stage=stage,
                            iteration_index=iteration_index,
                            stage_index=stage_index,
                            workspace=workspace,
                            scratch_dir=self._scratch_dir(session_id),
                            context=prior_handoff,
                            peer_context=peer_context,
                            require_final_answer=require_final_answer,
                            terminal=False,
                        )
                        call_slot = (
                            f"{scope_id}:i{iteration_index:03d}:s{stage_index:02d}:"
                            f"peer{index:02d}:r{round_index:02d}"
                        )
                        current.append(
                            self._call(
                                call_slot=call_slot,
                                stage_id=stage.id,
                                canonical_stage=stage.id,
                                role=stage.role,
                                agent_id=(
                                    f"{scope_id}:i{iteration_index:03d}:"
                                    f"{stage.id}:peer{index:02d}"
                                ),
                                session_id=session_id,
                                prompt=prompt,
                                workspace=workspace,
                                iteration_index=iteration_index,
                                stage_index=stage_index,
                                position_kind="peer",
                                site_id=site_id,
                                round_index=round_index,
                                peer_index=index,
                                require_final_answer=require_final_answer,
                            )
                        )

                peer_payload = [
                    {
                        "peer": index,
                        "handoff": artifact.handoff,
                        "claims": [
                            claim.model_dump(mode="json") for claim in artifact.claims
                        ],
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
                        "Synthesize the peer artifacts into one stage result. Weight evidence "
                        "rather than votes, identify unsupported convergence, and preserve "
                        "material dissent."
                    ),
                )
                consensus_session_id = (
                    f"{self.plan.run_id}:{scope_id}:i{iteration_index:03d}:"
                    f"{stage.id}:chair"
                )
                consensus_workspace = self._session_workspace(
                    source_workspace, consensus_session_id
                )
                prompt = self._base_prompt(
                    stage=consensus_stage,
                    iteration_index=iteration_index,
                    stage_index=stage_index,
                    workspace=consensus_workspace,
                    scratch_dir=self._scratch_dir(consensus_session_id),
                    context=prior_handoff,
                    peer_context=json.dumps(peer_payload, indent=2),
                    require_final_answer=require_final_answer,
                    terminal=terminal,
                )
                call_slot = (
                    f"{scope_id}:i{iteration_index:03d}:s{stage_index:02d}:chair"
                )
                consensus = self._call(
                    call_slot=call_slot,
                    stage_id=consensus_stage.id,
                    canonical_stage=stage.id,
                    role=consensus_stage.role,
                    agent_id=(
                        f"{scope_id}:i{iteration_index:03d}:{stage.id}:chair"
                    ),
                    session_id=consensus_session_id,
                    prompt=prompt,
                    workspace=consensus_workspace,
                    iteration_index=iteration_index,
                    stage_index=stage_index,
                    position_kind="chair",
                    site_id=site_id,
                    round_index=workflow.deliberation_rounds + 1,
                    authoritative=True,
                    require_final_answer=require_final_answer,
                    terminal=terminal,
                )
                prior_handoff = consensus.handoff
        assert consensus is not None
        return consensus

    def _run_site(
        self,
        *,
        source_workspace: Path,
        scope_id: str,
        site_id: str | None,
    ) -> AgentArtifact:
        if self.plan.workflow.mode == "deliberative":
            return self._run_deliberative(
                source_workspace=source_workspace,
                scope_id=scope_id,
                site_id=site_id,
            )
        return self._run_linear(
            source_workspace=source_workspace,
            scope_id=scope_id,
            site_id=site_id,
        )

    def _independent_verification(
        self,
        *,
        final_artifact: AgentArtifact,
        source_workspace: Path | None = None,
        peer_context: str = "",
        central: bool = False,
    ) -> AgentArtifact | None:
        if not self.plan.workflow.safeguards.independent_rerun:
            return None
        session_id = f"{self.plan.run_id}:independent-verifier"
        if central:
            workspace = self._central_workspace(session_id)
        else:
            if source_workspace is None:
                raise ValueError("Non-central verification requires a source workspace.")
            workspace = self._session_workspace(source_workspace, session_id)
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
            iteration_index=self.spec.iteration_policy.iterations,
            stage_index=len(self.spec.stages),
            workspace=workspace,
            scratch_dir=self._scratch_dir(session_id),
            context=final_artifact.handoff,
            peer_context=peer_context,
        )
        return self._call(
            call_slot="independent-verification",
            stage_id=stage.id,
            canonical_stage=stage.id,
            role=stage.role,
            agent_id="independent-verifier",
            session_id=session_id,
            prompt=prompt,
            workspace=workspace,
            iteration_index=self.spec.iteration_policy.iterations,
            stage_index=len(self.spec.stages),
            position_kind="verification",
            round_index=1,
        )

    def execute(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.recorder.emit(
            "run_resumed" if self.resumed_from_state else "run_started",
            {
                **self.plan.public_dict(),
                "iterations": self.spec.iteration_policy.iterations,
                "completion_mode": self.spec.iteration_policy.completion_mode,
                "substrate_hashes": self.substrate_hashes,
            },
        )
        self._write_state()
        if self.plan.workflow.federated:
            site_reports: dict[str, AgentArtifact] = {}
            for site_id, source in sorted(self.plan.task.site_workspaces.items()):
                site_reports[site_id] = self._run_site(
                    source_workspace=source,
                    scope_id=f"site-{site_id}",
                    site_id=site_id,
                )
            authorized_reports = {
                site_id: {"handoff": artifact.handoff} for site_id, artifact in site_reports.items()
            }
            central_session_id = f"{self.plan.run_id}:central-reviewer"
            central_workspace = self._central_workspace(central_session_id)
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
                iteration_index=self.spec.iteration_policy.iterations,
                stage_index=len(self.spec.stages),
                workspace=central_workspace,
                scratch_dir=self._scratch_dir(central_session_id),
                peer_context=json.dumps(authorized_reports, indent=2),
                central=True,
                require_final_answer=True,
                terminal=True,
            )
            final_artifact = self._call(
                call_slot="federated-central-synthesis",
                stage_id=central_stage.id,
                canonical_stage=central_stage.id,
                role=central_stage.role,
                agent_id="central-reviewer",
                session_id=central_session_id,
                prompt=central_prompt,
                workspace=central_workspace,
                iteration_index=self.spec.iteration_policy.iterations,
                stage_index=len(self.spec.stages),
                position_kind="central",
                round_index=1,
                authoritative=True,
                require_final_answer=True,
                terminal=True,
            )
            verification = self._independent_verification(
                final_artifact=final_artifact,
                peer_context=json.dumps(authorized_reports, indent=2),
                central=True,
            )
            site_handoffs = {
                site_id: artifact.handoff for site_id, artifact in site_reports.items()
            }
        else:
            final_artifact = self._run_site(
                source_workspace=self.plan.task.public_workspace,
                scope_id="single-site",
                site_id=None,
            )
            verification = self._independent_verification(
                final_artifact=final_artifact,
                source_workspace=self.plan.task.public_workspace,
            )
            site_handoffs = {}

        expected_calls = required_agent_calls(self.spec, self.plan.task, self.plan.workflow)
        if self.ledger.agent_calls != expected_calls:
            raise RuntimeError(
                f"Healthy execution produced {self.ledger.agent_calls} successful calls; "
                f"the frozen graph requires {expected_calls}."
            )

        result = {
            **self.plan.public_dict(),
            "status": "completed",
            "spec_fingerprint": self.fingerprint,
            "started_at": None,
            "ended_at": _utc_now(),
            "agent_calls": self.ledger.agent_calls,
            "call_attempts": self.call_index,
            "call_failures": self.call_failures,
            "timeout_count": sum(
                failure.get("error_type") in {"TimeoutError", "TimeoutExpired"}
                for failure in self.call_failures
            ),
            "usage": self.ledger.usage.model_dump(mode="json"),
            "iteration_policy": self.spec.iteration_policy.model_dump(mode="json"),
            "iterations_completed": self.iterations_completed,
            "terminal_iteration": self.terminal_iteration,
            "stop_reason": "fixed_iterations_complete",
            "substrate_hashes": self.substrate_hashes,
            "final_artifact": final_artifact.model_dump(mode="json"),
            "verification_artifact": (
                verification.model_dump(mode="json") if verification is not None else None
            ),
            "site_handoffs": site_handoffs,
            "artifact_count": len(self.artifacts),
        }
        _atomic_json(self.run_dir / "artifacts.json", self.artifacts)
        self._write_state(status="completed", stop_reason="fixed_iterations_complete")
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
        if prior.get("status") == "completed":
            if prior.get("spec_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Cannot resume completed run {plan.run_id}: spec fingerprint changed."
                )
            if prior.get("substrate_hashes") != _substrate_hashes(plan):
                raise RuntimeError(
                    f"Cannot resume completed run {plan.run_id}: public substrate changed."
                )
            return {**prior, "resumed": True}
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        archive_root = output_root / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive_path = archive_root / f"{plan.run_id}__{timestamp}"
        run_dir.replace(archive_path)

    started_at = _utc_now()
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
        resume=resume,
    )
    try:
        result = controller.execute()
        result["started_at"] = started_at
        result["resumed"] = controller.resumed_from_state
    except Exception as exc:
        controller._write_state(  # noqa: SLF001 - run boundary owns controller state
            status="failed", stop_reason=f"technical_failure:{type(exc).__name__}"
        )
        result = {
            **plan.public_dict(),
            "status": "failed",
            "spec_fingerprint": fingerprint,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "agent_calls": controller.ledger.agent_calls,
            "call_attempts": controller.call_index,
            "call_failures": controller.call_failures,
            "timeout_count": sum(
                failure.get("error_type") in {"TimeoutError", "TimeoutExpired"}
                for failure in controller.call_failures
            ),
            "usage": controller.ledger.usage.model_dump(mode="json"),
            "iteration_policy": spec.iteration_policy.model_dump(mode="json"),
            "iterations_completed": controller.iterations_completed,
            "terminal_iteration": controller.terminal_iteration,
            "stop_reason": f"technical_failure:{type(exc).__name__}",
            "substrate_hashes": controller.substrate_hashes,
            "resumed": controller.resumed_from_state,
        }
    finally:
        for scoped_runtime in scoped_runtimes.values():
            scoped_runtime.close()
        runtime.close()
    _atomic_json(run_path, result)
    return result


def _frozen_schedule(
    *, spec: ExperimentSpec, plans: list[RunPlan], root: Path, fingerprint: str
) -> list[RunPlan]:
    """Create or validate the seeded replicate-block schedule consumed by every mode."""

    schedule_path = root / "schedule.json"
    by_id = {plan.run_id: plan for plan in plans}
    if len(by_id) != len(plans):
        raise ValueError("Run IDs are not unique; refusing to construct a schedule.")
    if schedule_path.exists():
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        if payload.get("spec_fingerprint") != fingerprint:
            raise ValueError(
                f"Frozen schedule fingerprint does not match the experiment: {schedule_path}"
            )
        run_ids = payload.get("run_ids")
        if not isinstance(run_ids, list) or len(run_ids) != len(set(run_ids)):
            raise ValueError(f"Frozen schedule has malformed or duplicate run IDs: {schedule_path}")
        if set(run_ids) != set(by_id):
            missing = sorted(set(by_id) - set(run_ids))
            extra = sorted(set(run_ids) - set(by_id))
            raise ValueError(
                f"Frozen schedule run IDs do not match the resolved experiment; "
                f"missing={missing}, extra={extra}."
            )
        if payload.get("seed") != spec.schedule_seed:
            raise ValueError(f"Frozen schedule seed does not match: {schedule_path}")
        return [by_id[str(run_id)] for run_id in run_ids]

    rng = random.Random(spec.schedule_seed)
    blocks: list[dict[str, Any]] = []
    ordered: list[RunPlan] = []
    for replicate in sorted({plan.replicate for plan in plans}):
        block = sorted(
            (plan for plan in plans if plan.replicate == replicate),
            key=lambda plan: plan.run_id,
        )
        rng.shuffle(block)
        ordered.extend(block)
        blocks.append(
            {"replicate": replicate, "run_ids": [plan.run_id for plan in block]}
        )
    _atomic_json(
        schedule_path,
        {
            "schema_version": "1",
            "experiment_id": spec.experiment_id,
            "spec_fingerprint": fingerprint,
            "seed": spec.schedule_seed,
            "strategy": "replicate_blocks_seeded_cell_shuffle",
            "blocks": blocks,
            "run_ids": [plan.run_id for plan in ordered],
            "created_at": _utc_now(),
        },
    )
    return ordered


def _write_private_evaluation_index(spec: ExperimentSpec, root: Path) -> None:
    tasks: dict[str, dict[str, Any]] = {}
    for task in spec.tasks:
        if task.private_evaluation_path is None:
            continue
        path = task.private_evaluation_path.resolve(strict=True)
        tasks[task.id] = {
            "semantic_condition": task.semantic_condition,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    assets = {}
    for label, raw_path in sorted(spec.private_evaluator_assets.items()):
        path = raw_path.resolve(strict=True)
        assets[label] = {"path": str(path), "sha256": _sha256_file(path)}
    if tasks or assets:
        _atomic_json(
            root / "private_evaluation_index.json",
            {
                "schema_version": "1",
                "tasks": tasks,
                "assets": assets,
            },
        )


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
    plans = _frozen_schedule(
        spec=spec,
        plans=build_run_plans(spec),
        root=root,
        fingerprint=fingerprint,
    )
    resolved_payload = spec.model_dump(mode="json", exclude_none=True)
    resolved_payload.pop("private_evaluator_assets", None)
    for task in resolved_payload.get("tasks", []):
        task.pop("private_evaluation_path", None)
        task.pop("metadata", None)
    _atomic_json(root / "resolved_spec.json", resolved_payload)
    _write_private_evaluation_index(spec, root)
    plan_payload = [
        {
            **plan.public_dict(),
            "schedule_position": index,
            "planned_agent_calls": required_agent_calls(spec, plan.task, plan.workflow),
            "iterations": spec.iteration_policy.iterations,
        }
        for index, plan in enumerate(plans, start=1)
    ]
    _atomic_json(root / "plan.json", plan_payload)
    planned_calls = sum(item["planned_agent_calls"] for item in plan_payload)
    if dry_run:
        summary = {
            "experiment_id": spec.experiment_id,
            "status": "planned",
            "spec_fingerprint": fingerprint,
            "n_runs": len(plans),
            "planned_agent_calls": planned_calls,
            "schedule_path": str(root / "schedule.json"),
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
    schedule_positions = {plan.run_id: index for index, plan in enumerate(plans)}
    results.sort(key=lambda item: schedule_positions[str(item["run_id"])])
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
        "planned_agent_calls": planned_calls,
        "realized_agent_calls": sum(int(result.get("agent_calls", 0)) for result in results),
        "schedule_path": str(root / "schedule.json"),
        "output_root": str(root),
        "ended_at": _utc_now(),
        "runs": results,
    }
    _atomic_json(root / "summary.json", summary)
    return summary
