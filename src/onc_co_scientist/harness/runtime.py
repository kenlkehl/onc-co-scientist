"""Agent runtime adapters used by the co-scientist experiment controller."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .experiment import ModelSpec, ResourceBudget


class SubgroupPredicate(BaseModel):
    """One strict, machine-readable predicate defining a subgroup."""

    model_config = ConfigDict(extra="forbid", strict=True)

    variable: str = Field(min_length=1)
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"]
    value: str | int | float | bool


class ScientificClaim(BaseModel):
    """One normalized, independently scoreable scientific claim."""

    model_config = ConfigDict(extra="forbid", strict=True)

    exposure: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    direction: Literal["positive", "negative", "null", "uncertain"]
    subgroup: list[SubgroupPredicate] = Field(default_factory=list)
    comparator: str = ""
    effect_estimate: float | None = None
    effect_unit: str = ""
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    subgroup_n: int | None = Field(default=None, ge=0)
    exposed_n: int | None = Field(default=None, ge=0)
    comparator_n: int | None = Field(default=None, ge=0)
    supported: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class AgentArtifact(BaseModel):
    """Structured scientific artifact passed between controlled stages."""

    model_config = ConfigDict(extra="allow")

    summary: str = Field(min_length=1)
    handoff: str = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[ScientificClaim] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    minority_report: str = ""
    final_answer: dict[str, Any] | str | None = None


class AgentUsage(BaseModel):
    """Provider-neutral usage accounting for matched-budget checks."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0)

    def __add__(self, other: AgentUsage) -> AgentUsage:
        return AgentUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            tool_calls=self.tool_calls + other.tool_calls,
            cost_usd=self.cost_usd + other.cost_usd,
            duration_seconds=self.duration_seconds + other.duration_seconds,
        )


class AgentRequest(BaseModel):
    """One auditable call from the controller to an agent runtime."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    experiment_id: str
    run_id: str
    task_id: str
    workflow_id: str
    model_profile: str
    model_id: str
    reasoning_effort: str | None = None
    stage_id: str
    require_final_answer: bool = False
    iteration_index: int = Field(default=1, ge=1, le=20)
    max_iterations: int = Field(default=1, ge=1, le=20)
    stage_index: int = Field(default=0, ge=0)
    stage_position: int = Field(default=1, ge=1)
    terminal: bool = False
    role: str
    agent_id: str
    session_id: str
    prompt: str
    workspace: Path
    scratch_dir: Path
    call_dir: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_positions(self) -> AgentRequest:
        if self.iteration_index > self.max_iterations:
            raise ValueError("iteration_index may not exceed max_iterations.")
        if self.stage_position != self.stage_index + 1:
            raise ValueError("stage_position must equal stage_index + 1.")
        if self.terminal and self.iteration_index != self.max_iterations:
            raise ValueError("A terminal request must occur in the final iteration.")
        return self


class AgentResponse(BaseModel):
    """Normalized result from any supported runtime."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    artifact: AgentArtifact
    usage: AgentUsage = Field(default_factory=AgentUsage)
    raw_text: str = ""
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRuntime(Protocol):
    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse: ...

    def close(self) -> None: ...


def _runtime_environment(model: ModelSpec) -> dict[str, str]:
    """Build a small inherited environment plus explicitly allowed credentials."""

    inherited = (
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        # Transport-only failover controls used by the vLLM CLI adapter. They do
        # not alter the frozen experiment spec or conversational session identity.
        "OCS_VLLM_BASE_URL_OVERRIDE",
        "OCS_VLLM_MODEL_ID_OVERRIDE",
    )
    keys = {*inherited, *model.env_passthrough}
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _persist_runtime_success(request: AgentRequest, response: AgentResponse) -> None:
    """Leave an adoption marker before returning control to the orchestrator."""

    _write_json_atomic(
        request.call_dir / "runtime_success.json",
        {
            "schema_version": "1",
            "request_id": request.request_id,
            "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            "call_slot": request.metadata.get("call_slot"),
            "response": response.model_dump(mode="json"),
        },
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Kill a timed-out subprocess and every child in its process group."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except (AttributeError, PermissionError, OSError):
        # ``start_new_session`` is POSIX-only in practice, but retaining this
        # fallback keeps the helper safe if the runtime is exercised elsewhere.
        process.kill()


def run_subprocess_in_group(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in a disposable process group.

    ``subprocess.run`` kills only its direct child on timeout. Agent adapters
    launch nested sandboxes and analysis interpreters, so that behavior can
    leave expensive descendants alive after the harness has abandoned a call.
    """

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_agent_artifact(text: str) -> AgentArtifact:
    """Parse a JSON artifact, retaining plain-text output as a safe fallback."""

    stripped = _strip_json_fence(text)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match and match.group(0) != stripped:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "artifact" in payload and isinstance(payload["artifact"], dict):
            payload = payload["artifact"]
        if "summary" not in payload:
            payload["summary"] = stripped
        if "handoff" not in payload:
            payload["handoff"] = str(payload["summary"])
        return AgentArtifact.model_validate(payload)
    fallback = stripped or "Agent returned an empty response."
    return AgentArtifact(summary=fallback, handoff=fallback)


def artifact_output_instructions(*, require_final_answer: bool = False) -> str:
    """Stable output contract appended to every scientific stage prompt."""

    final_answer_example = (
        '{"conclusion": "final scientific conclusion", "supported_claim_indices": [0]}'
        if require_final_answer
        else "null"
    )
    final_answer_rule = (
        "This is a final synthesis stage. final_answer MUST be non-null and state the final "
        "scientific conclusion, referring to supported claims by their zero-based indices."
        if require_final_answer
        else "This is not a final synthesis stage. Set final_answer to null."
    )
    return (
        "\n\nReturn only one JSON object with this shape:\n"
        "{\n"
        '  "summary": "complete stage result",\n'
        '  "handoff": "self-contained written handoff for the next scientist",\n'
        '  "hypotheses": ["specific hypothesis"],\n'
        '  "analyses": [{"method": "...", "result": "..."}],\n'
        '  "claims": [{\n'
        '    "exposure": "treatment or biomarker",\n'
        '    "outcome": "outcome column",\n'
        '    "direction": "positive|negative|null|uncertain",\n'
        '    "subgroup": [{"variable": "column", "operator": "eq", "value": 1}],\n'
        '    "comparator": "comparison group",\n'
        '    "effect_estimate": 0.0,\n'
        '    "effect_unit": "outcome units",\n'
        '    "p_value": 0.05,\n'
        '    "subgroup_n": 0,\n'
        '    "exposed_n": 0,\n'
        '    "comparator_n": 0,\n'
        '    "supported": true,\n'
        '    "confidence": 0.0,\n'
        '    "evidence": ["analysis or statistic supporting this claim"]\n'
        "  }],\n"
        '  "evidence": ["file, statistic, or observation supporting the result"],\n'
        '  "concerns": ["limitation or competing explanation"],\n'
        '  "minority_report": "material unresolved disagreement, or empty string",\n'
        f'  "final_answer": {final_answer_example}\n'
        "}\n"
        "For each material relationship, emit one claim. Use direction='positive' when the "
        "exposure increases the named outcome, 'negative' when it decreases it, 'null' for a "
        "supported null result, and 'uncertain' otherwise. Encode every defining subgroup "
        "predicate as a separate {variable, operator, value} item, using only eq, ne, lt, le, "
        "gt, ge, in, or not_in. Values must be scalar strings, numbers, or booleans. Use an empty "
        "subgroup list only for a population-wide claim. Use null for unavailable numeric values "
        "rather than inventing them. Confidence and p_value must be between 0 and 1; sample-size "
        "fields must be nonnegative integers.\n"
        f"{final_answer_rule}\n"
        "Do not include Markdown fences or text outside the JSON object."
    )


def build_pi_command(model: ModelSpec) -> list[str]:
    """Resolve the reproducible Pi RPC command without starting a process."""

    command = list(model.command or ["pi"])
    command.extend(["--mode", "rpc", "--no-session", "--model", model.model_id])
    if model.provider:
        command.extend(["--provider", model.provider])
    if model.pi_cleanroom:
        command.extend(
            [
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--no-context-files",
            ]
        )
    if model.pi_system_prompt:
        command.extend(["--system-prompt", model.pi_system_prompt])
    if model.pi_tools:
        command.extend(["--tools", ",".join(model.pi_tools)])
    else:
        command.append("--no-tools")
    command.extend(model.extra_args)
    return command


class CliJsonRuntime:
    """Framework-neutral subprocess contract modeled on the clinical benchmark.

    The configured executable receives ``--request-file`` and ``--output``.
    It may write either an ``AgentArtifact`` directly or an ``AgentResponse``.
    """

    def __init__(self, model: ModelSpec):
        self.model = model

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        request.call_dir.mkdir(parents=True, exist_ok=True)
        request.scratch_dir.mkdir(parents=True, exist_ok=True)
        request_path = request.call_dir / "request.json"
        output_path = request.call_dir / "response.json"
        request_path.write_text(
            request.model_dump_json(indent=2, exclude={"call_dir"}) + "\n",
            encoding="utf-8",
        )
        if output_path.exists():
            output_path.unlink()

        replacements = {
            "{request_file}": str(request_path),
            "{output_file}": str(output_path),
        }
        configured_tokens = [*self.model.command, *self.model.extra_args]
        command = [replacements.get(token, token) for token in configured_tokens]
        if "{request_file}" not in configured_tokens:
            command.extend(["--request-file", str(request_path)])
        if "{output_file}" not in configured_tokens:
            command.extend(["--output", str(output_path)])

        env = _runtime_environment(self.model)
        started = time.monotonic()
        try:
            completed = run_subprocess_in_group(
                command,
                cwd=request.workspace,
                env=env,
                timeout=budget.max_runtime_seconds_per_call,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Agent call {request.request_id} exceeded {budget.max_runtime_seconds_per_call}s"
            ) from exc
        duration = time.monotonic() - started
        (request.call_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (request.call_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        (request.call_dir / "command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Agent command exited {completed.returncode}: {completed.stderr[-1000:]}"
            )
        if not output_path.exists():
            raise RuntimeError("Agent completed without writing its output file.")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "request_id" in payload and "artifact" in payload:
            response = AgentResponse.model_validate(payload)
            response.usage.duration_seconds = duration
            _persist_runtime_success(request, response)
            return response
        artifact = AgentArtifact.model_validate(payload)
        response = AgentResponse(
            request_id=request.request_id,
            artifact=artifact,
            usage=AgentUsage(duration_seconds=duration),
            raw_text=output_path.read_text(encoding="utf-8"),
            runtime_metadata={"adapter": "cli-json", "command": command[0]},
        )
        _persist_runtime_success(request, response)
        return response

    def close(self) -> None:
        return


class _PiSession:
    """One persistent Pi RPC subprocess."""

    def __init__(self, model: ModelSpec, workspace: Path):
        command = build_pi_command(model)
        env = _runtime_environment(model)
        env.setdefault("PI_SKIP_VERSION_CHECK", "1")
        env.setdefault("PI_TELEMETRY", "0")
        self.command = command
        self.process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            raise RuntimeError("Could not open Pi RPC pipes.")
        self.lines: queue.Queue[str] = queue.Queue()
        self.stderr_lines: list[str] = []
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self.last_stats = AgentUsage()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line)

    def send(self, payload: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Pi RPC process exited {self.process.returncode}: "
                f"{''.join(self.stderr_lines)[-1000:]}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def read_json(self, timeout: float) -> tuple[str, dict[str, Any]]:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for Pi RPC output.") from exc
        try:
            return line, json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Pi emitted non-JSON RPC output: {line[:500]}") from exc

    def request_response(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        event_sink: Callable[[str], None],
    ) -> dict[str, Any]:
        request_id = str(payload.setdefault("id", uuid.uuid4().hex))
        self.send(payload)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pi RPC command {payload['type']} timed out.")
            raw, event = self.read_json(remaining)
            event_sink(raw)
            if event.get("type") == "response" and str(event.get("id")) == request_id:
                if not event.get("success"):
                    raise RuntimeError(f"Pi RPC command failed: {event}")
                return event

    def prompt(
        self,
        message: str,
        *,
        timeout: float,
        event_sink: Callable[[str], None],
    ) -> str:
        request_id = uuid.uuid4().hex
        self.send({"id": request_id, "type": "prompt", "message": message})
        deadline = time.monotonic() + timeout
        accepted = False
        latest_text = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.send({"type": "abort"})
                raise TimeoutError("Pi agent prompt timed out.")
            raw, event = self.read_json(remaining)
            event_sink(raw)
            event_type = event.get("type")
            if event_type == "response" and str(event.get("id")) == request_id:
                if not event.get("success"):
                    raise RuntimeError(f"Pi rejected prompt: {event}")
                accepted = True
            elif event_type == "message_end":
                message_payload = event.get("message")
                if isinstance(message_payload, dict) and message_payload.get("role") == "assistant":
                    content = message_payload.get("content", [])
                    if isinstance(content, list):
                        text_parts = [
                            str(block.get("text", ""))
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        ]
                        if text_parts:
                            latest_text = "".join(text_parts)
            elif event_type == "agent_settled" and accepted:
                break
        if latest_text:
            return latest_text
        response = self.request_response(
            {"type": "get_last_assistant_text"},
            timeout=max(1.0, deadline - time.monotonic()),
            event_sink=event_sink,
        )
        return str((response.get("data") or {}).get("text") or "")

    def stats(self, *, timeout: float, event_sink: Callable[[str], None]) -> AgentUsage:
        response = self.request_response(
            {"type": "get_session_stats"},
            timeout=timeout,
            event_sink=event_sink,
        )
        data = response.get("data") or {}
        tokens = data.get("tokens") or {}
        cumulative = AgentUsage(
            input_tokens=int(tokens.get("input", 0) or 0),
            output_tokens=int(tokens.get("output", 0) or 0),
            tool_calls=int(data.get("toolCalls", 0) or 0),
            cost_usd=float(data.get("cost", 0.0) or 0.0),
        )
        delta = AgentUsage(
            input_tokens=max(0, cumulative.input_tokens - self.last_stats.input_tokens),
            output_tokens=max(0, cumulative.output_tokens - self.last_stats.output_tokens),
            tool_calls=max(0, cumulative.tool_calls - self.last_stats.tool_calls),
            cost_usd=max(0.0, cumulative.cost_usd - self.last_stats.cost_usd),
        )
        self.last_stats = cumulative
        return delta

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class PiRpcRuntime:
    """Python client for Pi's documented headless JSONL RPC protocol."""

    def __init__(self, model: ModelSpec):
        self.model = model
        self.sessions: dict[str, _PiSession] = {}

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        request.call_dir.mkdir(parents=True, exist_ok=True)
        request.scratch_dir.mkdir(parents=True, exist_ok=True)
        session = self.sessions.get(request.session_id)
        if session is None:
            session = _PiSession(self.model, request.workspace)
            self.sessions[request.session_id] = session
        events_path = request.call_dir / "events.jsonl"

        def sink(raw: str) -> None:
            with events_path.open("a", encoding="utf-8") as stream:
                stream.write(raw if raw.endswith("\n") else raw + "\n")

        started = time.monotonic()
        raw_text = session.prompt(
            request.prompt,
            timeout=float(budget.max_runtime_seconds_per_call),
            event_sink=sink,
        )
        duration = time.monotonic() - started
        usage = session.stats(timeout=30.0, event_sink=sink)
        usage.duration_seconds = duration
        (request.call_dir / "response.txt").write_text(raw_text, encoding="utf-8")
        return AgentResponse(
            request_id=request.request_id,
            artifact=parse_agent_artifact(raw_text),
            usage=usage,
            raw_text=raw_text,
            runtime_metadata={
                "adapter": "pi-rpc",
                "command": session.command,
                "session_id": request.session_id,
            },
        )

    def close(self) -> None:
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


class StubRuntime:
    """Deterministic runtime for dry integration tests and harness development."""

    def __init__(self, model: ModelSpec):
        self.model = model
        self.calls: list[AgentRequest] = []

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        self.calls.append(request)
        artifact = AgentArtifact(
            summary=f"Stub result for {request.stage_id} by {request.agent_id}.",
            handoff=f"Stub handoff from {request.stage_id}.",
            evidence=[f"workspace={request.workspace.name}"],
            minority_report="" if "minority" not in request.prompt.lower() else "No minority view.",
            final_answer=(
                {"conclusion": "Stub synthesis.", "supported_claim_indices": []}
                if "final_answer MUST be non-null" in request.prompt
                else None
            ),
        )
        request.call_dir.mkdir(parents=True, exist_ok=True)
        (request.call_dir / "response.json").write_text(
            artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return AgentResponse(
            request_id=request.request_id,
            artifact=artifact,
            usage=AgentUsage(input_tokens=100, output_tokens=40, duration_seconds=0.01),
            raw_text=artifact.model_dump_json(),
            runtime_metadata={"adapter": "stub"},
        )

    def close(self) -> None:
        return


def create_runtime(model: ModelSpec) -> AgentRuntime:
    if model.adapter == "pi-rpc":
        return PiRpcRuntime(model)
    if model.adapter == "cli-json":
        return CliJsonRuntime(model)
    return StubRuntime(model)
