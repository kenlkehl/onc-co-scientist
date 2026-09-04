from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.vllm_cli_json_adapter as vllm_adapter
from scripts.codex_cli_json_adapter import artifact_schema as codex_artifact_schema
from scripts.vllm_cli_json_adapter import (
    _decision_from_text,
    _python_resource_failure,
    _sandbox_command,
    _trim_history,
    artifact_schema,
    run_adapter,
)


class _FakeResponse:
    def __init__(
        self,
        content: str | None,
        *,
        input_tokens: int = 10,
        output_tokens: int = 5,
        finish_reason: str = "stop",
    ):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        self._content = content
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._finish_reason = finish_reason

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
            },
        }


class _FakeToolResponse:
    def __init__(
        self,
        name: str,
        arguments: dict[str, Any] | str,
        *,
        call_id: str,
        input_tokens: int = 10,
        output_tokens: int = 5,
        finish_reason: str = "tool_calls",
    ):
        encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
        function = SimpleNamespace(name=name, arguments=encoded)
        tool_call = SimpleNamespace(id=call_id, function=function)
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[tool_call]),
                finish_reason=finish_reason,
            )
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        self._name = name
        self._arguments = encoded
        self._call_id = call_id
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._finish_reason = finish_reason

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": self._call_id,
                                "type": "function",
                                "function": {
                                    "name": self._name,
                                    "arguments": self._arguments,
                                },
                            }
                        ],
                    },
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
            },
        }


class _FakeCompletions:
    def __init__(self, responses: list[Any]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[Any]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))

    def with_options(self, *, timeout: float) -> _FakeClient:
        assert timeout > 0
        return self


def _artifact(*, supported: bool) -> dict[str, Any]:
    return {
        "summary": "completed",
        "handoff": "carry this result forward",
        "hypotheses": ["candidate"],
        "analyses": [{"method": "test", "result": "estimate"}],
        "claims": [
            {
                "exposure": "treatment",
                "outcome": "pfs_months",
                "direction": "positive",
                "subgroup": [{"variable": "marker", "operator": "eq", "value": 1}],
                "comparator": "marker=0",
                "effect_estimate": 2.0,
                "effect_unit": "months",
                "p_value": 0.01,
                "subgroup_n": 100,
                "exposed_n": 50,
                "comparator_n": 50,
                "supported": supported,
                "confidence": 0.9,
                "evidence": ["reproducible model"],
            }
        ],
        "evidence": ["dataset.parquet"],
        "concerns": [],
        "minority_report": "",
        "final_answer": {
            "conclusion": "supported candidate",
            "supported_claim_indices": [0],
        },
    }


def _args(tmp_path: Path, request_file: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        request_file=request_file,
        output=output,
        base_url="http://example.test/v1",
        api_key="EMPTY",
        analysis_python=Path("/home/klkehl/thisenv/bin/python"),
        bwrap="/usr/bin/bwrap",
        timeout_seconds=60,
        api_timeout_seconds=30.0,
        python_timeout_seconds=10,
        python_memory_limit_mb=4096,
        max_api_retries=0,
        max_contract_repairs=2,
        max_tool_calls=3,
        min_tool_calls=0,
        max_controller_decisions=4,
        max_tool_output_chars=1_000,
        max_history_chars=10_000,
        max_tokens=1_024,
        max_decision_tokens=512,
        temperature=0.2,
        top_p=0.95,
        top_k=64,
        repetition_penalty=1.1,
        thinking_mode="enabled",
        interaction_mode="json-schema",
    )


def test_vllm_adapter_uses_exact_codex_stage_schema() -> None:
    assert artifact_schema(require_final_answer=False) == codex_artifact_schema(
        require_final_answer=False
    )
    assert artifact_schema(require_final_answer=True) == codex_artifact_schema(
        require_final_answer=True
    )


def test_decision_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="controller decision failed"):
        _decision_from_text(
            json.dumps(
                {
                    "action": "final",
                    "python_code": "",
                    "purpose": "done",
                    "unexpected": True,
                }
            )
        )


def test_history_trimming_removes_complete_oldest_exchange() -> None:
    history = [
        {"role": "user", "content": "a" * 10},
        {"role": "assistant", "content": "b" * 10},
        {"role": "user", "content": "c" * 10},
        {"role": "assistant", "content": "d" * 10},
    ]
    assert _trim_history(history, 25) == history[2:]


def test_sandbox_command_preserves_dynamic_loader_mount(tmp_path: Path) -> None:
    command = _sandbox_command(
        bwrap="/usr/bin/bwrap",
        python=Path("/home/klkehl/thisenv/bin/python"),
        environment_root=Path("/home/klkehl/thisenv"),
        python_root=Path("/home/klkehl/.local/share/uv/python/example"),
        workspace=tmp_path / "workspace",
        scratch_dir=tmp_path / "scratch",
        code_path=tmp_path / "scratch" / "code.py",
    )
    assert any(
        command[index : index + 3] == ["--ro-bind", "/lib64", "/lib64"]
        for index in range(len(command) - 2)
    )
    assert "--unshare-net" in command
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        position = command.index(variable)
        assert command[position - 1] == "--setenv"
        assert command[position + 1] == "1"


@pytest.mark.parametrize(
    ("stderr", "returncode", "reason"),
    [
        ("MemoryError", 1, "python_memory_error"),
        ("Unable to allocate 18.6 GiB", 1, "allocation_failure"),
        ("", -9, "analysis_process_sigkill"),
        ("Segmentation fault", 1, "analysis_process_sigsegv"),
    ],
)
def test_python_resource_failure_is_reported_to_controller(
    stderr: str,
    returncode: int,
    reason: str,
) -> None:
    assert _python_resource_failure(stderr, returncode) == reason


def test_adapter_repairs_semantic_contract_and_saves_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "persistent",
        "model_profile": "qwen",
        "model_id": "Qwen/Qwen3.8-27B",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Analyze the public dataset and return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    decision = {"action": "final", "python_code": "", "purpose": "ready"}
    client = _FakeClient(
        [
            _FakeResponse(json.dumps(decision)),
            _FakeResponse(json.dumps(_artifact(supported=False))),
            _FakeResponse(json.dumps(_artifact(supported=True))),
        ]
    )

    response = run_adapter(_args(tmp_path, request_file, output), client=client)

    assert response.artifact.claims[0].supported is True
    assert [
        request["max_tokens"] for request in client.chat.completions.requests
    ] == [512, 1_024, 1_024]
    assert all(
        request["extra_body"]
        == {
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 64,
            "repetition_penalty": 1.1,
        }
        for request in client.chat.completions.requests
    )
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 15
    assert response.runtime_metadata["contract_repair_count"] == 1
    assert output.exists()
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert audit["status"] == "accepted"
    assert audit["thinking_mode"] == "enabled"
    assert audit["interaction_mode"] == "json-schema"
    assert [turn["kind"] for turn in audit["turns"]] == [
        "decision",
        "artifact",
        "contract_repair",
    ]
    session_files = list((scratch.parent / ".vllm_sessions").glob("*.json"))
    assert len(session_files) == 1
    session = json.loads(session_files[0].read_text())
    assert len(session["history"]) == 2


def test_adapter_uses_native_tools_for_controller_and_artifact(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "persistent",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    client = _FakeClient(
        [
            _FakeToolResponse(
                "finish_stage", {"purpose": "ready"}, call_id="finish-1"
            ),
            _FakeToolResponse(
                "submit_stage_artifact",
                _artifact(supported=True),
                call_id="artifact-1",
            ),
        ]
    )
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 0

    response = run_adapter(args, client=client)

    requests = client.chat.completions.requests
    assert len(requests) == 2
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["max_tokens"] == 512
    assert {
        tool["function"]["name"] for tool in requests[0]["tools"]
    } == {"run_python", "finish_stage"}
    assert requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_stage_artifact"},
    }
    assert requests[1]["max_tokens"] == 1_024
    assert response.artifact.final_answer is not None
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert audit["interaction_mode"] == "native-tools"
    assert [turn["kind"] for turn in audit["turns"]] == ["decision", "artifact"]


def test_adapter_retries_native_tool_arguments_that_fail_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "persistent",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    client = _FakeClient(
        [
            _FakeToolResponse("finish_stage", {}, call_id="invalid-finish"),
            _FakeToolResponse(
                "finish_stage", {"purpose": "ready"}, call_id="valid-finish"
            ),
            _FakeToolResponse(
                "submit_stage_artifact",
                _artifact(supported=True),
                call_id="artifact-1",
            ),
        ]
    )
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 0
    args.max_api_retries = 1

    run_adapter(args, client=client)

    assert len(client.chat.completions.requests) == 3
    assert (
        client.chat.completions.requests[0]["seed"]
        != client.chat.completions.requests[1]["seed"]
    )
    assert "SCHEMA RECOVERY" in client.chat.completions.requests[1]["messages"][-1][
        "content"
    ]
    recovery = client.chat.completions.requests[1]["messages"][-1]["content"]
    assert "`finish_stage`" in recovery
    assert "`run_python`" in recovery
    attempt_root = call_dir / "vllm_turns" / "model_turn_0001_decision"
    first_error = json.loads((attempt_root / "attempt_0001" / "error.json").read_text())
    assert first_error["error_type"] == "ValueError"
    assert first_error["retryable"] is True
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert audit["turns"][0]["api_attempts"] == 2


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "native_schema",
        "plain_schema",
        "native_malformed_json",
        "native_markdown_fence",
        "plain_unterminated_fence",
    ],
)
def test_native_artifact_schema_failure_enters_contract_repair_loop(
    tmp_path: Path, invalid_mode: str
) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "persistent",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    invalid = _artifact(supported=True)
    invalid.pop("summary")
    invalid_response: Any
    if invalid_mode == "plain_schema":
        invalid_response = _FakeResponse(json.dumps(invalid))
    elif invalid_mode == "native_schema":
        invalid_response = _FakeToolResponse(
            "submit_stage_artifact", invalid, call_id="invalid-artifact"
        )
    elif invalid_mode == "native_malformed_json":
        invalid_response = _FakeToolResponse(
            "submit_stage_artifact", '{"summary":', call_id="invalid-artifact"
        )
    elif invalid_mode == "native_markdown_fence":
        invalid_response = _FakeToolResponse(
            "submit_stage_artifact",
            "```json\n" + json.dumps(_artifact(supported=True)) + "\n```",
            call_id="invalid-artifact",
        )
    else:
        invalid_response = _FakeResponse(
            "```json\n" + json.dumps(_artifact(supported=True))
        )
    client = _FakeClient(
        [
            _FakeToolResponse(
                "finish_stage", {"purpose": "ready"}, call_id="finish-1"
            ),
            invalid_response,
            _FakeToolResponse(
                "submit_stage_artifact",
                _artifact(supported=True),
                call_id="repaired-artifact",
            ),
        ]
    )
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 0
    args.max_api_retries = 2

    response = run_adapter(args, client=client)

    assert response.artifact.final_answer is not None
    assert len(client.chat.completions.requests) == 3
    repair_request = client.chat.completions.requests[-1]
    assert "CONTROLLER CONTRACT REPAIR" in repair_request["messages"][-1]["content"]
    if invalid_mode in {"native_schema", "plain_schema"}:
        assert "'summary' is a required property" in repair_request["messages"][-1][
            "content"
        ]
    elif invalid_mode == "plain_unterminated_fence":
        assert "unterminated Markdown fence" in repair_request["messages"][-1][
            "content"
        ]
    else:
        assert "was not valid JSON" in repair_request["messages"][-1]["content"]
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert [turn["kind"] for turn in audit["turns"]] == [
        "decision",
        "artifact",
        "contract_repair",
    ]
    assert audit["turns"][1]["status"] == "contract_rejected"
    assert audit["turns"][1]["api_attempts"] == 1
    assert audit["turns"][2]["status"] == "accepted"
    assert audit["contract_repair_count"] == 1


def test_adapter_accepts_exact_native_artifact_only_after_minimum_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "deliberative",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis_consensus",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist consensus chair",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    candidate = _artifact(supported=True)
    client = _FakeClient(
        [
            _FakeToolResponse(
                "submit_stage_artifact", candidate, call_id="too-early"
            ),
            _FakeToolResponse(
                "run_python",
                {"python_code": "print(1)", "purpose": "minimum inspection"},
                call_id="python-1",
            ),
            _FakeToolResponse(
                "submit_stage_artifact", candidate, call_id="accepted-artifact"
            ),
        ]
    )
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 1
    args.max_api_retries = 1
    monkeypatch.setattr(
        vllm_adapter,
        "_run_python_tool",
        lambda **_: {
            "tool": "python",
            "tool_number": 1,
            "status": "ok",
            "returncode": 0,
            "duration_seconds": 0.0,
            "stdout": "1\n",
            "stderr": "",
            "stdout_bytes": 2,
            "stderr_bytes": 0,
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
    )

    response = run_adapter(args, client=client)

    requests = client.chat.completions.requests
    assert len(requests) == 3
    assert all(
        {tool["function"]["name"] for tool in item["tools"]}
        == {"run_python", "finish_stage"}
        for item in requests
    )
    assert "`finish_stage`" in requests[1]["messages"][-1]["content"]
    assert "submit_stage_artifact" in requests[0]["messages"][-1]["content"]
    assert response.artifact.final_answer is not None
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert [turn["kind"] for turn in audit["turns"]] == [
        "decision",
        "python_tool",
        "content_artifact",
    ]
    assert audit["turns"][-1]["response_mode"] == (
        "validated_native_artifact_fallback"
    )
    first_error = json.loads(
        (
            call_dir
            / "vllm_turns"
            / "model_turn_0001_decision"
            / "attempt_0001"
            / "error.json"
        ).read_text()
    )
    assert "unavailable native tool" in first_error["error"]
    fallback_success = json.loads(
        (
            call_dir
            / "vllm_turns"
            / "model_turn_0002_decision"
            / "attempt_0001"
            / "success.json"
        ).read_text()
    )
    assert fallback_success["response_mode"] == (
        "validated_native_artifact_fallback"
    )
    assert fallback_success["content_fallback_schema_sha256"] == audit[
        "output_schema_sha256"
    ]


def test_adapter_recovers_from_length_exhaustion_with_fresh_seed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "deliberative",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    client = _FakeClient(
        [
            _FakeResponse(
                None,
                input_tokens=100,
                output_tokens=16_000,
                finish_reason="length",
            ),
            _FakeToolResponse(
                "finish_stage", {"purpose": "ready"}, call_id="finish-2"
            ),
            _FakeToolResponse(
                "submit_stage_artifact",
                _artifact(supported=True),
                call_id="artifact-1",
            ),
        ]
    )
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 0
    args.max_api_retries = 1

    response = run_adapter(args, client=client)

    requests = client.chat.completions.requests
    assert [item["max_tokens"] for item in requests] == [512, 512, 1_024]
    assert requests[0]["seed"] != requests[1]["seed"]
    assert requests[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert requests[1]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "GENERATION RECOVERY" in requests[1]["messages"][-1]["content"]
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 16_010
    first_attempt = (
        call_dir
        / "vllm_turns"
        / "model_turn_0001_decision"
        / "attempt_0001"
    )
    first_error = json.loads((first_attempt / "error.json").read_text())
    assert first_error["finish_reason"] == "length"
    assert first_error["output_tokens"] == 16_000
    assert (first_attempt / "request.json").exists()
    second_request = json.loads(
        (
            first_attempt.parent / "attempt_0002" / "request.json"
        ).read_text()
    )
    assert second_request["seed"] != json.loads(
        (first_attempt / "request.json").read_text()
    )["seed"]


@pytest.mark.parametrize("fallback_on_controller", [True, False])
def test_adapter_accepts_schema_valid_plain_content_from_native_server(
    tmp_path: Path, fallback_on_controller: bool
) -> None:
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "run" / "scratch" / "session"
    call_dir = tmp_path / "run" / "calls" / "call_0001"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    request = {
        "request_id": "run:c0001",
        "experiment_id": "experiment",
        "run_id": "run",
        "task_id": "task",
        "workflow_id": "persistent",
        "model_profile": "gemma",
        "model_id": "gemma4-31b",
        "reasoning_effort": None,
        "stage_id": "synthesis",
        "require_final_answer": True,
        "iteration_index": 1,
        "max_iterations": 1,
        "stage_index": 3,
        "stage_position": 4,
        "terminal": True,
        "role": "synthesis scientist",
        "agent_id": "agent",
        "session_id": "persistent-session",
        "prompt": "Return the required artifact.",
        "workspace": str(workspace),
        "scratch_dir": str(scratch),
        "metadata": {},
    }
    request_file = call_dir / "request.json"
    request_file.parent.mkdir(parents=True)
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = call_dir / "response.json"
    plain_artifact = _FakeResponse(
        "```json\n" + json.dumps(_artifact(supported=True)) + "\n```"
    )
    responses: list[Any] = [plain_artifact]
    if not fallback_on_controller:
        responses.insert(
            0,
            _FakeToolResponse(
                "finish_stage", {"purpose": "ready"}, call_id="finish-1"
            ),
        )
    client = _FakeClient(responses)
    args = _args(tmp_path, request_file, output)
    args.interaction_mode = "native-tools"
    args.min_tool_calls = 0

    response = run_adapter(args, client=client)

    assert response.artifact.final_answer is not None
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    model_turns = [
        turn for turn in audit["turns"] if turn["kind"] != "python_tool"
    ]
    fallback_turn = model_turns[-1]
    assert fallback_turn["response_mode"] == "validated_content_fallback"
    expected_kind = "content_artifact" if fallback_on_controller else "artifact"
    assert fallback_turn["kind"] == expected_kind
    success = json.loads(
        (
            call_dir
            / fallback_turn["directory"]
            / "attempt_0001"
            / "success.json"
        ).read_text()
    )
    assert success["response_mode"] == "validated_content_fallback"
    assert success["content_fallback_schema_sha256"] == audit["output_schema_sha256"]
