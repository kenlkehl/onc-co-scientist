from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.codex_cli_json_adapter import artifact_schema as codex_artifact_schema
from scripts.vllm_cli_json_adapter import (
    _decision_from_text,
    _sandbox_command,
    _trim_history,
    artifact_schema,
    run_adapter,
)


class _FakeResponse:
    def __init__(self, content: str, *, input_tokens: int = 10, output_tokens: int = 5):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        self._content = content

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]):
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]):
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
        max_api_retries=0,
        max_contract_repairs=2,
        max_tool_calls=3,
        min_tool_calls=0,
        max_controller_decisions=4,
        max_tool_output_chars=1_000,
        max_history_chars=10_000,
        max_tokens=1_024,
        temperature=0.2,
        top_p=0.95,
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
    assert all(
        request["max_tokens"] == 1_024
        for request in client.chat.completions.requests
    )
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 15
    assert response.runtime_metadata["contract_repair_count"] == 1
    assert output.exists()
    audit = json.loads((call_dir / "vllm_call.json").read_text())
    assert audit["status"] == "accepted"
    assert [turn["kind"] for turn in audit["turns"]] == [
        "decision",
        "artifact",
        "contract_repair",
    ]
    session_files = list((scratch.parent / ".vllm_sessions").glob("*.json"))
    assert len(session_files) == 1
    session = json.loads(session_files[0].read_text())
    assert len(session["history"]) == 2
