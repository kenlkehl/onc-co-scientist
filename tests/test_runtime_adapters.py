from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from onc_co_scientist.harness.experiment import ModelSpec, ResourceBudget
from onc_co_scientist.harness.runtime import (
    AgentRequest,
    CliJsonRuntime,
    ScientificClaim,
    SubgroupPredicate,
    build_pi_command,
    parse_agent_artifact,
)


def test_parse_agent_artifact_accepts_fenced_or_plain_output() -> None:
    fenced = parse_agent_artifact(
        '```json\n{"summary":"result","handoff":"next","hypotheses":["H1"]}\n```'
    )
    plain = parse_agent_artifact("A narrative result")

    assert fenced.hypotheses == ["H1"]
    assert fenced.handoff == "next"
    assert plain.summary == "A narrative result"
    assert plain.handoff == plain.summary
    assert plain.claims == []


def test_parse_agent_artifact_accepts_strict_scientific_claims() -> None:
    artifact = parse_agent_artifact(
        json.dumps(
            {
                "summary": "supported subgroup",
                "handoff": "carry forward",
                "claims": [
                    {
                        "exposure": "treatment_sotorasib",
                        "outcome": "pfs_months",
                        "direction": "positive",
                        "subgroup": [
                            {"variable": "kras_g12c", "operator": "eq", "value": 1},
                            {"variable": "sex_female", "operator": "eq", "value": 0},
                        ],
                        "comparator": "untreated patients in the same subgroup",
                        "effect_estimate": 4.9,
                        "effect_unit": "months",
                        "p_value": 0.001,
                        "subgroup_n": 3266,
                        "exposed_n": 1154,
                        "comparator_n": 2112,
                        "supported": True,
                        "confidence": 0.95,
                        "evidence": ["adjusted subgroup contrast"],
                    }
                ],
            }
        )
    )

    assert artifact.claims[0].effect_estimate == 4.9
    assert artifact.claims[0].subgroup == [
        SubgroupPredicate(variable="kras_g12c", operator="eq", value=1),
        SubgroupPredicate(variable="sex_female", operator="eq", value=0),
    ]


def test_scientific_claim_rejects_noncanonical_or_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ScientificClaim.model_validate(
            {
                "exposure": "treatment_sotorasib",
                "outcome": "pfs_months",
                "direction": "benefit",
                "unexpected": "not permitted",
            }
        )

    with pytest.raises(ValidationError):
        SubgroupPredicate.model_validate(
            {
                "variable": "kras_g12c",
                "operator": "equals",
                "value": 1,
            }
        )


def test_cli_json_adapter_uses_benchmark_style_file_contract(tmp_path: Path) -> None:
    adapter_script = tmp_path / "adapter.py"
    adapter_script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request-file", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request_file.read_text())
args.output.write_text(json.dumps({
    "summary": "completed " + request["stage_id"],
    "handoff": "next stage",
    "evidence": ["public.txt"]
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    request = AgentRequest(
        request_id="request-1",
        experiment_id="experiment",
        run_id="run",
        task_id="task",
        workflow_id="sequential",
        model_profile="test",
        model_id="test-model",
        stage_id="analysis",
        role="analyst",
        agent_id="agent",
        session_id="session",
        prompt="Analyze.",
        workspace=workspace,
        scratch_dir=tmp_path / "scratch",
        call_dir=tmp_path / "call",
    )
    runtime = CliJsonRuntime(
        ModelSpec(
            id="test",
            model_id="test-model",
            adapter="cli-json",
            command=[sys.executable, str(adapter_script)],
        )
    )

    response = runtime.run(request, ResourceBudget(max_runtime_seconds_per_call=30))

    assert response.request_id == "request-1"
    assert response.artifact.summary == "completed analysis"
    assert response.artifact.evidence == ["public.txt"]
    saved_request = json.loads((tmp_path / "call" / "request.json").read_text())
    assert "call_dir" not in saved_request


def test_pi_command_defaults_to_cleanroom_resources_and_tool_allowlist() -> None:
    command = build_pi_command(
        ModelSpec(
            id="pi",
            model_id="openai/test",
            provider="openai",
            adapter="pi-rpc",
            pi_tools=["read", "bash"],
        )
    )

    for flag in (
        "--mode",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--system-prompt",
        "--tools",
    ):
        assert flag in command
    assert command[command.index("--tools") + 1] == "read,bash"


def test_pi_command_can_disable_all_tools() -> None:
    command = build_pi_command(ModelSpec(id="pi", model_id="test", adapter="pi-rpc", pi_tools=[]))

    assert "--no-tools" in command
    assert "--tools" not in command
