import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.aim1_recovery.endpoint import RUNTIME, prepare_endpoint
from experiments.aim1_recovery.run_batch import validate_launch
from onc_co_scientist.harness.python_sandbox import ISOLATION_VERSION


def test_loose_endpoint_preparation_retains_contract_and_public_roles(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "endpoint"
    plan = prepare_endpoint(repo, out, Path(sys.executable), "local-model", "http://unused/v1", 1)
    assert len(plan["jobs"]) == 2
    assert plan["protocol"]["harness_id"] == "endpoint-claude-legacy-loose-v1"
    assert plan["protocol"]["endpoint_runtime"] == RUNTIME
    assert plan["protocol"]["isolation"] == ISOLATION_VERSION
    for job in plan["jobs"]:
        workspace = Path(job["workspace"])
        brief = (workspace / "agent_instructions.md").read_text()
        assert "submit_iteration tool" in brief
        assert "execute_python" in brief
        assert "isolated filesystem" in brief
        assert "working directory is /workspace" in brief
        assert "submit --workspace" not in brief
        assert "finalize --workspace" not in brief
        assert "Fixed exploration budget" not in brief
        assert "non-null structured finding" in brief
        metadata = json.loads((workspace / "metadata.json").read_text())
        assert metadata["harness_id"] == job["harness_id"] == plan["protocol"]["harness_id"]
        assert not metadata["fixed_research_budget"]
        assert metadata["filesystem_isolation"] == ISOLATION_VERSION
        assert metadata["execution_workspace"] == "/workspace"
        assert len(metadata["treatment_columns"]) == 4
        assert all(f"`{column}`" in brief for column in metadata["treatment_columns"])
    args = SimpleNamespace(model="local-model", backend="endpoint", reasoning_effort=None,
                           service_tier=None, **RUNTIME)
    validate_launch(plan, args)
    args.max_tokens_per_call = 1
    with pytest.raises(ValueError, match="frozen endpoint runtime"):
        validate_launch(plan, args)
