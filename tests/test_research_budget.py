import json
import os
from types import SimpleNamespace

import pytest

from experiments.aim1_recovery.run_batch import validate_launch
from onc_co_scientist.harness.structured_runner import StructuredRunner, finalize_workspace


def test_fixed_budget_requires_real_records_and_preserved_artifacts(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(dict(dataset_id="d", max_iterations=4, fixed_research_budget=True))
    )
    runner = StructuredRunner(tmp_path, base_url="http://localhost", model="m")
    prior = []
    assert "research_step" in runner._submit(dict(index=1, proposed_hypotheses=[]), prior, 4)
    for i, action in enumerate(("screen", "multivariable", "refine", "robustness"), 1):
        script, output = f"analysis_{i}.py", f"output_{i}.txt"
        (tmp_path / script).write_text(f"print({i})\n")
        (tmp_path / output).write_text(f"{i}\n")
        record = dict(
            index=i,
            research_step=dict(
                action=action,
                rationale="Test a new candidate",
                script_path=script,
                output_path=output,
            ),
            proposed_hypotheses=[
                dict(
                    id=f"h{i}",
                    text="Candidate",
                    finding=dict(
                        outcome="y",
                        exposure=None,
                        contrast="subgroup_difference",
                        direction=1,
                        subgroup=[],
                    ),
                )
            ],
            analyses=[
                dict(hypothesis_ids=[f"h{i}"], code=script, result_summary="Negative result")
            ],
        )
        if i == 2:
            (tmp_path / script).write_text("print(1)\n")
            assert "script reuse" in runner._submit(record, prior, 4)
            (tmp_path / script).write_text(f"print({i})\n")
        assert runner._submit(record, prior, 4).startswith("Accepted")
        if i < 4:
            with pytest.raises(ValueError, match="fixed research budget"):
                finalize_workspace(tmp_path)
            assert not (tmp_path / "transcript.json").exists()
    assert len(finalize_workspace(tmp_path).iterations) == 4
    (tmp_path / "output_2.txt").write_text("altered")
    with pytest.raises(ValueError, match="artifact integrity"):
        finalize_workspace(tmp_path, write_output=False)


def test_standard_launch_cannot_be_relabelled_priority():
    plan = {
        "protocol": dict(
            schema_version="aim1-structured-v2",
            model_id="gpt-5.6-luna",
            backend="work",
            reasoning_effort="medium",
            service_tier_requested="standard",
        )
    }
    args = SimpleNamespace(
        model="gpt-5.6-luna",
        backend="work",
        reasoning_effort="medium",
        service_tier="standard",
        work_advertised_tier="priority",
    )
    with pytest.raises(ValueError, match="does not verify"):
        validate_launch(plan, args)
    args.work_advertised_tier = "standard"
    validate_launch(plan, args)
    args.model = "other-model"
    with pytest.raises(ValueError, match="model_id"):
        validate_launch(plan, args)


def test_endpoint_continues_after_attempted_early_finish(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text(
        json.dumps(dict(dataset_id="d", max_iterations=4, fixed_research_budget=True))
    )
    responses = [{"role": "assistant", "content": "Finished early"}]
    for i, action in enumerate(("screen", "multivariable", "refine", "robustness"), 1):
        script, output = f"step{i}.py", f"step{i}.txt"
        (tmp_path / script).write_text(f"print({i})\n")
        (tmp_path / output).write_text(f"{i}\n")
        record = dict(
            index=i,
            research_step=dict(
                action=action,
                rationale="New research question",
                script_path=script,
                output_path=output,
            ),
            proposed_hypotheses=[
                dict(
                    id=f"h{i}",
                    text="Candidate",
                    finding=dict(
                        outcome="y",
                        exposure=None,
                        contrast="subgroup_difference",
                        direction=1,
                        subgroup=[],
                    ),
                )
            ],
            analyses=[dict(hypothesis_ids=[f"h{i}"], code=script, result_summary="Result")],
        )
        responses.append(
            dict(
                role="assistant",
                tool_calls=[
                    dict(
                        id=str(i),
                        type="function",
                        function=dict(
                            name="submit_iteration", arguments=json.dumps({"iteration": record})
                        ),
                    )
                ],
            )
        )
    responses.append({"role": "assistant", "content": "All work complete"})
    runner = StructuredRunner(tmp_path, base_url="http://localhost", model="m", max_turns=10)
    requests = []

    def request(messages):
        requests.append(list(messages))
        return dict(choices=[dict(message=responses.pop(0))], usage=dict(completion_tokens=1))

    monkeypatch.setattr(runner, "_request", request)
    transcript = runner.run()
    assert len(transcript.iterations) == 4
    assert "fixed research budget requires 4" in requests[1][-1]["content"]
    assert len(finalize_workspace(tmp_path, write_output=False).iterations) == 4


def test_new_protocol_rejects_precomputed_and_reused_outputs(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            dict(
                dataset_id="d",
                max_iterations=4,
                fixed_research_budget=True,
                require_sequential_outputs=True,
            )
        )
    )
    runner = StructuredRunner(tmp_path, base_url="http://localhost", model="m")
    prior = []
    for i, action in enumerate(("screen", "multivariable", "refine", "robustness"), 1):
        script, output = f"step{i}.py", f"step{i}.txt"
        (tmp_path / script).write_text(f"print({i})\n")
        (tmp_path / output).write_text(f"{i}\n")
        record = dict(
            index=i,
            research_step=dict(
                action=action, rationale=f"Question {i}", script_path=script, output_path=output
            ),
            proposed_hypotheses=[
                dict(
                    id=f"h{i}",
                    text="Candidate",
                    finding=dict(
                        outcome="y",
                        exposure=None,
                        contrast="subgroup_difference",
                        direction=1,
                        subgroup=[],
                    ),
                )
            ],
            analyses=[dict(hypothesis_ids=[f"h{i}"], code=script, result_summary="Result")],
        )
        if i == 2:
            record["research_step"]["output_path"] = "step1.txt"
            assert "separate saved output" in runner._submit(record, prior, 4)
            record["research_step"]["output_path"] = output
            os.utime(tmp_path / output, (1, 1))
            assert "predates the previous submission" in runner._submit(record, prior, 4)
            assert not (tmp_path / "iterations/002.json").exists()
            (tmp_path / output).write_text("fresh output\n")
        assert runner._submit(record, prior, 4).startswith("Accepted")
    assert len(finalize_workspace(tmp_path, write_output=False).iterations) == 4
