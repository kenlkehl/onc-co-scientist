from __future__ import annotations

import json
from pathlib import Path

import yaml

from onc_co_scientist.harness.experiment import (
    ClinicalBenchmarkSource,
    ExperimentSpec,
    ModelSpec,
    ResourceBudget,
    SafeguardSpec,
    TaskSpec,
    WorkflowSpec,
    import_clinical_benchmark_tasks,
    load_experiment_spec,
)
from onc_co_scientist.harness.orchestrator import RunController, RunPlan, run_experiment
from onc_co_scientist.harness.runtime import (
    AgentArtifact,
    AgentRequest,
    AgentResponse,
    AgentUsage,
)


def _workspace(tmp_path: Path, name: str = "workspace") -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "public.txt").write_text("agent-safe evidence\n", encoding="utf-8")
    return path


def _spec(tmp_path: Path, workflows: list[WorkflowSpec]) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="test-experiment",
        output_root=tmp_path / "out",
        tasks=[
            TaskSpec(
                id="task-1",
                prompt="Determine whether the evidence supports hypothesis A.",
                public_workspace=_workspace(tmp_path),
            )
        ],
        models=[ModelSpec(id="stub-model", model_id="stub", adapter="stub")],
        workflows=workflows,
        replicates=1,
    )


def test_stub_matrix_runs_all_three_workflow_policies(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            WorkflowSpec(id="persistent", mode="persistent"),
            WorkflowSpec(id="sequential", mode="sequential"),
            WorkflowSpec(
                id="deliberative",
                mode="deliberative",
                agents_per_stage=2,
                deliberation_rounds=1,
            ),
        ],
    )
    summary = run_experiment(spec)

    assert summary["n_runs"] == 3
    assert summary["n_completed"] == 3
    assert summary["n_failed"] == 0

    runs = {run["workflow_id"]: run for run in summary["runs"]}
    assert runs["persistent"]["agent_calls"] == 4
    assert runs["sequential"]["agent_calls"] == 4
    assert runs["deliberative"]["agent_calls"] == 12

    persistent_artifacts = json.loads(
        (tmp_path / "out" / "runs" / runs["persistent"]["run_id"] / "artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    sequential_artifacts = json.loads(
        (tmp_path / "out" / "runs" / runs["sequential"]["run_id"] / "artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    assert len({item["session_id"] for item in persistent_artifacts}) == 1
    assert len({item["session_id"] for item in sequential_artifacts}) == 4
    persistent_run_dir = tmp_path / "out" / "runs" / runs["persistent"]["run_id"]
    output_records = [
        json.loads(line)
        for line in (persistent_run_dir / "agent_outputs.jsonl").read_text().splitlines()
    ]
    assert len(output_records) == runs["persistent"]["agent_calls"]
    assert all(record["event_type"] == "agent_output" for record in output_records)
    assert all(record["payload"]["raw_text"] for record in output_records)
    first_call = persistent_run_dir / "calls" / "call_0001"
    assert json.loads((first_call / "request.json").read_text())["prompt"].startswith(
        "You are the hypothesis scientist"
    )
    assert (first_call / "raw_response.txt").read_text() == output_records[0]["payload"]["raw_text"]


def test_resume_reuses_completed_cells_with_same_fingerprint(tmp_path: Path) -> None:
    spec = _spec(tmp_path, [WorkflowSpec(id="sequential", mode="sequential")])
    first = run_experiment(spec)
    second = run_experiment(spec, resume=True)

    assert first["n_completed"] == 1
    assert second["n_completed"] == 1
    assert second["n_resumed"] == 1


def test_rerun_preserves_prior_attempt_outputs_and_call_files(tmp_path: Path) -> None:
    spec = _spec(tmp_path, [WorkflowSpec(id="sequential", mode="sequential")])
    first = run_experiment(spec)
    second = run_experiment(spec)

    first_run = first["runs"][0]
    second_run = second["runs"][0]
    run_dir = tmp_path / "out" / "runs" / first_run["run_id"]
    output_records = [
        json.loads(line) for line in (run_dir / "agent_outputs.jsonl").read_text().splitlines()
    ]

    assert first_run["attempt"] == 1
    assert second_run["attempt"] == 2
    assert len(output_records) == first_run["agent_calls"] + second_run["agent_calls"]
    assert {record["attempt"] for record in output_records} == {1, 2}
    assert (run_dir / "calls" / "call_0001" / "raw_response.txt").exists()
    assert (run_dir / "calls" / "call_0008" / "raw_response.txt").exists()


def test_budget_failure_is_recorded_without_crashing_matrix(tmp_path: Path) -> None:
    spec = _spec(tmp_path, [WorkflowSpec(id="sequential", mode="sequential")])
    spec.budget = ResourceBudget(max_agent_calls=2)
    summary = run_experiment(spec)

    assert summary["n_failed"] == 1
    run = summary["runs"][0]
    assert run["error_type"] == "BudgetExceeded"
    assert run["agent_calls"] == 2
    run_dir = tmp_path / "out" / "runs" / run["run_id"]
    assert (run_dir / "events.jsonl").exists()
    assert len(json.loads((run_dir / "artifacts.json").read_text())) == 2
    assert len((run_dir / "agent_outputs.jsonl").read_text().splitlines()) == 2


class _SensitiveRuntime:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        self.requests.append(request)
        site_id = request.metadata.get("site_id")
        if site_id:
            artifact = AgentArtifact(
                summary=f"RAW_PRIVATE_TRANSCRIPT_{site_id}",
                handoff=f"SAFE_STRUCTURED_REPORT_{site_id}",
            )
        else:
            artifact = AgentArtifact(summary="central synthesis", handoff="central synthesis")
        return AgentResponse(
            request_id=request.request_id,
            artifact=artifact,
            usage=AgentUsage(input_tokens=1, output_tokens=1),
        )

    def close(self) -> None:
        return


def test_federated_reviewer_receives_only_site_handoffs(tmp_path: Path) -> None:
    site_a = _workspace(tmp_path, "site-a")
    site_b = _workspace(tmp_path, "site-b")
    task = TaskSpec(
        id="federated-task",
        prompt="Synthesize the site evidence.",
        public_workspace=site_a,
        site_workspaces={"A": site_a, "B": site_b},
    )
    workflow = WorkflowSpec(
        id="federated-sequential",
        mode="sequential",
        federated=True,
        safeguards=SafeguardSpec(minority_report=True),
    )
    model = ModelSpec(
        id="mixed-deployment",
        model_id="default-model",
        adapter="stub",
        site_model_ids={"A": "site-model-a", "B": "site-model-b"},
        central_model_id="central-model",
    )
    spec = ExperimentSpec(
        experiment_id="federated-test",
        output_root=tmp_path / "out",
        tasks=[task],
        models=[model],
        workflows=[workflow],
    )
    plan = RunPlan(
        run_id="federated-run",
        task=task,
        workflow=workflow,
        model=model,
        replicate=1,
    )
    runtime = _SensitiveRuntime()
    controller = RunController(
        spec=spec,
        plan=plan,
        run_dir=tmp_path / "run",
        runtime=runtime,
        fingerprint=spec.fingerprint(),
    )

    result = controller.execute()

    central = next(
        request for request in runtime.requests if request.agent_id == "central-reviewer"
    )
    assert "SAFE_STRUCTURED_REPORT_A" in central.prompt
    assert "SAFE_STRUCTURED_REPORT_B" in central.prompt
    assert "RAW_PRIVATE_TRANSCRIPT" not in central.prompt
    assert central.model_id == "central-model"
    assert {
        request.model_id for request in runtime.requests if request.metadata.get("site_id") == "A"
    } == {"site-model-a"}
    assert set(result["site_handoffs"]) == {"A", "B"}


def test_clinical_importer_uses_gold_free_question_bank(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    questions.mkdir()
    data_root = tmp_path / "cohorts"
    _workspace(data_root, "nsclc")
    (questions / "nsclc.yaml").write_text(
        yaml.safe_dump(
            {
                "cohort": "nsclc",
                "questions": [
                    {"id": "Q1", "category": 3, "text": "What proportion has EGFR mutation?"},
                    {"id": "Q2", "category": 7, "text": "Estimate an adjusted hazard ratio."},
                ],
            }
        ),
        encoding="utf-8",
    )
    tasks = import_clinical_benchmark_tasks(
        ClinicalBenchmarkSource(
            questions_root=questions,
            cohort_data_root=data_root,
            categories=[3],
        )
    )

    assert [task.id for task in tasks] == ["Q1"]
    assert tasks[0].public_workspace == data_root / "nsclc"
    assert tasks[0].private_evaluation_path is None


def test_clinical_importer_rejects_gold_bearing_public_bank(tmp_path: Path) -> None:
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "nsclc.yaml").write_text(
        yaml.safe_dump(
            {
                "questions": [
                    {
                        "id": "Q1",
                        "category": 3,
                        "text": "Question",
                        "gold_answer": {"value": 42},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        import_clinical_benchmark_tasks(
            ClinicalBenchmarkSource(
                questions_root=questions,
                cohort_data_root=tmp_path / "data",
            )
        )
    except ValueError as exc:
        assert "Gold-bearing" in str(exc)
    else:
        raise AssertionError("Expected gold-bearing public bank to be rejected.")


def test_manifest_paths_are_resolved_relative_to_config(tmp_path: Path) -> None:
    _workspace(tmp_path, "public")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "relative-paths",
                "output_root": "results",
                "tasks": [
                    {
                        "id": "task",
                        "prompt": "Analyze.",
                        "public_workspace": "public",
                    }
                ],
                "models": [{"id": "stub", "model_id": "stub", "adapter": "stub"}],
                "workflows": [{"id": "sequential", "mode": "sequential"}],
            }
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(config)

    assert spec.output_root == (tmp_path / "results").resolve()
    assert spec.tasks[0].public_workspace == (tmp_path / "public").resolve()


def test_dry_run_redacts_private_evaluation_path(tmp_path: Path) -> None:
    public = _workspace(tmp_path)
    private = tmp_path / "gold" / "answers.json"
    private.parent.mkdir()
    private.write_text('{"answer": 42}\n', encoding="utf-8")
    spec = ExperimentSpec(
        experiment_id="redaction-test",
        output_root=tmp_path / "out",
        tasks=[
            TaskSpec(
                id="task",
                prompt="Analyze.",
                public_workspace=public,
                private_evaluation_path=private,
            )
        ],
        models=[ModelSpec(id="stub", model_id="stub", adapter="stub")],
        workflows=[WorkflowSpec(id="sequential", mode="sequential")],
    )

    run_experiment(spec, dry_run=True)
    resolved_text = (tmp_path / "out" / "resolved_spec.json").read_text(encoding="utf-8")

    assert str(private) not in resolved_text
    assert "private_evaluation_path" not in resolved_text
