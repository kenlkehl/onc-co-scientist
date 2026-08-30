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
from onc_co_scientist.harness.orchestrator import (
    RunController,
    RunPlan,
    build_run_plans,
    run_experiment,
)
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


def test_run_plans_interleave_workflows_within_each_replicate(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            WorkflowSpec(id="persistent", mode="persistent"),
            WorkflowSpec(id="sequential", mode="sequential"),
            WorkflowSpec(
                id="deliberative",
                mode="deliberative",
                agents_per_stage=2,
            ),
        ],
    )
    spec.replicates = 2

    plans = build_run_plans(spec)

    assert [(plan.replicate, plan.workflow.id) for plan in plans] == [
        (1, "persistent"),
        (1, "sequential"),
        (1, "deliberative"),
        (2, "persistent"),
        (2, "sequential"),
        (2, "deliberative"),
    ]


def test_resume_reuses_completed_cells_with_same_fingerprint(tmp_path: Path) -> None:
    spec = _spec(tmp_path, [WorkflowSpec(id="sequential", mode="sequential")])
    first = run_experiment(spec)
    second = run_experiment(spec, resume=True)

    assert first["n_completed"] == 1
    assert second["n_completed"] == 1
    assert second["n_resumed"] == 1


def test_budget_failure_is_recorded_without_crashing_matrix(tmp_path: Path) -> None:
    spec = _spec(tmp_path, [WorkflowSpec(id="sequential", mode="sequential")])
    spec.budget = ResourceBudget(max_agent_calls=2)
    summary = run_experiment(spec)

    assert summary["n_failed"] == 1
    run = summary["runs"][0]
    assert run["error_type"] == "BudgetExceeded"
    assert run["agent_calls"] == 2
    assert (tmp_path / "out" / "runs" / run["run_id"] / "events.jsonl").exists()
    partial_artifacts = json.loads(
        (tmp_path / "out" / "runs" / run["run_id"] / "artifacts.json").read_text()
    )
    assert len(partial_artifacts) == 2


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


class _RecordingRuntime:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(
            request_id=request.request_id,
            artifact=AgentArtifact(
                summary=f"completed {request.stage_id}",
                handoff=f"handoff {request.stage_id}",
            ),
            usage=AgentUsage(input_tokens=1, output_tokens=1),
        )

    def close(self) -> None:
        return


def test_copy_strategy_isolates_workspace_and_scratch_by_session(tmp_path: Path) -> None:
    workflows = [
        WorkflowSpec(id="persistent", mode="persistent"),
        WorkflowSpec(id="sequential", mode="sequential"),
        WorkflowSpec(
            id="deliberative",
            mode="deliberative",
            agents_per_stage=2,
            deliberation_rounds=2,
        ),
    ]
    expected_session_counts = {"persistent": 1, "sequential": 4, "deliberative": 12}

    for workflow in workflows:
        case_root = tmp_path / workflow.id
        source = _workspace(case_root)
        task = TaskSpec(id="task", prompt="Analyze.", public_workspace=source)
        model = ModelSpec(id="model", model_id="stub", adapter="stub")
        spec = ExperimentSpec(
            experiment_id=f"isolation-{workflow.id}",
            workspace_strategy="copy",
            output_root=case_root / "out",
            tasks=[task],
            models=[model],
            workflows=[workflow],
        )
        plan = RunPlan(
            run_id=f"run-{workflow.id}",
            task=task,
            workflow=workflow,
            model=model,
            replicate=1,
        )
        runtime = _RecordingRuntime()
        controller = RunController(
            spec=spec,
            plan=plan,
            run_dir=case_root / "run",
            runtime=runtime,
            fingerprint=spec.fingerprint(),
        )

        controller.execute()

        by_session: dict[str, list[AgentRequest]] = {}
        for request in runtime.requests:
            by_session.setdefault(request.session_id, []).append(request)
        assert len(by_session) == expected_session_counts[workflow.id]
        for session_requests in by_session.values():
            assert len({request.workspace for request in session_requests}) == 1
            assert len({request.scratch_dir for request in session_requests}) == 1
            workspace = session_requests[0].workspace
            assert workspace != source
            assert (workspace / "public.txt").read_text() == "agent-safe evidence\n"
        assert len({requests[0].workspace for requests in by_session.values()}) == len(by_session)
        assert len({requests[0].scratch_dir for requests in by_session.values()}) == len(by_session)


def test_only_final_synthesis_prompts_require_final_answer(tmp_path: Path) -> None:
    workflow = WorkflowSpec(id="persistent", mode="persistent")
    spec = _spec(tmp_path, [workflow])
    plan = RunPlan(
        run_id="final-contract",
        task=spec.tasks[0],
        workflow=workflow,
        model=spec.models[0],
        replicate=1,
    )
    runtime = _RecordingRuntime()
    controller = RunController(
        spec=spec,
        plan=plan,
        run_dir=tmp_path / "run",
        runtime=runtime,
        fingerprint=spec.fingerprint(),
    )

    controller.execute()

    for request in runtime.requests:
        if request.stage_id == "synthesis":
            assert "final_answer MUST be non-null" in request.prompt
            assert '"final_answer": {"conclusion"' in request.prompt
        else:
            assert "Set final_answer to null" in request.prompt
            assert '"final_answer": null' in request.prompt


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
