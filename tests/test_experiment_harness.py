from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from onc_co_scientist.harness.experiment import (
    ClinicalBenchmarkSource,
    ExperimentSpec,
    IterationPolicy,
    ModelSpec,
    ResourceBudget,
    SafeguardSpec,
    TaskSpec,
    WorkflowSpec,
    import_clinical_benchmark_tasks,
    load_experiment_spec,
    required_agent_calls,
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
    CliJsonRuntime,
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


def test_treatment_roles_change_fingerprint_without_invalidating_legacy_configs(tmp_path):
    spec = _spec(tmp_path, [WorkflowSpec(id="persistent", mode="persistent")])
    legacy = spec.model_dump(mode="json", exclude_none=True)
    for task in legacy["tasks"]:
        task.pop("treatment_columns")
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert spec.fingerprint() == expected
    spec.tasks[0].treatment_columns = ["feature_123"]
    assert spec.fingerprint() != expected
    first = spec.fingerprint()
    spec.tasks[0].treatment_columns = ["feature_124"]
    assert spec.fingerprint() != first


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
                final_answer=(
                    {"conclusion": "site synthesis", "supported_claim_indices": []}
                    if "final_answer MUST be non-null" in request.prompt
                    else None
                ),
            )
        else:
            artifact = AgentArtifact(
                summary="central synthesis",
                handoff="central synthesis",
                final_answer=(
                    {"conclusion": "central synthesis", "supported_claim_indices": []}
                    if "final_answer MUST be non-null" in request.prompt
                    else None
                ),
            )
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
                final_answer=(
                    {"conclusion": "synthesis", "supported_claim_indices": []}
                    if "final_answer MUST be non-null" in request.prompt
                    else None
                ),
            ),
            usage=AgentUsage(input_tokens=1, output_tokens=1),
        )

    def close(self) -> None:
        return


@pytest.mark.parametrize("mode", ["persistent", "sequential", "deliberative"])
@pytest.mark.parametrize("columns", [["treatment_example"], ["feature_123", "feature_007"]])
def test_treatment_roles_reach_every_workflow_call(tmp_path, mode, columns):
    workflow = WorkflowSpec(
        id=mode, mode=mode, agents_per_stage=2 if mode == "deliberative" else 1
    )
    spec = _spec(tmp_path, [workflow])
    spec.tasks[0].treatment_columns = columns
    spec.iteration_policy = IterationPolicy(iterations=2)
    runtime = _RecordingRuntime()
    controller = RunController(
        spec=spec, plan=build_run_plans(spec)[0], runtime=runtime,
        run_dir=tmp_path / "run", fingerprint=spec.fingerprint(),
    )
    assert controller.execute()["status"] == "completed"
    assert len(runtime.requests) == (24 if mode == "deliberative" else 8)
    for request in runtime.requests:
        assert "## Treatment variables" in request.prompt
        for column in columns:
            assert f"- `{column}`" in request.prompt
        assert "use the treatment being tested as the exposure" in request.prompt
        if columns[0].startswith("feature_"):
            assert "treatment_example" not in request.prompt


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
                metadata={"never_expose": "PRIVATE_TARGET_DETAIL"},
            )
        ],
        models=[ModelSpec(id="stub", model_id="stub", adapter="stub")],
        workflows=[WorkflowSpec(id="sequential", mode="sequential")],
        private_evaluator_assets={"mapping": private},
    )

    run_experiment(spec, dry_run=True)
    resolved_text = (tmp_path / "out" / "resolved_spec.json").read_text(encoding="utf-8")

    assert str(private) not in resolved_text
    assert "private_evaluation_path" not in resolved_text
    assert "private_evaluator_assets" not in resolved_text
    assert "PRIVATE_TARGET_DETAIL" not in resolved_text


def test_twenty_iteration_call_graph_order_sessions_and_final_contract(tmp_path: Path) -> None:
    workflows = [
        WorkflowSpec(id="persistent", mode="persistent"),
        WorkflowSpec(id="sequential", mode="sequential"),
        WorkflowSpec(
            id="deliberative", mode="deliberative", agents_per_stage=2, deliberation_rounds=1
        ),
    ]
    spec = _spec(tmp_path, workflows)
    spec.iteration_policy = IterationPolicy(iterations=20, completion_mode="fixed")
    spec.budget = ResourceBudget(max_agent_calls=240)

    summary = run_experiment(spec)

    assert summary["planned_agent_calls"] == 400
    runs = {run["workflow_id"]: run for run in summary["runs"]}
    assert {key: runs[key]["agent_calls"] for key in runs} == {
        "persistent": 80,
        "sequential": 80,
        "deliberative": 240,
    }
    assert all(run["iterations_completed"] == 20 for run in runs.values())
    assert all(run["terminal_iteration"] == 20 for run in runs.values())

    for workflow_id, expected_sessions in {
        "persistent": 1,
        "sequential": 80,
        "deliberative": 240,
    }.items():
        artifacts = json.loads(
            (
                tmp_path
                / "out"
                / "runs"
                / runs[workflow_id]["run_id"]
                / "artifacts.json"
            ).read_text(encoding="utf-8")
        )
        assert len({item["session_id"] for item in artifacts}) == expected_sessions
        checkpoints = [
            item
            for item in artifacts
            if item["position_kind"] in ({"chair"} if workflow_id == "deliberative" else {"linear"})
        ]
        assert [item["canonical_stage"] for item in checkpoints[:4]] == [
            "hypothesis_generation",
            "analysis",
            "critique",
            "synthesis",
        ]
        for item in artifacts:
            if item["canonical_stage"] == "synthesis":
                assert item["artifact"]["final_answer"] is not None
            else:
                assert item["artifact"]["final_answer"] is None


def test_iteration_validation_and_exact_call_ceiling(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task = TaskSpec(id="task", prompt="Analyze.", public_workspace=workspace)
    model = ModelSpec(id="stub", model_id="stub", adapter="stub")
    workflows = [
        WorkflowSpec(id="persistent", mode="persistent"),
        WorkflowSpec(id="sequential", mode="sequential"),
        WorkflowSpec(id="deliberative", mode="deliberative", agents_per_stage=2),
    ]
    spec = ExperimentSpec(
        experiment_id="call-ceilings",
        tasks=[task],
        models=[model],
        workflows=workflows,
        iteration_policy=IterationPolicy(iterations=20),
        budget=ResourceBudget(max_agent_calls=240),
    )
    assert [required_agent_calls(spec, task, workflow) for workflow in workflows] == [80, 80, 240]

    with pytest.raises(ValidationError, match="below the 240 calls required"):
        ExperimentSpec(
            experiment_id="too-small",
            tasks=[task],
            models=[model],
            workflows=workflows,
            iteration_policy=IterationPolicy(iterations=20),
            budget=ResourceBudget(max_agent_calls=239),
        )
    with pytest.raises(ValidationError):
        IterationPolicy(iterations=21)


class _InterruptAfterRuntime(_RecordingRuntime):
    def __init__(self, successful_calls: int) -> None:
        super().__init__()
        self.successful_calls = successful_calls

    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        if len(self.requests) >= self.successful_calls:
            raise RuntimeError("deliberate interruption")
        return super().run(request, budget)


class _UniqueHandoffRuntime(_RecordingRuntime):
    def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
        self.requests.append(request)
        handoff = (
            f"HANDOFF::{request.iteration_index}::{request.stage_id}::"
            f"{request.metadata['position_kind']}::{request.metadata.get('peer_index')}"
        )
        return AgentResponse(
            request_id=request.request_id,
            artifact=AgentArtifact(
                summary=handoff,
                handoff=handoff,
                final_answer=(
                    {"conclusion": handoff, "supported_claim_indices": []}
                    if "final_answer MUST be non-null" in request.prompt
                    else None
                ),
            ),
            usage=AgentUsage(input_tokens=1, output_tokens=1),
        )


def _canonical_artifacts(path: Path) -> list[tuple[str, dict, dict]]:
    artifacts = json.loads((path / "artifacts.json").read_text(encoding="utf-8"))
    return [
        (item["call_slot"], item["artifact"], item["usage"])
        for item in artifacts
    ]


def test_cross_iteration_handoffs_match_each_workflow_contract(tmp_path: Path) -> None:
    for workflow in (
        WorkflowSpec(id="persistent", mode="persistent"),
        WorkflowSpec(id="sequential", mode="sequential"),
        WorkflowSpec(id="deliberative", mode="deliberative", agents_per_stage=2),
    ):
        case = tmp_path / workflow.id
        task = TaskSpec(
            id="task", prompt="Analyze.", public_workspace=_workspace(case)
        )
        model = ModelSpec(id="stub", model_id="stub", adapter="stub")
        spec = ExperimentSpec(
            experiment_id=f"handoff-{workflow.id}",
            tasks=[task],
            models=[model],
            workflows=[workflow],
            iteration_policy=IterationPolicy(iterations=2),
            budget=ResourceBudget(max_agent_calls=24),
        )
        runtime = _UniqueHandoffRuntime()
        RunController(
            spec=spec,
            plan=RunPlan(
                run_id="run", task=task, workflow=workflow, model=model, replicate=1
            ),
            run_dir=case / "run",
            runtime=runtime,
            fingerprint=spec.fingerprint(),
        ).execute()

        if workflow.mode == "persistent":
            assert len({request.session_id for request in runtime.requests}) == 1
            assert all(
                "AUTHORIZED WRITTEN HANDOFF" not in request.prompt
                for request in runtime.requests
            )
        elif workflow.mode == "sequential":
            handoffs = [
                f"HANDOFF::{request.iteration_index}::{request.stage_id}::linear::None"
                for request in runtime.requests
            ]
            assert "AUTHORIZED WRITTEN HANDOFF" not in runtime.requests[0].prompt
            for index, request in enumerate(runtime.requests[1:], start=1):
                assert handoffs[index - 1] in request.prompt
                assert all(
                    older not in request.prompt for older in handoffs[: max(0, index - 1)]
                )
        else:
            prior_chair = ""
            requests = runtime.requests
            for offset in range(0, len(requests), 3):
                peer_one, peer_two, chair = requests[offset : offset + 3]
                peer_one_handoff = (
                    f"HANDOFF::{peer_one.iteration_index}::{peer_one.stage_id}::peer::1"
                )
                peer_two_handoff = (
                    f"HANDOFF::{peer_two.iteration_index}::{peer_two.stage_id}::peer::2"
                )
                if prior_chair:
                    assert prior_chair in peer_one.prompt
                    assert prior_chair in peer_two.prompt
                    assert prior_chair in chair.prompt
                assert peer_two_handoff not in peer_one.prompt
                assert peer_one_handoff not in peer_two.prompt
                assert peer_one_handoff in chair.prompt
                assert peer_two_handoff in chair.prompt
                prior_chair = (
                    f"HANDOFF::{chair.iteration_index}::{chair.stage_id}::chair::None"
                )


@pytest.mark.parametrize(
    ("workflow", "total_calls"),
    [
        (WorkflowSpec(id="persistent", mode="persistent"), 8),
        (WorkflowSpec(id="sequential", mode="sequential"), 8),
        (
            WorkflowSpec(id="deliberative", mode="deliberative", agents_per_stage=2),
            24,
        ),
    ],
)
def test_resume_after_every_call_position_matches_uninterrupted(
    tmp_path: Path, workflow: WorkflowSpec, total_calls: int
) -> None:
    for completed_before_interrupt in range(1, total_calls):
        case = tmp_path / f"{workflow.id}-{completed_before_interrupt}"
        source = _workspace(case)
        task = TaskSpec(id="task", prompt="Analyze.", public_workspace=source)
        model = ModelSpec(id="stub", model_id="stub", adapter="stub")
        spec = ExperimentSpec(
            experiment_id=f"resume-{workflow.id}",
            output_root=case / "out",
            tasks=[task],
            models=[model],
            workflows=[workflow],
            iteration_policy=IterationPolicy(iterations=2),
            budget=ResourceBudget(max_agent_calls=total_calls),
        )
        plan = RunPlan(
            run_id="run", task=task, workflow=workflow, model=model, replicate=1
        )
        interrupted_dir = case / "interrupted"
        first_runtime = _InterruptAfterRuntime(completed_before_interrupt)
        first = RunController(
            spec=spec,
            plan=plan,
            run_dir=interrupted_dir,
            runtime=first_runtime,
            fingerprint=spec.fingerprint(),
        )
        with pytest.raises(RuntimeError, match="deliberate interruption"):
            first.execute()
        resumed_runtime = _RecordingRuntime()
        resumed = RunController(
            spec=spec,
            plan=plan,
            run_dir=interrupted_dir,
            runtime=resumed_runtime,
            fingerprint=spec.fingerprint(),
            resume=True,
        )
        result = resumed.execute()
        assert result["agent_calls"] == total_calls
        assert len(resumed_runtime.requests) == total_calls - completed_before_interrupt
        if workflow.mode == "persistent":
            assert len(
                {
                    request.session_id
                    for request in [*first_runtime.requests, *resumed_runtime.requests]
                }
            ) == 1
        assert len({item[0] for item in _canonical_artifacts(interrupted_dir)}) == total_calls

        reference_dir = case / "reference"
        reference = RunController(
            spec=spec,
            plan=plan,
            run_dir=reference_dir,
            runtime=_RecordingRuntime(),
            fingerprint=spec.fingerprint(),
        )
        reference.execute()
        assert _canonical_artifacts(interrupted_dir) == _canonical_artifacts(reference_dir)


def test_resume_rejects_changed_fingerprint_or_substrate(tmp_path: Path) -> None:
    workflow = WorkflowSpec(id="sequential", mode="sequential")
    spec = _spec(tmp_path, [workflow])
    plan = RunPlan(
        run_id="run",
        task=spec.tasks[0],
        workflow=workflow,
        model=spec.models[0],
        replicate=1,
    )
    run_dir = tmp_path / "partial"
    controller = RunController(
        spec=spec,
        plan=plan,
        run_dir=run_dir,
        runtime=_InterruptAfterRuntime(1),
        fingerprint=spec.fingerprint(),
    )
    with pytest.raises(RuntimeError, match="deliberate interruption"):
        controller.execute()

    with pytest.raises(RuntimeError, match="spec_fingerprint changed"):
        RunController(
            spec=spec,
            plan=plan,
            run_dir=run_dir,
            runtime=_RecordingRuntime(),
            fingerprint="changed",
            resume=True,
        )
    (spec.tasks[0].public_workspace / "public.txt").write_text(
        "changed evidence\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="substrate_hashes changed"):
        RunController(
            spec=spec,
            plan=plan,
            run_dir=run_dir,
            runtime=_RecordingRuntime(),
            fingerprint=spec.fingerprint(),
            resume=True,
        )


def test_resume_adopts_runtime_success_before_controller_checkpoint(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "\n".join(
            [
                "import argparse, json",
                "from pathlib import Path",
                "p=argparse.ArgumentParser()",
                "p.add_argument('--request-file', type=Path, required=True)",
                "p.add_argument('--output', type=Path, required=True)",
                "a=p.parse_args()",
                "r=json.loads(a.request_file.read_text())",
                "final={'conclusion':'done','supported_claim_indices':[]} if "
                "'final_answer MUST be non-null' in r['prompt'] else None",
                "a.output.write_text(json.dumps({'summary':'done','handoff':'next',"
                "'final_answer':final}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    workspace = _workspace(tmp_path, "public")
    task = TaskSpec(id="task", prompt="Analyze.", public_workspace=workspace)
    model = ModelSpec(
        id="cli", model_id="test", adapter="cli-json", command=[sys.executable, str(adapter)]
    )
    workflow = WorkflowSpec(id="sequential", mode="sequential")
    spec = ExperimentSpec(
        experiment_id="adopt-success",
        tasks=[task],
        models=[model],
        workflows=[workflow],
        budget=ResourceBudget(max_agent_calls=4, max_runtime_seconds_per_call=30),
    )
    plan = RunPlan(run_id="run", task=task, workflow=workflow, model=model, replicate=1)

    class CrashAfterSuccess:
        def __init__(self) -> None:
            self.inner = CliJsonRuntime(model)

        def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
            self.inner.run(request, budget)
            raise RuntimeError("process died after runtime success")

        def close(self) -> None:
            return

    run_dir = tmp_path / "run"
    first = RunController(
        spec=spec,
        plan=plan,
        run_dir=run_dir,
        runtime=CrashAfterSuccess(),
        fingerprint=spec.fingerprint(),
    )
    with pytest.raises(RuntimeError, match="process died after runtime success"):
        first.execute()
    assert (run_dir / "calls" / "call_0001" / "runtime_success.json").is_file()

    class CountingRuntime:
        def __init__(self) -> None:
            self.inner = CliJsonRuntime(model)
            self.calls = 0

        def run(self, request: AgentRequest, budget: ResourceBudget) -> AgentResponse:
            self.calls += 1
            return self.inner.run(request, budget)

        def close(self) -> None:
            return

    runtime = CountingRuntime()
    resumed = RunController(
        spec=spec,
        plan=plan,
        run_dir=run_dir,
        runtime=runtime,
        fingerprint=spec.fingerprint(),
        resume=True,
    )
    result = resumed.execute()

    assert result["agent_calls"] == 4
    assert result["call_attempts"] == 4
    assert runtime.calls == 3
    assert len(_canonical_artifacts(run_dir)) == 4


def test_seeded_schedule_is_consumed_and_validated(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    tasks = [
        TaskSpec(
            id="named",
            semantic_condition="named",
            prompt="Analyze.",
            public_workspace=workspace,
        ),
        TaskSpec(
            id="masked",
            semantic_condition="masked",
            prompt="Analyze.",
            public_workspace=workspace,
        ),
    ]
    spec = ExperimentSpec(
        experiment_id="schedule-test",
        output_root=tmp_path / "scheduled",
        tasks=tasks,
        models=[ModelSpec(id="stub", model_id="stub", adapter="stub")],
        workflows=[
            WorkflowSpec(id="persistent", mode="persistent"),
            WorkflowSpec(id="sequential", mode="sequential"),
            WorkflowSpec(id="deliberative", mode="deliberative", agents_per_stage=2),
        ],
        replicates=2,
        schedule_seed=123,
    )
    first = run_experiment(spec, dry_run=True)
    schedule_path = Path(first["schedule_path"])
    frozen = schedule_path.read_text(encoding="utf-8")
    payload = json.loads(frozen)
    assert [len(block["run_ids"]) for block in payload["blocks"]] == [6, 6]
    assert run_experiment(spec, dry_run=True)["n_runs"] == 12
    assert schedule_path.read_text(encoding="utf-8") == frozen

    payload["run_ids"][0] = "not-a-real-run"
    schedule_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="run IDs do not match"):
        run_experiment(spec, dry_run=True)
