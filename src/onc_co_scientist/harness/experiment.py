"""Declarative experiment specifications for the co-scientist harness.

The schema keeps experimental design separate from any particular agent
runtime.  A single YAML file can therefore expand into a matched matrix of
tasks, workflow conditions, models, and replicates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

AdapterKind = Literal["cli-json", "pi-rpc", "stub"]
WorkflowMode = Literal["persistent", "sequential", "deliberative"]
WorkspaceStrategy = Literal["reference", "copy"]
CompletionMode = Literal["fixed"]

DEFAULT_PI_SYSTEM_PROMPT = (
    "You are a scientific analysis agent participating in a controlled experiment. "
    "Follow the user's task and experimental controls exactly. Use only explicitly "
    "permitted tools and evidence, and return the requested structured artifact."
)


class ResourceBudget(BaseModel):
    """Matched per-run limits applied to every experimental condition."""

    model_config = ConfigDict(extra="forbid")

    max_agent_calls: int = Field(default=64, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_runtime_seconds_per_call: int = Field(default=900, ge=1)


class IterationPolicy(BaseModel):
    """Ordered scientific-cycle policy for every planned run."""

    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(default=1, ge=1, le=20)
    completion_mode: CompletionMode = "fixed"


class StageSpec(BaseModel):
    """One scientific function in the co-scientist workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    instructions: str = Field(min_length=1)


def default_stages() -> list[StageSpec]:
    return [
        StageSpec(
            id="hypothesis_generation",
            role="hypothesis scientist",
            instructions=(
                "Generate specific, falsifiable hypotheses. State the expected direction, "
                "population, exposure or biomarker, outcome, and plausible alternatives."
            ),
        ),
        StageSpec(
            id="analysis",
            role="analysis scientist",
            instructions=(
                "Test the candidate hypotheses using only the permitted task data and tools. "
                "Report methods, effect estimates, uncertainty, diagnostics, and null findings."
            ),
        ),
        StageSpec(
            id="critique",
            role="critical reviewer",
            instructions=(
                "Audit the hypotheses and analyses for leakage, confounding, multiplicity, "
                "unsupported assumptions, coding errors, and plausible competing explanations."
            ),
        ),
        StageSpec(
            id="synthesis",
            role="synthesis scientist",
            instructions=(
                "Produce the final scientific report. Distinguish supported findings, negative "
                "findings, unresolved uncertainty, and the most informative next analysis."
            ),
        ),
    ]


class SafeguardSpec(BaseModel):
    """Auditable workflow safeguards rather than implicit prompt variations."""

    model_config = ConfigDict(extra="forbid")

    preliminary_commitment: bool = False
    evidence_ledger: bool = True
    minority_report: bool = False
    independent_rerun: bool = False


class ModelSpec(BaseModel):
    """One model/runtime deployment profile."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Stable profile identifier used in run IDs.")
    model_id: str = Field(min_length=1)
    provider: str | None = None
    adapter: AdapterKind = "pi-rpc"
    command: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    reasoning_effort: str | None = None
    env_passthrough: list[str] = Field(default_factory=list)
    pi_cleanroom: bool = True
    pi_tools: list[str] = Field(
        default_factory=lambda: ["read", "bash", "grep", "find", "ls"],
        description="Pi tool allowlist. An empty list passes --no-tools.",
    )
    pi_system_prompt: str = DEFAULT_PI_SYSTEM_PROMPT
    site_model_ids: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-site model IDs for federated deployment profiles.",
    )
    central_model_id: str | None = Field(
        default=None,
        description="Optional model ID for the federated central reviewer.",
    )

    @model_validator(mode="after")
    def validate_command(self) -> ModelSpec:
        if self.adapter == "cli-json" and not self.command:
            raise ValueError("cli-json model profiles require a non-empty command.")
        return self

    def for_scope(self, scope: str | None) -> ModelSpec:
        """Return a runtime profile for one site or the central reviewer."""

        model_id = self.model_id
        if scope == "central" and self.central_model_id:
            model_id = self.central_model_id
        elif scope is not None and scope in self.site_model_ids:
            model_id = self.site_model_ids[scope]
        return self.model_copy(
            update={
                "model_id": model_id,
                "site_model_ids": {},
                "central_model_id": None,
            }
        )


class WorkflowSpec(BaseModel):
    """Communication policy under study."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    mode: WorkflowMode
    agents_per_stage: int = Field(default=1, ge=1)
    deliberation_rounds: int = Field(default=1, ge=1, le=5)
    federated: bool = False
    safeguards: SafeguardSpec = Field(default_factory=SafeguardSpec)

    @model_validator(mode="after")
    def validate_shape(self) -> WorkflowSpec:
        if self.mode != "deliberative" and self.agents_per_stage != 1:
            raise ValueError("agents_per_stage may exceed 1 only for deliberative workflows.")
        if self.mode != "deliberative" and self.deliberation_rounds != 1:
            raise ValueError("deliberation_rounds applies only to deliberative workflows.")
        if self.mode == "deliberative" and self.agents_per_stage < 2:
            raise ValueError("deliberative workflows require at least two agents per stage.")
        return self


class TaskSpec(BaseModel):
    """Gold-free task material exposed to an agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    semantic_condition: str = Field(
        default="unspecified",
        min_length=1,
        description="Explicit experimental condition; never inferred from a filename.",
    )
    prompt: str = Field(min_length=1)
    treatment_columns: list[str] = Field(
        default_factory=list,
        description="All treatment variables in the public dataset's naming scheme; "
        "shown to every workflow participant. Never include a private name mapping.",
    )
    public_workspace: Path
    site_workspaces: dict[str, Path] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    private_evaluation_path: Path | None = Field(
        default=None,
        description="Harness-side evaluator material. Never included in an AgentRequest.",
    )


class ClinicalBenchmarkSource(BaseModel):
    """Import gold-free exercises from clin-genomic-analysis-benchmark."""

    model_config = ConfigDict(extra="forbid")

    questions_root: Path
    cohort_data_root: Path
    cohorts: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    categories: list[int] = Field(default_factory=list)
    limit_per_cohort: int | None = Field(default=None, ge=1)


class ExperimentSpec(BaseModel):
    """Top-level co-scientist experiment manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    experiment_id: str = Field(min_length=1)
    description: str = ""
    output_root: Path = Path("runs/co_scientist")
    workspace_strategy: WorkspaceStrategy = "reference"
    tasks: list[TaskSpec] = Field(default_factory=list)
    clinical_benchmark: ClinicalBenchmarkSource | None = None
    models: list[ModelSpec] = Field(min_length=1)
    workflows: list[WorkflowSpec] = Field(min_length=1)
    stages: list[StageSpec] = Field(default_factory=default_stages, min_length=1)
    iteration_policy: IterationPolicy = Field(default_factory=IterationPolicy)
    replicates: int = Field(default=1, ge=1)
    max_parallel: int = Field(default=1, ge=1)
    schedule_seed: int = 20260831
    private_evaluator_assets: dict[str, Path] = Field(
        default_factory=dict,
        description="Evaluator-only files. These are excluded from all public resolved specs.",
    )
    budget: ResourceBudget = Field(default_factory=ResourceBudget)

    @model_validator(mode="after")
    def validate_matrix(self) -> ExperimentSpec:
        if not self.tasks and self.clinical_benchmark is None:
            raise ValueError("Provide at least one task or a clinical_benchmark source.")
        for label, values in (
            ("task", [item.id for item in self.tasks]),
            ("model", [item.id for item in self.models]),
            ("workflow", [item.id for item in self.workflows]),
            ("stage", [item.id for item in self.stages]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} IDs are not allowed.")
        for task in self.tasks:
            for workflow in self.workflows:
                required = required_agent_calls(self, task, workflow)
                if self.budget.max_agent_calls < required:
                    raise ValueError(
                        f"max_agent_calls={self.budget.max_agent_calls} is below the "
                        f"{required} calls required by task {task.id!r}, workflow "
                        f"{workflow.id!r}, {len(self.stages)} stages, and "
                        f"{self.iteration_policy.iterations} iteration(s)."
                    )
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        # Preserve fingerprints of frozen experiments predating public treatment roles.
        for task in payload["tasks"]:
            if not task["treatment_columns"]:
                task.pop("treatment_columns")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_path(path: Path | None, base: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (base / path).resolve()


def _resolve_paths(spec: ExperimentSpec, config_dir: Path) -> ExperimentSpec:
    spec.output_root = _resolve_path(spec.output_root, config_dir) or spec.output_root
    for task in spec.tasks:
        task.public_workspace = (
            _resolve_path(task.public_workspace, config_dir) or task.public_workspace
        )
        task.private_evaluation_path = _resolve_path(task.private_evaluation_path, config_dir)
        task.site_workspaces = {
            site: _resolve_path(path, config_dir) or path
            for site, path in task.site_workspaces.items()
        }
    source = spec.clinical_benchmark
    if source is not None:
        source.questions_root = (
            _resolve_path(source.questions_root, config_dir) or source.questions_root
        )
        source.cohort_data_root = (
            _resolve_path(source.cohort_data_root, config_dir) or source.cohort_data_root
        )
    spec.private_evaluator_assets = {
        label: _resolve_path(path, config_dir) or path
        for label, path in spec.private_evaluator_assets.items()
    }
    return spec


def required_agent_calls(
    spec: ExperimentSpec,
    task: TaskSpec,
    workflow: WorkflowSpec,
) -> int:
    """Return the exact healthy-run call count implied by a matrix cell."""

    calls_per_stage = 1
    if workflow.mode == "deliberative":
        calls_per_stage = workflow.agents_per_stage * workflow.deliberation_rounds + 1
    scientific_calls = (
        spec.iteration_policy.iterations * len(spec.stages) * calls_per_stage
    )
    if workflow.federated:
        scientific_calls *= len(task.site_workspaces)
        scientific_calls += 1  # one evaluator-blind central synthesis
    if workflow.safeguards.independent_rerun:
        scientific_calls += 1
    return scientific_calls


def import_clinical_benchmark_tasks(source: ClinicalBenchmarkSource) -> list[TaskSpec]:
    """Load only the public question bank; reject accidental gold fields."""

    question_files = sorted(source.questions_root.glob("*.yaml"))
    available = {path.stem: path for path in question_files}
    chosen_cohorts = source.cohorts or sorted(available)
    unknown = sorted(set(chosen_cohorts) - set(available))
    if unknown:
        raise ValueError(
            f"Clinical benchmark cohort(s) not found under {source.questions_root}: "
            f"{', '.join(unknown)}"
        )

    wanted_ids = set(source.question_ids)
    wanted_categories = set(source.categories)
    forbidden = {
        "classification",
        "gold_answer",
        "analysis_spec",
        "disambiguation_concepts",
        "gold_supporting_evidence",
    }
    tasks: list[TaskSpec] = []
    for cohort in chosen_cohorts:
        path = available[cohort]
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
            raise ValueError(f"Malformed public question bank: {path}")
        selected = []
        for question in raw["questions"]:
            if not isinstance(question, dict):
                raise ValueError(f"Malformed question in {path}")
            leaked = forbidden.intersection(question)
            if leaked:
                raise ValueError(
                    f"Gold-bearing field(s) in public question bank {path}: "
                    f"{', '.join(sorted(leaked))}"
                )
            qid = str(question.get("id", ""))
            category = int(question.get("category", 0))
            if wanted_ids and qid not in wanted_ids:
                continue
            if wanted_categories and category not in wanted_categories:
                continue
            selected.append(question)
        if source.limit_per_cohort is not None:
            selected = selected[: source.limit_per_cohort]

        for question in selected:
            qid = str(question["id"])
            category = int(question["category"])
            text = str(question["text"]).strip()
            tasks.append(
                TaskSpec(
                    id=qid,
                    prompt=(
                        "Answer this translational clinico-genomic analysis exercise:\n\n"
                        f"{text}\n\n"
                        "First determine the estimand and analytic conventions needed. Use the "
                        "cohort files and data dictionary in the public workspace. Report "
                        "executable methods and a typed result when the question is answerable; "
                        "otherwise identify the concrete ambiguity or data limitation."
                    ),
                    public_workspace=source.cohort_data_root / cohort,
                    metadata={
                        "source": "clin-genomic-analysis-benchmark",
                        "cohort": cohort,
                        "category": category,
                        "question_bank": str(path),
                    },
                )
            )
    if wanted_ids:
        found = {task.id for task in tasks}
        missing = sorted(wanted_ids - found)
        if missing:
            raise ValueError(f"Requested clinical benchmark question(s) not found: {missing}")
    return tasks


def load_experiment_spec(path: Path | str) -> ExperimentSpec:
    """Read, resolve, and expand an experiment YAML manifest."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at {config_path}")
    spec = _resolve_paths(ExperimentSpec.model_validate(raw), config_path.parent)
    if spec.clinical_benchmark is not None:
        imported = import_clinical_benchmark_tasks(spec.clinical_benchmark)
        existing = {task.id for task in spec.tasks}
        duplicates = existing.intersection(task.id for task in imported)
        if duplicates:
            raise ValueError(f"Duplicate task IDs after benchmark import: {sorted(duplicates)}")
        spec.tasks.extend(imported)
    return spec
