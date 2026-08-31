"""Task bundles, transcript schemas, and controlled co-scientist orchestration.

This package builds the instructions bundle that an external agentic harness
(Claude Code, Codex, a custom loop, etc.) consumes, and defines the canonical
JSON transcript format that the harness must emit for scoring.  The experiment
modules add manifest-driven persistent, sequential, deliberative, and federated
workflows without coupling the benchmark to one model provider.
"""

from .experiment import (
    ExperimentSpec,
    IterationPolicy,
    ModelSpec,
    ResourceBudget,
    StageSpec,
    TaskSpec,
    WorkflowSpec,
    load_experiment_spec,
    required_agent_calls,
)
from .orchestrator import build_run_plans, run_experiment
from .runtime import (
    AgentArtifact,
    AgentRequest,
    AgentResponse,
    AgentUsage,
    ScientificClaim,
    SubgroupPredicate,
)
from .task_spec import TaskBundle, build_task
from .transcript import (
    AnalysisRecord,
    HypothesisRecord,
    IterationRecord,
    Transcript,
)

__all__ = [
    "AgentArtifact",
    "AgentRequest",
    "AgentResponse",
    "AgentUsage",
    "AnalysisRecord",
    "ExperimentSpec",
    "HypothesisRecord",
    "IterationRecord",
    "IterationPolicy",
    "ModelSpec",
    "ResourceBudget",
    "ScientificClaim",
    "StageSpec",
    "SubgroupPredicate",
    "TaskBundle",
    "TaskSpec",
    "Transcript",
    "WorkflowSpec",
    "build_task",
    "build_run_plans",
    "load_experiment_spec",
    "run_experiment",
    "required_agent_calls",
]
