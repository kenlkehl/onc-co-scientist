"""Archived Claude task brief with public treatment roles and the output contract."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.aim1_recovery.prepare import digest, write_json
from onc_co_scientist.harness.python_sandbox import ISOLATION_INSTRUCTIONS, ISOLATION_VERSION
from onc_co_scientist.harness.treatment_roles import render_treatment_roles

STYLE = "claude-legacy-loose-v1"
HARNESS = "codex-cli-claude-legacy-loose-v1"


def apply_loose_prompt(repo: Path, out: Path, plan: dict, python: Path) -> None:
    """Change fresh public prompts, never copy archived research into a workspace."""
    backend = plan["protocol"]["backend"]
    harness = f"{backend}-claude-legacy-loose-v1"
    if backend == "endpoint":
        python_instructions = "Use the execute_python tool for Python. "
        recording = (
            "Submit each iteration with the submit_iteration tool, passing its IterationRecord "
            "as the iteration argument. Use execute_python for analysis and file operations; "
            "each Python call starts a fresh process, so save and reload intermediate artifacts. "
            "After writing analysis_summary.txt, finish with a final message. The runner "
            "assembles transcript.json from your accepted submissions. Only record-validation "
            "feedback is available; there is no scientific scoring feedback during research.\n\n"
            + ISOLATION_INSTRUCTIONS + "\n\n"
        )
    else:
        python_instructions = f"Use `{python.absolute()}` for Python. "
        recording = (
            f"Submit: `{python.absolute()} -m onc_co_scientist.harness.structured_runner "
            "submit --workspace . --record iteration_record.json`\n\n"
            "After writing analysis_summary.txt, create transcript.json with "
            f"`{python.absolute()} -m onc_co_scientist.harness.structured_runner "
            "finalize --workspace .`. Only record-validation feedback is available; "
            "there is no scientific scoring feedback during research.\n\n"
        )
    sources = {}
    for job in plan["jobs"]:
        ws = Path(job["workspace"])
        relative = (
            "example_data_clinical_all_claude/ds001/tasks/nsclc/"
            f"{job['variant']}/agent_instructions.md"
        )
        source = repo / relative
        sources[relative] = digest(source)
        brief = source.read_text().replace("**Patients:** 50000", "**Patients:** 40000")
        brief += "\n" + render_treatment_roles(job["treatment_columns"])
        brief += (
            "\n## Runtime and structured recording\n\n"
            + python_instructions + "Read metadata.json for the exact "
            "dataset_id, model_id, harness_id, and max_iterations. pandas, scipy, "
            "statsmodels, scikit-learn and pyarrow are available.\n\n"
            "Every proposed hypothesis must also include a non-null structured finding "
            "using the supplied transcript_schema.json and fictional transcript_example.json. "
            "Use actual column names, signed direction, the tested contrast and all subgroup "
            "predicates. Link each analysis to its hypothesis IDs.\n\n"
            "Save each actual iteration as an IterationRecord JSON object with index, "
            "proposed_hypotheses and analyses. Submit it before starting the next iteration; "
            "do not reconstruct, renumber or pad records after the investigation. Keep the "
            "analysis code and results you execute in this workspace. You choose the analysis "
            "methods, sequence and stopping point within the 25-iteration cap.\n\n"
            + recording +
            "Inspect only this workspace's inputs and your own outputs. Do not inspect "
            "other jobs, prior research, repository source, answer keys or external sources, "
            "and do not delegate.\n"
        )
        (ws / "agent_instructions.md").write_text(brief)
        metadata = json.loads((ws / "metadata.json").read_text())
        metadata.update(
            harness_id=harness, prompt_style=STYLE, fixed_research_budget=False,
            require_sequential_outputs=False,
        )
        if backend == "endpoint":
            metadata.update(
                filesystem_isolation=ISOLATION_VERSION, execution_workspace="/workspace"
            )
        write_json(ws / "metadata.json", metadata)
        example = json.loads((ws / "transcript_example.json").read_text())
        example["harness_id"] = harness
        example["iterations"][0].pop("research_step")
        write_json(ws / "transcript_example.json", example)
        job.update(prompt_style=STYLE, harness_id=harness,
                   instructions_sha256=digest(ws / "agent_instructions.md"))
        job["public_input_sha256"] = {
            name: digest(ws / name) for name in job["public_input_sha256"]
        }
    protocol = plan["protocol"]
    protocol.pop("fixed_research_budget")
    protocol.update(
        prompt_style=STYLE, harness_id=harness,
        iteration_cap={"clinical": 25},
        archived_prompt_sha256=sources,
        stopping_rule="Agent-selected stopping after thorough exploration, at most 25 iterations",
        removed_workflow_requirements=[
            "Exactly 25 iterations", "20/40/25/remainder exploration phase allocation",
            "Four research action labels and completion quota",
            "Distinct script and sequential output file for every iteration",
        ],
        retained_contract=[
            "Explicit treatment roles in the visible column naming scheme",
            "Archived clinical task brief and systematic heterogeneity-search instruction",
            "Structured findings and contemporaneous immutable iteration receipts",
            "Code/results retention and final narrative",
            "DS001-only data, fixed 40000/10000 split and v2 scorer",
        ],
        submission_integrity="Iteration JSON hashes and ordered timestamped receipts; "
        "no per-script/output hashes or execution-time validation",
        budget_limitations="Same 25-iteration ceiling; early stopping allowed. "
        "Neither actual iterations nor tokens/compute are matched.",
        comparability="Workflow comparison with the prior structured CLI batch. "
        "Legacy brief plus explicit public treatment roles and current structured recording, "
        "not an exact Claude runtime replication.",
    )


def launch_prompt(job: dict) -> str:
    return (
        f"Analyze the oncology dataset in {job['workspace']}. "
        "Read agent_instructions.md and metadata.json and use the specified Python. "
        "Explore and test clinically meaningful patterns, refining ideas as evidence accumulates. "
        "Follow the task brief's stopping rule and recording instructions. "
        "Inspect only this workspace's inputs and own outputs; no other folders, repository "
        "source, prior runs, answer keys, external sources or delegation. "
        "Return job ID, iteration count and finalization status only."
    )
