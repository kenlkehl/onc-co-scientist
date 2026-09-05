#!/usr/bin/env python3
"""Prepare paired, truth-free workspaces for the structured Aim 1 pilot.

Run from the repository root. Reuses the archived pilot cohorts; it does not
regenerate or tune their planted associations. Evaluator files are separate
from every research workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from onc_co_scientist.harness.task_spec import _render_instructions
from onc_co_scientist.harness.transcript import Transcript


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(
    repo: Path,
    out: Path,
    python: Path,
    *,
    clinical_repeats: int = 20,
    depmap_repeats: int = 25,
    seed: int = 20260904,
    model: str = "gpt-5.6-luna",
    backend: str = "work",
    reasoning_effort: str | None = "medium",
    service_tier: str | None = "standard",
) -> dict:
    if (out / "plan.json").exists():
        raise ValueError(
            "An experiment already exists here; use its frozen plan or a new directory."
        )
    sources = [
        (
            "clinical",
            name,
            repo / "example_data_clinical_all_claude/ds001" / name,
            clinical_repeats,
            25,
        )
        for name in ("aml", "breast", "crc", "nsclc", "prostate")
    ] + [("depmap", "depmap", repo / "example_data_depmap_all_codex/depmap", depmap_repeats, 10)]
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "private"
    jobs = []
    sources_record = []
    schema = Transcript.model_json_schema()
    # Require structured claims in new jobs while preserving legacy Transcript reads.
    hschema = schema["$defs"]["HypothesisRecord"]
    hschema["required"] = sorted(set(hschema.get("required", [])) | {"finding"})
    hschema["properties"]["finding"] = {"$ref": "#/$defs/StructuredFinding"}
    for family, name, source, repeats, cap in sources:
        named = pd.read_parquet(source / "named/public/dataset.parquet")
        masked = pd.read_parquet(source / "anonymized/public/dataset.parquet")
        mapping = json.loads((source / "anonymized/column_mapping.json").read_text())
        inverse = {v: k for k, v in mapping.items()}
        pd.testing.assert_frame_equal(named, masked.rename(columns=inverse)[named.columns])
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(named))
        heldout = np.sort(indices[: len(named) // 5])
        discovery = np.sort(indices[len(named) // 5 :])
        ev = cache / name
        ev.mkdir(parents=True, exist_ok=True)
        named.iloc[heldout].to_parquet(ev / "evaluation.parquet", index=False)
        shutil.copyfile(source / "named/manifest.json", ev / "manifest.json")
        shutil.copyfile(source / "anonymized/column_mapping.json", ev / "column_mapping.json")
        write_json(
            ev / "split.json",
            {
                "seed": seed,
                "discovery_rows": discovery.tolist(),
                "evaluation_rows": heldout.tolist(),
                "index_basis": "original zero-based row",
            },
        )
        manifest = json.loads((ev / "manifest.json").read_text())
        kind = "crispr_depmap" if family == "depmap" else "clinical_cohort"
        for variant, frame in (("named", named), ("anonymized", masked)):
            discfile = ev / f"discovery_{variant}.parquet"
            frame.iloc[discovery].to_parquet(discfile, index=False)
            for repeat in range(1, repeats + 1):
                jobs.append(
                    {
                        "family": family,
                        "task": name,
                        "variant": variant,
                        "replicate": repeat,
                        "max_iterations": cap,
                        "dataset_id": manifest["dataset_id"],
                        "dataset_kind": kind,
                        "discovery_source": str(discfile.resolve()),
                        "description_source": str(
                            (source / variant / "public/dataset_description.md").resolve()
                        ),
                        "evaluator": str(ev.resolve()),
                    }
                )
        sources_record.append(
            {
                "family": family,
                "task": name,
                "source": str(source.relative_to(repo)),
                "source_sha256": digest(source / "named/public/dataset.parquet"),
                "manifest_sha256": digest(ev / "manifest.json"),
                "evaluation_sha256": digest(ev / "evaluation.parquet"),
                "source_n": len(named),
                "discovery_n": len(discovery),
                "evaluation_n": len(heldout),
                "paired_identity_verified": True,
            }
        )
    np.random.default_rng(seed + 1).shuffle(jobs)
    for i, job in enumerate(jobs, 1):
        job["job_id"] = f"job_{i:04}"
        ws = out / "public" / job["job_id"]
        ws.mkdir(parents=True, exist_ok=True)
        # Independent copies prevent one research agent changing another's data.
        shutil.copyfile(job.pop("discovery_source"), ws / "dataset.parquet")
        shutil.copyfile(job.pop("description_source"), ws / "dataset_description.md")
        write_json(ws / "transcript_schema.json", schema)
        metadata = {
            "dataset_id": job["dataset_id"],
            "max_iterations": job["max_iterations"],
            "model_id": model,
            "harness_id": f"{backend}-structured-v2",
            "reasoning_effort": reasoning_effort,
            "service_tier": service_tier,
            "fixed_research_budget": True,
            "require_sequential_outputs": True,
            "job_id": job["job_id"],
            "python": str(python.absolute()),
        }
        write_json(ws / "metadata.json", metadata)
        instructions = _render_instructions(
            job["dataset_id"],
            40000,
            job["dataset_kind"],
            job["max_iterations"],
            "dataset.parquet",
            "dataset_description.md",
            str(python.absolute().parent.parent),
        )
        instructions = instructions.replace(
            f"Protocol (up to {job['max_iterations']} iterations)",
            f"Protocol (exactly {job['max_iterations']} research iterations)",
        ).replace(
            "Stop when you have thoroughly probed the dataset or when you reach the iteration cap.",
            "Complete the full iteration budget, continuing to test refinements and alternatives "
            "after an initial finding. Early completion does not satisfy this experiment protocol.",
        )
        instructions += "\n## Contemporaneous submissions\n\n"
        instructions += (
            "Submit each iteration before beginning the next using the submission command below. "
            "Do not reconstruct or renumber iterations after completing the investigation. "
            "A new iteration must respond to evidence or test a new/refined hypothesis; do not "
            "pad the transcript with repeated findings. Complete the full fixed budget. "
            "Keep all executed analysis scripts and results in this workspace. Do not read "
            "any other "
            "job, prior transcript, repository source, answer key, or external source. "
            "Use the provided Python environment; pandas, scipy, statsmodels, scikit-learn and "
            "pyarrow are available. Do not create or search for other environments.\n\n"
            "Each submission is an IterationRecord JSON object with index, proposed_hypotheses, "
            "and analyses. Every hypothesis must contain a non-null structured finding as "
            "described "
            "above. Analysis code may reference a saved script you actually executed.\n\n"
            f"Submit: `{python.absolute()} -m onc_co_scientist.harness.structured_runner submit "
            "--workspace . --record iteration_record.json`\n\n"
            f"Finish: `{python.absolute()} -m onc_co_scientist.harness.structured_runner finalize "
            "--workspace .`\n\n"
            "Write analysis_summary.txt and then finalize. Only syntax/record validation feedback "
            "is available; no scientific evaluation feedback is provided during a run.\n"
        )
        instructions += (
            "\n## Fixed exploration budget\n\n"
            "Plan substantive research across the full budget. Use approximately the first 20% "
            "for broad outcome/exposure screening, the next 40% for multivariable and nested "
            "subgroup exploration, the next 25% for refining candidate rules and continuous "
            "cutoffs, and the remainder for robustness and alternative explanations. Search "
            "both the presence and absence of modifiers. Test whether adding or removing a "
            "condition changes the result; do not assume the first useful subgroup is complete. "
            "Apply this process to the supplied data without guessing a hidden answer.\n\n"
            "Each iteration must include a research_step object with action (screen, "
            "multivariable, refine, or robustness), rationale, script_path, and output_path. "
            "Save and execute a separate substantive script for that iteration and save its "
            "output BEFORE submitting. Use workspace-relative paths and reference script_path "
            "in the linked analysis code field. Preserve submitted scripts and outputs. The "
            "research_step rationales should describe their specific iteration's question and "
            "how earlier results motivated it where relevant; avoid generic repeated rationales. "
            "Each output file must be distinct and generated after the preceding iteration was "
            "submitted; precomputed output or a reused output file will be rejected. All four "
            "actions must appear before finalization; empty iterations or identical script reuse "
            "are rejected. A negative result is useful; do not manufacture significant findings. "
            "The supplied transcript_example.json uses fictional columns to illustrate the "
            "format only. Its numbers are fictional, not suggested analyses or results.\n"
        )
        write_json(
            ws / "transcript_example.json",
            {
                **{
                    k: metadata[k]
                    for k in ("dataset_id", "model_id", "harness_id", "max_iterations")
                },
                "iterations": [
                    {
                        "index": 1,
                        "research_step": {
                            "action": "screen",
                            "rationale": "Illustration of testing a candidate relationship.",
                            "script_path": "analysis_001.py",
                            "output_path": "result_001.txt",
                        },
                        "proposed_hypotheses": [
                            {
                                "id": "h1",
                                "text": "Fictional outcome is higher in the fictional subgroup.",
                                "finding": {
                                    "outcome": "outcome_example",
                                    "exposure": None,
                                    "contrast": "subgroup_difference",
                                    "direction": 1,
                                    "subgroup": [
                                        {"column": "feature_example", "operator": "eq", "value": 1}
                                    ],
                                },
                            }
                        ],
                        "analyses": [
                            {
                                "hypothesis_ids": ["h1"],
                                "code": "analysis_001.py",
                                "result_summary": "Fictional format illustration only.",
                                "p_value": 0.4,
                                "effect_estimate": 0.1,
                            }
                        ],
                    }
                ],
            },
        )
        (ws / "agent_instructions.md").write_text(instructions)
        job["workspace"] = str(ws.resolve())
        job["data_sha256"] = digest(ws / "dataset.parquet")
        job["instructions_sha256"] = digest(ws / "agent_instructions.md")
        job["public_input_sha256"] = {
            f: digest(ws / f)
            for f in (
                "metadata.json",
                "transcript_schema.json",
                "transcript_example.json",
                "dataset_description.md",
            )
        }
        job["status"] = "pending"
    protocol = {
        "schema_version": "aim1-structured-v2",
        "scorer_version": "structured-recovery-v2",
        "split_seed": seed,
        "discovery_fraction": 0.8,
        "model_id": model,
        "backend": backend,
        "reasoning_effort": reasoning_effort,
        "service_tier_requested": service_tier,
        "service_tier_evidence": (
            "Not launched; verify backend capability before dispatch, retain "
            "returned endpoint telemetry when available"
        ),
        "fixed_research_budget": {"clinical": 25, "depmap": 10},
        "submission_integrity": (
            "Distinct outputs generated after the preceding submission; 10ms filesystem "
            "timestamp tolerance; not proof of execution or an adversarial boundary"
        ),
        "budget_limitations": (
            "Matched iteration budgets and required research actions; not equal "
            "tokens, compute, or proof of scientific originality"
        ),
        "primary_definition": (
            "Complete correctly directed subgroup identity; treatment effect within"
            " subgroup or treatment interaction accepted; statistical confirmation "
            "separate"
        ),
        "secondary_confirmation": (
            "Candidate's declared contrast, finite linked discovery analysis, held-"
            "out signed evidence and online alpha spending; report interaction "
            "confirmation separately"
        ),
        "new_session_per_replicate": True,
        "context_fork": "none",
        "isolation": (
            "separate task directories and instructions; shared Work filesystem, not OS isolation"
        ),
        "primary_precision_min": 0.90,
        "primary_recall_min": 0.90,
        "strict_numeric_atol": 1e-9,
        "alpha": 0.05,
        "multiplicity": (
            "online alpha/[j(j+1)] per distinct structured submitted claim; no future look-ahead"
        ),
        "minimum_group_n": 10,
        "time_endpoint": (
            "earliest submitted complete finding; separate time to confirmation; censor at task cap"
        ),
        "threshold_status": (
            "development choice after archived-pilot inspection; frozen before these fresh runs"
        ),
        "sensitivity_precision_recall": [0.90, 0.95, 1.0],
        "novelty": "optional separate LLM scoring; not used for primary recovery",
        "technical_retry": (
            "resume same experiment context on technical interruption only; no retries "
            "selected by recovery"
        ),
        "comparability": (
            "new model, structured output, fixed exploration budget and neutral examples; not a "
            "causal comparison with legacy pilot"
        ),
    }
    plan = {"protocol": protocol, "sources": sources_record, "jobs": jobs}
    write_json(out / "plan.json", plan)
    write_json(out / "protocol.json", protocol)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--clinical-repeats", type=int, default=20)
    parser.add_argument("--depmap-repeats", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--backend", choices=["work", "endpoint"], default="work")
    parser.add_argument(
        "--reasoning-effort", default="medium", help="Use unspecified to omit for local endpoints"
    )
    parser.add_argument(
        "--service-tier",
        choices=["standard", "priority", "fast", "unspecified"],
        default="standard",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    result = prepare(
        repo,
        args.out.resolve(),
        args.python,
        clinical_repeats=args.clinical_repeats,
        depmap_repeats=args.depmap_repeats,
        model=args.model,
        backend=args.backend,
        reasoning_effort=None if args.reasoning_effort == "unspecified" else args.reasoning_effort,
        service_tier=None if args.service_tier == "unspecified" else args.service_tier,
    )
    print(json.dumps({"jobs": len(result["jobs"]), "plan": str(args.out / "plan.json")}))


if __name__ == "__main__":
    main()
