"""Export the new workflow without generating or rewriting numerical datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .prompting import schemas
from .schemas import FULL_RUN_ITERATIONS, PairSpec, ValidationPolicy, WorkflowVersions

PUBLIC_INSTRUCTIONS = """Investigate diverse, scientifically meaningful comparisons and use previous
results to allocate further investigation. Broad exploration and focused follow-up are both useful
throughout the run. Assess discoveries in this supplied dataset and use independent validation when
useful. Biological interpretation and generalizability may remain uncertain.

Register comparisons, anticipated directions, and initial assessments before requesting analyses.
The controller assigns claim references. For a changed claim, propose a new comparison and link
refinements to the earlier parent claim.
Clinical log_pfs_months is natural log progression-free survival in months, fully observed without
censoring. Dependency scores are continuous; more negative values mean stronger dependency.
Research signatures are constructed standardized assays in arbitrary units. Research markers D, E,
and F are constructed binary assays with no assigned gene, pathway, or clinical role.
A mean_difference is exposed minus comparator within eligibility and subgroup. An interaction
subtracts that comparison in the subgroup complement in the same eligible population.
Use your scientific judgment to assess the size, uncertainty, and importance of each effect
and decide which conclusions the evidence supports. Explain your reasoning in the narrative.
Returned estimates and intervals are ALREADY SIGNED to the indicated hypothesis direction. Do not
multiply them by direction again. For example, raw -0.4 with direction -1 is reported as +0.4.
Discovery analyses use 95% Welch intervals and need at least 20 observations in every
comparison cell.

Each iteration has explore, analyze, appraise, and synthesize stages. Every stage may register
hypotheses for later analysis. Analyze selects up to 12 previously registered hypotheses.
Appraise assesses every new discovery result, records active/deferred/closed investigation, and may
request validation after considering those results. At most one new voluntary validation request
is allowed per iteration and ten per run. Cached requests reuse the same evidence. The evaluator
may also deliver scheduled validation during the run. Independent intervals account for the maximum
number of validation comparisons in the run. Results arrive before synthesis. Assess newly released
validation immediately in synthesis and explicitly reassess the original claim when its response
deadline appears in the context, even if you also investigate a refinement. The controller maintains
the accepted claim set from your assessments; you do not need to repeat that list. Explanations are
retained for inspection. Acceptance does not require prior independent validation.
"""


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repackage(source: Path, out: Path) -> dict:
    """Preflight all hashes, then copy private provenance and public numerical bytes."""
    source, out = source.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    plans = []
    for pair in sorted((source / "private").glob("*/pair.json")):
        spec = PairSpec.model_validate_json(pair.read_text())
        if spec.generation_version != 2:
            raise ValueError("Workflow requires frozen numerical DGP v2")
        assignment = json.loads((pair.parent / "assignment.json").read_text())
        for version in ("expected", "surprising"):
            item = assignment[version]
            public = source / "public" / item["task_id"]
            if sha256(public / "dataset.parquet") != item["sha256"]:
                raise ValueError(f"Source dataset hash mismatch: {public}")
            ValidationPolicy.default(spec.profile, FULL_RUN_ITERATIONS)
        plans.append((pair, spec, assignment))
    if not plans:
        raise ValueError("No frozen pairs found")
    versions = WorkflowVersions().model_dump()
    manifest = {"versions": versions, "source_root": str(source), "tasks": []}
    out.mkdir(parents=True)
    for pair, spec, assignment in plans:
        private = out / "private" / spec.pair_id
        shutil.copytree(pair.parent, private)
        for version, item in assignment.items():
            original = source / "public" / item["task_id"]
            task_id = hashlib.sha256(
                f"{versions['workflow']}:{item['task_id']}".encode()
            ).hexdigest()[:16]
            public = out / "public" / task_id
            public.mkdir(parents=True)
            for name in ("dataset.parquet", "data_dictionary.json"):
                shutil.copyfile(original / name, public / name)
            task = json.loads((original / "task.json").read_text())
            # Effect-size cutoffs belong only to the evaluator, never to the agent's task.
            task["outcomes"] = [
                {key: outcome[key] for key in ("name", "units")} for outcome in task["outcomes"]
            ]
            # Repackaging upgrades historical cell-line tasks as well as fresh generation.
            task["iterations"] = FULL_RUN_ITERATIONS
            policy = ValidationPolicy.default(spec.profile, task["iterations"])
            # Actual release schedule and selection state remain private to the evaluator.
            task.update(task_id=task_id, versions=versions, validation_limit=10)
            (public / "task.json").write_text(json.dumps(task, indent=2))
            (public / "instructions.md").write_text(PUBLIC_INSTRUCTIONS)
            (public / "stage_schema.json").write_text(json.dumps(schemas(), indent=2))
            digest = sha256(public / "dataset.parquet")
            assert digest == item["sha256"]
            manifest["tasks"].append(
                {
                    "pair_id": spec.pair_id,
                    "version": version,
                    "task_id": task_id,
                    "source_task_id": item["task_id"],
                    "sha256": digest,
                    "source_sha256": item["sha256"],
                    "spec_sha256": sha256(pair),
                    "policy": policy.model_dump(),
                    "package_sha256": {p.name: sha256(p) for p in sorted(public.iterdir())},
                }
            )
            item.update(task_id=task_id, source_task_id=original.name, source_sha256=digest)
        (private / "assignment.json").write_text(json.dumps(assignment, indent=2))
        (private / "workflow.json").write_text(
            json.dumps(
                {
                    "versions": versions,
                    "policy": ValidationPolicy.default(
                        spec.profile, FULL_RUN_ITERATIONS
                    ).model_dump(),
                },
                indent=2,
            )
        )
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
