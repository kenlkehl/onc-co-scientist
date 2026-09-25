"""Prepare a reviewed, calibrated 40-run composite-outcome NSCLC campaign."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from onc_co_scientist.expected_surprising.ablation import collapse_outcomes
from onc_co_scientist.expected_surprising.evaluation import WorkflowValidationService, calibrate
from onc_co_scientist.expected_surprising.generation import (
    equations,
    version_discoveries,
    write_pair,
)
from onc_co_scientist.expected_surprising.masking import (
    MaskedValidationService,
    SemanticMask,
    mask_package,
)
from onc_co_scientist.expected_surprising.packaging import repackage, sha256
from onc_co_scientist.expected_surprising.research import Researcher, now
from onc_co_scientist.expected_surprising.review_policy import (
    REVIEW_POLICY_VERSION,
    require_current_pair,
    spec_digest,
)
from onc_co_scientist.expected_surprising.schemas import DGPAssessment, DGPReview, PairSpec
from onc_co_scientist.expected_surprising.scoring import comparison_key, workflow_discovery
from onc_co_scientist.harness.durable_io import atomic_write_json, atomic_write_text
from onc_co_scientist.harness.experiment import load_experiment_spec
from onc_co_scientist.harness.orchestrator import run_experiment
from onc_co_scientist.providers.vllm_openai import VLLMConfig, VLLMProvider

REPO = Path(__file__).resolve().parents[2]
SOURCE_PAIR = "es-v2-nsclc_depmap-42005"


def prepare(root: Path, parent: Path):
    if (root / "frozen_manifest.json").exists():
        raise FileExistsError("Campaign already frozen; use its manage.py to run or resume")
    prep = root / "preparation"
    prep.mkdir(parents=True, exist_ok=True)
    source_path = parent / "input_data/unmasked/private" / SOURCE_PAIR / "pair.json"
    source = PairSpec.model_validate_json(source_path.read_text())
    spec = collapse_outcomes(source)
    atomic_write_json(prep / "derived_unreviewed.json", spec.model_dump())
    shutil.copyfile(source_path, prep / "source_pair.json")
    calibration_path = prep / "calibration.json"
    if not calibration_path.exists():
        atomic_write_json(calibration_path, calibrate(spec, replicates=1000, reference_n=100000))
    calibration = json.loads(calibration_path.read_text())
    if calibration["pair_id"] != spec.pair_id or not calibration["calibration_passed"]:
        raise ValueError("The composite pair must pass full numerical calibration")

    reviewed_path = prep / "reviewed_pair.json"
    if reviewed_path.exists():
        reviewed = PairSpec.model_validate_json(reviewed_path.read_text())
        if spec_digest(reviewed) != spec_digest(spec):
            raise ValueError("Reviewed specification differs from the requested projection")
        spec = reviewed
    else:
        config = yaml.safe_load((parent / "persistent/unmasked/config.yaml").read_text())
        provider_config = dict(config["models"][0]["provider_config"])
        provider_config.pop("kind")
        provider = VLLMProvider(VLLMConfig(**provider_config))
        reviewer = Researcher(provider, prep / "independent_dgp_review")
        prompt = (
            "Independently review this composite dependency-outcome simulation design. Return "
            "DGPAssessment JSON. This is a user-requested controlled ablation of outcome count: "
            "six previously reviewed gene-specific signal components are added to ONE composite "
            "endpoint. The composite does not represent a single gene knockout. Expected, neutral "
            "and surprising labels explicitly inherit the SOURCE relationships; no new literature "
            "support for a shared knockout is asserted. Original evidence remains unmodified. "
            "Check preservation of all six exposure contrasts, eligibility, subgroups, effect "
            "coefficients and source directions. Check that the fixed surprising component is "
            "reversed in BOTH versions, while only the focal component changes direction between "
            "versions and retains its expected outside-subgroup effect. These reversals are "
            "intentional. The composite sums conditional SIGNAL components, with ONE residual "
            "noise draw at the original sigma 0.18; it is not the arithmetic sum of six noisy "
            "observed knockout columns. Covariates, sample size, seed and cutoff 0.15 "
            "are preserved. "
            "Its composite scale may extend beyond a single-knockout scale. Review full-DGP "
            "calibration for direction, focal effect/SNR balance, detectability and overall "
            "ordering. Expected_components_scoped concerns the source component scopes; "
            "plausible_outcome_scale concerns this declared composite. Neutral effects are "
            "nonzero by design. Reject or request actionable revisions for an actual "
            "inconsistency.\n"
            + json.dumps(
                {
                    "schema": DGPAssessment.model_json_schema(),
                    "source_components": [d.model_dump() for d in source.discoveries],
                    "derived_spec": spec.model_dump(exclude={"evidence", "dgp_review"}),
                    "equations": {v: equations(spec, v) for v in ("expected", "surprising")},
                    "calibration": calibration,
                }
            )
        )
        assessment = DGPAssessment.model_validate(reviewer.ask(prompt, max_tokens=24000))
        spec.dgp_review = DGPReview(
            assessment=assessment,
            policy_version=REVIEW_POLICY_VERSION,
            spec_sha256=spec_digest(spec),
            model=provider.model_id,
            reviewed_at=now(),
        )
        atomic_write_json(reviewed_path, spec.model_dump())
    require_current_pair(spec)
    print(json.dumps({"review": spec.dgp_review.assessment.model_dump()}), flush=True)
    spec.status = "locked"
    generated = prep / "generated"
    write_pair(spec, generated)
    private = generated / "private" / spec.pair_id
    shutil.copyfile(calibration_path, private / "calibration.json")
    shutil.copyfile(source_path, private / "source_pair.json")
    atomic_write_json(private / "derivation.json", spec.outcome_projection.model_dump())
    unmasked, masked = root / "input_data/unmasked", root / "input_data/masked"
    repackage(generated, unmasked)
    mask_package(unmasked, masked, seed=20260910)
    mp = masked / "private" / spec.pair_id
    mask = SemanticMask(json.loads((mp / "masking.json").read_text()))
    original_assignment = json.loads((source_path.parent / "assignment.json").read_text())
    assignments = {
        condition: json.loads(
            (
                root / "input_data" / condition / "private" / spec.pair_id / "assignment.json"
            ).read_text()
        )
        for condition in ("unmasked", "masked")
    }
    audit = {"rows": spec.n, "outcomes": 1, "planted_findings": 6, "versions": {}, "oracle": []}
    policy = json.loads((unmasked / "private" / spec.pair_id / "workflow.json").read_text())[
        "policy"
    ]
    for version in ("expected", "surprising"):
        frames = {
            c: pd.read_parquet(
                root
                / "input_data"
                / c
                / "public"
                / assignments[c][version]["task_id"]
                / "dataset.parquet"
            )
            for c in assignments
        }
        original = pd.read_parquet(
            parent
            / "input_data/unmasked/public"
            / original_assignment[version]["task_id"]
            / "dataset.parquet"
        )
        pd.testing.assert_frame_equal(
            original.drop(columns=[o.name for o in source.outcomes]),
            frames["unmasked"].drop(columns=[spec.outcomes[0].name]),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            mask.frame(frames["unmasked"]), frames["masked"], check_exact=True
        )
        assert "cell_line_id" not in mask.columns
        audit["versions"][version] = dict(
            covariates_equal=True,
            masked_values_equal=True,
            columns=len(frames["unmasked"].columns),
            predictors=len(frames["unmasked"].columns) - 2,
        )
        for repeat in range(1, 11):
            named = WorkflowValidationService(spec, version, f"replicate-{repeat:03d}", policy)
            opaque = MaskedValidationService(spec, version, f"replicate-{repeat:03d}", policy, mask)
            exact = []
            for service in (named, opaque):
                discoveries = version_discoveries(service.spec, version)
                claims = [d.hypothesis for d in discoveries]
                confirmation = service.confirm(claims)
                score = workflow_discovery(
                    claims, discoveries, {comparison_key(h) for h in claims}, confirmation
                )
                exact.append(score["exact"])
            assert exact[0] == exact[1] and exact[0]["D"] == 100, (version, repeat, exact)
            audit["oracle"].append(dict(version=version, repeat=repeat, exact_D=100, parity=True))
    atomic_write_json(root / "data_audit.json", audit)

    frozen = root / "source/src"
    shutil.copytree(REPO / "src", frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manager = (parent / "manage.py").read_text()
    start, end = manager.index("def prepare():"), manager.index("\ndef verify():")
    manager = (
        manager[:start]
        + 'def prepare():\n    raise RuntimeError("Already prepared")\n\n'
        + manager[end:]
    )
    manager = manager.replace(f'PAIR = "{SOURCE_PAIR}"', f'PAIR = "{spec.pair_id}"')
    manager = manager.replace(
        'WORKFLOWS = ("persistent", "sequential", "deliberative")', 'WORKFLOWS = ("persistent",)'
    )
    manager = manager.replace("PARALLEL = 10", "PARALLEL = 40")
    atomic_write_text(root / "manage.py", manager)
    inventory = []
    for condition in ("unmasked", "masked"):
        path = root / "persistent" / condition
        path.mkdir(parents=True)
        raw = yaml.safe_load((parent / "persistent" / condition / "config.yaml").read_text())
        raw.update(
            experiment_id=f"nsclc_depmap_composite_qwen38_20260923_persistent_{condition}",
            description="One composite dependency outcome with six source-derived findings.",
            output_root=str(path),
            max_parallel=40,
        )
        raw["expected_surprising"].update(
            root=str(root / "input_data" / condition), pair_ids=[spec.pair_id]
        )
        config_path = path / "config.yaml"
        atomic_write_text(config_path, yaml.safe_dump(raw, sort_keys=False))
        config = load_experiment_spec(config_path)
        dry = run_experiment(config, dry_run=True)
        plans = json.loads((path / "plan.json").read_text())
        assert dry["n_runs"] == 20
        assert collections.Counter(p["semantic_condition"] for p in plans) == {
            "expected": 10,
            "surprising": 10,
        }
        inventory.append(
            dict(
                workflow="persistent",
                masking=condition,
                runs=20,
                config=str(config_path),
                fingerprint=config.fingerprint(),
            )
        )
    atomic_write_json(root / "grid.json", inventory)
    shutil.copyfile(parent / "sampling.json", root / "sampling.json")
    protocol = f"""# NSCLC DepMap composite-outcome Qwen grid

40 persistent runs: ten repeats each of expected/masked, expected/unmasked,
surprising/masked, and surprising/unmasked. Each run has 25 iterations.
Run alongside the original 16-outcome campaign with 40 additional concurrent workers.

The dataset has 2,000 rows, 65 predictors, one identifier and ONE composite dependency
outcome. All original predictor values and six effect components are retained,
including the focal subgroup reversal and outside-subgroup effect. Intercept -0.25,
residual sigma 0.18 and absolute threshold 0.15 are retained. The composite adds the
six conditional signal components and one residual draw; it does not sum six noisy
outcome columns. Its marginal variance can differ from the individual endpoints.
Source categories and literature provenance are inherited, not reinterpreted as
evidence for a shared knockout. See private derivation.json and source_pair.json.

The independent DGP review passed. Calibration used 1,000 development repeats per
version and 100,000 reference rows: focal recovery was
{calibration["reference_recovery"]["expected"]:.1%} expected and
{calibration["reference_recovery"]["surprising"]:.1%} surprising. All calibration
criteria passed. These are numerical checks, not agent benchmark results. Oracle
confirmation recovers all six findings for every naming/version/repeat combination.

The new run uses the corrected identifier filtering, absolute threshold guidance,
and preserved assay descriptions (appraisal-3.3.0 / ledger-1.1.0 / masking v2).
The original 16-outcome run retains its frozen older protocol. Consequently the
comparison changes public context as well as outcome count and signal aggregation.

Model, sampling, xhigh reasoning, context budget, retries and failure accounting
match the original campaign. Model: Inferact/Qwen3.8-Flash-Next-NVFP4 at
http://sn4622130540:8001/v1. See sampling.json. DGP review is preparation work and
excluded from experimental call/token totals.

Run: /home/klkehl/thisenv/bin/python -u {root}/manage.py run --workflow persistent
Resume adds --resume only after verifying the prior host PID is stopped.
The manager holds execution.lock; do not launch duplicate workers. Source, packages,
plans and configurations are frozen by hash. A completed grid requires RESULTS.md
and RESULTS.json, then a report to the user and a reporting receipt.
"""
    atomic_write_text(root / "PROTOCOL.md", protocol)
    paths = [root / n for n in ("manage.py", "grid.json", "sampling.json", "data_audit.json")]
    paths += [
        p
        for directory in (root / "input_data", root / "source", prep)
        for p in directory.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    paths += [
        root / "persistent" / c / name
        for c in ("unmasked", "masked")
        for name in ("config.yaml", "plan.json", "schedule.json")
    ]
    atomic_write_json(
        root / "frozen_manifest.json",
        dict(
            prepared_at=now(),
            hashes={str(p.relative_to(root)): sha256(p) for p in sorted(paths)},
            design="1 workflow x 2 versions x 2 naming conditions x 10 repeats = 40 runs",
            iterations=25,
            max_parallel=40,
            parent_campaign=str(parent),
        ),
    )
    print(json.dumps(dict(root=str(root), runs=40, frozen=True, data_audit=audit)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.parent.resolve())
