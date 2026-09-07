"""Command-line entry points for the versioned Aim 1 workflow."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from ..providers.registry import get_provider
from ..synthetic.cancer_types import all_cancer_types
from .design import select_design
from .evaluation import calibrate
from .generation import write_pair
from .research import Researcher
from .schemas import PairSpec, ReviewedCandidate, StageRecord

app = typer.Typer(help="Generate and evaluate paired expected/surprising discovery tasks.")


@app.command()
def recheck(
    config: Path, specs: Path, out: Path = Path("data/expected_surprising_v2"), profile: str = "all"
):
    """Re-review historical inventories under current policy; retain rejection reasons."""
    from .review_policy import REVIEW_POLICY_VERSION, reject_nested_comparison

    settings = yaml.safe_load(config.read_text())
    provider = get_provider(settings["provider"])
    critic = get_provider(settings.get("realism_provider", settings["provider"]))
    for path in sorted(specs.glob("*.json")):
        spec = PairSpec.model_validate_json(path.read_text())
        if profile != "all" and spec.profile != profile:
            continue
        audit = out / "research" / spec.profile
        researcher = Researcher(provider, audit, realism_provider=critic)
        researcher.profile = spec.profile
        accepted, results = [], []
        for old in spec.evidence:
            try:
                item = researcher.review(old.candidate)
                reject_nested_comparison(item.candidate, accepted)
                accepted.append(item)
                results.append({"id": old.candidate.hypothesis.id, "accepted": True})
            except ValueError as exc:
                failure = {
                    "candidate": old.candidate.model_dump(),
                    "reason": str(exc),
                    "phase": "release_recheck",
                }
                researcher.log("rejected", failure)
                results.append(
                    {"id": old.candidate.hypothesis.id, "accepted": False, "reason": str(exc)}
                )
            (audit / f"{spec.profile}_reviewed.json").write_text(
                json.dumps([r.model_dump() for r in accepted], indent=2)
            )
            (audit / "recheck.json").write_text(
                json.dumps(
                    {
                        "source_pair_id": spec.pair_id,
                        "policy_version": REVIEW_POLICY_VERSION,
                        "results": results,
                    },
                    indent=2,
                )
            )
        typer.echo(f"{spec.profile}: retained {len(accepted)}/{len(spec.evidence)} candidates")


@app.command()
def research(config: Path, out: Path = Path("data/expected_surprising_v2"), profile: str = "all"):
    """Propose, search, review, and replace candidate discoveries with an LLM."""
    settings = yaml.safe_load(config.read_text())
    provider = get_provider(settings["provider"])
    profiles = [str(c) for c in all_cancer_types()] if profile == "all" else [profile]
    for cancer in profiles:
        critic = get_provider(settings.get("realism_provider", settings["provider"]))
        reviewed = Researcher(
            provider, out / "research" / cancer, realism_provider=critic
        ).generate(
            cancer,
            max_rounds=settings.get("max_candidate_rounds", 8),
            expected_count=settings.get("expected_candidates", {}).get(cancer, 4),
        )
        typer.echo(f"{cancer}: {len(reviewed)} reviewed candidates")


@app.command()
def generate(config: Path, out: Path = Path("data/expected_surprising_v2"), profile: str = "all"):
    """Compile reviewed candidates into new six-discovery paired datasets."""
    settings = yaml.safe_load(config.read_text())
    profiles = [str(c) for c in all_cancer_types()] if profile == "all" else [profile]
    all_profiles = [str(c) for c in all_cancer_types()]
    for cancer in profiles:
        reviewed_path = out / "research" / cancer / f"{cancer}_reviewed.json"
        reviewed = [
            ReviewedCandidate.model_validate(r) for r in json.loads(reviewed_path.read_text())
        ]
        index = all_profiles.index(cancer)
        spec, design_report = select_design(
            cancer,
            reviewed,
            seed=settings.get("seed", 42000) + index,
            n=settings["depmap_n" if cancer.endswith("depmap") else "clinical_n"],
            focal_mode="subgroup" if index % 2 else "overall",
            expected_count=settings.get("expected_candidates", {}).get(cancer, 4),
        )
        provider = get_provider(settings["provider"])
        critic = get_provider(settings.get("realism_provider", settings["provider"]))
        spec = Researcher(provider, out / "research" / cancer, realism_provider=critic).review_dgp(
            spec
        )
        mapping = write_pair(spec, out)
        (out / "private" / spec.pair_id / "design_selection.json").write_text(
            json.dumps(design_report, indent=2)
        )
        (out / "public" / mapping["expected"]["task_id"] / "stage_schema.json").write_text(
            json.dumps(StageRecord.model_json_schema(), indent=2)
        )
        (out / "public" / mapping["surprising"]["task_id"] / "stage_schema.json").write_text(
            json.dumps(StageRecord.model_json_schema(), indent=2)
        )
        typer.echo(f"{cancer}: {spec.pair_id} ({spec.focal_mode}; development)")


@app.command("calibrate")
def calibrate_command(pair: Path, replicates: int = 1000, reference_n: int = 100000):
    """Audit all embedded contrasts and reference recovery before formal evaluation."""
    spec = PairSpec.model_validate_json(pair.read_text())
    report = calibrate(spec, replicates=replicates, reference_n=reference_n)
    path = pair.parent / "calibration.json"
    path.write_text(json.dumps(report, indent=2))
    typer.echo(f"{path}: passed={report['calibration_passed']}")


@app.command("run")
def run_command(
    pair: Path,
    public_root: Path,
    config: Path,
    out: Path,
    run_id: str,
    version: str = "expected",
    iterations: int | None = None,
):
    """Run the reference stage harness with deterministic analysis and scoring."""
    from .rollout import run

    spec = PairSpec.model_validate_json(pair.read_text())
    assignment = json.loads((pair.parent / "assignment.json").read_text())
    settings = yaml.safe_load(config.read_text())
    provider = get_provider(settings["provider"])
    report = run(
        spec,
        version,
        public_root / assignment[version]["task_id"],
        out,
        provider,
        run_id=run_id,
        iterations=iterations,
        max_tokens_per_call=settings.get("max_tokens_per_call", 125000),
        max_retries_per_stage=settings.get("max_retries_per_stage", 2),
    )
    typer.echo(
        json.dumps(
            {
                "primary_recovery": report["confirmation"]["primary_recovery"],
                "protocol_errors": len(report["protocol_errors"]),
            }
        )
    )


@app.command("summarize")
def summarize_command(reports_root: Path, out: Path, bootstrap_replicates: int = 2000):
    """Estimate the paired recovery difference and a dataset-cluster bootstrap interval."""
    from .summary import paired_summary

    reports = [json.loads(p.read_text()) for p in reports_root.glob("*/report.json")]
    summary = paired_summary(reports, bootstrap_replicates=bootstrap_replicates)
    out.write_text(json.dumps(summary, indent=2))
    typer.echo(
        f"{summary['base_dataset_n']} base datasets; difference={summary['paired_difference']}"
    )


@app.command("materialize")
def materialize_command(specs: Path, out: Path, historical_replay: bool = False):
    """Reproduce paired data from frozen reviewed specifications without an LLM call."""
    for path in sorted(specs.glob("*.json")):
        spec = PairSpec.model_validate_json(path.read_text())
        mapping = write_pair(spec, out, historical_replay=historical_replay)
        for item in mapping.values():
            (out / "public" / item["task_id"] / "stage_schema.json").write_text(
                json.dumps(StageRecord.model_json_schema(), indent=2)
            )
        typer.echo(spec.pair_id)


if __name__ == "__main__":
    app()
