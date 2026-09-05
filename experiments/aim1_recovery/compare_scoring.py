"""Post hoc v1/v2 comparison of existing structured runs, without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from onc_co_scientist.harness.structured_runner import finalize_workspace
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring.structured_batch import (
    LEGACY_SCORER_VERSION,
    SCORER_VERSION,
    score_transcript,
)
from onc_co_scientist.synthetic.schemas import DatasetManifest

from .score import same_transcript_records, sha


def compare(plan_path: Path, saved_scores: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("Use a new comparison directory to preserve previous reports")
    plan = json.loads(plan_path.read_text())
    original = {s["job_id"]: s for s in json.loads(saved_scores.read_text())}
    cache, rows = {}, []
    for job in plan["jobs"]:
        ws, ev = Path(job["workspace"]), Path(job["evaluator"])
        if sha(ws / "dataset.parquet") != job["data_sha256"]:
            raise ValueError("Research data changed")
        if sha(ws / "agent_instructions.md") != job["instructions_sha256"]:
            raise ValueError("Instructions changed")
        transcript = Transcript.model_validate_json((ws / "transcript.json").read_text())
        if not same_transcript_records(transcript, finalize_workspace(ws, write_output=False)):
            raise ValueError("Saved transcript differs from original submitted records")
        if job["task"] not in cache:
            source = next(s for s in plan["sources"] if s["task"] == job["task"])
            for file, key in (
                ("manifest.json", "manifest_sha256"),
                ("evaluation.parquet", "evaluation_sha256"),
            ):
                if sha(ev / file) != source[key]:
                    raise ValueError("Evaluator inputs changed")
            cache[job["task"]] = (
                DatasetManifest.model_validate_json((ev / "manifest.json").read_text()),
                pd.read_parquet(ev / "evaluation.parquet"),
                json.loads((ev / "column_mapping.json").read_text()),
            )
        manifest, frame, mapping = cache[job["task"]]
        prior = score_transcript(
            manifest,
            transcript,
            frame,
            column_mapping=mapping,
            evidence_design="heldout_20_percent",
            scorer_version=LEGACY_SCORER_VERSION,
        )
        # Compare every score field, including all detailed claim judgments.
        if any(original[job["job_id"]][key] != value for key, value in prior.items()):
            raise ValueError(f"Frozen v1 scores no longer reproduce: {job['job_id']}")
        current = score_transcript(
            manifest,
            transcript,
            frame,
            column_mapping=mapping,
            evidence_design="heldout_20_percent",
            scorer_version=SCORER_VERSION,
        )
        row = {k: job[k] for k in ("job_id", "family", "task", "variant", "replicate")}
        row["v1_primary"] = prior["primary_recovered"]
        for key in (
            "primary_recovered",
            "strict_recovered",
            "confirmed_recovered",
            "interaction_confirmed_recovered",
            "primary_iteration",
            "confirmed_iteration",
        ):
            row[f"v2_{key}"] = current[key]
        rows.append(row)
    out.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "run_comparison.csv", index=False)
    groups = []
    for (family, variant), group in table.groupby(["family", "variant"]):
        groups.append(
            dict(
                family=family,
                variant=variant,
                n=len(group),
                v1_primary=int(group.v1_primary.sum()),
                v2_identity=int(group.v2_primary_recovered.sum()),
                v2_strict_identity=int(group.v2_strict_recovered.sum()),
                v2_confirmed=int(group.v2_confirmed_recovered.sum()),
                v2_interaction_confirmed=int(group.v2_interaction_confirmed_recovered.sum()),
            )
        )
    summary = dict(
        design="post_hoc_rescore_of_existing_20260904_runs_not_a_new_batch",
        v1_all_claim_fields_reproduced=True,
        n=len(rows),
        groups=groups,
        source_plan_sha256=sha(plan_path),
        source_scores_sha256=sha(saved_scores),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Post hoc v2 scoring of the completed September 4 runs",
        "",
        "These are the same 250 Luna medium priority runs. No new research was run. "
        "Every original v1 score field, including all claim judgments, reproduced exactly. "
        "Original runs and reports were read only. This comparison changes scoring alone; "
        "it does not implement the new fixed research budget or Standard service.",
        "",
        "| Family | Condition | N | V1 primary | V2 identity | V2 strict identity | V2 confirmed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['family']} | {g['variant']} | {g['n']} | {g['v1_primary']} | "
            f"{g['v2_identity']} | {g['v2_strict_identity']} | {g['v2_confirmed']} |"
        )
    lines += [
        "",
        "V2 accepts treatment effects within the complete subgroup and separates "
        "identity/completeness from statistical confirmation. Confirmation retains the "
        "original finite-analysis, held-out sign, sample-size, and online-alpha requirements. "
        "Changing the contrast label does not excuse a missing subgroup condition.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--saved-scores", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.plan, a.saved_scores, a.out)))
