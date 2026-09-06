"""Fresh Aim 1 preparation upgrades archived dependency outcome names privately."""

import copy
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from experiments.aim1_recovery.preflight import validate_inputs
from experiments.aim1_recovery.prepare import digest, prepare
from onc_co_scientist.scoring.deterministic import score_finding
from onc_co_scientist.synthetic.anonymize import extend_outcome_mapping
from onc_co_scientist.synthetic.io import read_manifest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("already_masked", [False, True])
def test_fresh_depmap_jobs_mask_targets_without_changing_archive_or_scores(
    tmp_path, already_masked
):
    repo = REPO
    relative = "example_data_depmap_all_codex/depmap"
    if already_masked:
        repo = tmp_path / "repo"
        shutil.copytree(REPO / relative, repo / relative)
        source = repo / relative
        mapping_path = source / "anonymized/column_mapping.json"
        mapping = json.loads(mapping_path.read_text())
        mapping = extend_outcome_mapping(mapping, read_manifest(source / "named").outcome_columns,
                                         seed=123)
        mapping_path.write_text(json.dumps(mapping))
        frame_path = source / "anonymized/public/dataset.parquet"
        pd.read_parquet(frame_path).rename(columns=mapping).to_parquet(frame_path, index=False)
    source = repo / relative
    tracked_sources = [
        source / "anonymized/column_mapping.json",
        source / "anonymized/public/dataset.parquet",
        source / "anonymized/public/dataset_description.md",
    ]
    original_hashes = {path: digest(path) for path in tracked_sources}
    out = tmp_path / "experiment"
    plan = prepare(repo, out, Path(sys.executable), depmap_repeats=1, tasks=("depmap",))
    assert validate_inputs(out / "plan.json")["input_validation"] == "passed"
    jobs = {job["variant"]: job for job in plan["jobs"]}
    named_ws, masked_ws = (Path(jobs[v]["workspace"]) for v in ("named", "anonymized"))
    private = Path(jobs["anonymized"]["evaluator"])
    mapping = json.loads((private / "column_mapping.json").read_text())
    original_mapping = json.loads(tracked_sources[0].read_text())
    assert all(mapping[name] == alias for name, alias in original_mapping.items())
    manifest = read_manifest(private)
    named = pd.read_parquet(named_ws / "dataset.parquet")
    masked = pd.read_parquet(masked_ws / "dataset.parquet")
    pd.testing.assert_frame_equal(named, masked.rename(columns={v: k for k, v in mapping.items()}))
    assert all(name.startswith("outcome_") for name in jobs["anonymized"]["outcome_columns"])
    assert jobs["named"]["outcome_columns"] == manifest.outcome_columns
    assert plan["protocol"]["depmap_masking_version"] == "depmap-features-and-outcomes-v1"
    assert set(manifest.outcome_columns).isdisjoint(masked.columns)
    for path in masked_ws.iterdir():
        assert path.name not in {"manifest.json", "column_mapping.json"}
        if path.suffix in {".md", ".json"}:
            text = path.read_text()
            assert not any(outcome in text for outcome in manifest.outcome_columns)
    description = (masked_ws / "dataset_description.md").read_text()
    assert "### Dependency outcomes" in description
    assert "more negative" in description.lower()
    assert "40000" in description
    assert original_hashes == {path: digest(path) for path in tracked_sources}

    # The same complete claim must retain both recovery and numerical evidence.
    spec = manifest.associations[0]
    predicates = []
    for column, value in spec.subgroup.predicate.items():
        if isinstance(value, dict):
            for bound, cutoff in value.items():
                predicates.append({"column": column, "operator": {"min": "ge", "max": "le"}[bound],
                                   "value": cutoff})
        else:
            predicates.append({"column": column, "operator": "eq", "value": value})
    finding = {"outcome": spec.outcome, "exposure": None, "contrast": "subgroup_difference",
               "direction": spec.direction, "subgroup": predicates}
    masked_finding = copy.deepcopy(finding)
    masked_finding["outcome"] = mapping[finding["outcome"]]
    for predicate in masked_finding["subgroup"]:
        predicate["column"] = mapping[predicate["column"]]
    evaluation = pd.read_parquet(private / "evaluation.parquet")
    expected = score_finding(finding, spec, manifest, evaluation)
    observed = score_finding(masked_finding, spec, manifest, evaluation, column_mapping=mapping)
    assert expected["recovered"]
    assert observed == expected
