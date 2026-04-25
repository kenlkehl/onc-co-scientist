"""Tests for the anonymized-twin layer over a generated DatasetBundle."""

import json
import re

import pytest

from onc_co_scientist.harness.task_spec import build_task
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import aggregate_datasets, score_dataset, write_report
from onc_co_scientist.synthetic.anonymize import (
    anonymize_bundle,
    build_column_mapping,
)
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import (
    COLUMN_MAPPING_FILENAME,
    DATASET_FILENAME,
    DESCRIPTION_FILENAME,
    MANIFEST_FILENAME,
    PUBLIC_SUBDIR,
    read_description,
    read_frame,
    read_manifest,
    write_bundle_pair,
)


_FEATURE_TOKEN = re.compile(r"^feature_\d{3,}$")


def _small_buried_config(**overrides) -> GeneratorConfig:
    base = dict(
        dataset_id="anon_ds",
        patient_n=200,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
    )
    base.update(overrides)
    return GeneratorConfig(**base)


def test_build_column_mapping_excludes_outcomes_and_ids():
    columns = ["patient_id", "age_years", "egfr_mutation", "pfs_months", "objective_response"]
    mapping = build_column_mapping(
        columns,
        outcome_columns=["pfs_months", "objective_response"],
        seed=0,
    )
    # Outcomes and id columns are excluded from the rename.
    assert "patient_id" not in mapping
    assert "pfs_months" not in mapping
    assert "objective_response" not in mapping
    # Every other column is renamed.
    assert set(mapping) == {"age_years", "egfr_mutation"}
    for new in mapping.values():
        assert _FEATURE_TOKEN.match(new), new


def test_build_column_mapping_is_deterministic_for_same_seed():
    columns = [f"col_{i:03d}" for i in range(20)]
    a = build_column_mapping(columns, outcome_columns=[], seed=42)
    b = build_column_mapping(columns, outcome_columns=[], seed=42)
    assert a == b
    c = build_column_mapping(columns, outcome_columns=[], seed=43)
    assert a != c, "different seeds should produce different mappings"


def test_anonymize_bundle_renames_frame_and_manifest():
    bundle = generate_dataset(_small_buried_config())
    anon, mapping = anonymize_bundle(bundle, seed=0)

    # Frame columns equal manifest.columns (preserves ordering invariant).
    assert list(anon.frame.columns) == list(anon.manifest.columns)

    # Outcome columns and patient_id are preserved verbatim.
    assert "patient_id" in anon.frame.columns
    for outcome in bundle.manifest.outcome_columns:
        assert outcome in anon.frame.columns

    # Every non-outcome non-id column is renamed to feature_NNN.
    for col in anon.frame.columns:
        if col == "patient_id" or col in anon.manifest.outcome_columns:
            continue
        assert _FEATURE_TOKEN.match(col), col

    # Mapping covers exactly the renamed columns.
    expected_renamed = set(bundle.frame.columns) - {"patient_id"} - set(
        bundle.manifest.outcome_columns
    )
    assert set(mapping.keys()) == expected_renamed


def test_anonymize_bundle_renames_association_variables_and_predicate():
    bundle = generate_dataset(_small_buried_config())
    anon, mapping = anonymize_bundle(bundle, seed=0)

    assert len(bundle.manifest.associations) == 1
    original = bundle.manifest.associations[0]
    renamed = anon.manifest.associations[0]

    # Variables: outcomes preserved, everything else renamed via mapping.
    expected_vars = [
        v if v in bundle.manifest.outcome_columns else mapping[v]
        for v in original.variables
    ]
    assert renamed.variables == expected_vars

    # Subgroup predicate keys are renamed; values are preserved verbatim.
    assert original.subgroup is not None
    assert renamed.subgroup is not None
    expected_predicate = {
        mapping.get(k, k): v for k, v in original.subgroup.predicate.items()
    }
    assert renamed.subgroup.predicate == expected_predicate


def test_anonymize_bundle_preserves_data_values():
    bundle = generate_dataset(_small_buried_config())
    anon, mapping = anonymize_bundle(bundle, seed=0)

    # Renaming must not touch the underlying values; reverse-map and compare.
    inverse = {v: k for k, v in mapping.items()}
    for col in anon.frame.columns:
        original_name = inverse.get(col, col)
        assert anon.frame[col].tolist() == bundle.frame[original_name].tolist(), col


def _hypothesis_text_for_anon_spec(spec) -> str:
    """Build a hypothesis text that the default RegexMatcher will accept for
    an anonymized AssociationSpec. The matcher requires at least one token
    from each variable's keyword set, so we just list every renamed variable
    plus every predicate column verbatim."""
    parts = list(spec.variables)
    if spec.subgroup is not None:
        parts.extend(spec.subgroup.predicate.keys())
    return " ".join(parts)


def test_anonymize_bundle_scoring_roundtrip(tmp_path):
    """Score a perfect transcript against the anonymized manifest; the
    renamed associations must still match when the hypothesis names the
    renamed variables."""
    bundle = generate_dataset(_small_buried_config(patient_n=120))
    anon, _mapping = anonymize_bundle(bundle, seed=0)

    iterations = [{"index": 1, "proposed_hypotheses": [], "analyses": []}]
    for spec in anon.manifest.associations:
        hyp_id = f"h_{spec.id}"
        iterations[0]["proposed_hypotheses"].append(
            {
                "id": hyp_id,
                "text": _hypothesis_text_for_anon_spec(spec),
                "kind": "novel",
            }
        )
        iterations[0]["analyses"].append(
            {
                "hypothesis_ids": [hyp_id],
                "result_summary": "matched ground truth",
                "p_value": 0.001,
                "effect_estimate": float(spec.effect_size),
                "significant": True,
            }
        )

    transcript = Transcript.model_validate(
        {
            "dataset_id": anon.manifest.dataset_id,
            "model_id": "test-model",
            "harness_id": "anon-roundtrip@0.1.0",
            "max_iterations": 3,
            "iterations": iterations,
        }
    )
    score = score_dataset(anon.manifest, transcript)
    pipeline = aggregate_datasets([score])
    write_report(pipeline, tmp_path / "score")
    # The buried finding is tagged hidden_novel; perfect transcript recovers it
    # in iteration 1.
    assert score.mean_iterations_hidden_novel == 1.0


def test_write_bundle_pair_layout(tmp_path):
    bundle = generate_dataset(_small_buried_config(patient_n=80))
    named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "ds", anon_seed=0)

    # Both directories have full bundle layout.
    for root in (named_dir, anon_dir):
        assert (root / MANIFEST_FILENAME).exists(), root
        public = root / PUBLIC_SUBDIR
        assert (public / DATASET_FILENAME).exists(), root
        assert (public / DESCRIPTION_FILENAME).exists(), root

    # Anonymized side has the column mapping at the top level (alongside
    # manifest, never inside public/).
    mapping_path = anon_dir / COLUMN_MAPPING_FILENAME
    assert mapping_path.exists()
    assert not (anon_dir / PUBLIC_SUBDIR / COLUMN_MAPPING_FILENAME).exists()

    mapping = json.loads(mapping_path.read_text())
    # Mapping is a real-name -> anonymized-name dict.
    for original, renamed in mapping.items():
        assert _FEATURE_TOKEN.match(renamed), renamed
        assert original != renamed


def test_anonymized_description_uses_renamed_columns(tmp_path):
    bundle = generate_dataset(_small_buried_config(patient_n=60))
    _named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "ds", anon_seed=0)

    desc = read_description(anon_dir).lower()
    # Outcome names are still real.
    for outcome in bundle.manifest.outcome_columns:
        assert outcome.lower() in desc, outcome
    # Real feature names are NOT present anywhere in the description.
    leaks = []
    for col in bundle.manifest.covariate_columns + bundle.manifest.treatment_columns:
        if col == "patient_id":
            continue
        if col.lower() in desc:
            leaks.append(col)
    assert leaks == [], f"anonymized description leaks real names: {leaks}"


def test_anonymized_bundle_is_readable_via_io_helpers(tmp_path):
    bundle = generate_dataset(_small_buried_config(patient_n=60))
    _named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "ds", anon_seed=0)
    anon_manifest = read_manifest(anon_dir)
    anon_frame = read_frame(anon_dir)
    assert list(anon_frame.columns) == list(anon_manifest.columns)


def test_anonymize_seed_changes_assignments():
    bundle = generate_dataset(_small_buried_config())
    _, mapping_a = anonymize_bundle(bundle, seed=0)
    _, mapping_b = anonymize_bundle(bundle, seed=1)
    assert mapping_a != mapping_b
    # Same set of original columns is renamed in both cases.
    assert set(mapping_a) == set(mapping_b)
    # Same set of new names (the prefix scheme is identical).
    assert set(mapping_a.values()) == set(mapping_b.values())


def test_build_task_against_anonymized_bundle_reads_renamed_columns(tmp_path):
    bundle = generate_dataset(_small_buried_config(patient_n=60))
    _named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "ds", anon_seed=0)
    task = build_task(anon_dir, tmp_path / "task", max_iterations=2)
    # The task's dataset.parquet has the anonymized column names.
    import pandas as pd

    df = pd.read_parquet(task.dataset_path)
    for col in df.columns:
        if col == "patient_id" or col in bundle.manifest.outcome_columns:
            continue
        assert _FEATURE_TOKEN.match(col), col
