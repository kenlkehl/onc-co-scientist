"""End-to-end smoke test that runs generator + task build + scoring against a
perfect, hand-crafted transcript."""

import json

from onc_co_scientist.harness.task_spec import build_task
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import aggregate_datasets, score_dataset, write_report
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import (
    read_manifest,
    write_bundle,
    write_bundle_pair,
)


def test_generate_build_score_pipeline(tmp_path):
    # 1. Generate a dataset bundle.
    cfg = GeneratorConfig(
        dataset_id="score_ds",
        patient_n=120,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(cfg)
    ds_dir = write_bundle(bundle, tmp_path / "ds")
    manifest = read_manifest(ds_dir)

    # 2. Build a harness task bundle from it.
    task = build_task(ds_dir, tmp_path / "task", max_iterations=3)
    assert task.instructions_path.exists()

    # 3. Hand-craft a transcript that uncovers every association in iteration 1.
    per_association_ids = []
    iterations = [{"index": 1, "proposed_hypotheses": [], "analyses": []}]
    for spec in manifest.associations:
        hyp_id = f"h_{spec.id}"
        per_association_ids.append(hyp_id)
        iterations[0]["proposed_hypotheses"].append(
            {
                "id": hyp_id,
                "text": spec.natural_language_description,
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
            "dataset_id": manifest.dataset_id,
            "model_id": "test-model",
            "harness_id": "e2e-smoke@0.1.0",
            "max_iterations": 3,
            "iterations": iterations,
        }
    )

    # 4. Score and write a report.
    score = score_dataset(manifest, transcript)
    pipeline = aggregate_datasets([score])
    report_dir = write_report(pipeline, tmp_path / "score")

    # Every association uncovered at iteration 1.
    assert score.mean_iterations_concordant == 1.0
    assert score.mean_iterations_discordant == 1.0
    assert score.mean_iterations_hidden_novel == 1.0
    assert score.paradigm_adherence == 0.0

    # Report artifacts materialized correctly.
    score_json = json.loads((report_dir / "score.json").read_text())
    assert score_json["n_datasets"] == 1
    report_md = (report_dir / "score.md").read_text()
    assert report_md.startswith("# Oncology Co-Scientist Benchmark")


def test_buried_finding_pipeline_against_anonymized_bundle(tmp_path):
    """Generate a buried-only dataset, materialize both named/anonymized
    variants, build a task against the anonymized side, and score a perfect
    transcript that names the renamed variables."""
    cfg = GeneratorConfig(
        dataset_id="buried_e2e",
        patient_n=200,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
    )
    bundle = generate_dataset(cfg)

    named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "ds", anon_seed=0)
    assert (named_dir / "manifest.json").exists()
    assert (anon_dir / "manifest.json").exists()

    anon_manifest = read_manifest(anon_dir)
    task = build_task(anon_dir, tmp_path / "task", max_iterations=2)
    assert task.instructions_path.exists()

    iterations = [{"index": 1, "proposed_hypotheses": [], "analyses": []}]
    for spec in anon_manifest.associations:
        hyp_id = f"h_{spec.id}"
        # Anonymized variables -> hypothesis text that names the renamed
        # columns so the default RegexMatcher accepts the match.
        parts = list(spec.variables)
        if spec.subgroup is not None:
            parts.extend(spec.subgroup.predicate.keys())
        iterations[0]["proposed_hypotheses"].append(
            {
                "id": hyp_id,
                "text": " ".join(parts),
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
            "dataset_id": anon_manifest.dataset_id,
            "model_id": "test-model",
            "harness_id": "buried-e2e@0.1.0",
            "max_iterations": 2,
            "iterations": iterations,
        }
    )

    score = score_dataset(anon_manifest, transcript)
    pipeline = aggregate_datasets([score])
    write_report(pipeline, tmp_path / "score")
    # Buried finding tagged hidden_novel; perfect transcript recovers it
    # in iteration 1.
    assert score.mean_iterations_hidden_novel == 1.0
