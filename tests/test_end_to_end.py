"""End-to-end smoke test: generator → task build → novelty + buried scoring.

Uses a ``StubJudge`` so no LLM is invoked. The buried-finding match
behavior is asserted by handing the stub a phrase that's guaranteed to
appear in the buried spec's natural-language description.
"""

import json

from onc_co_scientist.harness.task_spec import build_task
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import (
    StubJudge,
    aggregate_batch,
    aggregate_replicates,
    score_buried,
    score_novelty,
    write_batch_report,
)
from onc_co_scientist.scoring.aggregate import ReplicateScore
from onc_co_scientist.synthetic.cancer_types import CancerType
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import read_manifest, write_bundle, write_bundle_pair
from onc_co_scientist.synthetic.multi import (
    generate_multi_dataset,
    write_multi_bundle_pair,
)


def _perfect_transcript_for(manifest, *, hypothesis_text: str, max_iterations: int):
    iterations = [{"index": 1, "proposed_hypotheses": [], "analyses": []}]
    for spec in manifest.associations:
        hyp_id = f"h_{spec.id}"
        iterations[0]["proposed_hypotheses"].append(
            {"id": hyp_id, "text": hypothesis_text, "kind": "novel"}
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
    return Transcript.model_validate(
        {
            "dataset_id": manifest.dataset_id,
            "model_id": "test-model",
            "harness_id": "e2e@0.1.0",
            "max_iterations": max_iterations,
            "iterations": iterations,
        }
    )


def _make_replicate(manifest, transcript, judge) -> ReplicateScore:
    novelty = score_novelty(transcript, judge)
    buried = score_buried(manifest, transcript, judge)
    return ReplicateScore(
        dataset_id=manifest.dataset_id,
        model_id=transcript.model_id,
        harness_id=transcript.harness_id,
        max_iterations=transcript.max_iterations,
        novelty=novelty,
        buried=buried,
    )


def test_generate_build_score_pipeline(tmp_path):
    cfg = GeneratorConfig(
        dataset_id="score_ds",
        patient_n=120,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
    )
    bundle = generate_dataset(cfg)
    ds_dir = write_bundle(bundle, tmp_path / "ds")
    manifest = read_manifest(ds_dir)

    task = build_task(ds_dir, tmp_path / "task", max_iterations=3)
    assert task.instructions_path.exists()

    # Use a sentinel string that we can both put into the perfect transcript
    # AND wire into the stub judge so it matches the buried spec.
    sentinel = "BURIED_SENTINEL_PHRASE"
    transcript = _perfect_transcript_for(
        manifest, hypothesis_text=f"{sentinel} explains response", max_iterations=3
    )

    # Build a stub that:
    #   - marks anything containing "BURIED_SENTINEL" as novel,
    #   - matches the same phrase against any buried spec (we key on the
    #     spec's typical "subgroup" framing — most buried specs in the
    #     catalog mention "subgroup", "exceptional", or "buried").
    judge = StubJudge(
        novel_phrases=frozenset({"BURIED_SENTINEL"}),
        match_phrases={
            # Hit the most common natural-language framings used by the
            # buried-signature catalogs (`subgroup`, `feature combination`).
            "subgroup": frozenset({sentinel}),
            "feature combination": frozenset({sentinel}),
            "buried": frozenset({sentinel}),
            "exceptional": frozenset({sentinel}),
        },
    )

    rep = _make_replicate(manifest, transcript, judge)
    bundle_score = aggregate_replicates([rep])
    batch = aggregate_batch([bundle_score])
    report_dir = write_batch_report(batch, tmp_path / "score")

    # Perfect transcript flips both metrics to their ideal values.
    assert rep.frac_novel == 1.0
    assert rep.buried_score == 1
    assert rep.uncovered is True

    payload = json.loads((report_dir / "batch_score.json").read_text())
    assert payload["n_bundles"] == 1
    assert payload["frac_novel"] == 1.0
    assert payload["buried_score"] == 1
    md = (report_dir / "batch_score.md").read_text()
    assert md.startswith("# Oncology Co-Scientist Benchmark")


def test_buried_finding_pipeline_against_named_bundle(tmp_path):
    """Generate a buried-only dataset, materialize both named/anonymized
    variants (anonymized excluded from scoring), and score a perfect
    transcript against the named manifest with the new buried scorer."""
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

    named_manifest = read_manifest(named_dir)
    task = build_task(named_dir, tmp_path / "task", max_iterations=2)
    assert task.instructions_path.exists()

    sentinel = "BURIED_SENTINEL_PHRASE"
    transcript = _perfect_transcript_for(
        named_manifest, hypothesis_text=f"{sentinel} drives response", max_iterations=2
    )
    judge = StubJudge(
        novel_phrases=frozenset({"BURIED_SENTINEL"}),
        match_phrases={
            # Every buried-signature NL contains "conjunction"; using it as
            # the stub key reliably wires this transcript's hypothesis to
            # the buried spec across all cancer types.
            "conjunction": frozenset({sentinel}),
        },
    )

    rep = _make_replicate(named_manifest, transcript, judge)
    assert rep.uncovered is True
    assert rep.buried_score == 1
    assert rep.frac_novel == 1.0


def test_multi_cancer_pipeline_score_two_profiles(tmp_path):
    """Generate two cancer-type bundles, score perfect transcripts against
    each named manifest, and verify both replicates uncover the buried finding."""
    base_config = GeneratorConfig(
        dataset_id="multi_e2e",
        patient_n=200,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        n_extra_covariates=10,
    )
    chosen = [CancerType.crc, CancerType.breast]
    bundles = generate_multi_dataset(base_config, chosen)
    written = write_multi_bundle_pair(bundles, tmp_path / "ds", anon_seed=0)

    sentinel = "BURIED_SENTINEL_PHRASE"
    judge = StubJudge(
        novel_phrases=frozenset({"BURIED_SENTINEL"}),
        match_phrases={
            # Every buried-signature NL contains "conjunction"; using it as
            # the stub key reliably wires this transcript's hypothesis to
            # the buried spec across all cancer types.
            "conjunction": frozenset({sentinel}),
        },
    )

    for ct in chosen:
        named_dir, _anon_dir = written[ct]
        manifest = read_manifest(named_dir)
        assert manifest.cancer_type == ct.value
        assert manifest.dataset_id == f"multi_e2e_{ct.value}"

        task = build_task(named_dir, tmp_path / f"task_{ct.value}", max_iterations=2)
        assert task.instructions_path.exists()

        transcript = _perfect_transcript_for(
            manifest, hypothesis_text=f"{sentinel} drives response", max_iterations=2
        )
        rep = _make_replicate(manifest, transcript, judge)
        assert rep.uncovered, f"buried not uncovered for {ct.value}"
        assert rep.buried_score == 1
