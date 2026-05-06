from onc_co_scientist.harness.task_spec import build_task
from onc_co_scientist.harness.transcript import Transcript
from onc_co_scientist.scoring import StubJudge, score_buried
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.io import read_description, read_frame, write_bundle_pair


def _depmap_config(**overrides) -> GeneratorConfig:
    base = dict(
        dataset_id="crc_depmap_ds",
        cancer_type="crc_depmap",
        patient_n=2_000,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    base.update(overrides)
    return GeneratorConfig(**base)


def test_depmap_profile_generates_dependency_map_context():
    bundle = generate_dataset(_depmap_config())

    assert bundle.manifest.cancer_type == "crc_depmap"
    assert bundle.manifest.dataset_kind == "crispr_depmap"
    assert bundle.manifest.id_columns == ["cell_line_id"]
    assert "cell_line_id" in bundle.frame.columns
    assert "patient_id" not in bundle.frame.columns

    dependency_cols = [c for c in bundle.frame.columns if c.startswith("dependency_")]
    assert len(dependency_cols) >= 5
    assert set(dependency_cols) <= set(bundle.manifest.outcome_columns)
    assert set(dependency_cols).isdisjoint(set(bundle.manifest.covariate_columns))

    desc = bundle.public_description.lower()
    assert "crispr knockout dependency screen" in desc
    assert "ccle-style" in desc
    assert "cell-line records" in desc
    assert "more negative values indicate stronger dependency" in desc
    assert "commercial healthcare" not in desc
    assert "electronic health records" not in desc


def test_depmap_anonymized_description_and_task_prompt_use_cell_line_context(tmp_path):
    bundle = generate_dataset(_depmap_config(patient_n=300))
    named_dir, anon_dir = write_bundle_pair(bundle, tmp_path / "crc_depmap", anon_seed=0)

    anon_frame = read_frame(anon_dir)
    assert "cell_line_id" in anon_frame.columns
    assert "patient_id" not in anon_frame.columns

    anon_desc = read_description(anon_dir).lower()
    assert "crispr knockout dependency screen" in anon_desc
    assert "feature_001" in anon_desc
    assert "dependency_kif18a" in anon_desc
    assert "apc_mutation" not in anon_desc
    assert "commercial healthcare" not in anon_desc

    task = build_task(named_dir, tmp_path / "task", max_iterations=3)
    instructions = task.instructions_path.read_text(encoding="utf-8")
    instructions_lower = instructions.lower()
    assert "CRISPR Dependency Map Analysis" in instructions
    assert "**Cell lines:** 300" in instructions
    assert "DepMap/CCLE" in instructions
    assert "systematic dependency-heterogeneity search" in instructions
    assert "patient privacy" not in instructions_lower
    assert "commercial healthcare" not in instructions_lower
    assert "treatment-effect heterogeneity" not in instructions_lower
    example = task.example_path.read_text(encoding="utf-8")
    assert "dependency_KIF18A" in example
    assert "treatment_pembrolizumab" not in example


def test_depmap_buried_dependency_scores_with_existing_matcher():
    bundle = generate_dataset(_depmap_config(patient_n=1_000))
    transcript = Transcript(
        dataset_id=bundle.manifest.dataset_id,
        model_id="m",
        harness_id="h@depmap",
        max_iterations=3,
        iterations=[
            {
                "index": 1,
                "proposed_hypotheses": [
                    {
                        "id": "h1",
                        "text": (
                            "KIF18A dependency is more negative in colorectal "
                            "APC-mutant, SMAD4-intact cell lines with high WNT "
                            "activity."
                        ),
                    }
                ],
                "analyses": [
                    {
                        "hypothesis_ids": ["h1"],
                        "result_summary": "Subgroup mean dependency_KIF18A is lower.",
                        "p_value": 0.001,
                        "effect_estimate": -0.9,
                        "significant": True,
                    }
                ],
            }
        ],
    )
    judge = StubJudge(match_phrases={"KIF18A": frozenset({"KIF18A dependency"})})

    score = score_buried(bundle.manifest, transcript, judge, variant="named")

    assert score.uncovered is True
    assert score.earliest_iteration_uncovered == 1
    assert score.recovery_level == "exact"
