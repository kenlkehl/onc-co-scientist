import pytest

from onc_open_mindedness.synthetic.distractors import DEFAULT_DISTRACTOR_POOL
from onc_open_mindedness.synthetic.generator import GeneratorConfig, generate_dataset
from onc_open_mindedness.synthetic.schemas import ParadigmClass


def test_builtin_generator_produces_expected_mix():
    config = GeneratorConfig(
        dataset_id="test_ds",
        patient_n=200,
        seed=7,
        n_concordant=2,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(config)
    assert bundle.manifest.patient_n == 200
    assert len(bundle.frame) == 200
    assert set(bundle.manifest.outcome_columns) <= set(bundle.frame.columns)

    counts = {k: 0 for k in ParadigmClass}
    for spec in bundle.manifest.associations:
        counts[spec.paradigm_class] += 1
    assert counts[ParadigmClass.concordant] == 2
    assert counts[ParadigmClass.discordant] == 1
    assert counts[ParadigmClass.hidden_novel] == 1


def test_builtin_generator_is_deterministic():
    cfg = GeneratorConfig(
        dataset_id="det",
        patient_n=150,
        seed=11,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    a = generate_dataset(cfg)
    b = generate_dataset(cfg)
    # Columns present should match exactly.
    assert list(a.frame.columns) == list(b.frame.columns)
    # Outcome draws are stochastic-by-seed; the same seed must yield the same values.
    for col in a.manifest.outcome_columns:
        assert a.frame[col].tolist() == b.frame[col].tolist()


def test_generator_rejects_cross_class_variable_overlap():
    # concordant_pembrolizumab_egfr_pfs (index 0) and
    # discordant_pembrolizumab_egfr_inverted (index 1) share the same
    # (outcome, variable-set) key, so requesting them together must be
    # rejected by _assert_no_cross_class_contradictions.
    cfg = GeneratorConfig(
        dataset_id="bad_mix",
        patient_n=100,
        seed=0,
        n_concordant=1,
        n_discordant=2,  # second discordant is the inverted pembrolizumab+EGFR entry
        n_hidden_novel=0,
    )
    with pytest.raises(ValueError, match="paradigm-concordant and paradigm-discordant"):
        generate_dataset(cfg)


def test_default_generator_includes_many_extra_covariates():
    # The default config makes the task harder by appending ~100 distractor
    # covariates. Every distractor column should land in the manifest's
    # covariate list.
    config = GeneratorConfig(
        dataset_id="many_cov",
        patient_n=80,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(config)
    assert config.n_extra_covariates >= 50, "default should be substantial"
    expected_names = [spec.name for spec in DEFAULT_DISTRACTOR_POOL[: config.n_extra_covariates]]
    for name in expected_names:
        assert name in bundle.frame.columns
        assert name in bundle.manifest.covariate_columns
    # Distractors must never be classified as outcomes or treatments.
    distractor_set = set(expected_names)
    assert distractor_set.isdisjoint(set(bundle.manifest.outcome_columns))
    assert distractor_set.isdisjoint(set(bundle.manifest.treatment_columns))


def test_zero_extra_covariates_matches_original_column_set():
    # n_extra_covariates=0 preserves the pre-distractor covariate set so that
    # pinned callers / older fixtures keep the exact same column layout.
    config = GeneratorConfig(
        dataset_id="no_extra",
        patient_n=60,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        n_extra_covariates=0,
    )
    bundle = generate_dataset(config)
    baseline = {
        "age_years",
        "sex_female",
        "smoking_status",
        "ecog_ps",
        "histology",
        "stage_iv",
        "has_brain_mets",
        "egfr_mutation",
        "kras_g12c",
        "alk_fusion",
        "stk11_mutation",
        "brca2_mutation",
        "pdl1_tps",
        "tmb_high",
    }
    assert baseline <= set(bundle.manifest.covariate_columns)
    # No distractor columns leaked in.
    distractor_names = {spec.name for spec in DEFAULT_DISTRACTOR_POOL}
    assert distractor_names.isdisjoint(set(bundle.frame.columns))


def test_requested_extras_exceeding_pool_size_raises():
    config = GeneratorConfig(
        dataset_id="too_many",
        patient_n=30,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        n_extra_covariates=len(DEFAULT_DISTRACTOR_POOL) + 1,
    )
    with pytest.raises(ValueError, match="exceeds DEFAULT_DISTRACTOR_POOL"):
        generate_dataset(config)


def test_extra_covariates_are_independent_of_seed_for_base_frame():
    # With and without distractors, the paradigm-column values and outcome
    # values should match — distractors use a separate RNG stream.
    cfg_kwargs = dict(
        dataset_id="iso",
        patient_n=120,
        seed=5,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    a = generate_dataset(GeneratorConfig(**cfg_kwargs, n_extra_covariates=0))
    b = generate_dataset(GeneratorConfig(**cfg_kwargs, n_extra_covariates=50))
    shared = set(a.frame.columns) & set(b.frame.columns)
    # The paradigm-used and outcome columns must survive the transformation
    # unchanged (same RNG stream on both).
    for col in shared:
        if col == "patient_id":
            continue
        assert a.frame[col].tolist() == b.frame[col].tolist(), col


def test_public_description_excludes_ground_truth():
    config = GeneratorConfig(
        dataset_id="ds_x",
        patient_n=50,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
    )
    bundle = generate_dataset(config)
    desc = bundle.public_description.lower()
    # The public description should never reveal paradigm-class terminology.
    assert "concordant" not in desc
    assert "discordant" not in desc
    assert "hidden_novel" not in desc
    # Nor should it leak the benchmark's evaluation intent, nor betray that the
    # data are synthetic — the cohort should read as a realistic clinical dataset.
    for leak in (
        "open-mindedness",
        "deliberately inverted",
        "counter-intuitive",
        "paradigm",
        "willingness",
        "synthetic",
        "simulated",
        "benchmark",
        "data-generating",
        "ground truth",
        "generated for",
    ):
        assert leak not in desc, f"public_description leaks: {leak!r}"
