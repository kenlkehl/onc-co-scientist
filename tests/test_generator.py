import pytest

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
    # concordant_io_egfr and discordant_io_egfr_inverted share
    # (outcome=progression_free_months, vars={treatment_io, egfr_mutation, progression_free_months})
    # so requesting them together must be rejected.
    cfg = GeneratorConfig(
        dataset_id="bad_mix",
        patient_n=100,
        seed=0,
        n_concordant=1,
        n_discordant=2,  # second discordant is the inverted IO+EGFR
        n_hidden_novel=0,
    )
    with pytest.raises(ValueError, match="paradigm-concordant and paradigm-discordant"):
        generate_dataset(cfg)


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
