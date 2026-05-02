import numpy as np
import pytest

from onc_co_scientist.synthetic.distractors import DEFAULT_DISTRACTOR_POOL
from onc_co_scientist.synthetic.generator import GeneratorConfig, generate_dataset
from onc_co_scientist.synthetic.injector import BACKGROUND_PROGNOSTIC_VARIABLES
from onc_co_scientist.synthetic.paradigms import (
    DEFAULT_POOL,
    buried_signature_catalog,
)
from onc_co_scientist.synthetic.schemas import (
    AssociationForm,
    AssociationSpec,
    ParadigmClass,
)


def _treated_subgroup_count(bundle, spec: AssociationSpec) -> int:
    df = bundle.frame
    predicate_cols = set(spec.subgroup.predicate)
    drivers = [
        v for v in spec.variables if v != spec.outcome and v not in predicate_cols
    ]
    driver = drivers[0]
    mask = np.ones(len(df), dtype=bool)
    for col, val in spec.subgroup.predicate.items():
        col_vals = df[col].to_numpy()
        if isinstance(val, dict) and ({"min", "max"} & val.keys()):
            low = val.get("min", -np.inf)
            high = val.get("max", np.inf)
            mask &= (col_vals >= low) & (col_vals <= high)
        else:
            mask &= col_vals == val
    return int((mask & (df[driver].to_numpy() == 1)).sum())


def test_builtin_generator_produces_expected_mix():
    config = GeneratorConfig(
        dataset_id="test_ds",
        patient_n=200,
        seed=7,
        n_concordant=2,
        n_discordant=1,
        n_hidden_novel=1,
        n_buried_signatures=0,
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
        min_buried_treated_subgroup_n=0,
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
        n_buried_signatures=0,
    )
    with pytest.raises(ValueError, match="paradigm-concordant and paradigm-discordant"):
        generate_dataset(cfg)


def test_default_generator_includes_ten_extra_covariates():
    # The default config makes the task harder by appending 10 distractor
    # covariates. Every distractor column should land in the manifest's
    # covariate list.
    config = GeneratorConfig(
        dataset_id="many_cov",
        patient_n=80,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(config)
    assert config.n_extra_covariates == 10
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
        min_buried_treated_subgroup_n=0,
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
        # Disease-burden labs are part of the base frame so the
        # background-prognostic layer in injector.py can always read them.
        "albumin_g_dl",
        "ldh_u_l",
        "weight_loss_pct_6mo",
        "crp_mg_l",
        "nlr",
    }
    assert baseline <= set(bundle.manifest.covariate_columns)
    # No distractor columns leaked in.
    distractor_names = {spec.name for spec in DEFAULT_DISTRACTOR_POOL}
    assert distractor_names.isdisjoint(set(bundle.frame.columns))
    # The promoted labs must no longer live in the distractor pool — otherwise
    # we'd get a column collision at non-zero n_extra_covariates.
    promoted = {"albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo", "crp_mg_l", "nlr"}
    assert promoted.isdisjoint(distractor_names)


def test_requested_extras_exceeding_pool_size_raises():
    config = GeneratorConfig(
        dataset_id="too_many",
        patient_n=30,
        seed=0,
        n_concordant=1,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
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
        min_buried_treated_subgroup_n=0,
    )
    a = generate_dataset(GeneratorConfig(**cfg_kwargs, n_extra_covariates=0))
    b = generate_dataset(GeneratorConfig(**cfg_kwargs, n_extra_covariates=10))
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
        min_buried_treated_subgroup_n=0,
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
        "evaluat",
        "data-generating",
        "ground truth",
        "generated for",
    ):
        assert leak not in desc, f"public_description leaks: {leak!r}"


def test_background_prognostics_disjoint_from_paradigm_variables():
    # Load-bearing safety check: the background-prognostic layer must read
    # only disease-burden columns that are NEVER referenced by any paradigm
    # association. Overlap could allow an unscored "conventional" effect to
    # neutralize a discordant tag (e.g. a background "high TMB → better
    # response" would silently undo the discordant TMB×pembro injection).
    paradigm_vars: set[str] = set()
    for klass_specs in DEFAULT_POOL.values():
        for spec in klass_specs:
            paradigm_vars.update(spec.variables)
            if spec.subgroup is not None:
                paradigm_vars.update(spec.subgroup.predicate.keys())
    overlap = BACKGROUND_PROGNOSTIC_VARIABLES & paradigm_vars
    assert overlap == set(), (
        f"Background prognostic variables overlap paradigm variables: {overlap!r}. "
        "Move overlapping variables out of one or the other to keep the "
        "discordant signal uncontaminated."
    )


def test_buried_signature_catalog_predicates_are_disjoint_from_background():
    # Buried multi-feature signatures must use predicate columns that the
    # background-prognostic layer does NOT also drive — otherwise the buried
    # effect would be entangled with the unscored prognostic layer.
    for spec in buried_signature_catalog():
        assert spec.subgroup is not None, spec.id
        assert spec.form is AssociationForm.subgroup_conditional, spec.id
        assert spec.paradigm_class is ParadigmClass.hidden_novel, spec.id
        # Multi-feature requirement: at least three predicate keys.
        assert len(spec.subgroup.predicate) >= 3, spec.id
        overlap = set(spec.subgroup.predicate) & BACKGROUND_PROGNOSTIC_VARIABLES
        assert overlap == set(), (
            f"Buried-signature {spec.id!r} predicate overlaps "
            f"BACKGROUND_PROGNOSTIC_VARIABLES: {overlap!r}"
        )


def test_buried_only_config_injects_single_multifeature_finding():
    # Tiny cohort smoke test opts out of the treated-subgroup size floor.
    cfg = GeneratorConfig(
        dataset_id="buried_only",
        patient_n=400,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    assert len(bundle.manifest.associations) == 1
    spec = bundle.manifest.associations[0]
    assert spec.paradigm_class is ParadigmClass.hidden_novel
    assert spec.form is AssociationForm.subgroup_conditional
    assert spec.subgroup is not None
    assert len(spec.subgroup.predicate) >= 3
    # Predicate columns must all be present in the generated frame.
    for col in spec.subgroup.predicate:
        assert col in bundle.frame.columns


def test_buried_selection_respects_treated_subgroup_floor():
    cfg = GeneratorConfig(
        dataset_id="buried_floor",
        patient_n=50_000,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=1000,
        n_extra_covariates=0,
    )
    bundle = generate_dataset(cfg)
    spec = bundle.manifest.associations[0]
    assert _treated_subgroup_count(bundle, spec) >= 1000


def test_buried_selection_raises_when_floor_unmet():
    cfg = GeneratorConfig(
        dataset_id="buried_floor_fail",
        patient_n=400,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=1000,
        n_extra_covariates=0,
    )
    with pytest.raises(ValueError, match="min_buried_treated_subgroup_n=1000"):
        generate_dataset(cfg)


def test_buried_signature_signal_has_recoverable_effect_at_modest_n():
    # Power sanity check: with patient_n=20000 and the default buried finding,
    # comparing the targeted outcome inside vs. outside the predicate-defined
    # subgroup (treatment-active rows only) yields a difference whose sign
    # matches the signed effect size. This guards against a regression that
    # silently wires the buried spec into the manifest but not the outcome.
    cfg = GeneratorConfig(
        dataset_id="buried_power",
        patient_n=20_000,
        seed=0,
        n_concordant=0,
        n_discordant=0,
        n_hidden_novel=0,
        n_buried_signatures=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(cfg)
    spec = bundle.manifest.associations[0]
    df = bundle.frame
    # Driver = the variable in spec.variables that is neither the outcome nor
    # in the predicate (mirrors the injector's resolution logic).
    predicate_cols = set(spec.subgroup.predicate)
    drivers = [
        v for v in spec.variables if v != spec.outcome and v not in predicate_cols
    ]
    driver = drivers[0]

    # Build the subgroup mask using the same semantics as the injector.
    mask = np.ones(len(df), dtype=bool)
    for col, val in spec.subgroup.predicate.items():
        col_vals = df[col].to_numpy()
        if isinstance(val, dict) and ({"min", "max"} & val.keys()):
            low = val.get("min", -np.inf)
            high = val.get("max", np.inf)
            mask &= (col_vals >= low) & (col_vals <= high)
        else:
            mask &= col_vals == val

    treated = df[driver].to_numpy() == 1
    in_group = treated & mask
    out_of_group = treated & ~mask
    if in_group.sum() < 30 or out_of_group.sum() < 30:
        pytest.skip(
            f"Subgroup too small for power check: "
            f"in={int(in_group.sum())}, out={int(out_of_group.sum())}"
        )
    in_mean = float(df.loc[in_group, spec.outcome].mean())
    out_mean = float(df.loc[out_of_group, spec.outcome].mean())
    diff = in_mean - out_mean
    assert np.sign(diff) == np.sign(spec.effect_size), (
        f"Buried-signature outcome shift has wrong sign: in={in_mean:.3f}, "
        f"out={out_mean:.3f}, expected sign(effect_size)={np.sign(spec.effect_size)}"
    )


def test_background_prognostics_yield_realistic_r_squared():
    # The whole point of the background-prognostic layer is to keep agents
    # from instantly recognizing the cohort as synthetic. Concretely:
    # multivariable regression on classic disease-burden variables should
    # produce an R² in the realistic NSCLC literature band (~0.05 to ~0.40),
    # not the structurally near-zero value that gave the prior generator
    # away. This test doubles as a calibration check on the effect sizes
    # in injector._background_prognostic_contribution.
    config = GeneratorConfig(
        dataset_id="r2_check",
        patient_n=500,
        seed=42,
        n_concordant=2,
        n_discordant=1,
        n_hidden_novel=1,
        min_buried_treated_subgroup_n=0,
    )
    bundle = generate_dataset(config)
    df = bundle.frame
    # OLS via lstsq (avoids a statsmodels dependency in the test suite).
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["ecog_ps"].to_numpy(dtype=float),
            df["stage_iv"].to_numpy(dtype=float),
            df["has_brain_mets"].to_numpy(dtype=float),
            df["age_years"].to_numpy(dtype=float),
            df["albumin_g_dl"].to_numpy(dtype=float),
            df["weight_loss_pct_6mo"].to_numpy(dtype=float),
        ]
    )
    y = df["pfs_months"].to_numpy(dtype=float)
    coefs, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coefs
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    n, p = X.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    assert 0.05 <= adj_r2 <= 0.40, (
        f"PFM adjusted R² on disease-burden predictors = {adj_r2:.3f}, "
        "outside the realistic 0.05–0.40 band. Recalibrate the background "
        "prognostic effect sizes in injector._background_prognostic_contribution."
    )
