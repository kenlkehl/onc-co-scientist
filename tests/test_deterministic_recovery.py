import numpy as np
import pandas as pd
import pytest

from onc_co_scientist.harness.structured import Predicate, StructuredFinding
from onc_co_scientist.scoring.deterministic import evaluate_finding, score_finding
from onc_co_scientist.synthetic.schemas import AssociationSpec, DatasetManifest


def test_predicate_rejects_nonfinite_and_contradictory_bounds():
    with pytest.raises(ValueError):
        Predicate(column="x", operator="eq", value=float("nan"))
    with pytest.raises(ValueError):
        StructuredFinding(
            outcome="y",
            exposure="t",
            contrast="treatment_effect",
            direction=1,
            subgroup=[
                Predicate(column="x", operator="gt", value=2),
                Predicate(column="x", operator="lt", value=1),
            ],
        )


def test_evaluate_treatment_effect_uses_normalized_difference():
    df = pd.DataFrame(
        {
            "y": [5] * 10 + [1] * 10 + [10] * 10 + [10] * 10,
            "t": [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10,
            "x": [1] * 20 + [0] * 20,
        }
    )
    f = StructuredFinding(
        outcome="y",
        exposure="t",
        contrast="treatment_effect",
        direction=1,
        subgroup=[Predicate(column="x", operator="eq", value=1)],
    )
    ev = evaluate_finding(f, df)
    assert ev["effect"] == pytest.approx(4.0)
    assert ev["finite"] is True


def test_evaluate_subgroup_difference():
    df = pd.DataFrame({"y": [4] * 10 + [6] * 10 + [1] * 10 + [3] * 10, "x": [1] * 20 + [0] * 20})
    f = StructuredFinding(
        outcome="y",
        exposure=None,
        contrast="subgroup_difference",
        direction=1,
        subgroup=[Predicate(column="x", operator="eq", value=1)],
    )
    assert evaluate_finding(f, df)["effect"] == pytest.approx(3.0)


@pytest.fixture
def scenario():
    rng = np.random.default_rng(813)
    n = 20000
    df = pd.DataFrame(
        {
            "x": rng.integers(0, 2, n),
            "z": rng.integers(0, 2, n),
            "w": rng.uniform(0, 2, n),
            "t": rng.integers(0, 2, n),
        }
    )
    mask = df.x.eq(1) & df.z.eq(0) & df.w.ge(1)
    df["y"] = 5 * mask * df.t + rng.normal(0, 1, n)
    df["other_y"] = rng.normal(0, 1, n)
    spec = AssociationSpec(
        id="signal",
        paradigm_class="hidden_novel",
        form="subgroup_conditional",
        variables=["t", "x", "z", "w", "y"],
        outcome="y",
        direction=1,
        effect_size=5,
        subgroup={
            "name": "s",
            "predicate": {"x": 1, "z": 0, "w": {"min": 1.0}},
            "description": "subgroup",
        },
        natural_language_description="not read by the deterministic evaluator",
    )
    manifest = DatasetManifest(
        dataset_id="fixture",
        seed=1,
        patient_n=n,
        columns=list(df),
        treatment_columns=["t"],
        outcome_columns=["y", "other_y"],
        covariate_columns=["x", "z", "w"],
        associations=[spec],
    )
    finding = {
        "outcome": "y",
        "exposure": "t",
        "contrast": "treatment_interaction",
        "direction": 1,
        "subgroup": [
            {"column": "x", "operator": "eq", "value": 1},
            {"column": "z", "operator": "eq", "value": 0},
            {"column": "w", "operator": "ge", "value": 1.0},
        ],
    }
    return df, spec, manifest, finding


@pytest.mark.parametrize("cutoff", [1.0, 0.95, 0.819])
def test_exact_and_approximate_masked_invariant(scenario, cutoff):
    import copy

    df, spec, manifest, f = scenario
    f["subgroup"][-1]["value"] = cutoff
    named = score_finding(f, spec, manifest, df)
    mapping = {
        "x": "feature_1",
        "z": "feature_2",
        "w": "feature_3",
        "t": "feature_4",
        "y": "feature_5",
    }
    masked = copy.deepcopy(f)
    for key in ["outcome", "exposure"]:
        masked[key] = mapping[masked[key]]
    for p in masked["subgroup"]:
        p["column"] = mapping[p["column"]]
    scored = score_finding(masked, spec, manifest, df, column_mapping=mapping)
    assert named == scored
    assert scored["strict_match"] is (cutoff == 1.0)
    assert scored["recovered"] is (cutoff >= 0.95)
    # A named finding stays named even when a map is supplied.
    assert score_finding(f, spec, manifest, df, column_mapping=mapping) == named


def test_approximate_boundary_and_stricter_tolerance(scenario):
    df, spec, manifest, f = scenario
    f["subgroup"][-1]["value"] = 0.95
    result = score_finding(f, spec, manifest, df)
    assert result["recovered"] and not result["strict_match"]
    assert not score_finding(f, spec, manifest, df, config={"precision_min": 0.99})["recovered"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "wrong_value",
        "wrong_outcome",
        "wrong_exposure",
        "wrong_contrast",
        "wrong_direction",
    ],
)
def test_incomplete_or_wrong_claims_never_complete(scenario, mutation):
    df, spec, manifest, f = scenario
    if mutation == "missing":
        f["subgroup"].pop(1)
    if mutation == "extra":
        f["subgroup"].append({"column": "w", "operator": "le", "value": 1.99})
    if mutation == "wrong_value":
        f["subgroup"][0]["value"] = 0
    if mutation == "wrong_outcome":
        f["outcome"] = "other_y"
    if mutation == "wrong_exposure":
        f["exposure"] = "x"
        f["subgroup"] = f["subgroup"][1:]
    if mutation == "wrong_contrast":
        f["contrast"] = "treatment_effect"
    if mutation == "wrong_direction":
        f["direction"] = -1
    result = score_finding(f, spec, manifest, df)
    assert not result["match_recovered"]
    assert not result["recovered"]


def test_omission_fails_even_when_functionally_identical(scenario):
    df, spec, manifest, f = scenario
    df["z"] = 0
    f["subgroup"].pop(1)
    result = score_finding(f, spec, manifest, df)
    assert result["functional"]["precision"] == 1
    assert result["functional"]["recall"] == 1
    assert not result["match_recovered"]


def test_order_duplicate_and_redundant_bounds(scenario):
    df, spec, manifest, f = scenario
    f["subgroup"] = f["subgroup"][::-1] + [
        dict(f["subgroup"][0]),
        {"column": "w", "operator": "ge", "value": 0.5},
    ]
    result = score_finding(f, spec, manifest, df)
    assert result["recovered"] and result["strict_match"]


def test_evidence_cannot_be_fixed_by_answer_key(scenario):
    df, spec, manifest, f = scenario
    f["outcome"] = "other_y"
    result = score_finding(f, spec, manifest, df)
    assert abs(result["functional"]["effect"]) < 0.2
    assert not result["recovered"]


def test_correct_words_with_opposite_data_do_not_recover(scenario):
    df, spec, manifest, f = scenario
    df["y"] = -df.y
    result = score_finding(f, spec, manifest, df)
    assert result["match_recovered"]
    assert not result["evidence_direction_ok"] and not result["recovered"]


def test_four_group_welch_matches_manual(scenario):
    from scipy import stats

    df, _, _, f = scenario
    mask = df.x.eq(1) & df.z.eq(0) & df.w.ge(1)
    groups = [
        df.loc[m & df.t.eq(t), "y"] for m, t in [(mask, 1), (mask, 0), (~mask, 1), (~mask, 0)]
    ]
    variances = np.array([g.var(ddof=1) / len(g) for g in groups])
    variance = variances.sum()
    degrees = variance**2 / sum(
        v * v / (len(g) - 1) for v, g in zip(variances, groups, strict=True)
    )
    evidence = evaluate_finding(f, df)
    assert evidence["df"] == pytest.approx(degrees)
    assert evidence["standard_error"] == pytest.approx(np.sqrt(variance))
    assert evidence["p_value"] == pytest.approx(
        2 * stats.t.sf(abs(evidence["effect"]) / np.sqrt(variance), degrees)
    )


@pytest.mark.parametrize(
    "predicate",
    [
        {"column": "x", "operator": "eq", "value": None},
        {"column": "x", "operator": "in", "value": 1},
        {"column": "x", "operator": "ge", "value": "high"},
        {"column": "x", "operator": "eq", "value": [1]},
        {"column": "x", "operator": "in", "value": [[1]]},
    ],
)
def test_malformed_predicates_rejected(predicate):
    with pytest.raises(ValueError):
        Predicate.model_validate(predicate)


def test_missing_and_nonbinary_exposure(scenario):
    df, _, _, f = scenario
    df.loc[:99, "t"] = np.nan
    result = evaluate_finding(f, df)
    assert sum(result["cell_n"].values()) == len(df) - 100
    df.loc[100, "t"] = 2
    with pytest.raises(ValueError, match="binary"):
        evaluate_finding(f, df)


@pytest.mark.parametrize("field", ["p_value", "effect_estimate"])
def test_nonfinite_discovery_analysis_cannot_confer_recovery(scenario, field):
    from onc_co_scientist.harness.transcript import Transcript
    from onc_co_scientist.scoring.structured_batch import score_transcript

    df, _, manifest, finding = scenario
    analysis = {
        "hypothesis_ids": ["h1"],
        "code": "# Retained executed analysis",
        "result_summary": "Claimed support",
        "p_value": 1e-10,
        "effect_estimate": 5.0,
        "significant": True,
    }
    payload = {
        "dataset_id": manifest.dataset_id,
        "model_id": "fixture",
        "harness_id": "fixture",
        "max_iterations": 1,
        "iterations": [
            {
                "index": 1,
                "proposed_hypotheses": [{"id": "h1", "text": "Claim", "finding": finding}],
                "analyses": [analysis],
            }
        ],
    }
    assert score_transcript(manifest, Transcript.model_validate(payload), df)["primary_recovered"]
    analysis[field] = float("nan")
    scored = score_transcript(manifest, Transcript.model_validate(payload), df)
    assert scored["primary_recovered"]
    assert not scored["confirmed_recovered"]
    legacy = score_transcript(
        manifest, Transcript.model_validate(payload), df, scorer_version="structured-recovery-v1"
    )
    assert not legacy["primary_recovered"]
    assert not scored["claim_scores"][0]["training_evidence_present"]


@pytest.mark.parametrize("contrast", ["treatment_effect", "treatment_interaction"])
@pytest.mark.parametrize("masked", [False, True])
def test_v2_identity_and_confirmation_are_distinct(scenario, contrast, masked):
    import copy

    from onc_co_scientist.harness.transcript import Transcript
    from onc_co_scientist.scoring.structured_batch import score_transcript

    df, _, manifest, finding = scenario
    finding["contrast"] = contrast
    mapping = {c: f"feature_{i}" for i, c in enumerate(df.columns)}
    if masked:
        finding = copy.deepcopy(finding)
        for key in ("outcome", "exposure"):
            finding[key] = mapping[finding[key]]
        for pred in finding["subgroup"]:
            pred["column"] = mapping[pred["column"]]
    payload = dict(
        dataset_id=manifest.dataset_id,
        model_id="m",
        harness_id="h",
        max_iterations=2,
        iterations=[
            dict(
                index=1,
                proposed_hypotheses=[dict(id="h1", text="Candidate", finding=finding)],
                analyses=[],
            )
        ],
    )
    first = score_transcript(
        manifest, Transcript.model_validate(payload), df, column_mapping=mapping
    )
    assert first["primary_recovered"] and first["strict_recovered"]
    assert first["primary_iteration"] == 1 and not first["confirmed_recovered"]
    payload["iterations"].append(
        dict(
            index=2,
            proposed_hypotheses=[],
            analyses=[
                dict(
                    hypothesis_ids=["h1"],
                    code="analysis.py",
                    result_summary="Tested",
                    effect_estimate=5,
                    p_value=1e-10,
                )
            ],
        )
    )
    after = score_transcript(
        manifest, Transcript.model_validate(payload), df, column_mapping=mapping
    )
    assert after["primary_iteration"] == 1 and after["confirmed_iteration"] == 2
    assert after["interaction_confirmed_recovered"] == (contrast == "treatment_interaction")
    legacy = score_transcript(
        manifest,
        Transcript.model_validate(payload),
        df,
        column_mapping=mapping,
        scorer_version="structured-recovery-v1",
    )
    assert legacy["primary_recovered"] == (contrast == "treatment_interaction")
    if contrast == "treatment_interaction":
        assert legacy["primary_iteration"] == 2
    # A missing gate still fails even when the treatment contrast is accepted.
    payload["iterations"][0]["proposed_hypotheses"][0]["finding"]["subgroup"].pop(0)
    assert not score_transcript(
        manifest, Transcript.model_validate(payload), df, column_mapping=mapping
    )["primary_recovered"]
