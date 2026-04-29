"""Iterative hypothesis-driven analysis of ds001_aml.

Produces a Python dict capturing 25 iterations of (hypotheses, analyses) which is then
written to transcript.json by a separate writer.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
df = pd.read_parquet(HERE / "dataset.parquet")

OUT = {"iterations": []}


def add_iter(idx, hypotheses, analyses):
    OUT["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def chi2_2x2(df, group_col, outcome_col="objective_response"):
    tab = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    r1 = df.loc[df[group_col] == 1, outcome_col].mean()
    r0 = df.loc[df[group_col] == 0, outcome_col].mean()
    return p, (r1 - r0), r1, r0, int((df[group_col] == 1).sum()), int((df[group_col] == 0).sum())


def rate_diff(df, mask1, mask0, outcome_col="objective_response"):
    s1 = df.loc[mask1, outcome_col]
    s0 = df.loc[mask0, outcome_col]
    if len(s1) == 0 or len(s0) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, 0
    r1, r0 = s1.mean(), s0.mean()
    n1, n0 = len(s1), len(s0)
    p1 = (s1.sum() + s0.sum()) / (n1 + n0)
    se = math.sqrt(p1 * (1 - p1) * (1 / n1 + 1 / n0))
    z = (r1 - r0) / se if se > 0 else 0
    pval = 2 * (1 - stats.norm.cdf(abs(z))) if se > 0 else 1.0
    return pval, (r1 - r0), r1, r0, n1, n0


def logit_test(df, x_cols, outcome_col="objective_response", interaction=None):
    """Logistic regression returning the coefficient and p-value of the LAST (or interaction) term."""
    import statsmodels.api as sm
    X = df[x_cols].astype(float).copy()
    if interaction is not None:
        a, b = interaction
        X[f"{a}_x_{b}"] = X[a] * X[b]
        target = f"{a}_x_{b}"
    else:
        target = x_cols[-1]
    X = sm.add_constant(X)
    y = df[outcome_col].astype(float)
    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=200)
        coef = float(model.params[target])
        pv = float(model.pvalues[target])
        return coef, pv
    except Exception as e:
        return float("nan"), float("nan")


def cont_logit(df, x_col, outcome_col="objective_response", adjust=None):
    """Logistic regression of outcome on continuous x (and optional adjustments)."""
    import statsmodels.api as sm
    cols = [x_col]
    if adjust:
        cols = adjust + [x_col]
    X = df[cols].astype(float)
    X = sm.add_constant(X)
    y = df[outcome_col].astype(float)
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=200)
        return float(m.params[x_col]), float(m.pvalues[x_col])
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Iteration 1: foundational main effects of "good-prognosis" and "bad-prognosis"
# AML markers on objective response.
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h1", "text": "Patients with tp53_mutation=1 have a lower objective_response rate than patients with tp53_mutation=0.", "kind": "novel"},
    {"id": "h2", "text": "Patients with npm1_mutation=1 have a higher objective_response rate than patients with npm1_mutation=0.", "kind": "novel"},
    {"id": "h3", "text": "Patients with complex_karyotype=1 have a lower objective_response rate than patients with complex_karyotype=0.", "kind": "novel"},
    {"id": "h4", "text": "Patients with secondary_aml=1 have a lower objective_response rate than patients with secondary_aml=0.", "kind": "novel"},
]
analyses = []
for hid, col, direction in [("h1", "tp53_mutation", -1), ("h2", "npm1_mutation", +1), ("h3", "complex_karyotype", -1), ("h4", "secondary_aml", -1)]:
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on {col} x objective_response",
        "result_summary": f"Response rate {col}=1: {r1:.4f} (n={n1}); {col}=0: {r0:.4f} (n={n0}); diff={diff:+.4f}; chi2 p={p:.4f}.",
        "p_value": p,
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(1, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 2: IDH1/IDH2 main effects and FLT3 main effects
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h5", "text": "Patients with idh1_mutation=1 have a higher objective_response rate than patients with idh1_mutation=0.", "kind": "novel"},
    {"id": "h6", "text": "Patients with idh2_mutation=1 have a higher objective_response rate than patients with idh2_mutation=0.", "kind": "novel"},
    {"id": "h7", "text": "Patients with flt3_itd=1 have a different objective_response rate than patients with flt3_itd=0.", "kind": "novel"},
    {"id": "h8", "text": "Patients with flt3_tkd=1 have a different objective_response rate than patients with flt3_tkd=0.", "kind": "novel"},
]
analyses = []
for hid, col in [("h5", "idh1_mutation"), ("h6", "idh2_mutation"), ("h7", "flt3_itd"), ("h8", "flt3_tkd")]:
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on {col} x objective_response",
        "result_summary": f"Response rate {col}=1: {r1:.4f} (n={n1}); {col}=0: {r0:.4f} (n={n0}); diff={diff:+.4f}; chi2 p={p:.4f}.",
        "p_value": p,
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(2, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 3: Treatment main effects on response
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h9", "text": "Patients receiving treatment_venetoclax_azacitidine=1 have a higher objective_response rate than patients with treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h10", "text": "Patients receiving treatment_7plus3=1 have a higher objective_response rate than patients with treatment_7plus3=0.", "kind": "novel"},
    {"id": "h11", "text": "Patients receiving treatment_midostaurin=1 have a higher objective_response rate than patients with treatment_midostaurin=0.", "kind": "novel"},
    {"id": "h12", "text": "Patients receiving treatment_gilteritinib=1 have a higher objective_response rate than patients with treatment_gilteritinib=0.", "kind": "novel"},
    {"id": "h13", "text": "Patients receiving treatment_ivosidenib=1 have a higher objective_response rate than patients with treatment_ivosidenib=0.", "kind": "novel"},
    {"id": "h14", "text": "Patients receiving treatment_enasidenib=1 have a higher objective_response rate than patients with treatment_enasidenib=0.", "kind": "novel"},
]
analyses = []
for hid, col in [("h9", "treatment_venetoclax_azacitidine"), ("h10", "treatment_7plus3"), ("h11", "treatment_midostaurin"), ("h12", "treatment_gilteritinib"), ("h13", "treatment_ivosidenib"), ("h14", "treatment_enasidenib")]:
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"chi2_contingency on {col} x objective_response",
        "result_summary": f"Response rate {col}=1: {r1:.4f} (n={n1}); {col}=0: {r0:.4f} (n={n0}); diff={diff:+.4f}; chi2 p={p:.4f}.",
        "p_value": p,
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(3, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 4: Targeted-therapy x mutation interactions for FLT3 inhibitors
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h15", "text": "Within flt3_itd=1 patients, those receiving treatment_midostaurin=1 have a higher objective_response rate than flt3_itd=1 patients with treatment_midostaurin=0 (positive subgroup interaction).", "kind": "novel"},
    {"id": "h16", "text": "Within flt3_itd=1 patients, those receiving treatment_gilteritinib=1 have a higher objective_response rate than flt3_itd=1 patients with treatment_gilteritinib=0.", "kind": "novel"},
    {"id": "h17", "text": "There is a positive interaction between flt3_itd and treatment_midostaurin on objective_response in a logistic regression (interaction term coefficient > 0).", "kind": "novel"},
    {"id": "h18", "text": "There is a positive interaction between flt3_itd and treatment_gilteritinib on objective_response in a logistic regression (interaction term coefficient > 0).", "kind": "novel"},
]
analyses = []
# subgroup analyses
sub = df[df["flt3_itd"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_midostaurin")
analyses.append({
    "hypothesis_ids": ["h15"],
    "code": "chi2_contingency on treatment_midostaurin x objective_response within flt3_itd==1",
    "result_summary": f"In flt3_itd+ (n={len(sub)}): on midostaurin {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_gilteritinib")
analyses.append({
    "hypothesis_ids": ["h16"],
    "code": "chi2_contingency on treatment_gilteritinib x objective_response within flt3_itd==1",
    "result_summary": f"In flt3_itd+ (n={len(sub)}): on gilteritinib {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
# logistic regression interactions
coef, pv = logit_test(df, ["flt3_itd", "treatment_midostaurin"], interaction=("flt3_itd", "treatment_midostaurin"))
analyses.append({
    "hypothesis_ids": ["h17"],
    "code": "Logit objective_response ~ flt3_itd + treatment_midostaurin + flt3_itd:treatment_midostaurin",
    "result_summary": f"Interaction coefficient (logit): {coef:+.4f}, p={pv:.4f}.",
    "p_value": pv,
    "effect_estimate": coef,
    "significant": bool(pv < 0.05),
})
coef, pv = logit_test(df, ["flt3_itd", "treatment_gilteritinib"], interaction=("flt3_itd", "treatment_gilteritinib"))
analyses.append({
    "hypothesis_ids": ["h18"],
    "code": "Logit objective_response ~ flt3_itd + treatment_gilteritinib + flt3_itd:treatment_gilteritinib",
    "result_summary": f"Interaction coefficient (logit): {coef:+.4f}, p={pv:.4f}.",
    "p_value": pv,
    "effect_estimate": coef,
    "significant": bool(pv < 0.05),
})
add_iter(4, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 5: IDH1/IDH2 targeted therapy interactions
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h19", "text": "Within idh1_mutation=1 patients, those receiving treatment_ivosidenib=1 have a higher objective_response rate than idh1_mutation=1 patients with treatment_ivosidenib=0.", "kind": "novel"},
    {"id": "h20", "text": "Within idh2_mutation=1 patients, those receiving treatment_enasidenib=1 have a higher objective_response rate than idh2_mutation=1 patients with treatment_enasidenib=0.", "kind": "novel"},
    {"id": "h21", "text": "There is a positive interaction between idh1_mutation and treatment_ivosidenib on objective_response in a logistic regression.", "kind": "novel"},
    {"id": "h22", "text": "There is a positive interaction between idh2_mutation and treatment_enasidenib on objective_response in a logistic regression.", "kind": "novel"},
]
analyses = []
sub = df[df["idh1_mutation"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_ivosidenib")
analyses.append({
    "hypothesis_ids": ["h19"],
    "code": "chi2_contingency on treatment_ivosidenib x objective_response within idh1_mutation==1",
    "result_summary": f"In idh1+ (n={len(sub)}): on ivosidenib {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
sub = df[df["idh2_mutation"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_enasidenib")
analyses.append({
    "hypothesis_ids": ["h20"],
    "code": "chi2_contingency on treatment_enasidenib x objective_response within idh2_mutation==1",
    "result_summary": f"In idh2+ (n={len(sub)}): on enasidenib {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
coef, pv = logit_test(df, ["idh1_mutation", "treatment_ivosidenib"], interaction=("idh1_mutation", "treatment_ivosidenib"))
analyses.append({
    "hypothesis_ids": ["h21"],
    "code": "Logit objective_response ~ idh1_mutation + treatment_ivosidenib + idh1_mutation:treatment_ivosidenib",
    "result_summary": f"Interaction coefficient (logit): {coef:+.4f}, p={pv:.4f}.",
    "p_value": pv,
    "effect_estimate": coef,
    "significant": bool(pv < 0.05),
})
coef, pv = logit_test(df, ["idh2_mutation", "treatment_enasidenib"], interaction=("idh2_mutation", "treatment_enasidenib"))
analyses.append({
    "hypothesis_ids": ["h22"],
    "code": "Logit objective_response ~ idh2_mutation + treatment_enasidenib + idh2_mutation:treatment_enasidenib",
    "result_summary": f"Interaction coefficient (logit): {coef:+.4f}, p={pv:.4f}.",
    "p_value": pv,
    "effect_estimate": coef,
    "significant": bool(pv < 0.05),
})
add_iter(5, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 6: Venetoclax+azacitidine in unfit patients & with TP53 mutation
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h23", "text": "Within unfit_for_intensive=1 patients, those receiving treatment_venetoclax_azacitidine=1 have a higher objective_response rate than unfit patients with treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h24", "text": "There is a positive interaction between unfit_for_intensive and treatment_venetoclax_azacitidine on objective_response in a logistic regression.", "kind": "novel"},
    {"id": "h25", "text": "Within tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 vs 0 yields a different objective_response rate (sign undetermined a priori).", "kind": "novel"},
    {"id": "h26", "text": "Within tp53_mutation=1 patients, treatment_7plus3=1 vs 0 yields a lower objective_response rate (intensive chemo is hypothesized to be unhelpful in tp53+).", "kind": "novel"},
]
analyses = []
sub = df[df["unfit_for_intensive"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_venetoclax_azacitidine")
analyses.append({
    "hypothesis_ids": ["h23"],
    "code": "chi2_contingency on treatment_venetoclax_azacitidine x objective_response within unfit_for_intensive==1",
    "result_summary": f"In unfit (n={len(sub)}): on VEN+AZA {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
coef, pv = logit_test(df, ["unfit_for_intensive", "treatment_venetoclax_azacitidine"], interaction=("unfit_for_intensive", "treatment_venetoclax_azacitidine"))
analyses.append({
    "hypothesis_ids": ["h24"],
    "code": "Logit objective_response ~ unfit_for_intensive + treatment_venetoclax_azacitidine + interaction",
    "result_summary": f"Interaction coefficient (logit): {coef:+.4f}, p={pv:.4f}.",
    "p_value": pv,
    "effect_estimate": coef,
    "significant": bool(pv < 0.05),
})
sub = df[df["tp53_mutation"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_venetoclax_azacitidine")
analyses.append({
    "hypothesis_ids": ["h25"],
    "code": "chi2_contingency on treatment_venetoclax_azacitidine x objective_response within tp53_mutation==1",
    "result_summary": f"In tp53+ (n={len(sub)}): on VEN+AZA {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_7plus3")
analyses.append({
    "hypothesis_ids": ["h26"],
    "code": "chi2_contingency on treatment_7plus3 x objective_response within tp53_mutation==1",
    "result_summary": f"In tp53+ (n={len(sub)}): on 7+3 {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.",
    "p_value": p,
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
add_iter(6, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 7: Demographic main effects on response — age, sex, ECOG, BMI
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h27", "text": "Older age_years is associated with a lower objective_response rate (logistic regression coefficient on age_years is negative).", "kind": "novel"},
    {"id": "h28", "text": "Female patients (sex_female=1) have a different objective_response rate than male patients.", "kind": "novel"},
    {"id": "h29", "text": "Higher ecog_ps is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h30", "text": "Higher bmi is associated with a different objective_response rate (sign undetermined).", "kind": "novel"},
]
analyses = []
coef, pv = cont_logit(df, "age_years")
analyses.append({"hypothesis_ids": ["h27"], "code": "Logit objective_response ~ age_years", "result_summary": f"Logit coefficient on age_years: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
p, diff, r1, r0, n1, n0 = chi2_2x2(df, "sex_female")
analyses.append({"hypothesis_ids": ["h28"], "code": "chi2 sex_female x objective_response", "result_summary": f"sex_female=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
coef, pv = cont_logit(df, "ecog_ps")
analyses.append({"hypothesis_ids": ["h29"], "code": "Logit objective_response ~ ecog_ps", "result_summary": f"Logit coefficient on ecog_ps: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
coef, pv = cont_logit(df, "bmi")
analyses.append({"hypothesis_ids": ["h30"], "code": "Logit objective_response ~ bmi", "result_summary": f"Logit coefficient on bmi: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(7, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 8: Lab markers — WBC, blast %, albumin, LDH, CRP, NLR
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h31", "text": "Higher wbc_k_per_ul is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h32", "text": "Higher blast_pct_marrow is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h33", "text": "Higher albumin_g_dl is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h34", "text": "Higher ldh_u_l is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h35", "text": "Higher crp_mg_l is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h36", "text": "Higher nlr (neutrophil-lymphocyte ratio) is associated with a lower objective_response rate.", "kind": "novel"},
]
analyses = []
for hid, col in [("h31", "wbc_k_per_ul"), ("h32", "blast_pct_marrow"), ("h33", "albumin_g_dl"), ("h34", "ldh_u_l"), ("h35", "crp_mg_l"), ("h36", "nlr")]:
    coef, pv = cont_logit(df, col)
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {col}", "result_summary": f"Logit coefficient on {col}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(8, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 9: Symptom and PRO grades on response
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h37", "text": "Higher fatigue_grade is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h38", "text": "Higher pain_nrs is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h39", "text": "Higher dyspnea_grade is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h40", "text": "Higher cough_grade is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h41", "text": "Higher appetite_loss_grade is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h42", "text": "Higher weight_loss_pct_6mo is associated with a lower objective_response rate.", "kind": "novel"},
]
analyses = []
for hid, col in [("h37", "fatigue_grade"), ("h38", "pain_nrs"), ("h39", "dyspnea_grade"), ("h40", "cough_grade"), ("h41", "appetite_loss_grade"), ("h42", "weight_loss_pct_6mo")]:
    coef, pv = cont_logit(df, col)
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {col}", "result_summary": f"Logit coefficient on {col}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(9, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 10: Comorbidity main effects
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h43", "text": "diabetes_mellitus=1 patients have a different objective_response rate than diabetes_mellitus=0 patients.", "kind": "novel"},
    {"id": "h44", "text": "heart_failure=1 patients have a lower objective_response rate than heart_failure=0 patients.", "kind": "novel"},
    {"id": "h45", "text": "chronic_kidney_disease=1 patients have a lower objective_response rate than chronic_kidney_disease=0 patients.", "kind": "novel"},
    {"id": "h46", "text": "copd=1 patients have a lower objective_response rate than copd=0 patients.", "kind": "novel"},
    {"id": "h47", "text": "prior_malignancy=1 patients have a lower objective_response rate than prior_malignancy=0 patients.", "kind": "novel"},
    {"id": "h48", "text": "hypertension=1 patients have a different objective_response rate than hypertension=0 patients.", "kind": "novel"},
]
analyses = []
for hid, col in [("h43", "diabetes_mellitus"), ("h44", "heart_failure"), ("h45", "chronic_kidney_disease"), ("h46", "copd"), ("h47", "prior_malignancy"), ("h48", "hypertension")]:
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, col)
    analyses.append({"hypothesis_ids": [hid], "code": f"chi2 {col} x objective_response", "result_summary": f"{col}=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(10, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 11: Demographic/social determinants — race, insurance, rural, education
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h49", "text": "Patients with rural_residence=1 have a different objective_response rate than rural_residence=0 patients.", "kind": "novel"},
    {"id": "h50", "text": "Higher education_years is associated with a different objective_response rate.", "kind": "novel"},
    {"id": "h51", "text": "Higher smoking_pack_years is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h52", "text": "Objective response rate differs across categories of race_ethnicity.", "kind": "novel"},
    {"id": "h53", "text": "Objective response rate differs across categories of insurance_type.", "kind": "novel"},
]
analyses = []
p, diff, r1, r0, n1, n0 = chi2_2x2(df, "rural_residence")
analyses.append({"hypothesis_ids": ["h49"], "code": "chi2 rural_residence x objective_response", "result_summary": f"rural=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
coef, pv = cont_logit(df, "education_years")
analyses.append({"hypothesis_ids": ["h50"], "code": "Logit objective_response ~ education_years", "result_summary": f"Logit coefficient on education_years: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
coef, pv = cont_logit(df, "smoking_pack_years")
analyses.append({"hypothesis_ids": ["h51"], "code": "Logit objective_response ~ smoking_pack_years", "result_summary": f"Logit coefficient on smoking_pack_years: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
tab = pd.crosstab(df["race_ethnicity"], df["objective_response"])
chi2v, pv, _, _ = stats.chi2_contingency(tab)
rates = df.groupby("race_ethnicity")["objective_response"].mean()
spread = float(rates.max() - rates.min())
analyses.append({"hypothesis_ids": ["h52"], "code": "chi2_contingency race_ethnicity x objective_response", "result_summary": f"Response rates: {rates.round(4).to_dict()}; chi2 p={pv:.4f}.", "p_value": pv, "effect_estimate": spread, "significant": bool(pv < 0.05)})
tab = pd.crosstab(df["insurance_type"], df["objective_response"])
chi2v, pv, _, _ = stats.chi2_contingency(tab)
rates = df.groupby("insurance_type")["objective_response"].mean()
spread = float(rates.max() - rates.min())
analyses.append({"hypothesis_ids": ["h53"], "code": "chi2_contingency insurance_type x objective_response", "result_summary": f"Response rates: {rates.round(4).to_dict()}; chi2 p={pv:.4f}.", "p_value": pv, "effect_estimate": spread, "significant": bool(pv < 0.05)})
add_iter(11, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 12: Treatment lines and prior therapy main effects
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h54", "text": "Higher prior_lines_of_therapy is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h55", "text": "prior_chemotherapy=1 is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h56", "text": "prior_targeted_therapy=1 is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h57", "text": "Higher years_since_diagnosis is associated with a lower objective_response rate.", "kind": "novel"},
]
analyses = []
coef, pv = cont_logit(df, "prior_lines_of_therapy")
analyses.append({"hypothesis_ids": ["h54"], "code": "Logit objective_response ~ prior_lines_of_therapy", "result_summary": f"Logit coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
p, diff, r1, r0, n1, n0 = chi2_2x2(df, "prior_chemotherapy")
analyses.append({"hypothesis_ids": ["h55"], "code": "chi2 prior_chemotherapy x objective_response", "result_summary": f"prior_chemotherapy=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
p, diff, r1, r0, n1, n0 = chi2_2x2(df, "prior_targeted_therapy")
analyses.append({"hypothesis_ids": ["h56"], "code": "chi2 prior_targeted_therapy x objective_response", "result_summary": f"prior_targeted=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
coef, pv = cont_logit(df, "years_since_diagnosis")
analyses.append({"hypothesis_ids": ["h57"], "code": "Logit objective_response ~ years_since_diagnosis", "result_summary": f"Logit coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(12, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 13: Adjusted multivariable model — IDH1 main effect after adjusting
# for treatment, age, ECOG, secondary AML, complex karyotype, TP53. Refines h5.
# Also test other key biomarkers in same adjusted model.
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h58", "text": "After adjusting for age_years, ecog_ps, secondary_aml, complex_karyotype, tp53_mutation, and the six treatments, idh1_mutation remains positively associated with objective_response (refines h5).", "kind": "refined"},
    {"id": "h59", "text": "After the same adjustment, npm1_mutation is associated with objective_response with a positive coefficient (refines h2).", "kind": "refined"},
    {"id": "h60", "text": "After the same adjustment, complex_karyotype is associated with objective_response with a negative coefficient (refines h3).", "kind": "refined"},
    {"id": "h61", "text": "After the same adjustment, tp53_mutation is associated with objective_response with a negative coefficient (refines h1).", "kind": "refined"},
]
analyses = []
adjust = ["age_years", "ecog_ps", "secondary_aml", "complex_karyotype", "tp53_mutation",
          "treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib",
          "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]
import statsmodels.api as sm
for hid, target in [("h58", "idh1_mutation"), ("h59", "npm1_mutation"), ("h60", "complex_karyotype"), ("h61", "tp53_mutation")]:
    cols = [c for c in adjust if c != target] + [target]
    X = df[cols].astype(float)
    X = sm.add_constant(X)
    y = df["objective_response"].astype(float)
    m = sm.Logit(y, X).fit(disp=False, maxiter=200)
    coef = float(m.params[target])
    pv = float(m.pvalues[target])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {target} + adjustments", "result_summary": f"Adjusted logit coefficient on {target}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(13, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 14: SNP main effects — initial scan over all 26 SNPs
# ---------------------------------------------------------------------------
snps = [c for c in df.columns if c.startswith("snp_")]
hyps = [{"id": f"h_snp_{s}", "text": f"snp_{s.split('_')[1]}=1 is associated with a different objective_response rate than {s}=0.", "kind": "novel"} for s in snps]
# rename to clean ids
hyps = [{"id": f"h{62+i}", "text": f"{snps[i]}=1 patients have a different objective_response rate than {snps[i]}=0 patients.", "kind": "novel"} for i in range(len(snps))]
analyses = []
for i, s in enumerate(snps):
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, s)
    analyses.append({"hypothesis_ids": [f"h{62+i}"], "code": f"chi2 {s} x objective_response", "result_summary": f"{s}=1: {r1:.4f} (n={n1}); =0: {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(14, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 15: Marker subset — bone marrow, lymphocyte counts, hemoglobin
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_hgb", "text": "Higher hemoglobin_g_dl is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h_plt", "text": "Higher platelets_k_ul is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h_anc", "text": "Higher anc_k_ul is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h_alc", "text": "Higher alc_k_ul is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h_bil", "text": "Higher total_bilirubin_mg_dl is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h_cr", "text": "Higher creatinine_mg_dl is associated with a lower objective_response rate.", "kind": "novel"},
]
analyses = []
for hid, col in [("h_hgb", "hemoglobin_g_dl"), ("h_plt", "platelets_k_ul"), ("h_anc", "anc_k_ul"), ("h_alc", "alc_k_ul"), ("h_bil", "total_bilirubin_mg_dl"), ("h_cr", "creatinine_mg_dl")]:
    coef, pv = cont_logit(df, col)
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {col}", "result_summary": f"Logit coefficient on {col}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(15, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 16: Cardiovascular labs and vitals
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_sbp", "text": "Higher systolic_bp_mmhg is associated with a different objective_response rate.", "kind": "novel"},
    {"id": "h_hr", "text": "Higher heart_rate_bpm is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h_spo2", "text": "Higher spo2_pct is associated with a higher objective_response rate.", "kind": "novel"},
    {"id": "h_alb_alt", "text": "Higher alt_u_l is associated with a different objective_response rate.", "kind": "novel"},
    {"id": "h_inr", "text": "Higher inr is associated with a lower objective_response rate.", "kind": "novel"},
    {"id": "h_glu", "text": "Higher glucose_mg_dl is associated with a different objective_response rate.", "kind": "novel"},
]
analyses = []
for hid, col in [("h_sbp", "systolic_bp_mmhg"), ("h_hr", "heart_rate_bpm"), ("h_spo2", "spo2_pct"), ("h_alb_alt", "alt_u_l"), ("h_inr", "inr"), ("h_glu", "glucose_mg_dl")]:
    coef, pv = cont_logit(df, col)
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {col}", "result_summary": f"Logit coefficient on {col}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(16, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 17: Three-way refinement — IDH1 effect within ivosidenib-treated and
# ivosidenib-untreated groups (does idh1 main effect depend on treatment?)
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_idh1_iv1", "text": "Among patients receiving treatment_ivosidenib=1, idh1_mutation=1 is associated with a higher objective_response rate than idh1_mutation=0.", "kind": "refined"},
    {"id": "h_idh1_iv0", "text": "Among patients receiving treatment_ivosidenib=0, idh1_mutation=1 is associated with a higher objective_response rate than idh1_mutation=0.", "kind": "refined"},
    {"id": "h_idh2_en1", "text": "Among patients receiving treatment_enasidenib=1, idh2_mutation=1 vs 0 is associated with a different objective_response rate.", "kind": "refined"},
    {"id": "h_idh2_en0", "text": "Among patients receiving treatment_enasidenib=0, idh2_mutation=1 vs 0 is associated with a different objective_response rate.", "kind": "refined"},
]
analyses = []
sub = df[df["treatment_ivosidenib"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "idh1_mutation")
analyses.append({"hypothesis_ids": ["h_idh1_iv1"], "code": "chi2 idh1_mutation x objective_response within treatment_ivosidenib==1", "result_summary": f"In ivosidenib-treated (n={len(sub)}): idh1+ {r1:.4f} (n={n1}) vs idh1- {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
sub = df[df["treatment_ivosidenib"] == 0]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "idh1_mutation")
analyses.append({"hypothesis_ids": ["h_idh1_iv0"], "code": "chi2 idh1_mutation x objective_response within treatment_ivosidenib==0", "result_summary": f"In ivosidenib-untreated (n={len(sub)}): idh1+ {r1:.4f} (n={n1}) vs idh1- {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
sub = df[df["treatment_enasidenib"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "idh2_mutation")
analyses.append({"hypothesis_ids": ["h_idh2_en1"], "code": "chi2 idh2_mutation x objective_response within treatment_enasidenib==1", "result_summary": f"In enasidenib-treated (n={len(sub)}): idh2+ {r1:.4f} (n={n1}) vs idh2- {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
sub = df[df["treatment_enasidenib"] == 0]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "idh2_mutation")
analyses.append({"hypothesis_ids": ["h_idh2_en0"], "code": "chi2 idh2_mutation x objective_response within treatment_enasidenib==0", "result_summary": f"In enasidenib-untreated (n={len(sub)}): idh2+ {r1:.4f} (n={n1}) vs idh2- {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(17, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 18: Subgroup — VEN+AZA effect by NPM1 and TP53
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_va_npm1", "text": "Within npm1_mutation=1, treatment_venetoclax_azacitidine=1 yields a higher objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h_va_tp53", "text": "Within tp53_mutation=1, treatment_venetoclax_azacitidine=1 yields a higher objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h_va_ck", "text": "Within complex_karyotype=1, treatment_venetoclax_azacitidine=1 yields a different objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h_va_idh1", "text": "Within idh1_mutation=1, treatment_venetoclax_azacitidine=1 yields a higher objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
]
analyses = []
for hid, sub_col in [("h_va_npm1", "npm1_mutation"), ("h_va_tp53", "tp53_mutation"), ("h_va_ck", "complex_karyotype"), ("h_va_idh1", "idh1_mutation")]:
    sub = df[df[sub_col] == 1]
    p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_venetoclax_azacitidine")
    analyses.append({"hypothesis_ids": [hid], "code": f"chi2 treatment_venetoclax_azacitidine x objective_response within {sub_col}==1", "result_summary": f"In {sub_col}+ (n={len(sub)}): on VEN+AZA {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(18, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 19: 7+3 subgroup analyses — by age, ECOG, secondary AML
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_73_age", "text": "There is a negative interaction between age_years and treatment_7plus3 on objective_response in a logistic regression (treatment effect declines with age).", "kind": "novel"},
    {"id": "h_73_ecog", "text": "There is a negative interaction between ecog_ps and treatment_7plus3 on objective_response in a logistic regression.", "kind": "novel"},
    {"id": "h_73_sec", "text": "Within secondary_aml=1, treatment_7plus3=1 yields a different objective_response rate than treatment_7plus3=0.", "kind": "novel"},
]
analyses = []
coef, pv = logit_test(df, ["age_years", "treatment_7plus3"], interaction=("age_years", "treatment_7plus3"))
analyses.append({"hypothesis_ids": ["h_73_age"], "code": "Logit objective_response ~ age_years + 7+3 + age:7+3", "result_summary": f"Interaction coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
coef, pv = logit_test(df, ["ecog_ps", "treatment_7plus3"], interaction=("ecog_ps", "treatment_7plus3"))
analyses.append({"hypothesis_ids": ["h_73_ecog"], "code": "Logit objective_response ~ ecog_ps + 7+3 + ecog:7+3", "result_summary": f"Interaction coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
sub = df[df["secondary_aml"] == 1]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_7plus3")
analyses.append({"hypothesis_ids": ["h_73_sec"], "code": "chi2 treatment_7plus3 x objective_response within secondary_aml==1", "result_summary": f"In secondary_aml+ (n={len(sub)}): on 7+3 {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(19, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 20: Mutation-by-mutation co-occurrence interactions on response
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_npm1_flt3", "text": "There is an interaction between npm1_mutation and flt3_itd on objective_response (logit interaction term).", "kind": "novel"},
    {"id": "h_npm1_tp53", "text": "There is an interaction between npm1_mutation and tp53_mutation on objective_response.", "kind": "novel"},
    {"id": "h_tp53_ck", "text": "There is an interaction between tp53_mutation and complex_karyotype on objective_response.", "kind": "novel"},
    {"id": "h_idh1_idh2", "text": "There is an interaction between idh1_mutation and idh2_mutation on objective_response.", "kind": "novel"},
]
analyses = []
for hid, a, b in [("h_npm1_flt3", "npm1_mutation", "flt3_itd"), ("h_npm1_tp53", "npm1_mutation", "tp53_mutation"), ("h_tp53_ck", "tp53_mutation", "complex_karyotype"), ("h_idh1_idh2", "idh1_mutation", "idh2_mutation")]:
    coef, pv = logit_test(df, [a, b], interaction=(a, b))
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {a} + {b} + {a}:{b}", "result_summary": f"Interaction coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(20, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 21: SNP × treatment interactions for nominal SNP hits flagged in iter 14.
# We'll test interactions for each SNP with each major treatment as a screen.
# Then keep the most significant ones for highlighting.
# ---------------------------------------------------------------------------
# First let's actually find SNPs with nominally significant main effects
snp_main = []
for s in snps:
    p, diff, r1, r0, n1, n0 = chi2_2x2(df, s)
    snp_main.append((s, p, diff))
snp_main.sort(key=lambda x: x[1])
top_snps = [s for s, p, _ in snp_main[:5]]

hyps = []
analyses = []
trt_focus = ["treatment_venetoclax_azacitidine", "treatment_7plus3"]
for i, s in enumerate(top_snps):
    for j, t in enumerate(trt_focus):
        hid = f"h_snp_int_{i}_{j}"
        hyps.append({"id": hid, "text": f"There is an interaction between {s} and {t} on objective_response (logit interaction term).", "kind": "novel"})
        coef, pv = logit_test(df, [s, t], interaction=(s, t))
        analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {s} + {t} + interaction", "result_summary": f"Interaction coef: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(21, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 22: Adjusted IDH1 effect, isolating ivosidenib subset and adjusting
# for age, ECOG. Also, adjusted treatment effects after baseline covariates.
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_idh1_adj_iv1", "text": "Within treatment_ivosidenib=1 patients, idh1_mutation is positively associated with objective_response after adjusting for age_years and ecog_ps.", "kind": "refined"},
    {"id": "h_idh1_adj_iv0", "text": "Within treatment_ivosidenib=0 patients, idh1_mutation is positively associated with objective_response after adjusting for age_years and ecog_ps.", "kind": "refined"},
    {"id": "h_va_adj", "text": "After adjusting for age_years, ecog_ps, secondary_aml, complex_karyotype, and tp53_mutation, treatment_venetoclax_azacitidine is positively associated with objective_response.", "kind": "refined"},
    {"id": "h_73_adj", "text": "After the same adjustment, treatment_7plus3 is positively associated with objective_response.", "kind": "refined"},
]
analyses = []
import statsmodels.api as sm
for hid, mask in [("h_idh1_adj_iv1", df["treatment_ivosidenib"] == 1), ("h_idh1_adj_iv0", df["treatment_ivosidenib"] == 0)]:
    sub = df[mask]
    X = sub[["age_years", "ecog_ps", "idh1_mutation"]].astype(float)
    X = sm.add_constant(X)
    y = sub["objective_response"].astype(float)
    m = sm.Logit(y, X).fit(disp=False, maxiter=200)
    coef = float(m.params["idh1_mutation"])
    pv = float(m.pvalues["idh1_mutation"])
    analyses.append({"hypothesis_ids": [hid], "code": "Logit objective_response ~ age_years + ecog_ps + idh1_mutation", "result_summary": f"Adjusted logit coefficient on idh1_mutation in subgroup: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})

adjust = ["age_years", "ecog_ps", "secondary_aml", "complex_karyotype", "tp53_mutation"]
for hid, target in [("h_va_adj", "treatment_venetoclax_azacitidine"), ("h_73_adj", "treatment_7plus3")]:
    cols = adjust + [target]
    X = df[cols].astype(float)
    X = sm.add_constant(X)
    y = df["objective_response"].astype(float)
    m = sm.Logit(y, X).fit(disp=False, maxiter=200)
    coef = float(m.params[target])
    pv = float(m.pvalues[target])
    analyses.append({"hypothesis_ids": [hid], "code": f"Logit objective_response ~ {target} + adjustments", "result_summary": f"Adjusted logit coefficient on {target}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(22, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 23: Three-way interactions — does ven+aza benefit in unfit depend on TP53?
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_va_unfit_tp53_pos", "text": "Within unfit_for_intensive=1 AND tp53_mutation=1, treatment_venetoclax_azacitidine=1 yields a different objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h_va_unfit_tp53_neg", "text": "Within unfit_for_intensive=1 AND tp53_mutation=0, treatment_venetoclax_azacitidine=1 yields a higher objective_response rate than treatment_venetoclax_azacitidine=0.", "kind": "novel"},
    {"id": "h_three_way", "text": "There is a three-way interaction unfit_for_intensive x tp53_mutation x treatment_venetoclax_azacitidine on objective_response in a logistic regression.", "kind": "novel"},
]
analyses = []
sub = df[(df["unfit_for_intensive"] == 1) & (df["tp53_mutation"] == 1)]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_venetoclax_azacitidine")
analyses.append({"hypothesis_ids": ["h_va_unfit_tp53_pos"], "code": "chi2 VEN+AZA x outcome within unfit_for_intensive==1 & tp53==1", "result_summary": f"In unfit & tp53+ (n={len(sub)}): on VEN+AZA {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
sub = df[(df["unfit_for_intensive"] == 1) & (df["tp53_mutation"] == 0)]
p, diff, r1, r0, n1, n0 = chi2_2x2(sub, "treatment_venetoclax_azacitidine")
analyses.append({"hypothesis_ids": ["h_va_unfit_tp53_neg"], "code": "chi2 VEN+AZA x outcome within unfit_for_intensive==1 & tp53==0", "result_summary": f"In unfit & tp53- (n={len(sub)}): on VEN+AZA {r1:.4f} (n={n1}) vs off {r0:.4f} (n={n0}); diff={diff:+.4f}; p={p:.4f}.", "p_value": p, "effect_estimate": float(diff), "significant": bool(p < 0.05)})
import statsmodels.api as sm
X = df[["unfit_for_intensive", "tp53_mutation", "treatment_venetoclax_azacitidine"]].astype(float).copy()
X["a_b"] = X["unfit_for_intensive"] * X["tp53_mutation"]
X["a_c"] = X["unfit_for_intensive"] * X["treatment_venetoclax_azacitidine"]
X["b_c"] = X["tp53_mutation"] * X["treatment_venetoclax_azacitidine"]
X["a_b_c"] = X["unfit_for_intensive"] * X["tp53_mutation"] * X["treatment_venetoclax_azacitidine"]
X = sm.add_constant(X)
y = df["objective_response"].astype(float)
m = sm.Logit(y, X).fit(disp=False, maxiter=200)
coef = float(m.params["a_b_c"])
pv = float(m.pvalues["a_b_c"])
analyses.append({"hypothesis_ids": ["h_three_way"], "code": "Logit with all 2-way and 3-way interactions of unfit, tp53, ven+aza", "result_summary": f"Three-way interaction coefficient: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(23, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 24: Hepatic, renal, and lab adjusted hypotheses for IDH1 robustness
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_idh1_full", "text": "After adjusting for age_years, ecog_ps, secondary_aml, complex_karyotype, tp53_mutation, npm1_mutation, flt3_itd, treatment_ivosidenib, treatment_venetoclax_azacitidine, treatment_7plus3, blast_pct_marrow, wbc_k_per_ul, albumin_g_dl, ldh_u_l, idh1_mutation remains positively associated with objective_response.", "kind": "refined"},
    {"id": "h_dose_treat_count", "text": "The total number of concurrent treatments received (sum of the six treatment_* indicators) is associated with objective_response.", "kind": "novel"},
    {"id": "h_unfit_dose", "text": "Among unfit_for_intensive=1 patients, the count of concurrent treatments is associated with objective_response.", "kind": "novel"},
]
analyses = []
import statsmodels.api as sm
adj_cols = ["age_years", "ecog_ps", "secondary_aml", "complex_karyotype", "tp53_mutation",
            "npm1_mutation", "flt3_itd", "treatment_ivosidenib", "treatment_venetoclax_azacitidine",
            "treatment_7plus3", "blast_pct_marrow", "wbc_k_per_ul", "albumin_g_dl", "ldh_u_l",
            "idh1_mutation"]
X = df[adj_cols].astype(float)
X = sm.add_constant(X)
y = df["objective_response"].astype(float)
m = sm.Logit(y, X).fit(disp=False, maxiter=400)
coef = float(m.params["idh1_mutation"])
pv = float(m.pvalues["idh1_mutation"])
analyses.append({"hypothesis_ids": ["h_idh1_full"], "code": "Logit objective_response ~ idh1_mutation + many adjustments", "result_summary": f"Fully-adjusted logit coefficient on idh1_mutation: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})

trts_all = ["treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib", "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]
df["_trt_count"] = df[trts_all].sum(axis=1)
coef, pv = cont_logit(df, "_trt_count")
analyses.append({"hypothesis_ids": ["h_dose_treat_count"], "code": "Logit objective_response ~ trt_count (sum of 6 treatments)", "result_summary": f"Logit coefficient on treatment_count: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
sub = df[df["unfit_for_intensive"] == 1].copy()
coef, pv = cont_logit(sub, "_trt_count")
analyses.append({"hypothesis_ids": ["h_unfit_dose"], "code": "Logit objective_response ~ trt_count within unfit_for_intensive==1", "result_summary": f"In unfit (n={len(sub)}), logit coefficient on treatment_count: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(24, hyps, analyses)


# ---------------------------------------------------------------------------
# Iteration 25: Final synthesis — confirm IDH1 main effect and ECOG main effect
# in a single multivariable model; provide overall direction summary.
# ---------------------------------------------------------------------------
hyps = [
    {"id": "h_final_idh1", "text": "In a final multivariable logistic regression (objective_response ~ idh1_mutation + age_years + sex_female + ecog_ps + tp53_mutation + npm1_mutation + flt3_itd + complex_karyotype + secondary_aml + unfit_for_intensive + treatment_ivosidenib + treatment_venetoclax_azacitidine + treatment_7plus3 + treatment_midostaurin + treatment_gilteritinib + treatment_enasidenib), idh1_mutation has a positive coefficient (i.e., higher response).", "kind": "refined"},
    {"id": "h_final_ecog", "text": "In the same final multivariable model, ecog_ps has a coefficient with a sign matching the iter-7 univariable result.", "kind": "refined"},
    {"id": "h_final_age", "text": "In the same final multivariable model, age_years has a coefficient with a sign matching the iter-7 univariable result.", "kind": "refined"},
]
analyses = []
final_cols = ["idh1_mutation", "age_years", "sex_female", "ecog_ps", "tp53_mutation",
              "npm1_mutation", "flt3_itd", "complex_karyotype", "secondary_aml",
              "unfit_for_intensive", "treatment_ivosidenib", "treatment_venetoclax_azacitidine",
              "treatment_7plus3", "treatment_midostaurin", "treatment_gilteritinib",
              "treatment_enasidenib"]
X = df[final_cols].astype(float)
X = sm.add_constant(X)
y = df["objective_response"].astype(float)
m = sm.Logit(y, X).fit(disp=False, maxiter=400)
for hid, target in [("h_final_idh1", "idh1_mutation"), ("h_final_ecog", "ecog_ps"), ("h_final_age", "age_years")]:
    coef = float(m.params[target])
    pv = float(m.pvalues[target])
    analyses.append({"hypothesis_ids": [hid], "code": "Final multivariable logit (see model in result_summary)", "result_summary": f"Final-model coefficient on {target}: {coef:+.6f}, p={pv:.4f}.", "p_value": pv, "effect_estimate": coef, "significant": bool(pv < 0.05)})
add_iter(25, hyps, analyses)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
transcript = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@named-aml-1",
    "max_iterations": 25,
    "iterations": OUT["iterations"],
}
(HERE / "transcript.json").write_text(json.dumps(transcript, indent=2, default=float))

# Also pickle results for the summary writer
import pickle
(HERE / "_results.pkl").write_bytes(pickle.dumps(OUT))
print("DONE — wrote transcript.json with", len(OUT["iterations"]), "iterations")
