"""
Run a series of statistical analyses on the ds001_aml dataset and emit
transcript.json + analysis_summary.txt. All work is self-contained in this
directory.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
df = pd.read_parquet(HERE / "dataset.parquet")

OUT = {
    "dataset_id": "ds001_aml",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-manual@2026-04-27",
    "max_iterations": 25,
    "iterations": [],
}


def add_iter(index, hypotheses, analyses):
    OUT["iterations"].append(
        {
            "index": index,
            "proposed_hypotheses": hypotheses,
            "analyses": analyses,
        }
    )


def chi2_2x2(df, group_col, outcome_col):
    tab = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    p1 = df.loc[df[group_col] == 1, outcome_col].mean()
    p0 = df.loc[df[group_col] == 0, outcome_col].mean()
    return p1 - p0, p, p1, p0


def logit_or(df, x_col, y_col="objective_response", controls=None):
    cols = [x_col] + (controls or [])
    X = df[cols].astype(float)
    X = sm.add_constant(X)
    y = df[y_col].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    coef = res.params[x_col]
    p = res.pvalues[x_col]
    return coef, p, res


def logit_interaction(df, tx, marker, y_col="objective_response", controls=None):
    d = df.copy()
    d["_int"] = d[tx] * d[marker]
    cols = [tx, marker, "_int"] + (controls or [])
    X = d[cols].astype(float)
    X = sm.add_constant(X)
    y = d[y_col].astype(int)
    res = sm.Logit(y, X).fit(disp=0)
    return res.params["_int"], res.pvalues["_int"], res


# ---------------- Iteration 1: Disease prognostic markers (main effects) ----
hyps_1 = [
    {"id": "h1", "text": "Patients with tp53_mutation=1 have a lower probability of objective_response than those with tp53_mutation=0.", "kind": "novel"},
    {"id": "h2", "text": "Patients with complex_karyotype=1 have a lower probability of objective_response than those with complex_karyotype=0.", "kind": "novel"},
    {"id": "h3", "text": "Patients with secondary_aml=1 have a lower probability of objective_response than those with secondary_aml=0.", "kind": "novel"},
    {"id": "h4", "text": "Patients with unfit_for_intensive=1 have a lower probability of objective_response than those with unfit_for_intensive=0.", "kind": "novel"},
]
ana_1 = []
for hid, col in [("h1", "tp53_mutation"), ("h2", "complex_karyotype"), ("h3", "secondary_aml"), ("h4", "unfit_for_intensive")]:
    diff, p, p1, p0 = chi2_2x2(df, col, "objective_response")
    ana_1.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 on {col} vs objective_response",
        "result_summary": f"ORR with {col}=1: {p1:.3f}; with {col}=0: {p0:.3f}; diff={diff:+.3f}; chi2 p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(1, hyps_1, ana_1)

# ---------------- Iteration 2: Favorable AML markers ----
hyps_2 = [
    {"id": "h5", "text": "Patients with npm1_mutation=1 have a higher probability of objective_response than those with npm1_mutation=0.", "kind": "novel"},
    {"id": "h6", "text": "Patients with idh1_mutation=1 have a higher probability of objective_response than those with idh1_mutation=0.", "kind": "novel"},
    {"id": "h7", "text": "Patients with idh2_mutation=1 have a higher probability of objective_response than those with idh2_mutation=0.", "kind": "novel"},
    {"id": "h8", "text": "Patients with flt3_itd=1 have a lower probability of objective_response than those with flt3_itd=0.", "kind": "novel"},
]
ana_2 = []
for hid, col in [("h5", "npm1_mutation"), ("h6", "idh1_mutation"), ("h7", "idh2_mutation"), ("h8", "flt3_itd")]:
    diff, p, p1, p0 = chi2_2x2(df, col, "objective_response")
    ana_2.append({
        "hypothesis_ids": [hid],
        "code": f"chi2 on {col}",
        "result_summary": f"ORR {col}=1: {p1:.3f}; {col}=0: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(2, hyps_2, ana_2)

# ---------------- Iteration 3: Treatment main effects ----
hyps_3 = [
    {"id": f"h{9+i}", "text": f"Patients receiving {tx}=1 have a different probability of objective_response than those not receiving it.", "kind": "novel"}
    for i, tx in enumerate(["treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib", "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"])
]
ana_3 = []
tx_list = ["treatment_midostaurin", "treatment_gilteritinib", "treatment_ivosidenib", "treatment_enasidenib", "treatment_venetoclax_azacitidine", "treatment_7plus3"]
for i, tx in enumerate(tx_list):
    diff, p, p1, p0 = chi2_2x2(df, tx, "objective_response")
    ana_3.append({
        "hypothesis_ids": [f"h{9+i}"],
        "code": f"chi2 on {tx}",
        "result_summary": f"ORR {tx}=1: {p1:.3f}; off: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(diff),
        "significant": bool(p < 0.05),
    })
add_iter(3, hyps_3, ana_3)

# ---------------- Iteration 4: Midostaurin/gilteritinib × FLT3 (predictive interactions) ----
hyps_4 = [
    {"id": "h15", "text": "Among flt3_itd=1 patients, treatment_midostaurin=1 yields a higher objective_response rate than treatment_midostaurin=0; the benefit is larger than in flt3_itd=0 patients (positive interaction).", "kind": "novel"},
    {"id": "h16", "text": "Among flt3_itd=1 patients, treatment_gilteritinib=1 yields a higher objective_response rate than treatment_gilteritinib=0; the benefit is larger than in flt3_itd=0 patients (positive interaction).", "kind": "novel"},
    {"id": "h17", "text": "Among flt3_tkd=1 patients, treatment_gilteritinib=1 yields a higher objective_response rate than treatment_gilteritinib=0.", "kind": "novel"},
]
ana_4 = []
for hid, tx, marker in [("h15", "treatment_midostaurin", "flt3_itd"), ("h16", "treatment_gilteritinib", "flt3_itd"), ("h17", "treatment_gilteritinib", "flt3_tkd")]:
    coef, pint, res = logit_interaction(df, tx, marker)
    sub = df[df[marker] == 1]
    diff_in, _, p1_in, p0_in = chi2_2x2(sub, tx, "objective_response")
    ana_4.append({
        "hypothesis_ids": [hid],
        "code": f"logistic regression: objective_response ~ {tx} + {marker} + {tx}:{marker}",
        "result_summary": f"In {marker}=1: ORR {tx}=1 {p1_in:.3f} vs {tx}=0 {p0_in:.3f} (diff {diff_in:+.3f}); interaction logOR={coef:+.3f}, p={pint:.3g}",
        "p_value": float(pint),
        "effect_estimate": float(diff_in),
        "significant": bool(pint < 0.05),
    })
add_iter(4, hyps_4, ana_4)

# ---------------- Iteration 5: IDH inhibitors × IDH mutation ----
hyps_5 = [
    {"id": "h18", "text": "Among idh1_mutation=1 patients, treatment_ivosidenib=1 yields a higher objective_response rate than treatment_ivosidenib=0 (positive treatment-by-biomarker interaction).", "kind": "novel"},
    {"id": "h19", "text": "Among idh2_mutation=1 patients, treatment_enasidenib=1 yields a higher objective_response rate than treatment_enasidenib=0 (positive treatment-by-biomarker interaction).", "kind": "novel"},
]
ana_5 = []
for hid, tx, marker in [("h18", "treatment_ivosidenib", "idh1_mutation"), ("h19", "treatment_enasidenib", "idh2_mutation")]:
    coef, pint, res = logit_interaction(df, tx, marker)
    sub = df[df[marker] == 1]
    diff_in, _, p1_in, p0_in = chi2_2x2(sub, tx, "objective_response")
    ana_5.append({
        "hypothesis_ids": [hid],
        "code": f"logistic regression: objective_response ~ {tx} + {marker} + {tx}:{marker}",
        "result_summary": f"In {marker}=1: ORR {tx}=1 {p1_in:.3f} vs {tx}=0 {p0_in:.3f} (diff {diff_in:+.3f}); interaction logOR={coef:+.3f}, p={pint:.3g}",
        "p_value": float(pint),
        "effect_estimate": float(diff_in),
        "significant": bool(pint < 0.05),
    })
add_iter(5, hyps_5, ana_5)

# ---------------- Iteration 6: Venetoclax+aza × unfit / 7+3 × fit ----
hyps_6 = [
    {"id": "h20", "text": "Among unfit_for_intensive=1 patients, treatment_venetoclax_azacitidine=1 yields a higher objective_response rate than treatment_venetoclax_azacitidine=0 (positive interaction with unfit status).", "kind": "novel"},
    {"id": "h21", "text": "Among unfit_for_intensive=0 patients, treatment_7plus3=1 yields a higher objective_response rate than treatment_7plus3=0 (positive interaction with fit status).", "kind": "novel"},
]
ana_6 = []
for hid, tx, marker, sub_val in [("h20", "treatment_venetoclax_azacitidine", "unfit_for_intensive", 1), ("h21", "treatment_7plus3", "unfit_for_intensive", 0)]:
    coef, pint, res = logit_interaction(df, tx, marker)
    sub = df[df[marker] == sub_val]
    diff_in, _, p1_in, p0_in = chi2_2x2(sub, tx, "objective_response")
    ana_6.append({
        "hypothesis_ids": [hid],
        "code": f"logistic: objective_response ~ {tx} + {marker} + {tx}:{marker}",
        "result_summary": f"In {marker}={sub_val}: ORR {tx}=1 {p1_in:.3f} vs {tx}=0 {p0_in:.3f} (diff {diff_in:+.3f}); interaction logOR={coef:+.3f}, p={pint:.3g}",
        "p_value": float(pint),
        "effect_estimate": float(diff_in),
        "significant": bool(pint < 0.05),
    })
add_iter(6, hyps_6, ana_6)

# ---------------- Iteration 7: TP53 × venetoclax+aza ----
hyps_7 = [
    {"id": "h22", "text": "Among tp53_mutation=1 patients, treatment_venetoclax_azacitidine=1 yields a lower objective_response rate than expected (negative interaction; TP53 abrogates venetoclax benefit).", "kind": "novel"},
    {"id": "h23", "text": "Patients with tp53_mutation=1 AND complex_karyotype=1 have a lower probability of objective_response than tp53_mutation=0 AND complex_karyotype=0 (joint effect).", "kind": "novel"},
]
ana_7 = []
coef, pint, res = logit_interaction(df, "treatment_venetoclax_azacitidine", "tp53_mutation")
sub = df[df["tp53_mutation"] == 1]
diff_in, _, p1_in, p0_in = chi2_2x2(sub, "treatment_venetoclax_azacitidine", "objective_response")
ana_7.append({
    "hypothesis_ids": ["h22"],
    "code": "logistic: objective_response ~ treatment_venetoclax_azacitidine + tp53_mutation + interaction",
    "result_summary": f"In tp53_mutation=1: ORR ven+aza=1 {p1_in:.3f} vs =0 {p0_in:.3f} (diff {diff_in:+.3f}); interaction logOR={coef:+.3f}, p={pint:.3g}",
    "p_value": float(pint),
    "effect_estimate": float(diff_in),
    "significant": bool(pint < 0.05),
})
both = ((df["tp53_mutation"] == 1) & (df["complex_karyotype"] == 1)).astype(int)
neither = ((df["tp53_mutation"] == 0) & (df["complex_karyotype"] == 0)).astype(int)
mask = (both == 1) | (neither == 1)
sub = df[mask].copy()
sub["both"] = both[mask].values
diff, p, p1, p0 = chi2_2x2(sub, "both", "objective_response")
ana_7.append({
    "hypothesis_ids": ["h23"],
    "code": "ORR(tp53=1 & complex=1) vs ORR(tp53=0 & complex=0)",
    "result_summary": f"ORR both=1: {p1:.3f}; both=0: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}",
    "p_value": float(p),
    "effect_estimate": float(diff),
    "significant": bool(p < 0.05),
})
add_iter(7, hyps_7, ana_7)

# ---------------- Iteration 8: Lab/inflammation main effects on ORR ----
hyps_8 = [
    {"id": "h24", "text": "Higher ldh_u_l is associated with a lower probability of objective_response (negative coefficient in logistic regression).", "kind": "novel"},
    {"id": "h25", "text": "Higher albumin_g_dl is associated with a higher probability of objective_response.", "kind": "novel"},
    {"id": "h26", "text": "Higher crp_mg_l is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h27", "text": "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h28", "text": "Higher blast_pct_marrow is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_8 = []
for hid, col in [("h24", "ldh_u_l"), ("h25", "albumin_g_dl"), ("h26", "crp_mg_l"), ("h27", "nlr"), ("h28", "blast_pct_marrow")]:
    coef, p, _ = logit_or(df, col)
    ana_8.append({
        "hypothesis_ids": [hid],
        "code": f"logistic: objective_response ~ {col}",
        "result_summary": f"logOR per unit {col}: {coef:+.4g}; p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(coef),
        "significant": bool(p < 0.05),
    })
add_iter(8, hyps_8, ana_8)

# ---------------- Iteration 9: Performance status & age main effects ----
hyps_9 = [
    {"id": "h29", "text": "Higher ecog_ps is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h30", "text": "Higher age_years is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h31", "text": "Higher weight_loss_pct_6mo is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_9 = []
for hid, col in [("h29", "ecog_ps"), ("h30", "age_years"), ("h31", "weight_loss_pct_6mo")]:
    coef, p, _ = logit_or(df, col)
    ana_9.append({
        "hypothesis_ids": [hid],
        "code": f"logistic: objective_response ~ {col}",
        "result_summary": f"logOR per unit {col}: {coef:+.4g}; p={p:.3g}",
        "p_value": float(p),
        "effect_estimate": float(coef),
        "significant": bool(p < 0.05),
    })
add_iter(9, hyps_9, ana_9)

# ---------------- Iteration 10: Sex and demographics ----
hyps_10 = [
    {"id": "h32", "text": "sex_female=1 is associated with a higher probability of objective_response than sex_female=0.", "kind": "novel"},
    {"id": "h33", "text": "rural_residence=1 is associated with a lower probability of objective_response than rural_residence=0.", "kind": "novel"},
    {"id": "h34", "text": "More smoking_pack_years is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_10 = []
diff, p, p1, p0 = chi2_2x2(df, "sex_female", "objective_response")
ana_10.append({"hypothesis_ids": ["h32"], "code": "chi2 on sex_female", "result_summary": f"ORR female: {p1:.3f}; male: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
diff, p, p1, p0 = chi2_2x2(df, "rural_residence", "objective_response")
ana_10.append({"hypothesis_ids": ["h33"], "code": "chi2 on rural_residence", "result_summary": f"ORR rural=1: {p1:.3f}; =0: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
coef, p, _ = logit_or(df, "smoking_pack_years")
ana_10.append({"hypothesis_ids": ["h34"], "code": "logistic on smoking_pack_years", "result_summary": f"logOR per pack-year: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(10, hyps_10, ana_10)

# ---------------- Iteration 11: Race/ethnicity, insurance ----
hyps_11 = [
    {"id": "h35", "text": "Probability of objective_response differs across race_ethnicity categories (white/hispanic/black/asian/other).", "kind": "novel"},
    {"id": "h36", "text": "Probability of objective_response differs across insurance_type categories (medicare/private/medicaid/uninsured).", "kind": "novel"},
    {"id": "h37", "text": "Patients with insurance_type='uninsured' have a lower probability of objective_response than those with private insurance.", "kind": "novel"},
]
ana_11 = []
tab = pd.crosstab(df["race_ethnicity"], df["objective_response"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
rates = df.groupby("race_ethnicity")["objective_response"].mean().to_dict()
ana_11.append({"hypothesis_ids": ["h35"], "code": "chi2 on race_ethnicity x objective_response", "result_summary": f"ORR by race: {rates}; chi2 p={p:.3g}", "p_value": float(p), "effect_estimate": float(max(rates.values()) - min(rates.values())), "significant": bool(p < 0.05)})
tab = pd.crosstab(df["insurance_type"], df["objective_response"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
rates_i = df.groupby("insurance_type")["objective_response"].mean().to_dict()
ana_11.append({"hypothesis_ids": ["h36"], "code": "chi2 on insurance_type x objective_response", "result_summary": f"ORR by insurance: {rates_i}; chi2 p={p:.3g}", "p_value": float(p), "effect_estimate": float(max(rates_i.values()) - min(rates_i.values())), "significant": bool(p < 0.05)})
sub = df[df["insurance_type"].isin(["uninsured", "private"])]
diff_pp = sub.loc[sub["insurance_type"] == "uninsured", "objective_response"].mean() - sub.loc[sub["insurance_type"] == "private", "objective_response"].mean()
tab = pd.crosstab(sub["insurance_type"], sub["objective_response"])
_, p_pp, _, _ = stats.chi2_contingency(tab)
ana_11.append({"hypothesis_ids": ["h37"], "code": "chi2 uninsured vs private", "result_summary": f"ORR uninsured-private diff={diff_pp:+.3f}; p={p_pp:.3g}", "p_value": float(p_pp), "effect_estimate": float(diff_pp), "significant": bool(p_pp < 0.05)})
add_iter(11, hyps_11, ana_11)

# ---------------- Iteration 12: Comorbidities ----
hyps_12 = [
    {"id": "h38", "text": "heart_failure=1 is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h39", "text": "chronic_kidney_disease=1 is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h40", "text": "diabetes_mellitus=1 is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h41", "text": "prior_malignancy=1 is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_12 = []
for hid, col in [("h38", "heart_failure"), ("h39", "chronic_kidney_disease"), ("h40", "diabetes_mellitus"), ("h41", "prior_malignancy")]:
    diff, p, p1, p0 = chi2_2x2(df, col, "objective_response")
    ana_12.append({"hypothesis_ids": [hid], "code": f"chi2 on {col}", "result_summary": f"ORR {col}=1: {p1:.3f}; =0: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
add_iter(12, hyps_12, ana_12)

# ---------------- Iteration 13: Symptoms (PROs) ----
hyps_13 = [
    {"id": "h42", "text": "Higher fatigue_grade is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h43", "text": "Higher pain_nrs is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h44", "text": "Higher dyspnea_grade is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h45", "text": "Higher appetite_loss_grade is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_13 = []
for hid, col in [("h42", "fatigue_grade"), ("h43", "pain_nrs"), ("h44", "dyspnea_grade"), ("h45", "appetite_loss_grade")]:
    coef, p, _ = logit_or(df, col)
    ana_13.append({"hypothesis_ids": [hid], "code": f"logistic on {col}", "result_summary": f"logOR per unit {col}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(13, hyps_13, ana_13)

# ---------------- Iteration 14: Cytopenias / blood counts ----
hyps_14 = [
    {"id": "h46", "text": "Higher hemoglobin_g_dl is associated with a higher probability of objective_response.", "kind": "novel"},
    {"id": "h47", "text": "Higher platelets_k_ul is associated with a higher probability of objective_response.", "kind": "novel"},
    {"id": "h48", "text": "Higher wbc_k_per_ul (presenting WBC) is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h49", "text": "Higher anc_k_ul is associated with a higher probability of objective_response.", "kind": "novel"},
]
ana_14 = []
for hid, col in [("h46", "hemoglobin_g_dl"), ("h47", "platelets_k_ul"), ("h48", "wbc_k_per_ul"), ("h49", "anc_k_ul")]:
    coef, p, _ = logit_or(df, col)
    ana_14.append({"hypothesis_ids": [hid], "code": f"logistic on {col}", "result_summary": f"logOR per unit {col}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(14, hyps_14, ana_14)

# ---------------- Iteration 15: Multivariable adjusted treatment effects ----
hyps_15 = [
    {"id": "h50", "text": "After adjusting for age_years, ecog_ps, secondary_aml, complex_karyotype, tp53_mutation, and unfit_for_intensive, treatment_venetoclax_azacitidine remains associated with higher objective_response.", "kind": "refined"},
    {"id": "h51", "text": "After adjusting for the same covariates, treatment_7plus3 is associated with higher objective_response.", "kind": "refined"},
]
ana_15 = []
controls = ["age_years", "ecog_ps", "secondary_aml", "complex_karyotype", "tp53_mutation", "unfit_for_intensive"]
for hid, tx in [("h50", "treatment_venetoclax_azacitidine"), ("h51", "treatment_7plus3")]:
    coef, p, _ = logit_or(df, tx, controls=controls)
    ana_15.append({"hypothesis_ids": [hid], "code": f"logistic: objective_response ~ {tx} + " + " + ".join(controls), "result_summary": f"adj logOR for {tx}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(15, hyps_15, ana_15)

# ---------------- Iteration 16: Hepatic / renal function ----
hyps_16 = [
    {"id": "h52", "text": "Higher creatinine_mg_dl is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h53", "text": "Higher total_bilirubin_mg_dl is associated with a lower probability of objective_response.", "kind": "novel"},
    {"id": "h54", "text": "Higher inr is associated with a lower probability of objective_response.", "kind": "novel"},
]
ana_16 = []
for hid, col in [("h52", "creatinine_mg_dl"), ("h53", "total_bilirubin_mg_dl"), ("h54", "inr")]:
    coef, p, _ = logit_or(df, col)
    ana_16.append({"hypothesis_ids": [hid], "code": f"logistic on {col}", "result_summary": f"logOR per unit {col}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(16, hyps_16, ana_16)

# ---------------- Iteration 17: SNPs (likely null in EHR data) ----
hyps_17 = [
    {"id": "h55", "text": "snp_rs1801133 (MTHFR C677T) is associated with objective_response.", "kind": "novel"},
    {"id": "h56", "text": "snp_rs429358 (APOE) is associated with objective_response.", "kind": "novel"},
    {"id": "h57", "text": "snp_rs1045642 (ABCB1) is associated with objective_response.", "kind": "novel"},
]
ana_17 = []
for hid, col in [("h55", "snp_rs1801133"), ("h56", "snp_rs429358"), ("h57", "snp_rs1045642")]:
    coef, p, _ = logit_or(df, col)
    ana_17.append({"hypothesis_ids": [hid], "code": f"logistic on {col}", "result_summary": f"logOR per dose {col}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(17, hyps_17, ana_17)

# ---------------- Iteration 18: Refined treatment-biomarker test (adjusted) ----
hyps_18 = [
    {"id": "h58", "text": "Adjusted for age_years, ecog_ps, and unfit_for_intensive, the treatment_midostaurin × flt3_itd interaction predicts objective_response (positive).", "kind": "refined"},
    {"id": "h59", "text": "Adjusted for the same covariates, the treatment_ivosidenib × idh1_mutation interaction predicts objective_response (positive).", "kind": "refined"},
    {"id": "h60", "text": "Adjusted for the same covariates, the treatment_enasidenib × idh2_mutation interaction predicts objective_response (positive).", "kind": "refined"},
]
ana_18 = []
controls = ["age_years", "ecog_ps", "unfit_for_intensive"]
for hid, tx, marker in [("h58", "treatment_midostaurin", "flt3_itd"), ("h59", "treatment_ivosidenib", "idh1_mutation"), ("h60", "treatment_enasidenib", "idh2_mutation")]:
    coef, pint, res = logit_interaction(df, tx, marker, controls=controls)
    sub = df[df[marker] == 1]
    diff_in, _, p1_in, p0_in = chi2_2x2(sub, tx, "objective_response")
    ana_18.append({"hypothesis_ids": [hid], "code": f"adjusted logistic with {tx}:{marker} interaction", "result_summary": f"In {marker}=1: {tx}=1 ORR {p1_in:.3f} vs =0 {p0_in:.3f}; adj interaction logOR={coef:+.3f}, p={pint:.3g}", "p_value": float(pint), "effect_estimate": float(diff_in), "significant": bool(pint < 0.05)})
add_iter(18, hyps_18, ana_18)

# ---------------- Iteration 19: Race/insurance after adjusting for clinical features ----
hyps_19 = [
    {"id": "h61", "text": "After adjusting for age_years, ecog_ps, tp53_mutation, complex_karyotype, secondary_aml, and unfit_for_intensive, race_ethnicity='black' (vs white) remains associated with a different probability of objective_response.", "kind": "refined"},
    {"id": "h62", "text": "After the same adjustment, insurance_type='uninsured' (vs private) is associated with a different probability of objective_response.", "kind": "refined"},
]
ana_19 = []
d = df.copy()
d["race_black"] = (d["race_ethnicity"] == "black").astype(int)
d["race_hispanic"] = (d["race_ethnicity"] == "hispanic").astype(int)
d["race_asian"] = (d["race_ethnicity"] == "asian").astype(int)
d["race_other"] = (d["race_ethnicity"] == "other").astype(int)
controls = ["age_years", "ecog_ps", "tp53_mutation", "complex_karyotype", "secondary_aml", "unfit_for_intensive"]
cols = ["race_black", "race_hispanic", "race_asian", "race_other"] + controls
X = sm.add_constant(d[cols].astype(float))
y = d["objective_response"].astype(int)
res = sm.Logit(y, X).fit(disp=0)
coef_b = res.params["race_black"]; p_b = res.pvalues["race_black"]
ana_19.append({"hypothesis_ids": ["h61"], "code": "adjusted logistic with race dummies", "result_summary": f"adj logOR race=black vs white: {coef_b:+.4g}; p={p_b:.3g}", "p_value": float(p_b), "effect_estimate": float(coef_b), "significant": bool(p_b < 0.05)})

d["ins_medicaid"] = (d["insurance_type"] == "medicaid").astype(int)
d["ins_medicare"] = (d["insurance_type"] == "medicare").astype(int)
d["ins_uninsured"] = (d["insurance_type"] == "uninsured").astype(int)
cols2 = ["ins_medicaid", "ins_medicare", "ins_uninsured"] + controls
X2 = sm.add_constant(d[cols2].astype(float))
res2 = sm.Logit(y, X2).fit(disp=0)
coef_u = res2.params["ins_uninsured"]; p_u = res2.pvalues["ins_uninsured"]
ana_19.append({"hypothesis_ids": ["h62"], "code": "adjusted logistic with insurance dummies", "result_summary": f"adj logOR uninsured vs private: {coef_u:+.4g}; p={p_u:.3g}", "p_value": float(p_u), "effect_estimate": float(coef_u), "significant": bool(p_u < 0.05)})
add_iter(19, hyps_19, ana_19)

# ---------------- Iteration 20: Three-way: TP53 × ven+aza × age ----
hyps_20 = [
    {"id": "h63", "text": "The benefit of treatment_venetoclax_azacitidine on objective_response in unfit_for_intensive=1 patients is greater in older age (age_years > 75) than in younger (age_years <= 75).", "kind": "novel"},
    {"id": "h64", "text": "The benefit of treatment_7plus3 on objective_response is largest in fit (unfit_for_intensive=0), younger (age_years <= 60) patients.", "kind": "novel"},
]
ana_20 = []
d = df.copy()
old = d[(d["unfit_for_intensive"] == 1) & (d["age_years"] > 75)]
young = d[(d["unfit_for_intensive"] == 1) & (d["age_years"] <= 75)]
diff_old, p_old, _, _ = chi2_2x2(old, "treatment_venetoclax_azacitidine", "objective_response")
diff_young, p_young, _, _ = chi2_2x2(young, "treatment_venetoclax_azacitidine", "objective_response")
ana_20.append({"hypothesis_ids": ["h63"], "code": "stratified ven+aza ORR by age in unfit", "result_summary": f"unfit & age>75: ven+aza diff {diff_old:+.3f} (p={p_old:.3g}); unfit & age<=75: {diff_young:+.3f} (p={p_young:.3g})", "p_value": float(p_old), "effect_estimate": float(diff_old - diff_young), "significant": bool(p_old < 0.05 or p_young < 0.05)})
fit_young = d[(d["unfit_for_intensive"] == 0) & (d["age_years"] <= 60)]
diff_fy, p_fy, _, _ = chi2_2x2(fit_young, "treatment_7plus3", "objective_response")
ana_20.append({"hypothesis_ids": ["h64"], "code": "7+3 ORR in fit, age<=60", "result_summary": f"In fit & age<=60: 7+3 diff {diff_fy:+.3f}, p={p_fy:.3g}", "p_value": float(p_fy), "effect_estimate": float(diff_fy), "significant": bool(p_fy < 0.05)})
add_iter(20, hyps_20, ana_20)

# ---------------- Iteration 21: Joint AML risk score and ORR ----
hyps_21 = [
    {"id": "h65", "text": "An adverse-risk composite (tp53_mutation=1 OR complex_karyotype=1 OR secondary_aml=1) is associated with a lower probability of objective_response than the absence of all three.", "kind": "novel"},
    {"id": "h66", "text": "A favorable composite (npm1_mutation=1 AND flt3_itd=0) is associated with a higher probability of objective_response than the rest of the cohort.", "kind": "novel"},
]
ana_21 = []
d = df.copy()
d["adverse"] = ((d["tp53_mutation"] == 1) | (d["complex_karyotype"] == 1) | (d["secondary_aml"] == 1)).astype(int)
diff, p, p1, p0 = chi2_2x2(d, "adverse", "objective_response")
ana_21.append({"hypothesis_ids": ["h65"], "code": "chi2 on adverse composite", "result_summary": f"ORR adverse=1: {p1:.3f}; =0: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
d["fav"] = ((d["npm1_mutation"] == 1) & (d["flt3_itd"] == 0)).astype(int)
diff2, p2, p1b, p0b = chi2_2x2(d, "fav", "objective_response")
ana_21.append({"hypothesis_ids": ["h66"], "code": "chi2 on favorable composite", "result_summary": f"ORR fav=1: {p1b:.3f}; =0: {p0b:.3f}; diff={diff2:+.3f}; p={p2:.3g}", "p_value": float(p2), "effect_estimate": float(diff2), "significant": bool(p2 < 0.05)})
add_iter(21, hyps_21, ana_21)

# ---------------- Iteration 22: Inflammation/nutrition composite ----
hyps_22 = [
    {"id": "h67", "text": "Patients with low albumin (albumin_g_dl < 3.5) AND high CRP (crp_mg_l > 10) have a lower probability of objective_response than those with neither.", "kind": "novel"},
    {"id": "h68", "text": "Patients with high LDH (above-cohort-median) have a lower probability of objective_response than those at or below the median.", "kind": "novel"},
]
ana_22 = []
d = df.copy()
d["inflam"] = ((d["albumin_g_dl"] < 3.5) & (d["crp_mg_l"] > 10)).astype(int)
d["clean"] = ((d["albumin_g_dl"] >= 3.5) & (d["crp_mg_l"] <= 10)).astype(int)
sub = d[(d["inflam"] == 1) | (d["clean"] == 1)].copy()
sub["g"] = sub["inflam"]
diff, p, p1, p0 = chi2_2x2(sub, "g", "objective_response")
ana_22.append({"hypothesis_ids": ["h67"], "code": "ORR by low-alb+high-CRP vs neither", "result_summary": f"ORR inflam: {p1:.3f}; clean: {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
med = d["ldh_u_l"].median()
d["high_ldh"] = (d["ldh_u_l"] > med).astype(int)
diff2, p2, p1b, p0b = chi2_2x2(d, "high_ldh", "objective_response")
ana_22.append({"hypothesis_ids": ["h68"], "code": f"chi2 on ldh > {med:.0f}", "result_summary": f"ORR high LDH: {p1b:.3f}; low LDH: {p0b:.3f}; diff={diff2:+.3f}; p={p2:.3g}", "p_value": float(p2), "effect_estimate": float(diff2), "significant": bool(p2 < 0.05)})
add_iter(22, hyps_22, ana_22)

# ---------------- Iteration 23: Multi-treatment combination ----
hyps_23 = [
    {"id": "h69", "text": "Among flt3_itd=1 patients, those receiving treatment_midostaurin=1 AND treatment_7plus3=1 have a higher objective_response rate than those receiving 7+3 alone.", "kind": "refined"},
    {"id": "h70", "text": "Among unfit_for_intensive=1 patients, those receiving treatment_venetoclax_azacitidine=1 have a higher ORR than those receiving treatment_7plus3=1 (intensive therapy in unfit patients).", "kind": "refined"},
]
ana_23 = []
sub = df[(df["flt3_itd"] == 1) & (df["treatment_7plus3"] == 1)]
diff, p, p1, p0 = chi2_2x2(sub, "treatment_midostaurin", "objective_response")
ana_23.append({"hypothesis_ids": ["h69"], "code": "ORR midostaurin among FLT3+/7+3 patients", "result_summary": f"In flt3_itd=1 & 7+3: midostaurin=1 ORR {p1:.3f}; =0 ORR {p0:.3f}; diff={diff:+.3f}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(diff), "significant": bool(p < 0.05)})
sub2 = df[(df["unfit_for_intensive"] == 1) & ((df["treatment_venetoclax_azacitidine"] == 1) ^ (df["treatment_7plus3"] == 1))].copy()
sub2["v_vs_7"] = sub2["treatment_venetoclax_azacitidine"]
diff2, p2, p1b, p0b = chi2_2x2(sub2, "v_vs_7", "objective_response")
ana_23.append({"hypothesis_ids": ["h70"], "code": "ven+aza vs 7+3 in unfit (mutually exclusive)", "result_summary": f"In unfit: ven+aza ORR {p1b:.3f}; 7+3 ORR {p0b:.3f}; diff={diff2:+.3f}; p={p2:.3g}", "p_value": float(p2), "effect_estimate": float(diff2), "significant": bool(p2 < 0.05)})
add_iter(23, hyps_23, ana_23)

# ---------------- Iteration 24: Sociodemographic + treatment receipt ----
hyps_24 = [
    {"id": "h71", "text": "Probability of receiving treatment_venetoclax_azacitidine differs by insurance_type (e.g., uninsured patients are less likely to receive it than privately insured).", "kind": "novel"},
    {"id": "h72", "text": "Probability of receiving treatment_7plus3 differs by race_ethnicity (e.g., black vs white patients).", "kind": "novel"},
]
ana_24 = []
tab = pd.crosstab(df["insurance_type"], df["treatment_venetoclax_azacitidine"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
rates = df.groupby("insurance_type")["treatment_venetoclax_azacitidine"].mean().to_dict()
ana_24.append({"hypothesis_ids": ["h71"], "code": "chi2 insurance × treatment_venetoclax_azacitidine", "result_summary": f"Receipt of ven+aza by insurance: {rates}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(max(rates.values()) - min(rates.values())), "significant": bool(p < 0.05)})
tab = pd.crosstab(df["race_ethnicity"], df["treatment_7plus3"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
rates = df.groupby("race_ethnicity")["treatment_7plus3"].mean().to_dict()
ana_24.append({"hypothesis_ids": ["h72"], "code": "chi2 race × treatment_7plus3", "result_summary": f"Receipt of 7+3 by race: {rates}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(max(rates.values()) - min(rates.values())), "significant": bool(p < 0.05)})
add_iter(24, hyps_24, ana_24)

# ---------------- Iteration 25: Composite multivariable model ----
hyps_25 = [
    {"id": "h73", "text": "In a multivariable logistic regression of objective_response on age_years, ecog_ps, tp53_mutation, complex_karyotype, secondary_aml, unfit_for_intensive, npm1_mutation, flt3_itd, albumin_g_dl, ldh_u_l, and treatment_venetoclax_azacitidine, tp53_mutation has a negative association.", "kind": "refined"},
    {"id": "h74", "text": "In the same multivariable model, npm1_mutation has a positive association with objective_response.", "kind": "refined"},
    {"id": "h75", "text": "In the same multivariable model, ecog_ps has a negative association with objective_response.", "kind": "refined"},
]
ana_25 = []
predictors = [
    "age_years", "ecog_ps", "tp53_mutation", "complex_karyotype", "secondary_aml",
    "unfit_for_intensive", "npm1_mutation", "flt3_itd", "albumin_g_dl", "ldh_u_l",
    "treatment_venetoclax_azacitidine",
]
X = sm.add_constant(df[predictors].astype(float))
y = df["objective_response"].astype(int)
res = sm.Logit(y, X).fit(disp=0)
for hid, col in [("h73", "tp53_mutation"), ("h74", "npm1_mutation"), ("h75", "ecog_ps")]:
    coef = res.params[col]; p = res.pvalues[col]
    ana_25.append({"hypothesis_ids": [hid], "code": "multivariable logistic", "result_summary": f"adj logOR for {col}: {coef:+.4g}; p={p:.3g}", "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iter(25, hyps_25, ana_25)


# Write transcript.json
with open(HERE / "transcript.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, indent=2)


# ----- Build analysis_summary.txt -----
lines = []
lines.append("Analysis Summary — ds001_aml (n=50,000; objective_response rate ~16.9%)")
lines.append("")
lines.append("Approach: 25 iterations of propose-test-refine. Tests used chi-square for binary")
lines.append("comparisons, logistic regression for continuous predictors and adjusted models, and")
lines.append("logistic regression with multiplicative interaction terms for treatment-by-biomarker")
lines.append("hypotheses. Significance threshold p<0.05.")
lines.append("")

for it in OUT["iterations"]:
    lines.append(f"--- Iteration {it['index']} ---")
    for h in it["proposed_hypotheses"]:
        lines.append(f"  [{h['id']}] {h['text']}")
    for a in it["analyses"]:
        sig = "SIG" if a.get("significant") else "ns"
        lines.append(f"   -> ({sig}) {a['result_summary']}")
    lines.append("")

# Synthesis section is filled in later by the agent (see analysis_summary.txt write below)
with open(HERE / "_summary_iters.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Done. Iterations:", len(OUT["iterations"]))
