"""Iterative analysis of ds001_nsclc dataset.

Runs 25 iterations of propose-test-refine and writes:
- transcript.json
- analysis_summary.txt
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
N = len(DF)
ORR = DF["objective_response"].mean()

iterations = []  # transcript iterations


def add_iteration(index, hypotheses, analyses):
    iterations.append({
        "index": index,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def hyp(hid, text, kind="novel"):
    return {"id": hid, "text": text, "kind": kind}


def fisher_or(df, group_col, outcome_col="objective_response"):
    tab = pd.crosstab(df[group_col], df[outcome_col])
    if tab.shape != (2, 2):
        return None
    a = tab.loc[1, 1] if 1 in tab.index else 0
    b = tab.loc[1, 0] if 1 in tab.index else 0
    c = tab.loc[0, 1] if 0 in tab.index else 0
    d = tab.loc[0, 0] if 0 in tab.index else 0
    odds_ratio, p = stats.fisher_exact([[a, b], [c, d]])
    rate1 = a / (a + b) if (a + b) else float("nan")
    rate0 = c / (c + d) if (c + d) else float("nan")
    return odds_ratio, p, rate1, rate0


def logit_fit(df, formula):
    return smf.logit(formula, data=df).fit(disp=0)


def signed_lr_test(model_full, model_reduced):
    lr = 2 * (model_full.llf - model_reduced.llf)
    df_diff = model_full.df_model - model_reduced.df_model
    p = stats.chi2.sf(lr, df_diff)
    return lr, p


# ------------------------------------------------------------------
# Iter 1 — main treatment effects
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h1.1", "Among all patients, treatment_pembrolizumab is associated with a higher rate of objective_response than no pembrolizumab."),
    hyp("h1.2", "Among all patients, treatment_sotorasib is associated with a higher rate of objective_response than no sotorasib."),
    hyp("h1.3", "Among all patients, treatment_olaparib is associated with a higher rate of objective_response than no olaparib."),
    hyp("h1.4", "Among all patients, treatment_osimertinib is associated with a higher rate of objective_response than no osimertinib."),
]
for hid, col in zip(["h1.1", "h1.2", "h1.3", "h1.4"],
                    ["treatment_pembrolizumab", "treatment_sotorasib", "treatment_olaparib", "treatment_osimertinib"]):
    odds, p, r1, r0 = fisher_or(DF, col)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"fisher_exact: objective_response by {col}",
        "result_summary": f"ORR {r1:.4f} on {col}=1 vs {r0:.4f} on {col}=0; OR={odds:.3f}, Fisher p={p:.3g}.",
        "p_value": float(p),
        "effect_estimate": float(r1 - r0),
        "significant": bool(p < 0.05),
    })
add_iteration(1, hyps, analyses)


# ------------------------------------------------------------------
# Iter 2 — osimertinib x egfr
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h2.1", "In patients with egfr_mutation=1, treatment_osimertinib is associated with a higher rate of objective_response than no osimertinib."),
    hyp("h2.2", "In patients with egfr_mutation=0, treatment_osimertinib has no benefit on objective_response."),
    hyp("h2.3", "There is a positive multiplicative interaction between treatment_osimertinib and egfr_mutation on objective_response (logit scale)."),
]
sub_pos = DF[DF["egfr_mutation"] == 1]
sub_neg = DF[DF["egfr_mutation"] == 0]
o, p, r1, r0 = fisher_or(sub_pos, "treatment_osimertinib")
analyses.append({"hypothesis_ids": ["h2.1"],
                 "result_summary": f"EGFR+ subset (n={len(sub_pos)}): ORR {r1:.4f} osimertinib vs {r0:.4f} no osimertinib; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
o, p, r1, r0 = fisher_or(sub_neg, "treatment_osimertinib")
analyses.append({"hypothesis_ids": ["h2.2"],
                 "result_summary": f"EGFR- subset (n={len(sub_neg)}): ORR {r1:.4f} osimertinib vs {r0:.4f} no osimertinib; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
m_full = logit_fit(DF, "objective_response ~ treatment_osimertinib * egfr_mutation")
m_red = logit_fit(DF, "objective_response ~ treatment_osimertinib + egfr_mutation")
_, lrp = signed_lr_test(m_full, m_red)
inter_coef = m_full.params["treatment_osimertinib:egfr_mutation"]
analyses.append({"hypothesis_ids": ["h2.3"],
                 "code": "logit objective_response ~ treatment_osimertinib * egfr_mutation",
                 "result_summary": f"Interaction log-OR={inter_coef:.3f}; LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(inter_coef), "significant": bool(lrp < 0.05)})
add_iteration(2, hyps, analyses)


# ------------------------------------------------------------------
# Iter 3 — sotorasib x kras_g12c
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h3.1", "In patients with kras_g12c=1, treatment_sotorasib is associated with a higher rate of objective_response than no sotorasib."),
    hyp("h3.2", "In patients with kras_g12c=0, treatment_sotorasib has no association with objective_response."),
    hyp("h3.3", "There is a positive multiplicative interaction between treatment_sotorasib and kras_g12c on objective_response."),
]
sub_pos = DF[DF["kras_g12c"] == 1]
sub_neg = DF[DF["kras_g12c"] == 0]
o, p, r1, r0 = fisher_or(sub_pos, "treatment_sotorasib")
analyses.append({"hypothesis_ids": ["h3.1"],
                 "result_summary": f"KRAS G12C+ subset (n={len(sub_pos)}): ORR {r1:.4f} sotorasib vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
o, p, r1, r0 = fisher_or(sub_neg, "treatment_sotorasib")
analyses.append({"hypothesis_ids": ["h3.2"],
                 "result_summary": f"KRAS G12C- subset (n={len(sub_neg)}): ORR {r1:.4f} sotorasib vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
m_full = logit_fit(DF, "objective_response ~ treatment_sotorasib * kras_g12c")
m_red = logit_fit(DF, "objective_response ~ treatment_sotorasib + kras_g12c")
_, lrp = signed_lr_test(m_full, m_red)
inter_coef = m_full.params["treatment_sotorasib:kras_g12c"]
analyses.append({"hypothesis_ids": ["h3.3"],
                 "code": "logit objective_response ~ treatment_sotorasib * kras_g12c",
                 "result_summary": f"Interaction log-OR={inter_coef:.3f}; LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(inter_coef), "significant": bool(lrp < 0.05)})
add_iteration(3, hyps, analyses)


# ------------------------------------------------------------------
# Iter 4 — olaparib x brca2
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h4.1", "In patients with brca2_mutation=1, treatment_olaparib is associated with a higher rate of objective_response than no olaparib."),
    hyp("h4.2", "In patients with brca2_mutation=0, treatment_olaparib has no effect on objective_response."),
    hyp("h4.3", "There is a positive multiplicative interaction between treatment_olaparib and brca2_mutation on objective_response."),
]
sub_pos = DF[DF["brca2_mutation"] == 1]
sub_neg = DF[DF["brca2_mutation"] == 0]
o, p, r1, r0 = fisher_or(sub_pos, "treatment_olaparib")
analyses.append({"hypothesis_ids": ["h4.1"],
                 "result_summary": f"BRCA2+ subset (n={len(sub_pos)}): ORR {r1:.4f} olaparib vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
o, p, r1, r0 = fisher_or(sub_neg, "treatment_olaparib")
analyses.append({"hypothesis_ids": ["h4.2"],
                 "result_summary": f"BRCA2- subset (n={len(sub_neg)}): ORR {r1:.4f} olaparib vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
m_full = logit_fit(DF, "objective_response ~ treatment_olaparib * brca2_mutation")
m_red = logit_fit(DF, "objective_response ~ treatment_olaparib + brca2_mutation")
_, lrp = signed_lr_test(m_full, m_red)
inter_coef = m_full.params["treatment_olaparib:brca2_mutation"]
analyses.append({"hypothesis_ids": ["h4.3"],
                 "code": "logit objective_response ~ treatment_olaparib * brca2_mutation",
                 "result_summary": f"Interaction log-OR={inter_coef:.3f}; LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(inter_coef), "significant": bool(lrp < 0.05)})
add_iteration(4, hyps, analyses)


# ------------------------------------------------------------------
# Iter 5 — pembrolizumab x pdl1_tps
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h5.1", "Higher pdl1_tps is associated with higher objective_response rate among pembrolizumab-treated patients."),
    hyp("h5.2", "There is a positive multiplicative interaction between treatment_pembrolizumab and pdl1_tps on objective_response."),
    hyp("h5.3", "Patients with pdl1_tps>=0.5 (high PD-L1) treated with pembrolizumab have higher objective_response than high-PD-L1 patients without pembrolizumab."),
]
sub_treated = DF[DF["treatment_pembrolizumab"] == 1]
m = logit_fit(sub_treated, "objective_response ~ pdl1_tps")
analyses.append({"hypothesis_ids": ["h5.1"],
                 "code": "logit objective_response ~ pdl1_tps in pembrolizumab=1",
                 "result_summary": f"Among pembro-treated (n={len(sub_treated)}): pdl1_tps coef={m.params['pdl1_tps']:.4f}, p={m.pvalues['pdl1_tps']:.3g}.",
                 "p_value": float(m.pvalues['pdl1_tps']), "effect_estimate": float(m.params['pdl1_tps']),
                 "significant": bool(m.pvalues['pdl1_tps'] < 0.05)})
m_full = logit_fit(DF, "objective_response ~ treatment_pembrolizumab * pdl1_tps")
m_red = logit_fit(DF, "objective_response ~ treatment_pembrolizumab + pdl1_tps")
_, lrp = signed_lr_test(m_full, m_red)
inter_coef = m_full.params["treatment_pembrolizumab:pdl1_tps"]
analyses.append({"hypothesis_ids": ["h5.2"],
                 "code": "logit objective_response ~ treatment_pembrolizumab * pdl1_tps",
                 "result_summary": f"Interaction coef={inter_coef:.4f} per unit pdl1_tps; LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(inter_coef), "significant": bool(lrp < 0.05)})
df50 = DF[DF["pdl1_tps"] >= 0.5]
o, p, r1, r0 = fisher_or(df50, "treatment_pembrolizumab")
analyses.append({"hypothesis_ids": ["h5.3"],
                 "result_summary": f"pdl1_tps>=0.5 subset (n={len(df50)}): ORR {r1:.4f} pembro vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(5, hyps, analyses)


# ------------------------------------------------------------------
# Iter 6 — pembrolizumab x tmb_high, x stk11, x keap1
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h6.1", "In tmb_high=1 patients, pembrolizumab yields higher objective_response than non-pembrolizumab; interaction is positive."),
    hyp("h6.2", "In stk11_mutation=1 patients, pembrolizumab benefit on objective_response is reduced (negative interaction) vs stk11_mutation=0."),
    hyp("h6.3", "In keap1_mutation=1 patients, pembrolizumab benefit on objective_response is reduced (negative interaction) vs keap1_mutation=0."),
]
for hid, mod in zip(["h6.1", "h6.2", "h6.3"], ["tmb_high", "stk11_mutation", "keap1_mutation"]):
    sub_pos = DF[DF[mod] == 1]
    sub_neg = DF[DF[mod] == 0]
    o1, p1, r1a, r0a = fisher_or(sub_pos, "treatment_pembrolizumab")
    o0, p0, r1b, r0b = fisher_or(sub_neg, "treatment_pembrolizumab")
    m_full = logit_fit(DF, f"objective_response ~ treatment_pembrolizumab * {mod}")
    m_red = logit_fit(DF, f"objective_response ~ treatment_pembrolizumab + {mod}")
    _, lrp = signed_lr_test(m_full, m_red)
    inter_coef = m_full.params[f"treatment_pembrolizumab:{mod}"]
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"stratified Fisher + logit ~ treatment_pembrolizumab * {mod}",
                     "result_summary": (
                         f"{mod}+ (n={len(sub_pos)}): pembro ORR {r1a:.4f} vs {r0a:.4f}, OR={o1:.3f} p={p1:.3g}. "
                         f"{mod}- (n={len(sub_neg)}): pembro ORR {r1b:.4f} vs {r0b:.4f}, OR={o0:.3f} p={p0:.3g}. "
                         f"Interaction log-OR={inter_coef:.3f}, LR p={lrp:.3g}."
                     ),
                     "p_value": float(lrp), "effect_estimate": float(inter_coef),
                     "significant": bool(lrp < 0.05)})
add_iteration(6, hyps, analyses)


# ------------------------------------------------------------------
# Iter 7 — ECOG, stage_iv, brain mets main effects
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h7.1", "Higher ecog_ps (worse performance status) is associated with lower objective_response rate."),
    hyp("h7.2", "stage_iv=1 (metastatic) is associated with lower objective_response rate than stage_iv=0."),
    hyp("h7.3", "has_brain_mets=1 is associated with lower objective_response rate than has_brain_mets=0."),
]
m = logit_fit(DF, "objective_response ~ ecog_ps")
analyses.append({"hypothesis_ids": ["h7.1"],
                 "code": "logit objective_response ~ ecog_ps",
                 "result_summary": f"ecog_ps log-OR per unit = {m.params['ecog_ps']:.3f}, p={m.pvalues['ecog_ps']:.3g}.",
                 "p_value": float(m.pvalues['ecog_ps']), "effect_estimate": float(m.params['ecog_ps']),
                 "significant": bool(m.pvalues['ecog_ps'] < 0.05)})
o, p, r1, r0 = fisher_or(DF, "stage_iv")
analyses.append({"hypothesis_ids": ["h7.2"],
                 "result_summary": f"stage_iv=1 ORR {r1:.4f} vs stage_iv=0 {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
o, p, r1, r0 = fisher_or(DF, "has_brain_mets")
analyses.append({"hypothesis_ids": ["h7.3"],
                 "result_summary": f"has_brain_mets=1 ORR {r1:.4f} vs has_brain_mets=0 {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(7, hyps, analyses)


# ------------------------------------------------------------------
# Iter 8 — labs / prognostic markers
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h8.1", "Higher albumin_g_dl is associated with higher objective_response rate."),
    hyp("h8.2", "Higher ldh_u_l is associated with lower objective_response rate."),
    hyp("h8.3", "Higher nlr is associated with lower objective_response rate."),
    hyp("h8.4", "Higher crp_mg_l is associated with lower objective_response rate."),
    hyp("h8.5", "Greater weight_loss_pct_6mo is associated with lower objective_response rate."),
]
for hid, var in zip(["h8.1", "h8.2", "h8.3", "h8.4", "h8.5"],
                    ["albumin_g_dl", "ldh_u_l", "nlr", "crp_mg_l", "weight_loss_pct_6mo"]):
    m = logit_fit(DF, f"objective_response ~ {var}")
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"logit objective_response ~ {var}",
                     "result_summary": f"{var} log-OR per unit = {m.params[var]:.4f}, p={m.pvalues[var]:.3g}.",
                     "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
                     "significant": bool(m.pvalues[var] < 0.05)})
add_iteration(8, hyps, analyses)


# ------------------------------------------------------------------
# Iter 9 — histology and smoking
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h9.1", "Squamous histology has different objective_response than adenocarcinoma overall."),
    hyp("h9.2", "Among pembrolizumab-treated patients, ever-smokers have higher objective_response than never-smokers."),
    hyp("h9.3", "There is a positive interaction between treatment_pembrolizumab and ever-smoker status (smoking_status != 'never') on objective_response."),
]
DF2 = DF.copy()
DF2["squamous"] = (DF2["histology"] == "squamous").astype(int)
DF2["ever_smoker"] = (DF2["smoking_status"] != "never").astype(int)
o, p, r1, r0 = fisher_or(DF2, "squamous")
analyses.append({"hypothesis_ids": ["h9.1"],
                 "result_summary": f"squamous ORR {r1:.4f} vs adenocarcinoma {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
sub_treated = DF2[DF2["treatment_pembrolizumab"] == 1]
o, p, r1, r0 = fisher_or(sub_treated, "ever_smoker")
analyses.append({"hypothesis_ids": ["h9.2"],
                 "result_summary": f"Among pembro-treated, ever-smoker ORR {r1:.4f} vs never-smoker {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
m_full = logit_fit(DF2, "objective_response ~ treatment_pembrolizumab * ever_smoker")
m_red = logit_fit(DF2, "objective_response ~ treatment_pembrolizumab + ever_smoker")
_, lrp = signed_lr_test(m_full, m_red)
coef = m_full.params["treatment_pembrolizumab:ever_smoker"]
analyses.append({"hypothesis_ids": ["h9.3"],
                 "code": "logit objective_response ~ treatment_pembrolizumab * ever_smoker",
                 "result_summary": f"Interaction log-OR={coef:.3f}, LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(coef), "significant": bool(lrp < 0.05)})
add_iteration(9, hyps, analyses)


# ------------------------------------------------------------------
# Iter 10 — metastatic site burden
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h10.1", "liver_mets=1 is associated with lower objective_response rate than liver_mets=0."),
    hyp("h10.2", "bone_mets=1 is associated with lower objective_response rate than bone_mets=0."),
    hyp("h10.3", "adrenal_mets=1 is associated with lower objective_response rate than adrenal_mets=0."),
    hyp("h10.4", "pleural_effusion=1 is associated with lower objective_response rate."),
]
for hid, var in zip(["h10.1", "h10.2", "h10.3", "h10.4"],
                    ["liver_mets", "bone_mets", "adrenal_mets", "pleural_effusion"]):
    o, p, r1, r0 = fisher_or(DF, var)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{var}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(10, hyps, analyses)


# ------------------------------------------------------------------
# Iter 11 — actionable mutations effects
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h11.1", "alk_fusion=1 patients have a different objective_response rate than alk_fusion=0."),
    hyp("h11.2", "Within egfr_mutation=1, treatment_osimertinib drives objective_response (lower without osimertinib)."),
    hyp("h11.3", "tp53_mutation=1 is associated with lower objective_response rate than tp53_mutation=0."),
    hyp("h11.4", "msi_high=1 is associated with higher objective_response rate than msi_high=0."),
]
for hid, var in zip(["h11.1", "h11.3", "h11.4"], ["alk_fusion", "tp53_mutation", "msi_high"]):
    o, p, r1, r0 = fisher_or(DF, var)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{var}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
sub_egfr = DF[DF["egfr_mutation"] == 1]
o, p, r1, r0 = fisher_or(sub_egfr, "treatment_osimertinib")
analyses.append({"hypothesis_ids": ["h11.2"],
                 "result_summary": f"Within EGFR+ (n={len(sub_egfr)}): ORR osimertinib {r1:.4f} vs no osimertinib {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(11, hyps, analyses)


# ------------------------------------------------------------------
# Iter 12 — demographics and access
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h12.1", "Higher age_years is negatively associated with objective_response."),
    hyp("h12.2", "sex_female=1 is associated with different objective_response than sex_female=0."),
    hyp("h12.3", "insurance_type is associated with differences in objective_response."),
    hyp("h12.4", "rural_residence=1 is associated with lower objective_response rate."),
    hyp("h12.5", "race_ethnicity is associated with differences in objective_response."),
]
m = logit_fit(DF, "objective_response ~ age_years")
analyses.append({"hypothesis_ids": ["h12.1"],
                 "code": "logit objective_response ~ age_years",
                 "result_summary": f"age_years log-OR per year = {m.params['age_years']:.4f}, p={m.pvalues['age_years']:.3g}.",
                 "p_value": float(m.pvalues['age_years']), "effect_estimate": float(m.params['age_years']),
                 "significant": bool(m.pvalues['age_years'] < 0.05)})
o, p, r1, r0 = fisher_or(DF, "sex_female")
analyses.append({"hypothesis_ids": ["h12.2"],
                 "result_summary": f"sex_female=1 ORR {r1:.4f} vs male {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
ins_table = pd.crosstab(DF["insurance_type"], DF["objective_response"])
chi2, p_ins, dof, _ = stats.chi2_contingency(ins_table)
rates = DF.groupby("insurance_type")["objective_response"].mean().to_dict()
analyses.append({"hypothesis_ids": ["h12.3"],
                 "result_summary": f"ORR by insurance: {rates}. chi2 p={p_ins:.3g}.",
                 "p_value": float(p_ins),
                 "effect_estimate": float(max(rates.values()) - min(rates.values())),
                 "significant": bool(p_ins < 0.05)})
o, p, r1, r0 = fisher_or(DF, "rural_residence")
analyses.append({"hypothesis_ids": ["h12.4"],
                 "result_summary": f"rural_residence=1 ORR {r1:.4f} vs urban {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
race_table = pd.crosstab(DF["race_ethnicity"], DF["objective_response"])
chi2, p_race, dof, _ = stats.chi2_contingency(race_table)
race_rates = DF.groupby("race_ethnicity")["objective_response"].mean().to_dict()
analyses.append({"hypothesis_ids": ["h12.5"],
                 "result_summary": f"ORR by race_ethnicity: {race_rates}. chi2 p={p_race:.3g}.",
                 "p_value": float(p_race),
                 "effect_estimate": float(max(race_rates.values()) - min(race_rates.values())),
                 "significant": bool(p_race < 0.05)})
add_iteration(12, hyps, analyses)


# ------------------------------------------------------------------
# Iter 13 — comorbidities
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h13.1", "interstitial_lung_disease_history=1 is associated with lower objective_response."),
    hyp("h13.2", "autoimmune_disease=1 is associated with different objective_response."),
    hyp("h13.3", "chronic_kidney_disease=1 is associated with lower objective_response."),
    hyp("h13.4", "depression_anxiety_diagnosis=1 is associated with lower objective_response."),
    hyp("h13.5", "copd=1 is associated with lower objective_response."),
]
for hid, var in zip(["h13.1", "h13.2", "h13.3", "h13.4", "h13.5"],
                    ["interstitial_lung_disease_history", "autoimmune_disease",
                     "chronic_kidney_disease", "depression_anxiety_diagnosis", "copd"]):
    o, p, r1, r0 = fisher_or(DF, var)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{var}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(13, hyps, analyses)


# ------------------------------------------------------------------
# Iter 14 — symptom burden
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h14.1", "Higher fatigue_grade is associated with lower objective_response."),
    hyp("h14.2", "Higher pain_nrs is associated with lower objective_response."),
    hyp("h14.3", "Higher dyspnea_grade is associated with lower objective_response."),
    hyp("h14.4", "Higher appetite_loss_grade is associated with lower objective_response."),
]
for hid, var in zip(["h14.1", "h14.2", "h14.3", "h14.4"],
                    ["fatigue_grade", "pain_nrs", "dyspnea_grade", "appetite_loss_grade"]):
    m = logit_fit(DF, f"objective_response ~ {var}")
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"logit objective_response ~ {var}",
                     "result_summary": f"{var} log-OR per unit = {m.params[var]:.4f}, p={m.pvalues[var]:.3g}.",
                     "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
                     "significant": bool(m.pvalues[var] < 0.05)})
add_iteration(14, hyps, analyses)


# ------------------------------------------------------------------
# Iter 15 — prior therapy
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h15.1", "More prior_lines_of_therapy is associated with lower objective_response."),
    hyp("h15.2", "prior_immunotherapy=1 is associated with lower objective_response."),
    hyp("h15.3", "prior_chemotherapy=1 is associated with lower objective_response."),
    hyp("h15.4", "prior_targeted_therapy=1 is associated with lower objective_response."),
]
m = logit_fit(DF, "objective_response ~ prior_lines_of_therapy")
analyses.append({"hypothesis_ids": ["h15.1"],
                 "code": "logit objective_response ~ prior_lines_of_therapy",
                 "result_summary": f"prior_lines_of_therapy log-OR per line = {m.params['prior_lines_of_therapy']:.4f}, p={m.pvalues['prior_lines_of_therapy']:.3g}.",
                 "p_value": float(m.pvalues['prior_lines_of_therapy']),
                 "effect_estimate": float(m.params['prior_lines_of_therapy']),
                 "significant": bool(m.pvalues['prior_lines_of_therapy'] < 0.05)})
for hid, var in zip(["h15.2", "h15.3", "h15.4"],
                    ["prior_immunotherapy", "prior_chemotherapy", "prior_targeted_therapy"]):
    o, p, r1, r0 = fisher_or(DF, var)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{var}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(15, hyps, analyses)


# ------------------------------------------------------------------
# Iter 16 — adjusted biomarker x treatment interactions
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h16.1", "After adjustment for ECOG, stage_iv, brain mets, age, sex, albumin, LDH, NLR, CRP, the EGFR+ x osimertinib interaction remains positive and significant."),
    hyp("h16.2", "After the same adjustment, KRAS G12C+ x sotorasib interaction remains positive and significant."),
    hyp("h16.3", "After the same adjustment, BRCA2+ x olaparib interaction remains positive and significant."),
    hyp("h16.4", "After the same adjustment, PD-L1 x pembrolizumab interaction remains positive and significant."),
]
covar = ("ecog_ps + stage_iv + has_brain_mets + age_years + sex_female + "
         "albumin_g_dl + ldh_u_l + nlr + crp_mg_l")
for hid, txt in [
    ("h16.1", "treatment_osimertinib * egfr_mutation"),
    ("h16.2", "treatment_sotorasib * kras_g12c"),
    ("h16.3", "treatment_olaparib * brca2_mutation"),
    ("h16.4", "treatment_pembrolizumab * pdl1_tps"),
]:
    formula = f"objective_response ~ {txt} + {covar}"
    m = logit_fit(DF, formula)
    inter = txt.replace(" * ", ":")
    p = m.pvalues[inter]; coef = m.params[inter]
    analyses.append({"hypothesis_ids": [hid],
                     "code": "logit " + formula,
                     "result_summary": f"Adjusted interaction {inter} log-OR={coef:.4f}, Wald p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
add_iteration(16, hyps, analyses)


# ------------------------------------------------------------------
# Iter 17 — ECOG-stratified pembro effect; PDL1<1 pembro effect
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h17.1", "There is a positive 3-way interaction (treatment_pembrolizumab x tmb_high x pdl1_tps) on objective_response, indicating that pembrolizumab benefit is greatest when both pdl1 and tmb are favorable."),
    hyp("h17.2", "In ecog_ps=2 patients, pembrolizumab benefit is smaller than in ecog_ps=0 patients (negative interaction)."),
    hyp("h17.3", "Among low PD-L1 (pdl1_tps<0.01) patients, pembrolizumab has no benefit on objective_response."),
]
m_full = logit_fit(DF, "objective_response ~ treatment_pembrolizumab * tmb_high * pdl1_tps")
m_red = logit_fit(DF, "objective_response ~ treatment_pembrolizumab + tmb_high + pdl1_tps + treatment_pembrolizumab:tmb_high + treatment_pembrolizumab:pdl1_tps + tmb_high:pdl1_tps")
_, lrp = signed_lr_test(m_full, m_red)
three = m_full.params.get("treatment_pembrolizumab:tmb_high:pdl1_tps", float('nan'))
analyses.append({"hypothesis_ids": ["h17.1"],
                 "code": "logit ~ treatment_pembrolizumab*tmb_high*pdl1_tps vs without 3-way",
                 "result_summary": f"3-way interaction coef={three:.4f}; LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(three),
                 "significant": bool(lrp < 0.05)})
sub2 = DF[DF["ecog_ps"] == 2]
sub0 = DF[DF["ecog_ps"] == 0]
o2, p2, r1_2, r0_2 = fisher_or(sub2, "treatment_pembrolizumab")
o0, p0, r1_0, r0_0 = fisher_or(sub0, "treatment_pembrolizumab")
analyses.append({"hypothesis_ids": ["h17.2"],
                 "result_summary": (f"ECOG=2 (n={len(sub2)}): pembro ORR {r1_2:.4f} vs {r0_2:.4f}, OR={o2:.3f}, p={p2:.3g}. "
                                    f"ECOG=0 (n={len(sub0)}): pembro ORR {r1_0:.4f} vs {r0_0:.4f}, OR={o0:.3f}, p={p0:.3g}."),
                 "p_value": float(p2), "effect_estimate": float((r1_2 - r0_2) - (r1_0 - r0_0)),
                 "significant": bool(p2 < 0.05 or p0 < 0.05)})
sub_neg_pdl1 = DF[DF["pdl1_tps"] < 0.01]
o, p, r1, r0 = fisher_or(sub_neg_pdl1, "treatment_pembrolizumab")
analyses.append({"hypothesis_ids": ["h17.3"],
                 "result_summary": f"pdl1_tps<0.01 (n={len(sub_neg_pdl1)}): pembro ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(17, hyps, analyses)


# ------------------------------------------------------------------
# Iter 18 — Other actionable molecular alterations and ORR
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h18.1", "alk_fusion=1 patients have higher objective_response than alk_fusion=0 patients (refines h11.1)."),
    hyp("h18.2", "ros1_fusion=1 patients have different objective_response than ros1_fusion=0 patients."),
    hyp("h18.3", "braf_v600e=1 patients have different objective_response than braf_v600e=0 patients."),
    hyp("h18.4", "met_exon14_skipping=1 patients have different objective_response than met_exon14_skipping=0."),
    hyp("h18.5", "ret_fusion=1 patients have different objective_response than ret_fusion=0."),
    hyp("h18.6", "ntrk_fusion=1 patients have different objective_response than ntrk_fusion=0."),
    hyp("h18.7", "her2_amplification=1 patients have different objective_response than her2_amplification=0."),
]
for hid, var in zip(["h18.1", "h18.2", "h18.3", "h18.4", "h18.5", "h18.6", "h18.7"],
                    ["alk_fusion", "ros1_fusion", "braf_v600e", "met_exon14_skipping",
                     "ret_fusion", "ntrk_fusion", "her2_amplification"]):
    o, p, r1, r0 = fisher_or(DF, var)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{var}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(18, hyps, analyses)


# ------------------------------------------------------------------
# Iter 19 — tumor markers
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h19.1", "Higher cea_ng_ml is associated with lower objective_response."),
    hyp("h19.2", "Higher ca_125_u_ml is associated with lower objective_response."),
    hyp("h19.3", "psa_ng_ml is not associated with objective_response (cohort is NSCLC)."),
]
for hid, var in zip(["h19.1", "h19.2", "h19.3"], ["cea_ng_ml", "ca_125_u_ml", "psa_ng_ml"]):
    m = logit_fit(DF, f"objective_response ~ {var}")
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"logit objective_response ~ {var}",
                     "result_summary": f"{var} log-OR per unit = {m.params[var]:.5g}, p={m.pvalues[var]:.3g}.",
                     "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
                     "significant": bool(m.pvalues[var] < 0.05)})
add_iteration(19, hyps, analyses)


# ------------------------------------------------------------------
# Iter 20 — SNPs
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h20.1", "None of the genotyped SNPs (snp_*) is individually associated with objective_response after Bonferroni correction."),
    hyp("h20.2", "snp_rs1045642 is not associated with objective_response in the unadjusted univariable test."),
    hyp("h20.3", "snp_rs429358 is not associated with objective_response in the unadjusted univariable test."),
]
snp_cols = [c for c in DF.columns if c.startswith("snp_")]
ps = []
for s in snp_cols:
    m = logit_fit(DF, f"objective_response ~ {s}")
    ps.append(m.pvalues[s])
ps = np.array(ps)
bonf_thresh = 0.05 / len(snp_cols)
n_pass = int((ps < bonf_thresh).sum())
min_p = float(ps.min())
analyses.append({"hypothesis_ids": ["h20.1"],
                 "code": "univariable logit per snp_* with Bonferroni",
                 "result_summary": f"{len(snp_cols)} SNPs tested. Bonferroni threshold {bonf_thresh:.3g}. Min raw p={min_p:.3g}. SNPs passing Bonferroni: {n_pass}.",
                 "p_value": float(min_p), "effect_estimate": float(n_pass), "significant": bool(n_pass > 0)})
for hid, var in [("h20.2", "snp_rs1045642"), ("h20.3", "snp_rs429358")]:
    m = logit_fit(DF, f"objective_response ~ {var}")
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"logit objective_response ~ {var}",
                     "result_summary": f"{var} log-OR per allele/dose = {m.params[var]:.4f}, raw p={m.pvalues[var]:.3g}.",
                     "p_value": float(m.pvalues[var]), "effect_estimate": float(m.params[var]),
                     "significant": bool(m.pvalues[var] < 0.05)})
add_iteration(20, hyps, analyses)


# ------------------------------------------------------------------
# Iter 21 — pembro x histology, x liver_mets, x brain_mets
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h21.1", "There is an interaction between treatment_pembrolizumab and squamous histology on objective_response."),
    hyp("h21.2", "There is a negative interaction between treatment_pembrolizumab and liver_mets on objective_response."),
    hyp("h21.3", "There is a negative interaction between treatment_pembrolizumab and has_brain_mets on objective_response."),
]
DF2 = DF.copy()
DF2["squamous"] = (DF2["histology"] == "squamous").astype(int)
for hid, mod in zip(["h21.1", "h21.2", "h21.3"], ["squamous", "liver_mets", "has_brain_mets"]):
    m_full = logit_fit(DF2, f"objective_response ~ treatment_pembrolizumab * {mod}")
    m_red = logit_fit(DF2, f"objective_response ~ treatment_pembrolizumab + {mod}")
    _, lrp = signed_lr_test(m_full, m_red)
    coef = m_full.params[f"treatment_pembrolizumab:{mod}"]
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"logit objective_response ~ treatment_pembrolizumab * {mod}",
                     "result_summary": f"Interaction (pembro:{mod}) log-OR={coef:.3f}, LR p={lrp:.3g}.",
                     "p_value": float(lrp), "effect_estimate": float(coef), "significant": bool(lrp < 0.05)})
add_iteration(21, hyps, analyses)


# ------------------------------------------------------------------
# Iter 22 — comprehensive multivariable model
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h22.1", "In a comprehensive multivariable logistic regression including all four treatments, key biomarker interactions, and clinical covariates, the treatment_osimertinib:egfr_mutation interaction remains positive and significant."),
    hyp("h22.2", "In the same model, treatment_sotorasib:kras_g12c remains positive and significant."),
    hyp("h22.3", "In the same model, treatment_olaparib:brca2_mutation remains positive and significant."),
    hyp("h22.4", "In the same model, treatment_pembrolizumab:pdl1_tps remains positive and significant."),
    hyp("h22.5", "ecog_ps and stage_iv remain negatively associated with objective_response in the multivariable model."),
]
formula = ("objective_response ~ "
           "treatment_pembrolizumab*pdl1_tps + treatment_pembrolizumab*tmb_high + treatment_pembrolizumab*stk11_mutation + treatment_pembrolizumab*keap1_mutation + "
           "treatment_osimertinib*egfr_mutation + treatment_sotorasib*kras_g12c + treatment_olaparib*brca2_mutation + "
           "ecog_ps + stage_iv + has_brain_mets + age_years + sex_female + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + "
           "liver_mets + bone_mets + tp53_mutation")
m = logit_fit(DF, formula)
for hid, term in [
    ("h22.1", "treatment_osimertinib:egfr_mutation"),
    ("h22.2", "treatment_sotorasib:kras_g12c"),
    ("h22.3", "treatment_olaparib:brca2_mutation"),
    ("h22.4", "treatment_pembrolizumab:pdl1_tps"),
]:
    p = m.pvalues[term]; coef = m.params[term]
    analyses.append({"hypothesis_ids": [hid],
                     "code": "comprehensive multivariable logit",
                     "result_summary": f"Adjusted {term} log-OR={coef:.4f}, Wald p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(coef), "significant": bool(p < 0.05)})
ecog_p = m.pvalues["ecog_ps"]; ecog_c = m.params["ecog_ps"]
stage_p = m.pvalues["stage_iv"]; stage_c = m.params["stage_iv"]
analyses.append({"hypothesis_ids": ["h22.5"],
                 "code": "comprehensive multivariable logit",
                 "result_summary": f"Adjusted ecog_ps log-OR={ecog_c:.4f} (p={ecog_p:.3g}); stage_iv log-OR={stage_c:.4f} (p={stage_p:.3g}).",
                 "p_value": float(min(ecog_p, stage_p)), "effect_estimate": float(ecog_c),
                 "significant": bool((ecog_p < 0.05) and (stage_p < 0.05))})
add_iteration(22, hyps, analyses)


# ------------------------------------------------------------------
# Iter 23 — off-target null effects
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h23.1", "Among EGFR-negative patients, treatment_osimertinib has no benefit on objective_response."),
    hyp("h23.2", "Among KRAS G12C-negative patients, treatment_sotorasib has no benefit on objective_response."),
    hyp("h23.3", "Among BRCA2-negative patients, treatment_olaparib has no benefit on objective_response."),
    hyp("h23.4", "Among low PD-L1 (pdl1_tps<0.01) patients, treatment_pembrolizumab has no benefit on objective_response."),
]
for hid, treat, biom_col, biom_val, label in [
    ("h23.1", "treatment_osimertinib", "egfr_mutation", 0, "EGFR- subset"),
    ("h23.2", "treatment_sotorasib", "kras_g12c", 0, "KRAS G12C- subset"),
    ("h23.3", "treatment_olaparib", "brca2_mutation", 0, "BRCA2- subset"),
    ("h23.4", "treatment_pembrolizumab", None, None, "pdl1_tps<0.01 subset"),
]:
    if biom_col is None:
        sub = DF[DF["pdl1_tps"] < 0.01]
    else:
        sub = DF[DF[biom_col] == biom_val]
    o, p, r1, r0 = fisher_or(sub, treat)
    analyses.append({"hypothesis_ids": [hid],
                     "result_summary": f"{label} (n={len(sub)}): {treat}=1 ORR {r1:.4f} vs {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                     "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
add_iteration(23, hyps, analyses)


# ------------------------------------------------------------------
# Iter 24 — smoking pack-years x pembrolizumab
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h24.1", "Among pembrolizumab-treated patients, higher smoking_pack_years is associated with higher objective_response (consistent with TMB-driven response in smokers)."),
    hyp("h24.2", "Among non-pembrolizumab patients, smoking_pack_years has no association with objective_response."),
    hyp("h24.3", "There is a positive interaction between smoking_pack_years and treatment_pembrolizumab on objective_response."),
]
sub_p = DF[DF["treatment_pembrolizumab"] == 1]
sub_n = DF[DF["treatment_pembrolizumab"] == 0]
m_p = logit_fit(sub_p, "objective_response ~ smoking_pack_years")
m_n = logit_fit(sub_n, "objective_response ~ smoking_pack_years")
analyses.append({"hypothesis_ids": ["h24.1"],
                 "code": "logit objective_response ~ smoking_pack_years in pembro=1",
                 "result_summary": f"pembro=1 (n={len(sub_p)}): smoking_pack_years coef={m_p.params['smoking_pack_years']:.5g}, p={m_p.pvalues['smoking_pack_years']:.3g}.",
                 "p_value": float(m_p.pvalues['smoking_pack_years']),
                 "effect_estimate": float(m_p.params['smoking_pack_years']),
                 "significant": bool(m_p.pvalues['smoking_pack_years'] < 0.05)})
analyses.append({"hypothesis_ids": ["h24.2"],
                 "code": "logit objective_response ~ smoking_pack_years in pembro=0",
                 "result_summary": f"pembro=0 (n={len(sub_n)}): smoking_pack_years coef={m_n.params['smoking_pack_years']:.5g}, p={m_n.pvalues['smoking_pack_years']:.3g}.",
                 "p_value": float(m_n.pvalues['smoking_pack_years']),
                 "effect_estimate": float(m_n.params['smoking_pack_years']),
                 "significant": bool(m_n.pvalues['smoking_pack_years'] < 0.05)})
m_full = logit_fit(DF, "objective_response ~ treatment_pembrolizumab * smoking_pack_years")
m_red = logit_fit(DF, "objective_response ~ treatment_pembrolizumab + smoking_pack_years")
_, lrp = signed_lr_test(m_full, m_red)
coef = m_full.params["treatment_pembrolizumab:smoking_pack_years"]
analyses.append({"hypothesis_ids": ["h24.3"],
                 "code": "logit objective_response ~ treatment_pembrolizumab * smoking_pack_years",
                 "result_summary": f"Interaction coef={coef:.5g}, LR p={lrp:.3g}.",
                 "p_value": float(lrp), "effect_estimate": float(coef), "significant": bool(lrp < 0.05)})
add_iteration(24, hyps, analyses)


# ------------------------------------------------------------------
# Iter 25 — race, education, biomarker confounding
# ------------------------------------------------------------------
analyses = []
hyps = [
    hyp("h25.1", "Asian race (race_ethnicity='asian') is associated with higher prevalence of egfr_mutation than non-Asian race."),
    hyp("h25.2", "After adjusting for biomarker status (egfr, kras_g12c, brca2, pdl1_tps, tmb_high), ECOG, and stage, race_ethnicity is not independently associated with objective_response."),
    hyp("h25.3", "education_years is not associated with objective_response after adjustment for ECOG and stage."),
]
DF2 = DF.copy()
DF2["asian"] = (DF2["race_ethnicity"] == "asian").astype(int)
DF2["_y"] = DF2["egfr_mutation"]
o, p, r1, r0 = fisher_or(DF2, "asian", outcome_col="_y")
analyses.append({"hypothesis_ids": ["h25.1"],
                 "code": "fisher_exact: egfr_mutation by asian vs non-asian",
                 "result_summary": f"Asian: EGFR rate {r1:.4f} vs non-Asian {r0:.4f}; OR={o:.3f}, p={p:.3g}.",
                 "p_value": float(p), "effect_estimate": float(r1 - r0), "significant": bool(p < 0.05)})
m_full = logit_fit(DF, "objective_response ~ C(race_ethnicity) + egfr_mutation + kras_g12c + brca2_mutation + pdl1_tps + tmb_high + ecog_ps + stage_iv")
m_red = logit_fit(DF, "objective_response ~ egfr_mutation + kras_g12c + brca2_mutation + pdl1_tps + tmb_high + ecog_ps + stage_iv")
_, lrp = signed_lr_test(m_full, m_red)
race_coefs = m_full.params.filter(like="C(race_ethnicity)")
race_max_diff = float(race_coefs.max() - race_coefs.min()) if len(race_coefs) else 0.0
analyses.append({"hypothesis_ids": ["h25.2"],
                 "code": "logit objective_response ~ race_ethnicity + biomarkers + ecog + stage",
                 "result_summary": f"LR test for race_ethnicity after adjustment p={lrp:.3g}; max log-OR diff among race coefs={race_max_diff:.3f}.",
                 "p_value": float(lrp), "effect_estimate": float(race_max_diff),
                 "significant": bool(lrp < 0.05)})
m = logit_fit(DF, "objective_response ~ education_years + ecog_ps + stage_iv")
analyses.append({"hypothesis_ids": ["h25.3"],
                 "code": "logit objective_response ~ education_years + ecog_ps + stage_iv",
                 "result_summary": f"education_years log-OR per year = {m.params['education_years']:.5g}, p={m.pvalues['education_years']:.3g}.",
                 "p_value": float(m.pvalues['education_years']),
                 "effect_estimate": float(m.params['education_years']),
                 "significant": bool(m.pvalues['education_years'] < 0.05)})
add_iteration(25, hyps, analyses)


# ------------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------------
transcript = {
    "dataset_id": "ds001_nsclc",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@2026-04-28",
    "max_iterations": 25,
    "iterations": iterations,
}
Path("transcript.json").write_text(json.dumps(transcript, indent=2))
print(f"Wrote transcript.json with {len(iterations)} iterations")

# Print results to drive narrative drafting
for it in iterations:
    print(f"--- Iter {it['index']} ---")
    for a in it["analyses"]:
        sig = "*" if a.get("significant") else " "
        print(f"  {sig} {a['hypothesis_ids']} est={a.get('effect_estimate')} p={a.get('p_value')}")
        print(f"      {a['result_summary']}")
