"""
Iterative analysis of ds001_breast.
Builds transcript.json and analysis_summary.txt.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

DF = pd.read_parquet("dataset.parquet")
N = len(DF)
OUT = "pfs_months"

iterations = []


def add_iter(idx, hypotheses, analyses):
    iterations.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def ols_effect(formula, df=DF, term=None):
    m = smf.ols(formula, data=df).fit()
    if term is None:
        term = [k for k in m.params.index if k != "Intercept"][0]
    coef = float(m.params[term])
    p = float(m.pvalues[term])
    return coef, p, int(m.nobs), m


def ttest(group1, group0):
    t, p = stats.ttest_ind(group1, group0, equal_var=False)
    return float(np.mean(group1) - np.mean(group0)), float(p)


def group_means_pfs(mask):
    a = DF.loc[mask, OUT]
    b = DF.loc[~mask, OUT]
    diff, p = ttest(a, b)
    return diff, p, a.mean(), b.mean(), len(a), len(b)


# ============ Iteration 1: treatment main effects ============
hyps_1 = [
    {"id": "h1_tamoxifen", "text": "Patients receiving treatment_tamoxifen have longer pfs_months (mean) than patients not receiving treatment_tamoxifen.", "kind": "novel"},
    {"id": "h1_palbociclib", "text": "Patients receiving treatment_palbociclib have longer pfs_months than patients not receiving treatment_palbociclib.", "kind": "novel"},
    {"id": "h1_trastuzumab", "text": "Patients receiving treatment_trastuzumab have longer pfs_months than patients not receiving treatment_trastuzumab.", "kind": "novel"},
    {"id": "h1_olaparib", "text": "Patients receiving treatment_olaparib have longer pfs_months than patients not receiving treatment_olaparib.", "kind": "novel"},
    {"id": "h1_saci", "text": "Patients receiving treatment_sacituzumab_govitecan have longer pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1_pembro", "text": "Patients receiving treatment_pembrolizumab have longer pfs_months than patients not receiving it.", "kind": "novel"},
]
analyses_1 = []
for tx, hid in [
    ("treatment_tamoxifen", "h1_tamoxifen"),
    ("treatment_palbociclib", "h1_palbociclib"),
    ("treatment_trastuzumab", "h1_trastuzumab"),
    ("treatment_olaparib", "h1_olaparib"),
    ("treatment_sacituzumab_govitecan", "h1_saci"),
    ("treatment_pembrolizumab", "h1_pembro"),
]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[tx] == 1)
    analyses_1.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{tx}']==1,'pfs_months'], df.loc[df['{tx}']==0,'pfs_months'])",
        "result_summary": f"Mean pfs_months {m1:.3f} ({tx}=1, n={n1}) vs {m0:.3f} ({tx}=0, n={n0}); diff={diff:.3f}, Welch t-test p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(1, hyps_1, analyses_1)


# ============ Iteration 2: biomarker main effects ============
hyps_2 = [
    {"id": "h2_er", "text": "ER-positive patients (er_positive=1) have longer pfs_months than ER-negative patients.", "kind": "novel"},
    {"id": "h2_pr", "text": "PR-positive patients (pr_positive=1) have longer pfs_months than PR-negative patients.", "kind": "novel"},
    {"id": "h2_her2", "text": "HER2-positive patients (her2_positive=1) have shorter pfs_months than HER2-negative patients (worse natural history).", "kind": "novel"},
    {"id": "h2_brca1", "text": "BRCA1-mutated patients (brca1_mutation=1) have shorter pfs_months than BRCA1 wild-type.", "kind": "novel"},
    {"id": "h2_brca2", "text": "BRCA2-mutated patients (brca2_mutation=1) have different (likely shorter) pfs_months than BRCA2 wild-type.", "kind": "novel"},
    {"id": "h2_pik3ca", "text": "PIK3CA-mutated patients have different pfs_months from wild-type.", "kind": "novel"},
    {"id": "h2_tp53", "text": "TP53-mutated patients have shorter pfs_months than TP53 wild-type.", "kind": "novel"},
    {"id": "h2_msi", "text": "MSI-high patients have different pfs_months from MSI-low/stable patients.", "kind": "novel"},
]
analyses_2 = []
for col, hid in [
    ("er_positive", "h2_er"),
    ("pr_positive", "h2_pr"),
    ("her2_positive", "h2_her2"),
    ("brca1_mutation", "h2_brca1"),
    ("brca2_mutation", "h2_brca2"),
    ("pik3ca_mutation", "h2_pik3ca"),
    ("tp53_mutation", "h2_tp53"),
    ("msi_high", "h2_msi"),
]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[col] == 1)
    analyses_2.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{col}']==1,'pfs_months'], df.loc[df['{col}']==0,'pfs_months'])",
        "result_summary": f"Mean pfs_months {m1:.3f} ({col}=1, n={n1}) vs {m0:.3f} ({col}=0, n={n0}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
add_iter(2, hyps_2, analyses_2)


# ============ Iteration 3: stage and metastases ============
hyps_3 = [
    {"id": "h3_stage4", "text": "Patients with stage_iv=1 have shorter pfs_months than stage_iv=0.", "kind": "novel"},
    {"id": "h3_brain", "text": "Patients with has_brain_mets=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h3_liver", "text": "Patients with liver_mets=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h3_bone", "text": "Patients with bone_mets=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h3_node", "text": "Patients with node_positive=1 have shorter pfs_months than node-negative.", "kind": "novel"},
    {"id": "h3_ecog", "text": "Higher ecog_ps is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
analyses_3 = []
for col, hid in [
    ("stage_iv", "h3_stage4"),
    ("has_brain_mets", "h3_brain"),
    ("liver_mets", "h3_liver"),
    ("bone_mets", "h3_bone"),
    ("node_positive", "h3_node"),
]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[col] == 1)
    analyses_3.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind grouped by {col}",
        "result_summary": f"Mean pfs_months {m1:.3f} ({col}=1) vs {m0:.3f} ({col}=0); diff={diff:.3f}, p={p:.3g}.",
        "p_value": p,
        "effect_estimate": diff,
        "significant": bool(p < 0.05),
    })
coef, p, n, _ = ols_effect("pfs_months ~ ecog_ps", term="ecog_ps")
analyses_3.append({
    "hypothesis_ids": ["h3_ecog"],
    "code": "smf.ols('pfs_months ~ ecog_ps', df).fit()",
    "result_summary": f"OLS slope of pfs_months on ecog_ps: {coef:.3f} per unit ECOG, p={p:.3g}.",
    "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
})
add_iter(3, hyps_3, analyses_3)


# ============ Iteration 4: demographics ============
hyps_4 = [
    {"id": "h4_age", "text": "Older age_years is associated with shorter pfs_months (negative slope of pfs_months on age_years).", "kind": "novel"},
    {"id": "h4_sex", "text": "Female patients (sex_female=1) have different pfs_months from males in this breast-cancer cohort.", "kind": "novel"},
    {"id": "h4_postmen", "text": "Postmenopausal patients (postmenopausal=1) have different pfs_months from premenopausal patients.", "kind": "novel"},
    {"id": "h4_bmi", "text": "Higher bmi is associated with different pfs_months (slope test).", "kind": "novel"},
]
analyses_4 = []
for term, hid in [("age_years", "h4_age"), ("bmi", "h4_bmi")]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {term}", term=term)
    analyses_4.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {term}', df).fit()",
        "result_summary": f"OLS slope on {term}: {coef:.4f} per unit, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
for col, hid in [("sex_female", "h4_sex"), ("postmenopausal", "h4_postmen")]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[col] == 1)
    analyses_4.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by {col}",
        "result_summary": f"Mean pfs_months {m1:.3f} ({col}=1) vs {m0:.3f} ({col}=0); diff={diff:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
add_iter(4, hyps_4, analyses_4)


# ============ Iteration 5: labs / inflammation / nutrition ============
hyps_5 = [
    {"id": "h5_albumin", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
    {"id": "h5_ldh", "text": "Higher ldh_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h5_nlr", "text": "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h5_crp", "text": "Higher crp_mg_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h5_wtloss", "text": "Greater weight_loss_pct_6mo is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h5_hgb", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months.", "kind": "novel"},
]
analyses_5 = []
for term, hid in [
    ("albumin_g_dl", "h5_albumin"),
    ("ldh_u_l", "h5_ldh"),
    ("nlr", "h5_nlr"),
    ("crp_mg_l", "h5_crp"),
    ("weight_loss_pct_6mo", "h5_wtloss"),
    ("hemoglobin_g_dl", "h5_hgb"),
]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {term}", term=term)
    analyses_5.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {term}', df).fit()",
        "result_summary": f"OLS slope on {term}: {coef:.4f} per unit, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(5, hyps_5, analyses_5)


# ============ Iteration 6: predictive interactions (treatment x biomarker) ============
hyps_6 = [
    {"id": "h6_tam_er", "text": "The pfs_months benefit of treatment_tamoxifen is greater in er_positive=1 patients than in er_positive=0 patients (positive interaction tamoxifen x ER).", "kind": "novel"},
    {"id": "h6_palb_er", "text": "The pfs_months benefit of treatment_palbociclib is greater in er_positive=1 patients than in er_positive=0 (positive interaction palbociclib x ER).", "kind": "novel"},
    {"id": "h6_tras_her2", "text": "The pfs_months benefit of treatment_trastuzumab is greater in her2_positive=1 patients than in her2_positive=0 (positive interaction trastuzumab x HER2).", "kind": "novel"},
]
analyses_6 = []
for tx, bm, hid in [
    ("treatment_tamoxifen", "er_positive", "h6_tam_er"),
    ("treatment_palbociclib", "er_positive", "h6_palb_er"),
    ("treatment_trastuzumab", "her2_positive", "h6_tras_her2"),
]:
    f = f"pfs_months ~ {tx} * {bm}"
    m = smf.ols(f, data=DF).fit()
    interaction = f"{tx}:{bm}"
    coef = float(m.params[interaction])
    p = float(m.pvalues[interaction])
    s11 = DF.loc[(DF[tx] == 1) & (DF[bm] == 1), OUT].mean()
    s10 = DF.loc[(DF[tx] == 1) & (DF[bm] == 0), OUT].mean()
    s01 = DF.loc[(DF[tx] == 0) & (DF[bm] == 1), OUT].mean()
    s00 = DF.loc[(DF[tx] == 0) & (DF[bm] == 0), OUT].mean()
    analyses_6.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('{f}', df).fit() and stratified means",
        "result_summary": (
            f"Interaction {tx} x {bm}: coef={coef:.3f}, p={p:.3g}. "
            f"Stratified mean pfs: tx=1&bm=1:{s11:.3f}, tx=1&bm=0:{s10:.3f}, "
            f"tx=0&bm=1:{s01:.3f}, tx=0&bm=0:{s00:.3f}. "
            f"Within-bm tx effect: bm=1: {s11-s01:.3f}, bm=0: {s10-s00:.3f}."
        ),
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(6, hyps_6, analyses_6)


# ============ Iteration 7: more predictive interactions ============
hyps_7 = [
    {"id": "h7_olap_brca", "text": "The pfs_months benefit of treatment_olaparib is greater in patients with any BRCA mutation (brca1_mutation=1 or brca2_mutation=1) than in BRCA wild-type (positive interaction olaparib x BRCA).", "kind": "novel"},
    {"id": "h7_pembro_msi", "text": "The pfs_months benefit of treatment_pembrolizumab is greater in msi_high=1 patients than in msi_high=0 (positive interaction pembrolizumab x MSI-high).", "kind": "novel"},
    {"id": "h7_saci_her2low", "text": "The pfs_months benefit of treatment_sacituzumab_govitecan differs (likely greater) in her2_low=1 patients vs her2_low=0.", "kind": "novel"},
    {"id": "h7_saci_tnbc", "text": "The pfs_months benefit of treatment_sacituzumab_govitecan is greater in triple-negative patients (er_positive=0 & pr_positive=0 & her2_positive=0) than in non-triple-negative.", "kind": "novel"},
]
analyses_7 = []
DF["brca_any"] = ((DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1)).astype(int)
DF["tnbc"] = ((DF["er_positive"] == 0) & (DF["pr_positive"] == 0) & (DF["her2_positive"] == 0)).astype(int)
for tx, bm, hid in [
    ("treatment_olaparib", "brca_any", "h7_olap_brca"),
    ("treatment_pembrolizumab", "msi_high", "h7_pembro_msi"),
    ("treatment_sacituzumab_govitecan", "her2_low", "h7_saci_her2low"),
    ("treatment_sacituzumab_govitecan", "tnbc", "h7_saci_tnbc"),
]:
    f = f"pfs_months ~ {tx} * {bm}"
    m = smf.ols(f, data=DF).fit()
    interaction = f"{tx}:{bm}"
    coef = float(m.params[interaction]); p = float(m.pvalues[interaction])
    s11 = DF.loc[(DF[tx] == 1) & (DF[bm] == 1), OUT].mean()
    s10 = DF.loc[(DF[tx] == 1) & (DF[bm] == 0), OUT].mean()
    s01 = DF.loc[(DF[tx] == 0) & (DF[bm] == 1), OUT].mean()
    s00 = DF.loc[(DF[tx] == 0) & (DF[bm] == 0), OUT].mean()
    n11 = int(((DF[tx] == 1) & (DF[bm] == 1)).sum())
    analyses_7.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('{f}', df).fit()",
        "result_summary": (
            f"Interaction {tx} x {bm}: coef={coef:.3f}, p={p:.3g}. "
            f"Mean pfs tx=1&bm=1:{s11:.3f} (n={n11}); tx=1&bm=0:{s10:.3f}; "
            f"tx=0&bm=1:{s01:.3f}; tx=0&bm=0:{s00:.3f}. "
            f"Within-bm tx effect: bm=1: {s11-s01:.3f}, bm=0: {s10-s00:.3f}."
        ),
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(7, hyps_7, analyses_7)


# ============ Iteration 8: multivariable model adjusting for prognostic factors ============
hyps_8 = [
    {"id": "h8_mv_palbociclib", "text": "After adjusting for age, ECOG, stage_iv, brain/liver/bone metastases, ER, HER2, BRCA, and labs, treatment_palbociclib retains an independent positive association with pfs_months.", "kind": "refined"},
    {"id": "h8_mv_trastuzumab", "text": "After adjusting for the same covariates, treatment_trastuzumab retains an independent positive association with pfs_months.", "kind": "refined"},
    {"id": "h8_mv_tamoxifen", "text": "After adjusting for the same covariates, treatment_tamoxifen retains an independent positive association with pfs_months.", "kind": "refined"},
    {"id": "h8_mv_pembro", "text": "After adjusting for the same covariates, treatment_pembrolizumab does NOT have a significant positive main-effect coefficient on pfs_months (effect concentrated in MSI-high subgroup).", "kind": "refined"},
]
covars = (
    "age_years + ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + "
    "er_positive + pr_positive + her2_positive + her2_low + brca_any + msi_high + "
    "albumin_g_dl + ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo + hemoglobin_g_dl"
)
treatments = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]
formula_mv = f"pfs_months ~ " + " + ".join(treatments) + " + " + covars
mv = smf.ols(formula_mv, data=DF).fit()
analyses_8 = []
for tx, hid_map in [
    ("treatment_tamoxifen", "h8_mv_tamoxifen"),
    ("treatment_palbociclib", "h8_mv_palbociclib"),
    ("treatment_trastuzumab", "h8_mv_trastuzumab"),
    ("treatment_pembrolizumab", "h8_mv_pembro"),
]:
    coef = float(mv.params[tx]); p = float(mv.pvalues[tx])
    analyses_8.append({
        "hypothesis_ids": [hid_map],
        "code": "smf.ols(pfs_months ~ all treatments + covariates).fit()",
        "result_summary": f"Adjusted coefficient for {tx}: {coef:.3f} months (p={p:.3g}); model adjusts for age, ECOG, stage, mets, ER/PR/HER2/HER2-low/BRCA/MSI, albumin, LDH, NLR, CRP, weight loss, Hgb.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(8, hyps_8, analyses_8)


# ============ Iteration 9: race/ethnicity, insurance, rural, education ============
hyps_9 = [
    {"id": "h9_race", "text": "Mean pfs_months differs across race_ethnicity categories (white, black, hispanic, asian, other).", "kind": "novel"},
    {"id": "h9_insurance", "text": "Mean pfs_months differs across insurance_type categories (private, medicare, medicaid, uninsured).", "kind": "novel"},
    {"id": "h9_rural", "text": "Patients with rural_residence=1 have shorter pfs_months than non-rural patients.", "kind": "novel"},
    {"id": "h9_education", "text": "More years of education_years are associated with longer pfs_months.", "kind": "novel"},
]
analyses_9 = []
# One-way ANOVA via groupby
race_groups = [DF.loc[DF["race_ethnicity"] == r, OUT].values for r in DF["race_ethnicity"].unique()]
f_race, p_race = stats.f_oneway(*race_groups)
race_means = ", ".join(f"{r}={DF.loc[DF['race_ethnicity']==r,OUT].mean():.3f}(n={int((DF['race_ethnicity']==r).sum())})" for r in ['white', 'black', 'hispanic', 'asian', 'other'])
analyses_9.append({
    "hypothesis_ids": ["h9_race"],
    "code": "stats.f_oneway across race_ethnicity groups",
    "result_summary": f"One-way ANOVA pfs_months by race_ethnicity: F={float(f_race):.3f}, p={float(p_race):.3g}. Group means: {race_means}.",
    "p_value": float(p_race),
    "effect_estimate": float(DF.loc[DF['race_ethnicity']=='black', OUT].mean() - DF.loc[DF['race_ethnicity']=='white', OUT].mean()),
    "significant": bool(float(p_race) < 0.05),
})
ins_groups = [DF.loc[DF["insurance_type"] == ins, OUT].values for ins in DF["insurance_type"].unique()]
f_ins, p_ins = stats.f_oneway(*ins_groups)
ins_means = ", ".join(f"{ins}={DF.loc[DF['insurance_type']==ins,OUT].mean():.3f}(n={int((DF['insurance_type']==ins).sum())})" for ins in ['private', 'medicare', 'medicaid', 'uninsured'])
analyses_9.append({
    "hypothesis_ids": ["h9_insurance"],
    "code": "stats.f_oneway across insurance_type groups",
    "result_summary": f"One-way ANOVA pfs_months by insurance_type: F={float(f_ins):.3f}, p={float(p_ins):.3g}. Means: {ins_means}.",
    "p_value": float(p_ins),
    "effect_estimate": float(DF.loc[DF['insurance_type']=='uninsured', OUT].mean() - DF.loc[DF['insurance_type']=='private', OUT].mean()),
    "significant": bool(float(p_ins) < 0.05),
})
diff, p, m1, m0, n1, n0 = group_means_pfs(DF["rural_residence"] == 1)
analyses_9.append({
    "hypothesis_ids": ["h9_rural"],
    "code": "t-test pfs_months by rural_residence",
    "result_summary": f"pfs rural=1: {m1:.3f} (n={n1}) vs rural=0: {m0:.3f} (n={n0}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
coef, p, n, _ = ols_effect("pfs_months ~ education_years", term="education_years")
analyses_9.append({
    "hypothesis_ids": ["h9_education"],
    "code": "smf.ols('pfs_months ~ education_years').fit()",
    "result_summary": f"OLS slope on education_years: {coef:.4f} months per year, p={p:.3g}.",
    "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
})
add_iter(9, hyps_9, analyses_9)


# ============ Iteration 10: comorbidities ============
hyps_10 = [
    {"id": "h10_dm", "text": "Patients with diabetes_mellitus=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h10_htn", "text": "Patients with hypertension=1 have different pfs_months from those without.", "kind": "novel"},
    {"id": "h10_ckd", "text": "Patients with chronic_kidney_disease=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h10_hf", "text": "Patients with heart_failure=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h10_priormalig", "text": "Patients with prior_malignancy=1 have different pfs_months from those without.", "kind": "novel"},
]
analyses_10 = []
for col, hid in [
    ("diabetes_mellitus", "h10_dm"),
    ("hypertension", "h10_htn"),
    ("chronic_kidney_disease", "h10_ckd"),
    ("heart_failure", "h10_hf"),
    ("prior_malignancy", "h10_priormalig"),
]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[col] == 1)
    analyses_10.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by {col}",
        "result_summary": f"pfs {col}=1: {m1:.3f} (n={n1}) vs 0: {m0:.3f} (n={n0}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
add_iter(10, hyps_10, analyses_10)


# ============ Iteration 11: prior therapy and lines ============
hyps_11 = [
    {"id": "h11_prior_chemo", "text": "Patients with prior_chemotherapy=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h11_prior_lines", "text": "Higher prior_lines_of_therapy is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h11_years_dx", "text": "Greater years_since_diagnosis is associated with different pfs_months.", "kind": "novel"},
    {"id": "h11_prior_targeted", "text": "Patients with prior_targeted_therapy=1 have shorter pfs_months than those without.", "kind": "novel"},
]
analyses_11 = []
for col, hid in [("prior_chemotherapy", "h11_prior_chemo"), ("prior_targeted_therapy", "h11_prior_targeted")]:
    diff, p, m1, m0, n1, n0 = group_means_pfs(DF[col] == 1)
    analyses_11.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by {col}",
        "result_summary": f"pfs {col}=1: {m1:.3f} (n={n1}) vs 0: {m0:.3f} (n={n0}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
    })
for term, hid in [("prior_lines_of_therapy", "h11_prior_lines"), ("years_since_diagnosis", "h11_years_dx")]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {term}", term=term)
    analyses_11.append({
        "hypothesis_ids": [hid],
        "code": f"OLS slope on {term}",
        "result_summary": f"OLS slope on {term}: {coef:.4f} months per unit, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(11, hyps_11, analyses_11)


# ============ Iteration 12: symptoms ============
hyps_12 = [
    {"id": "h12_fatigue", "text": "Higher fatigue_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h12_pain", "text": "Higher pain_nrs is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h12_dyspnea", "text": "Higher dyspnea_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h12_appetite", "text": "Higher appetite_loss_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
analyses_12 = []
for term, hid in [
    ("fatigue_grade", "h12_fatigue"),
    ("pain_nrs", "h12_pain"),
    ("dyspnea_grade", "h12_dyspnea"),
    ("appetite_loss_grade", "h12_appetite"),
]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {term}", term=term)
    analyses_12.append({
        "hypothesis_ids": [hid],
        "code": f"OLS slope on {term}",
        "result_summary": f"OLS slope on {term}: {coef:.4f} months per unit, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(12, hyps_12, analyses_12)


# ============ Iteration 13: tumor markers / size / proliferation ============
hyps_13 = [
    {"id": "h13_ca125", "text": "Higher ca_125_u_ml is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h13_cea", "text": "Higher cea_ng_ml is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h13_alkphos", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h13_ki67", "text": "Higher ki67_pct is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h13_tumor_size", "text": "Larger tumor_size_cm is associated with shorter pfs_months.", "kind": "novel"},
]
analyses_13 = []
for term, hid in [
    ("ca_125_u_ml", "h13_ca125"),
    ("cea_ng_ml", "h13_cea"),
    ("alkaline_phosphatase_u_l", "h13_alkphos"),
    ("ki67_pct", "h13_ki67"),
    ("tumor_size_cm", "h13_tumor_size"),
]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {term}", term=term)
    analyses_13.append({
        "hypothesis_ids": [hid],
        "code": f"OLS slope on {term}",
        "result_summary": f"OLS slope on {term}: {coef:.6f} months per unit, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(13, hyps_13, analyses_13)


# ============ Iteration 14: stratified treatment-biomarker subgroup analyses ============
hyps_14 = [
    {"id": "h14_tam_in_er", "text": "Within er_positive=1 patients only, treatment_tamoxifen is associated with longer pfs_months than no tamoxifen.", "kind": "refined"},
    {"id": "h14_tam_in_erneg", "text": "Within er_positive=0 patients only, treatment_tamoxifen has no benefit (or is inferior) on pfs_months.", "kind": "refined"},
    {"id": "h14_tras_in_her2pos", "text": "Within her2_positive=1, treatment_trastuzumab is associated with longer pfs_months.", "kind": "refined"},
    {"id": "h14_tras_in_her2neg", "text": "Within her2_positive=0, treatment_trastuzumab has no benefit on pfs_months.", "kind": "refined"},
    {"id": "h14_olap_in_brca", "text": "Within brca_any=1, treatment_olaparib is associated with longer pfs_months.", "kind": "refined"},
    {"id": "h14_pembro_in_msi", "text": "Within msi_high=1, treatment_pembrolizumab is associated with longer pfs_months.", "kind": "refined"},
]
analyses_14 = []
for tx, sub_col, sub_val, hid in [
    ("treatment_tamoxifen", "er_positive", 1, "h14_tam_in_er"),
    ("treatment_tamoxifen", "er_positive", 0, "h14_tam_in_erneg"),
    ("treatment_trastuzumab", "her2_positive", 1, "h14_tras_in_her2pos"),
    ("treatment_trastuzumab", "her2_positive", 0, "h14_tras_in_her2neg"),
    ("treatment_olaparib", "brca_any", 1, "h14_olap_in_brca"),
    ("treatment_pembrolizumab", "msi_high", 1, "h14_pembro_in_msi"),
]:
    sub = DF[DF[sub_col] == sub_val]
    a = sub.loc[sub[tx] == 1, OUT]
    b = sub.loc[sub[tx] == 0, OUT]
    if len(a) > 5 and len(b) > 5:
        diff, p = ttest(a, b)
    else:
        diff, p = float("nan"), float("nan")
    analyses_14.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by {tx} within {sub_col}=={sub_val}",
        "result_summary": f"In {sub_col}={sub_val} stratum (n={len(sub)}): pfs {tx}=1: {a.mean():.3f}(n={len(a)}) vs {tx}=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": None if np.isnan(p) else p,
        "effect_estimate": None if np.isnan(diff) else diff,
        "significant": (bool(p < 0.05) if not np.isnan(p) else None),
    })
add_iter(14, hyps_14, analyses_14)


# ============ Iteration 15: representative SNPs main effects ============
hyps_15 = [
    {"id": "h15_apoe_e4", "text": "Carriers of snp_rs429358 (APOE epsilon-4 marker) have different pfs_months from non-carriers.", "kind": "novel"},
    {"id": "h15_mthfr", "text": "snp_rs1801133 (MTHFR C677T) genotype is associated with pfs_months (slope test).", "kind": "novel"},
    {"id": "h15_mdr1", "text": "snp_rs1045642 (MDR1/ABCB1) genotype is associated with pfs_months.", "kind": "novel"},
    {"id": "h15_cyp2d6", "text": "snp_rs1065852 (CYP2D6*10) genotype is associated with pfs_months overall.", "kind": "novel"},
    {"id": "h15_cyp2c19", "text": "snp_rs4244285 (CYP2C19*2) genotype is associated with pfs_months overall.", "kind": "novel"},
]
analyses_15 = []
for snp, hid in [
    ("snp_rs429358", "h15_apoe_e4"),
    ("snp_rs1801133", "h15_mthfr"),
    ("snp_rs1045642", "h15_mdr1"),
    ("snp_rs1065852", "h15_cyp2d6"),
    ("snp_rs4244285", "h15_cyp2c19"),
]:
    coef, p, n, _ = ols_effect(f"pfs_months ~ {snp}", term=snp)
    analyses_15.append({
        "hypothesis_ids": [hid],
        "code": f"OLS slope on {snp}",
        "result_summary": f"OLS slope on {snp}: {coef:.4f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(15, hyps_15, analyses_15)


# ============ Iteration 16: tamoxifen x CYP2D6 (snp_rs1065852) interaction ============
hyps_16 = [
    {"id": "h16_tam_cyp2d6", "text": "The pfs_months benefit of treatment_tamoxifen is smaller in patients carrying snp_rs1065852 (CYP2D6*10 variant, reduced metabolism) than in non-carriers (negative interaction tamoxifen x snp_rs1065852).", "kind": "novel"},
    {"id": "h16_tam_cyp2d6_in_erpos", "text": "The negative interaction tamoxifen x snp_rs1065852 is present specifically among er_positive=1 patients.", "kind": "refined"},
]
analyses_16 = []
m = smf.ols("pfs_months ~ treatment_tamoxifen * snp_rs1065852", data=DF).fit()
coef = float(m.params["treatment_tamoxifen:snp_rs1065852"]); p = float(m.pvalues["treatment_tamoxifen:snp_rs1065852"])
strat = []
for tam, snp in [(1, 1), (1, 0), (0, 1), (0, 0)]:
    mask = (DF["treatment_tamoxifen"] == tam) & (DF["snp_rs1065852"] == snp)
    strat.append((tam, snp, DF.loc[mask, OUT].mean(), int(mask.sum())))
analyses_16.append({
    "hypothesis_ids": ["h16_tam_cyp2d6"],
    "code": "smf.ols('pfs_months ~ treatment_tamoxifen * snp_rs1065852', df).fit()",
    "result_summary": f"Interaction tamoxifen x snp_rs1065852: coef={coef:.3f}, p={p:.3g}. Stratified means (tam,snp,mean,n): " + "; ".join(f"({a},{b},{c:.3f},{d})" for a, b, c, d in strat),
    "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
})
sub = DF[DF["er_positive"] == 1]
m2 = smf.ols("pfs_months ~ treatment_tamoxifen * snp_rs1065852", data=sub).fit()
coef2 = float(m2.params["treatment_tamoxifen:snp_rs1065852"]); p2 = float(m2.pvalues["treatment_tamoxifen:snp_rs1065852"])
sub_eff_carrier = sub.loc[(sub['treatment_tamoxifen']==1)&(sub['snp_rs1065852']==1), OUT].mean() - sub.loc[(sub['treatment_tamoxifen']==0)&(sub['snp_rs1065852']==1), OUT].mean()
sub_eff_noncarrier = sub.loc[(sub['treatment_tamoxifen']==1)&(sub['snp_rs1065852']==0), OUT].mean() - sub.loc[(sub['treatment_tamoxifen']==0)&(sub['snp_rs1065852']==0), OUT].mean()
analyses_16.append({
    "hypothesis_ids": ["h16_tam_cyp2d6_in_erpos"],
    "code": "smf.ols on er_positive==1 subset",
    "result_summary": f"Within er_positive=1 (n={len(sub)}): interaction tamoxifen x snp_rs1065852 coef={coef2:.3f}, p={p2:.3g}. Tamoxifen effect by carrier status: snp=1: {sub_eff_carrier:.3f}, snp=0: {sub_eff_noncarrier:.3f}.",
    "p_value": p2, "effect_estimate": coef2, "significant": bool(p2 < 0.05),
})
add_iter(16, hyps_16, analyses_16)


# ============ Iteration 17: HER2 amplification, overlap with HER2+, treatment effect ============
hyps_17 = [
    {"id": "h17_her2amp", "text": "her2_amplification=1 patients have different pfs_months from her2_amplification=0 patients.", "kind": "novel"},
    {"id": "h17_her2amp_overlap", "text": "her2_amplification=1 is largely concordant with her2_positive=1 (chi-square test of independence is highly significant).", "kind": "novel"},
    {"id": "h17_tras_her2amp", "text": "Within her2_amplification=1, treatment_trastuzumab improves pfs_months relative to no trastuzumab.", "kind": "refined"},
]
analyses_17 = []
diff, p, m1, m0, n1, n0 = group_means_pfs(DF["her2_amplification"] == 1)
analyses_17.append({
    "hypothesis_ids": ["h17_her2amp"],
    "code": "t-test pfs by her2_amplification",
    "result_summary": f"pfs her2_amp=1: {m1:.3f}(n={n1}) vs =0: {m0:.3f}(n={n0}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
ct = pd.crosstab(DF["her2_positive"], DF["her2_amplification"])
chi2, p_overlap, dof, _ = stats.chi2_contingency(ct)
overlap_eff = float(ct.values[1, 1] / max(1, ct.values[1, :].sum()) - ct.values[0, 1] / max(1, ct.values[0, :].sum()))
analyses_17.append({
    "hypothesis_ids": ["h17_her2amp_overlap"],
    "code": "chi2_contingency(her2_positive, her2_amplification)",
    "result_summary": f"Crosstab her2_positive x her2_amplification: {ct.values.tolist()} (rows=her2_positive 0/1, cols=her2_amp 0/1). Chi2={chi2:.2f}, p={p_overlap:.3g}. Diff in P(her2_amp=1 | her2_positive): {overlap_eff:.3f}.",
    "p_value": float(p_overlap), "effect_estimate": overlap_eff, "significant": bool(p_overlap < 0.05),
})
sub = DF[DF["her2_amplification"] == 1]
if len(sub) > 20:
    a = sub.loc[sub["treatment_trastuzumab"] == 1, OUT]
    b = sub.loc[sub["treatment_trastuzumab"] == 0, OUT]
    if len(a) > 5 and len(b) > 5:
        diff, p = ttest(a, b)
    else:
        diff, p = float('nan'), float('nan')
    analyses_17.append({
        "hypothesis_ids": ["h17_tras_her2amp"],
        "code": "t-test pfs by trastuzumab within her2_amplification==1",
        "result_summary": f"Within her2_amplification=1 (n={len(sub)}): pfs tras=1: {a.mean():.3f}(n={len(a)}) vs tras=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": None if np.isnan(p) else p,
        "effect_estimate": None if np.isnan(diff) else diff,
        "significant": (bool(p < 0.05) if not np.isnan(p) else None),
    })
add_iter(17, hyps_17, analyses_17)


# ============ Iteration 18: full multivariable model with key interactions ============
hyps_18 = [
    {"id": "h18_full_model", "text": "In a full multivariable OLS model with all six treatments and key biomarker interactions (tamoxifen x er_positive, palbociclib x er_positive, trastuzumab x her2_positive, olaparib x brca_any, pembrolizumab x msi_high, sacituzumab x her2_low), the predictive interactions remain statistically significant after adjustment for demographics, comorbidities, and labs.", "kind": "refined"},
]
analyses_18 = []
formula_full = (
    "pfs_months ~ treatment_tamoxifen*er_positive + treatment_palbociclib*er_positive + "
    "treatment_trastuzumab*her2_positive + treatment_olaparib*brca_any + "
    "treatment_pembrolizumab*msi_high + treatment_sacituzumab_govitecan*her2_low + "
    "age_years + ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + "
    "albumin_g_dl + ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo + hemoglobin_g_dl + "
    "ki67_pct + alkaline_phosphatase_u_l + tp53_mutation"
)
mfull = smf.ols(formula_full, data=DF).fit()
interaction_terms = [
    "treatment_tamoxifen:er_positive",
    "treatment_palbociclib:er_positive",
    "treatment_trastuzumab:her2_positive",
    "treatment_olaparib:brca_any",
    "treatment_pembrolizumab:msi_high",
    "treatment_sacituzumab_govitecan:her2_low",
]
summary_int = []
for it in interaction_terms:
    if it in mfull.params.index:
        summary_int.append(f"{it}: coef={float(mfull.params[it]):.3f}, p={float(mfull.pvalues[it]):.3g}")
analyses_18.append({
    "hypothesis_ids": ["h18_full_model"],
    "code": f"smf.ols(full predictive+prognostic formula).fit()",
    "result_summary": "Adjusted interaction effects: " + "; ".join(summary_int) + f". Model R^2={mfull.rsquared:.3f}, n={int(mfull.nobs)}.",
    "p_value": float(mfull.pvalues["treatment_palbociclib:er_positive"]),
    "effect_estimate": float(mfull.params["treatment_palbociclib:er_positive"]),
    "significant": bool(float(mfull.pvalues["treatment_palbociclib:er_positive"]) < 0.05),
})
add_iter(18, hyps_18, analyses_18)


# ============ Iteration 19: postmenopausal interactions with endocrine/CDK4-6 ============
hyps_19 = [
    {"id": "h19_tam_postmen", "text": "Within postmenopausal=1 patients, treatment_tamoxifen is associated with longer pfs_months than no tamoxifen.", "kind": "refined"},
    {"id": "h19_palb_postmen", "text": "Within postmenopausal=1 & er_positive=1 patients, treatment_palbociclib is associated with longer pfs_months than no palbociclib.", "kind": "refined"},
]
analyses_19 = []
sub = DF[DF["postmenopausal"] == 1]
a = sub.loc[sub["treatment_tamoxifen"] == 1, OUT]; b = sub.loc[sub["treatment_tamoxifen"] == 0, OUT]
diff, p = ttest(a, b)
analyses_19.append({
    "hypothesis_ids": ["h19_tam_postmen"],
    "code": "t-test pfs by tamoxifen in postmenopausal==1",
    "result_summary": f"In postmenopausal=1 (n={len(sub)}): pfs tam=1: {a.mean():.3f}(n={len(a)}) vs tam=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
sub = DF[(DF["postmenopausal"] == 1) & (DF["er_positive"] == 1)]
a = sub.loc[sub["treatment_palbociclib"] == 1, OUT]; b = sub.loc[sub["treatment_palbociclib"] == 0, OUT]
diff, p = ttest(a, b)
analyses_19.append({
    "hypothesis_ids": ["h19_palb_postmen"],
    "code": "t-test pfs by palbociclib in postmenopausal==1 & er_positive==1",
    "result_summary": f"In postmenopausal=1 & ER+=1 (n={len(sub)}): pfs palb=1: {a.mean():.3f}(n={len(a)}) vs palb=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
add_iter(19, hyps_19, analyses_19)


# ============ Iteration 20: brain mets and treatment interactions ============
hyps_20 = [
    {"id": "h20_brain_pfs_short", "text": "After multivariable adjustment, has_brain_mets=1 is associated with shorter pfs_months than has_brain_mets=0 (negative coefficient).", "kind": "refined"},
    {"id": "h20_pembro_brain", "text": "The pembrolizumab pfs effect differs by has_brain_mets status (interaction pembrolizumab x has_brain_mets is non-zero).", "kind": "novel"},
    {"id": "h20_tras_brain", "text": "The trastuzumab pfs effect differs by has_brain_mets status (interaction trastuzumab x has_brain_mets is non-zero).", "kind": "novel"},
]
analyses_20 = []
mv2 = smf.ols(formula_mv, data=DF).fit()
coef = float(mv2.params["has_brain_mets"]); p = float(mv2.pvalues["has_brain_mets"])
analyses_20.append({
    "hypothesis_ids": ["h20_brain_pfs_short"],
    "code": "Adjusted multivariable OLS, has_brain_mets coefficient",
    "result_summary": f"Adjusted coefficient for has_brain_mets in main MV model: {coef:.3f} months (p={p:.3g}).",
    "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
})
for tx, hid in [("treatment_pembrolizumab", "h20_pembro_brain"), ("treatment_trastuzumab", "h20_tras_brain")]:
    m = smf.ols(f"pfs_months ~ {tx} * has_brain_mets", data=DF).fit()
    iterm = f"{tx}:has_brain_mets"
    coef = float(m.params[iterm]); p = float(m.pvalues[iterm])
    analyses_20.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('pfs_months ~ {tx} * has_brain_mets').fit()",
        "result_summary": f"Interaction {iterm}: coef={coef:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(20, hyps_20, analyses_20)


# ============ Iteration 21: TNBC subgroup, pembrolizumab and saci ============
hyps_21 = [
    {"id": "h21_tnbc_pfs", "text": "Triple-negative (er_positive=0, pr_positive=0, her2_positive=0) patients have shorter pfs_months than non-TNBC.", "kind": "novel"},
    {"id": "h21_pembro_tnbc", "text": "Within TNBC patients, treatment_pembrolizumab is associated with longer pfs_months than no pembrolizumab.", "kind": "refined"},
    {"id": "h21_saci_tnbc_strat", "text": "Within TNBC patients, treatment_sacituzumab_govitecan is associated with longer pfs_months than no sacituzumab.", "kind": "refined"},
]
analyses_21 = []
diff, p, m1, m0, n1, n0 = group_means_pfs(DF["tnbc"] == 1)
analyses_21.append({
    "hypothesis_ids": ["h21_tnbc_pfs"],
    "code": "t-test pfs by tnbc",
    "result_summary": f"pfs tnbc=1: {m1:.3f}(n={n1}) vs tnbc=0: {m0:.3f}(n={n0}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": p, "effect_estimate": diff, "significant": bool(p < 0.05),
})
for tx, hid in [("treatment_pembrolizumab", "h21_pembro_tnbc"), ("treatment_sacituzumab_govitecan", "h21_saci_tnbc_strat")]:
    sub = DF[DF["tnbc"] == 1]
    a = sub.loc[sub[tx] == 1, OUT]; b = sub.loc[sub[tx] == 0, OUT]
    if len(a) > 5 and len(b) > 5:
        diff, p = ttest(a, b)
    else:
        diff, p = float("nan"), float("nan")
    analyses_21.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by {tx} within tnbc==1",
        "result_summary": f"In tnbc=1 (n={len(sub)}): pfs {tx}=1: {a.mean():.3f}(n={len(a)}) vs {tx}=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": None if np.isnan(p) else p,
        "effect_estimate": None if np.isnan(diff) else diff,
        "significant": (bool(p < 0.05) if not np.isnan(p) else None),
    })
add_iter(21, hyps_21, analyses_21)


# ============ Iteration 22: refined olaparib in BRCA, brca1 vs brca2 ============
hyps_22 = [
    {"id": "h22_olap_brca1", "text": "Within brca1_mutation=1 patients, treatment_olaparib is associated with longer pfs_months.", "kind": "refined"},
    {"id": "h22_olap_brca2", "text": "Within brca2_mutation=1 patients, treatment_olaparib is associated with longer pfs_months.", "kind": "refined"},
    {"id": "h22_olap_nobrca", "text": "Within brca_any=0 (no germline BRCA), treatment_olaparib has no benefit on pfs_months.", "kind": "refined"},
]
analyses_22 = []
for sub_col, sub_val, hid in [
    ("brca1_mutation", 1, "h22_olap_brca1"),
    ("brca2_mutation", 1, "h22_olap_brca2"),
    ("brca_any", 0, "h22_olap_nobrca"),
]:
    sub = DF[DF[sub_col] == sub_val]
    a = sub.loc[sub["treatment_olaparib"] == 1, OUT]; b = sub.loc[sub["treatment_olaparib"] == 0, OUT]
    if len(a) > 5 and len(b) > 5:
        diff, p = ttest(a, b)
    else:
        diff, p = float("nan"), float("nan")
    analyses_22.append({
        "hypothesis_ids": [hid],
        "code": f"t-test pfs by olaparib within {sub_col}=={sub_val}",
        "result_summary": f"In {sub_col}={sub_val} (n={len(sub)}): pfs olap=1: {a.mean():.3f}(n={len(a)}) vs olap=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
        "p_value": None if np.isnan(p) else p,
        "effect_estimate": None if np.isnan(diff) else diff,
        "significant": (bool(p < 0.05) if not np.isnan(p) else None),
    })
add_iter(22, hyps_22, analyses_22)


# ============ Iteration 23: HER2-low / HER2-zero refinement; premenopausal palb ============
hyps_23 = [
    {"id": "h23_saci_her2low_strat", "text": "Within her2_low=1 patients, treatment_sacituzumab_govitecan is associated with longer pfs_months than no sacituzumab.", "kind": "refined"},
    {"id": "h23_saci_her2neg_nonlow", "text": "Within her2_positive=0 & her2_low=0 (HER2-zero) patients, treatment_sacituzumab_govitecan effect on pfs_months differs from the her2_low subgroup.", "kind": "refined"},
    {"id": "h23_palb_premen", "text": "Within er_positive=1 & postmenopausal=0 (premenopausal ER+) patients, treatment_palbociclib is associated with longer pfs_months than no palbociclib.", "kind": "refined"},
]
analyses_23 = []
sub = DF[DF["her2_low"] == 1]
a = sub.loc[sub["treatment_sacituzumab_govitecan"] == 1, OUT]; b = sub.loc[sub["treatment_sacituzumab_govitecan"] == 0, OUT]
diff, p = ttest(a, b) if (len(a) > 5 and len(b) > 5) else (float('nan'), float('nan'))
analyses_23.append({
    "hypothesis_ids": ["h23_saci_her2low_strat"],
    "code": "t-test pfs by sacituzumab in her2_low==1",
    "result_summary": f"In her2_low=1 (n={len(sub)}): pfs saci=1: {a.mean():.3f}(n={len(a)}) vs saci=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": None if np.isnan(p) else p,
    "effect_estimate": None if np.isnan(diff) else diff,
    "significant": (bool(p < 0.05) if not np.isnan(p) else None),
})
sub = DF[(DF["her2_positive"] == 0) & (DF["her2_low"] == 0)]
a = sub.loc[sub["treatment_sacituzumab_govitecan"] == 1, OUT]; b = sub.loc[sub["treatment_sacituzumab_govitecan"] == 0, OUT]
diff, p = ttest(a, b) if (len(a) > 5 and len(b) > 5) else (float('nan'), float('nan'))
analyses_23.append({
    "hypothesis_ids": ["h23_saci_her2neg_nonlow"],
    "code": "t-test pfs by sacituzumab in HER2-zero (her2_positive==0 & her2_low==0)",
    "result_summary": f"In HER2-zero (n={len(sub)}): pfs saci=1: {a.mean():.3f}(n={len(a)}) vs saci=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": None if np.isnan(p) else p,
    "effect_estimate": None if np.isnan(diff) else diff,
    "significant": (bool(p < 0.05) if not np.isnan(p) else None),
})
sub = DF[(DF["er_positive"] == 1) & (DF["postmenopausal"] == 0)]
a = sub.loc[sub["treatment_palbociclib"] == 1, OUT]; b = sub.loc[sub["treatment_palbociclib"] == 0, OUT]
diff, p = ttest(a, b) if (len(a) > 5 and len(b) > 5) else (float('nan'), float('nan'))
analyses_23.append({
    "hypothesis_ids": ["h23_palb_premen"],
    "code": "t-test pfs by palbociclib in er_positive==1 & postmenopausal==0",
    "result_summary": f"In ER+ & premenopausal (n={len(sub)}): pfs palb=1: {a.mean():.3f}(n={len(a)}) vs palb=0: {b.mean():.3f}(n={len(b)}); diff={diff:.3f}, p={p:.3g}.",
    "p_value": None if np.isnan(p) else p,
    "effect_estimate": None if np.isnan(diff) else diff,
    "significant": (bool(p < 0.05) if not np.isnan(p) else None),
})
add_iter(23, hyps_23, analyses_23)


# ============ Iteration 24: adjusted treatment x biomarker interactions ============
hyps_24 = [
    {"id": "h24_adj_palb_er", "text": "After adjusting for age, ECOG, stage, mets, and labs, the treatment_palbociclib x er_positive interaction on pfs_months remains positive and significant.", "kind": "refined"},
    {"id": "h24_adj_olap_brca", "text": "After adjusting for the same covariates, the treatment_olaparib x brca_any interaction remains positive and significant.", "kind": "refined"},
    {"id": "h24_adj_pembro_msi", "text": "After adjusting for the same covariates, the treatment_pembrolizumab x msi_high interaction remains positive and significant.", "kind": "refined"},
    {"id": "h24_adj_tras_her2", "text": "After adjusting for the same covariates, the treatment_trastuzumab x her2_positive interaction remains positive and significant.", "kind": "refined"},
]
analyses_24 = []


def adj_interaction(tx, bm):
    f = (
        f"pfs_months ~ {tx} * {bm} + age_years + ecog_ps + stage_iv + has_brain_mets + "
        "liver_mets + bone_mets + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + "
        "weight_loss_pct_6mo + hemoglobin_g_dl + ki67_pct"
    )
    m = smf.ols(f, data=DF).fit()
    iterm = f"{tx}:{bm}"
    return float(m.params[iterm]), float(m.pvalues[iterm])


for tx, bm, hid in [
    ("treatment_palbociclib", "er_positive", "h24_adj_palb_er"),
    ("treatment_olaparib", "brca_any", "h24_adj_olap_brca"),
    ("treatment_pembrolizumab", "msi_high", "h24_adj_pembro_msi"),
    ("treatment_trastuzumab", "her2_positive", "h24_adj_tras_her2"),
]:
    coef, p = adj_interaction(tx, bm)
    analyses_24.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols pfs_months ~ {tx}*{bm} + covariates",
        "result_summary": f"Adjusted interaction {tx} x {bm}: coef={coef:.3f}, p={p:.3g}.",
        "p_value": p, "effect_estimate": coef, "significant": bool(p < 0.05),
    })
add_iter(24, hyps_24, analyses_24)


# ============ Iteration 25: comprehensive summary model — variance explained, top predictors ============
hyps_25 = [
    {"id": "h25_variance", "text": "A comprehensive multivariable model with treatments, biomarkers, predictive interactions, and prognostic factors explains a meaningful fraction of variance in pfs_months (R-squared > 0.10) and the overall F-test is highly significant.", "kind": "refined"},
    {"id": "h25_top_factors", "text": "Among non-treatment factors in the full model, ecog_ps, stage_iv, albumin_g_dl, ldh_u_l, has_brain_mets, weight_loss_pct_6mo, and ki67_pct are among the strongest independent predictors of pfs_months (smallest p-values).", "kind": "refined"},
]
analyses_25 = []
mfinal = smf.ols(formula_full, data=DF).fit()
prog_terms = [
    "ecog_ps", "stage_iv", "has_brain_mets", "liver_mets", "bone_mets",
    "albumin_g_dl", "ldh_u_l", "nlr", "crp_mg_l", "weight_loss_pct_6mo",
    "hemoglobin_g_dl", "ki67_pct", "alkaline_phosphatase_u_l", "tp53_mutation",
    "age_years",
]
top_summaries = []
for t in prog_terms:
    if t in mfinal.params.index:
        top_summaries.append((t, float(mfinal.params[t]), float(mfinal.pvalues[t])))
top_summaries.sort(key=lambda x: x[2])
top_str = "; ".join(f"{t}: coef={c:.4f}, p={p:.3g}" for t, c, p in top_summaries[:8])
analyses_25.append({
    "hypothesis_ids": ["h25_variance"],
    "code": "smf.ols(full predictive+prognostic formula).fit() — overall fit",
    "result_summary": f"Full model R^2={mfinal.rsquared:.3f}, adj-R^2={mfinal.rsquared_adj:.3f}, n={int(mfinal.nobs)}. Overall F p-value={float(mfinal.f_pvalue):.3g}.",
    "p_value": float(mfinal.f_pvalue),
    "effect_estimate": float(mfinal.rsquared),
    "significant": bool(float(mfinal.f_pvalue) < 0.05),
})
analyses_25.append({
    "hypothesis_ids": ["h25_top_factors"],
    "code": "Sort prognostic-factor coefficients by p-value in full model",
    "result_summary": f"Top prognostic factors by p-value: {top_str}.",
    "p_value": top_summaries[0][2] if top_summaries else None,
    "effect_estimate": top_summaries[0][1] if top_summaries else None,
    "significant": bool(top_summaries[0][2] < 0.05) if top_summaries else None,
})
add_iter(25, hyps_25, analyses_25)


# ============ Build transcript ============
transcript = {
    "dataset_id": "ds001_breast",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code-custom@1.0",
    "max_iterations": 25,
    "iterations": iterations,
}

with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2, default=str)

print("Wrote transcript.json with", len(iterations), "iterations,",
      sum(len(it["analyses"]) for it in iterations), "analyses,",
      sum(len(it["proposed_hypotheses"]) for it in iterations), "hypotheses.")

lines = []
for it in iterations:
    lines.append(f"\n=== Iteration {it['index']} ===")
    for h in it["proposed_hypotheses"]:
        lines.append(f"[{h['id']}] {h['text']}")
    for a in it["analyses"]:
        sig = a.get("significant")
        sig_str = "SIG" if sig else ("NS" if sig is False else "?")
        lines.append(f"  -> ({','.join(a['hypothesis_ids'])}) [{sig_str}] effect={a.get('effect_estimate')}, p={a.get('p_value')}")
        lines.append(f"     {a['result_summary']}")

with open("_iter_log.txt", "w") as f:
    f.write("\n".join(lines))

print("Wrote _iter_log.txt")
