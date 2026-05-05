"""
Comprehensive iterative analysis of ds001_breast.
Runs 25 iterations of hypothesis-test loops and writes results to results.json.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
np.random.seed(0)

df = pd.read_parquet("dataset.parquet")

TRT_COLS = [
    "treatment_tamoxifen",
    "treatment_palbociclib",
    "treatment_trastuzumab",
    "treatment_olaparib",
    "treatment_sacituzumab_govitecan",
    "treatment_pembrolizumab",
]

CONT_FEATURES = [
    "age_years", "ki67_pct", "tumor_size_cm", "albumin_g_dl", "ldh_u_l",
    "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl",
    "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l", "total_bilirubin_mg_dl",
    "creatinine_mg_dl", "bun_mg_dl", "sodium_meq_l", "potassium_meq_l",
    "calcium_mg_dl",
]
BIN_FEATURES = [
    "sex_female", "stage_iv", "has_brain_mets", "node_positive",
    "postmenopausal", "er_positive", "pr_positive", "her2_positive",
    "her2_low", "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
]
ORD_FEATURES = ["ecog_ps"]

OUTCOME = "pfs_months"

results = {"iterations": []}

def add_iter(idx, hypotheses, analyses):
    results["iterations"].append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })

def fmt_p(p):
    return float(p) if (p is not None and np.isfinite(p)) else None

def fmt_e(e):
    return float(e) if (e is not None and np.isfinite(e)) else None

def ttest_by(df, group, outcome=OUTCOME):
    a = df.loc[df[group] == 1, outcome]
    b = df.loc[df[group] == 0, outcome]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    eff = a.mean() - b.mean()
    return {"mean_1": float(a.mean()), "mean_0": float(b.mean()),
            "n_1": int(len(a)), "n_0": int(len(b)),
            "effect": float(eff), "t": float(t), "p": float(p)}

def ols_simple(df, x, outcome=OUTCOME):
    X = sm.add_constant(df[x].astype(float))
    m = sm.OLS(df[outcome], X).fit()
    return {"beta": float(m.params[x]), "p": float(m.pvalues[x]),
            "r2": float(m.rsquared), "n": int(m.nobs)}

def ols_formula(df, formula):
    m = smf.ols(formula, data=df).fit()
    return m

# =====================================================================
# Iteration 1: outcome distribution + age and sex main effects
# =====================================================================
hyps = [
    {"id": "h1.1", "text": "Older age (age_years, continuous) is associated with shorter pfs_months overall (negative slope).", "kind": "novel"},
    {"id": "h1.2", "text": "Female sex (sex_female=1) is associated with longer pfs_months than male sex.", "kind": "novel"},
]
a_age = ols_simple(df, "age_years")
a_sex = ttest_by(df, "sex_female")
analyses = [
    {"hypothesis_ids": ["h1.1"],
     "code": "sm.OLS(pfs_months ~ age_years).fit()",
     "result_summary": f"Slope of pfs_months on age_years = {a_age['beta']:.4f} (p={a_age['p']:.3g}); R^2={a_age['r2']:.4f}.",
     "p_value": fmt_p(a_age["p"]), "effect_estimate": fmt_e(a_age["beta"]),
     "significant": bool(a_age["p"] < 0.05)},
    {"hypothesis_ids": ["h1.2"],
     "code": "ttest_ind(pfs_months[sex_female==1], pfs_months[sex_female==0])",
     "result_summary": f"Mean pfs_months: female={a_sex['mean_1']:.3f} vs male={a_sex['mean_0']:.3f}; difference={a_sex['effect']:.3f}; p={a_sex['p']:.3g}.",
     "p_value": fmt_p(a_sex["p"]), "effect_estimate": fmt_e(a_sex["effect"]),
     "significant": bool(a_sex["p"] < 0.05)},
]
add_iter(1, hyps, analyses)

# =====================================================================
# Iteration 2: ECOG performance status and stage IV
# =====================================================================
hyps = [
    {"id": "h2.1", "text": "Higher ECOG performance status (ecog_ps, ordinal 0/1/2) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h2.2", "text": "Stage IV disease (stage_iv=1) is associated with shorter pfs_months than non-stage IV.", "kind": "novel"},
    {"id": "h2.3", "text": "Brain metastases (has_brain_mets=1) are associated with shorter pfs_months than no brain mets.", "kind": "novel"},
]
a_ecog = ols_simple(df, "ecog_ps")
a_stiv = ttest_by(df, "stage_iv")
a_brm = ttest_by(df, "has_brain_mets")
analyses = [
    {"hypothesis_ids": ["h2.1"], "code": "OLS pfs ~ ecog_ps",
     "result_summary": f"Slope of pfs on ECOG = {a_ecog['beta']:.4f} (p={a_ecog['p']:.3g}).",
     "p_value": fmt_p(a_ecog["p"]), "effect_estimate": fmt_e(a_ecog["beta"]),
     "significant": bool(a_ecog["p"] < 0.05)},
    {"hypothesis_ids": ["h2.2"], "code": "ttest stage_iv",
     "result_summary": f"Mean pfs stage_iv=1 vs 0: {a_stiv['mean_1']:.3f} vs {a_stiv['mean_0']:.3f}; diff={a_stiv['effect']:.3f}; p={a_stiv['p']:.3g}.",
     "p_value": fmt_p(a_stiv["p"]), "effect_estimate": fmt_e(a_stiv["effect"]),
     "significant": bool(a_stiv["p"] < 0.05)},
    {"hypothesis_ids": ["h2.3"], "code": "ttest has_brain_mets",
     "result_summary": f"Mean pfs brain_mets=1 vs 0: {a_brm['mean_1']:.3f} vs {a_brm['mean_0']:.3f}; diff={a_brm['effect']:.3f}; p={a_brm['p']:.3g}.",
     "p_value": fmt_p(a_brm["p"]), "effect_estimate": fmt_e(a_brm["effect"]),
     "significant": bool(a_brm["p"] < 0.05)},
]
add_iter(2, hyps, analyses)

# =====================================================================
# Iteration 3: nodal status, postmenopausal, tumor size, ki67
# =====================================================================
hyps = [
    {"id": "h3.1", "text": "Node-positive disease (node_positive=1) is associated with shorter pfs_months than node-negative.", "kind": "novel"},
    {"id": "h3.2", "text": "Postmenopausal status (postmenopausal=1) is associated with longer pfs_months than premenopausal.", "kind": "novel"},
    {"id": "h3.3", "text": "Larger tumor size (tumor_size_cm, continuous) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h3.4", "text": "Higher Ki-67 proliferation index (ki67_pct, continuous) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
a_np = ttest_by(df, "node_positive")
a_pm = ttest_by(df, "postmenopausal")
a_tu = ols_simple(df, "tumor_size_cm")
a_ki = ols_simple(df, "ki67_pct")
analyses = [
    {"hypothesis_ids": ["h3.1"], "code": "ttest node_positive",
     "result_summary": f"Mean pfs node_pos=1 vs 0: {a_np['mean_1']:.3f} vs {a_np['mean_0']:.3f}; diff={a_np['effect']:.3f}; p={a_np['p']:.3g}.",
     "p_value": fmt_p(a_np["p"]), "effect_estimate": fmt_e(a_np["effect"]),
     "significant": bool(a_np["p"] < 0.05)},
    {"hypothesis_ids": ["h3.2"], "code": "ttest postmenopausal",
     "result_summary": f"Mean pfs postmen=1 vs 0: {a_pm['mean_1']:.3f} vs {a_pm['mean_0']:.3f}; diff={a_pm['effect']:.3f}; p={a_pm['p']:.3g}.",
     "p_value": fmt_p(a_pm["p"]), "effect_estimate": fmt_e(a_pm["effect"]),
     "significant": bool(a_pm["p"] < 0.05)},
    {"hypothesis_ids": ["h3.3"], "code": "OLS pfs ~ tumor_size_cm",
     "result_summary": f"Slope of pfs on tumor_size_cm = {a_tu['beta']:.4f} (p={a_tu['p']:.3g}).",
     "p_value": fmt_p(a_tu["p"]), "effect_estimate": fmt_e(a_tu["beta"]),
     "significant": bool(a_tu["p"] < 0.05)},
    {"hypothesis_ids": ["h3.4"], "code": "OLS pfs ~ ki67_pct",
     "result_summary": f"Slope of pfs on ki67_pct = {a_ki['beta']:.4f} (p={a_ki['p']:.3g}).",
     "p_value": fmt_p(a_ki["p"]), "effect_estimate": fmt_e(a_ki["beta"]),
     "significant": bool(a_ki["p"] < 0.05)},
]
add_iter(3, hyps, analyses)

# =====================================================================
# Iteration 4: hormone receptor and HER2 status
# =====================================================================
hyps = [
    {"id": "h4.1", "text": "ER-positive disease (er_positive=1) is associated with longer pfs_months than ER-negative.", "kind": "novel"},
    {"id": "h4.2", "text": "PR-positive disease (pr_positive=1) is associated with longer pfs_months than PR-negative.", "kind": "novel"},
    {"id": "h4.3", "text": "HER2-positive disease (her2_positive=1) is associated with shorter pfs_months than HER2-negative (without targeted therapy as a univariate effect).", "kind": "novel"},
    {"id": "h4.4", "text": "HER2-low disease (her2_low=1) is associated with different pfs_months than HER2-not-low.", "kind": "novel"},
]
a_er = ttest_by(df, "er_positive")
a_pr = ttest_by(df, "pr_positive")
a_hp = ttest_by(df, "her2_positive")
a_hl = ttest_by(df, "her2_low")
analyses = [
    {"hypothesis_ids": ["h4.1"], "code": "ttest er_positive",
     "result_summary": f"Mean pfs ER+ vs ER-: {a_er['mean_1']:.3f} vs {a_er['mean_0']:.3f}; diff={a_er['effect']:.3f}; p={a_er['p']:.3g}.",
     "p_value": fmt_p(a_er["p"]), "effect_estimate": fmt_e(a_er["effect"]),
     "significant": bool(a_er["p"] < 0.05)},
    {"hypothesis_ids": ["h4.2"], "code": "ttest pr_positive",
     "result_summary": f"Mean pfs PR+ vs PR-: {a_pr['mean_1']:.3f} vs {a_pr['mean_0']:.3f}; diff={a_pr['effect']:.3f}; p={a_pr['p']:.3g}.",
     "p_value": fmt_p(a_pr["p"]), "effect_estimate": fmt_e(a_pr["effect"]),
     "significant": bool(a_pr["p"] < 0.05)},
    {"hypothesis_ids": ["h4.3"], "code": "ttest her2_positive",
     "result_summary": f"Mean pfs HER2+ vs HER2-: {a_hp['mean_1']:.3f} vs {a_hp['mean_0']:.3f}; diff={a_hp['effect']:.3f}; p={a_hp['p']:.3g}.",
     "p_value": fmt_p(a_hp["p"]), "effect_estimate": fmt_e(a_hp["effect"]),
     "significant": bool(a_hp["p"] < 0.05)},
    {"hypothesis_ids": ["h4.4"], "code": "ttest her2_low",
     "result_summary": f"Mean pfs HER2-low vs not: {a_hl['mean_1']:.3f} vs {a_hl['mean_0']:.3f}; diff={a_hl['effect']:.3f}; p={a_hl['p']:.3g}.",
     "p_value": fmt_p(a_hl["p"]), "effect_estimate": fmt_e(a_hl["effect"]),
     "significant": bool(a_hl["p"] < 0.05)},
]
add_iter(4, hyps, analyses)

# =====================================================================
# Iteration 5: BRCA1, BRCA2, PIK3CA mutations
# =====================================================================
hyps = [
    {"id": "h5.1", "text": "BRCA1 mutation (brca1_mutation=1) is associated with different pfs_months than wild-type.", "kind": "novel"},
    {"id": "h5.2", "text": "BRCA2 mutation (brca2_mutation=1) is associated with different pfs_months than wild-type.", "kind": "novel"},
    {"id": "h5.3", "text": "PIK3CA mutation (pik3ca_mutation=1) is associated with different pfs_months than wild-type.", "kind": "novel"},
]
a_b1 = ttest_by(df, "brca1_mutation")
a_b2 = ttest_by(df, "brca2_mutation")
a_pk = ttest_by(df, "pik3ca_mutation")
analyses = [
    {"hypothesis_ids": ["h5.1"], "code": "ttest brca1",
     "result_summary": f"Mean pfs BRCA1+ vs BRCA1-: {a_b1['mean_1']:.3f} vs {a_b1['mean_0']:.3f}; diff={a_b1['effect']:.3f}; p={a_b1['p']:.3g}.",
     "p_value": fmt_p(a_b1["p"]), "effect_estimate": fmt_e(a_b1["effect"]),
     "significant": bool(a_b1["p"] < 0.05)},
    {"hypothesis_ids": ["h5.2"], "code": "ttest brca2",
     "result_summary": f"Mean pfs BRCA2+ vs BRCA2-: {a_b2['mean_1']:.3f} vs {a_b2['mean_0']:.3f}; diff={a_b2['effect']:.3f}; p={a_b2['p']:.3g}.",
     "p_value": fmt_p(a_b2["p"]), "effect_estimate": fmt_e(a_b2["effect"]),
     "significant": bool(a_b2["p"] < 0.05)},
    {"hypothesis_ids": ["h5.3"], "code": "ttest pik3ca",
     "result_summary": f"Mean pfs PIK3CA+ vs PIK3CA-: {a_pk['mean_1']:.3f} vs {a_pk['mean_0']:.3f}; diff={a_pk['effect']:.3f}; p={a_pk['p']:.3g}.",
     "p_value": fmt_p(a_pk["p"]), "effect_estimate": fmt_e(a_pk["effect"]),
     "significant": bool(a_pk["p"] < 0.05)},
]
add_iter(5, hyps, analyses)

# =====================================================================
# Iteration 6: lab markers — albumin, LDH, weight loss
# =====================================================================
hyps = [
    {"id": "h6.1", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
    {"id": "h6.2", "text": "Higher ldh_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h6.3", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
a_alb = ols_simple(df, "albumin_g_dl")
a_ldh = ols_simple(df, "ldh_u_l")
a_wl = ols_simple(df, "weight_loss_pct_6mo")
analyses = [
    {"hypothesis_ids": ["h6.1"], "code": "OLS pfs ~ albumin",
     "result_summary": f"Slope pfs on albumin = {a_alb['beta']:.4f} (p={a_alb['p']:.3g}).",
     "p_value": fmt_p(a_alb["p"]), "effect_estimate": fmt_e(a_alb["beta"]),
     "significant": bool(a_alb["p"] < 0.05)},
    {"hypothesis_ids": ["h6.2"], "code": "OLS pfs ~ ldh",
     "result_summary": f"Slope pfs on LDH = {a_ldh['beta']:.6f} (p={a_ldh['p']:.3g}).",
     "p_value": fmt_p(a_ldh["p"]), "effect_estimate": fmt_e(a_ldh["beta"]),
     "significant": bool(a_ldh["p"] < 0.05)},
    {"hypothesis_ids": ["h6.3"], "code": "OLS pfs ~ weight_loss",
     "result_summary": f"Slope pfs on weight_loss_pct_6mo = {a_wl['beta']:.4f} (p={a_wl['p']:.3g}).",
     "p_value": fmt_p(a_wl["p"]), "effect_estimate": fmt_e(a_wl["beta"]),
     "significant": bool(a_wl["p"] < 0.05)},
]
add_iter(6, hyps, analyses)

# =====================================================================
# Iteration 7: inflammation/blood — CRP, NLR, hemoglobin
# =====================================================================
hyps = [
    {"id": "h7.1", "text": "Higher crp_mg_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h7.2", "text": "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h7.3", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
]
a_crp = ols_simple(df, "crp_mg_l")
a_nlr = ols_simple(df, "nlr")
a_hb = ols_simple(df, "hemoglobin_g_dl")
analyses = [
    {"hypothesis_ids": ["h7.1"], "code": "OLS pfs ~ crp",
     "result_summary": f"Slope pfs on CRP = {a_crp['beta']:.5f} (p={a_crp['p']:.3g}).",
     "p_value": fmt_p(a_crp["p"]), "effect_estimate": fmt_e(a_crp["beta"]),
     "significant": bool(a_crp["p"] < 0.05)},
    {"hypothesis_ids": ["h7.2"], "code": "OLS pfs ~ nlr",
     "result_summary": f"Slope pfs on NLR = {a_nlr['beta']:.5f} (p={a_nlr['p']:.3g}).",
     "p_value": fmt_p(a_nlr["p"]), "effect_estimate": fmt_e(a_nlr["beta"]),
     "significant": bool(a_nlr["p"] < 0.05)},
    {"hypothesis_ids": ["h7.3"], "code": "OLS pfs ~ hemoglobin",
     "result_summary": f"Slope pfs on hemoglobin = {a_hb['beta']:.4f} (p={a_hb['p']:.3g}).",
     "p_value": fmt_p(a_hb["p"]), "effect_estimate": fmt_e(a_hb["beta"]),
     "significant": bool(a_hb["p"] < 0.05)},
]
add_iter(7, hyps, analyses)

# =====================================================================
# Iteration 8: liver/renal labs
# =====================================================================
hyps = [
    {"id": "h8.1", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h8.2", "text": "Higher ast_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h8.3", "text": "Higher alt_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h8.4", "text": "Higher total_bilirubin_mg_dl is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h8.5", "text": "Higher creatinine_mg_dl is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
a_alp = ols_simple(df, "alkaline_phosphatase_u_l")
a_ast = ols_simple(df, "ast_u_l")
a_alt = ols_simple(df, "alt_u_l")
a_tb = ols_simple(df, "total_bilirubin_mg_dl")
a_cr = ols_simple(df, "creatinine_mg_dl")
analyses = [
    {"hypothesis_ids": ["h8.1"], "code": "OLS pfs ~ alp",
     "result_summary": f"Slope pfs on ALP = {a_alp['beta']:.6f} (p={a_alp['p']:.3g}).",
     "p_value": fmt_p(a_alp["p"]), "effect_estimate": fmt_e(a_alp["beta"]),
     "significant": bool(a_alp["p"] < 0.05)},
    {"hypothesis_ids": ["h8.2"], "code": "OLS pfs ~ ast",
     "result_summary": f"Slope pfs on AST = {a_ast['beta']:.6f} (p={a_ast['p']:.3g}).",
     "p_value": fmt_p(a_ast["p"]), "effect_estimate": fmt_e(a_ast["beta"]),
     "significant": bool(a_ast["p"] < 0.05)},
    {"hypothesis_ids": ["h8.3"], "code": "OLS pfs ~ alt",
     "result_summary": f"Slope pfs on ALT = {a_alt['beta']:.6f} (p={a_alt['p']:.3g}).",
     "p_value": fmt_p(a_alt["p"]), "effect_estimate": fmt_e(a_alt["beta"]),
     "significant": bool(a_alt["p"] < 0.05)},
    {"hypothesis_ids": ["h8.4"], "code": "OLS pfs ~ bilirubin",
     "result_summary": f"Slope pfs on bilirubin = {a_tb['beta']:.4f} (p={a_tb['p']:.3g}).",
     "p_value": fmt_p(a_tb["p"]), "effect_estimate": fmt_e(a_tb["beta"]),
     "significant": bool(a_tb["p"] < 0.05)},
    {"hypothesis_ids": ["h8.5"], "code": "OLS pfs ~ creatinine",
     "result_summary": f"Slope pfs on creatinine = {a_cr['beta']:.4f} (p={a_cr['p']:.3g}).",
     "p_value": fmt_p(a_cr["p"]), "effect_estimate": fmt_e(a_cr["beta"]),
     "significant": bool(a_cr["p"] < 0.05)},
]
add_iter(8, hyps, analyses)

# =====================================================================
# Iteration 9: electrolytes & calcium
# =====================================================================
hyps = [
    {"id": "h9.1", "text": "Higher sodium_meq_l is associated with longer pfs_months (positive slope).", "kind": "novel"},
    {"id": "h9.2", "text": "Higher potassium_meq_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h9.3", "text": "Higher calcium_mg_dl is associated with shorter pfs_months (hypercalcemia is unfavorable).", "kind": "novel"},
    {"id": "h9.4", "text": "Higher bun_mg_dl is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
a_na = ols_simple(df, "sodium_meq_l")
a_k = ols_simple(df, "potassium_meq_l")
a_ca = ols_simple(df, "calcium_mg_dl")
a_bun = ols_simple(df, "bun_mg_dl")
analyses = [
    {"hypothesis_ids": ["h9.1"], "code": "OLS pfs ~ sodium",
     "result_summary": f"Slope pfs on sodium = {a_na['beta']:.4f} (p={a_na['p']:.3g}).",
     "p_value": fmt_p(a_na["p"]), "effect_estimate": fmt_e(a_na["beta"]),
     "significant": bool(a_na["p"] < 0.05)},
    {"hypothesis_ids": ["h9.2"], "code": "OLS pfs ~ potassium",
     "result_summary": f"Slope pfs on potassium = {a_k['beta']:.4f} (p={a_k['p']:.3g}).",
     "p_value": fmt_p(a_k["p"]), "effect_estimate": fmt_e(a_k["beta"]),
     "significant": bool(a_k["p"] < 0.05)},
    {"hypothesis_ids": ["h9.3"], "code": "OLS pfs ~ calcium",
     "result_summary": f"Slope pfs on calcium = {a_ca['beta']:.4f} (p={a_ca['p']:.3g}).",
     "p_value": fmt_p(a_ca["p"]), "effect_estimate": fmt_e(a_ca["beta"]),
     "significant": bool(a_ca["p"] < 0.05)},
    {"hypothesis_ids": ["h9.4"], "code": "OLS pfs ~ bun",
     "result_summary": f"Slope pfs on BUN = {a_bun['beta']:.4f} (p={a_bun['p']:.3g}).",
     "p_value": fmt_p(a_bun["p"]), "effect_estimate": fmt_e(a_bun["beta"]),
     "significant": bool(a_bun["p"] < 0.05)},
]
add_iter(9, hyps, analyses)

# =====================================================================
# Iteration 10: multivariable baseline model — does each prognostic stay?
# =====================================================================
hyps = [
    {"id": "h10.1", "text": "In a multivariable OLS adjusting for ECOG, stage_iv, has_brain_mets, age_years, albumin_g_dl, ldh_u_l, weight_loss_pct_6mo, crp_mg_l, nlr, hemoglobin_g_dl, ki67_pct, tumor_size_cm, alkaline_phosphatase_u_l, calcium_mg_dl, ECOG remains negatively associated with pfs_months.", "kind": "novel"},
    {"id": "h10.2", "text": "In the same multivariable model, albumin_g_dl remains positively associated with pfs_months.", "kind": "novel"},
    {"id": "h10.3", "text": "In the same multivariable model, weight_loss_pct_6mo remains negatively associated with pfs_months.", "kind": "novel"},
]
m10 = ols_formula(df, "pfs_months ~ ecog_ps + stage_iv + has_brain_mets + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl + ki67_pct + tumor_size_cm + alkaline_phosphatase_u_l + calcium_mg_dl")
def grab(model, name):
    return float(model.params[name]), float(model.pvalues[name])
b_e, p_e = grab(m10, "ecog_ps")
b_a, p_a = grab(m10, "albumin_g_dl")
b_w, p_w = grab(m10, "weight_loss_pct_6mo")
analyses = [
    {"hypothesis_ids": ["h10.1"], "code": "smf.ols(...).fit() multivariable",
     "result_summary": f"Adjusted ECOG slope = {b_e:.4f} (p={p_e:.3g}).",
     "p_value": fmt_p(p_e), "effect_estimate": fmt_e(b_e),
     "significant": bool(p_e < 0.05)},
    {"hypothesis_ids": ["h10.2"], "code": "same model",
     "result_summary": f"Adjusted albumin slope = {b_a:.4f} (p={p_a:.3g}).",
     "p_value": fmt_p(p_a), "effect_estimate": fmt_e(b_a),
     "significant": bool(p_a < 0.05)},
    {"hypothesis_ids": ["h10.3"], "code": "same model",
     "result_summary": f"Adjusted weight_loss slope = {b_w:.4f} (p={p_w:.3g}); model R^2 = {m10.rsquared:.4f}.",
     "p_value": fmt_p(p_w), "effect_estimate": fmt_e(b_w),
     "significant": bool(p_w < 0.05)},
]
add_iter(10, hyps, analyses)

# =====================================================================
# Iteration 11: marginal treatment main effects (each treatment alone)
# =====================================================================
hyps = []
analyses = []
for i, t in enumerate(TRT_COLS, 1):
    hid = f"h11.{i}"
    hyps.append({"id": hid, "text": f"Patients receiving {t} have different mean pfs_months than patients not receiving it (unadjusted main effect).", "kind": "novel"})
    res = ttest_by(df, t)
    analyses.append({
        "hypothesis_ids": [hid], "code": f"ttest {t}",
        "result_summary": f"Mean pfs on {t}=1 vs 0: {res['mean_1']:.3f} vs {res['mean_0']:.3f}; diff={res['effect']:.3f}; p={res['p']:.3g}.",
        "p_value": fmt_p(res["p"]), "effect_estimate": fmt_e(res["effect"]),
        "significant": bool(res["p"] < 0.05)
    })
add_iter(11, hyps, analyses)

# =====================================================================
# Iteration 12: treatment effects adjusted for prognostics
# =====================================================================
adj_terms = "ecog_ps + stage_iv + has_brain_mets + age_years + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl + ki67_pct + tumor_size_cm + alkaline_phosphatase_u_l + calcium_mg_dl + er_positive + pr_positive + her2_positive + her2_low + brca1_mutation + brca2_mutation + pik3ca_mutation + node_positive + postmenopausal + sex_female"
hyps = []
analyses = []
for i, t in enumerate(TRT_COLS, 1):
    hid = f"h12.{i}"
    hyps.append({"id": hid, "text": f"Adjusting for prognostic features (ECOG, stage_iv, brain_mets, age, albumin, LDH, weight_loss, CRP, NLR, Hb, Ki67, tumor size, ALP, Ca, ER/PR/HER2/HER2-low, BRCA1/2, PIK3CA, node, postmenopausal, sex), {t} remains independently associated with pfs_months (signed effect).", "kind": "refined"})
    m = ols_formula(df, f"pfs_months ~ {t} + {adj_terms}")
    b, p = grab(m, t)
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols(pfs_months ~ {t} + adj).fit()",
        "result_summary": f"Adjusted effect of {t} on pfs_months = {b:.4f} months (p={p:.3g}); model R^2 = {m.rsquared:.4f}.",
        "p_value": fmt_p(p), "effect_estimate": fmt_e(b),
        "significant": bool(p < 0.05)
    })
add_iter(12, hyps, analyses)

# =====================================================================
# Iteration 13: tamoxifen × ER, PR (the textbook interaction)
# =====================================================================
hyps = [
    {"id": "h13.1", "text": "treatment_tamoxifen prolongs pfs_months only in ER-positive (er_positive=1) patients; the er_positive × treatment_tamoxifen interaction is positive (treatment effect larger in ER+).", "kind": "novel"},
    {"id": "h13.2", "text": "Among ER-positive patients, treatment_tamoxifen has a positive effect on pfs_months; among ER-negative patients, the effect is null or negative.", "kind": "refined"},
    {"id": "h13.3", "text": "treatment_tamoxifen × pr_positive interaction: tamoxifen effect is larger when pr_positive=1.", "kind": "novel"},
]
m13a = ols_formula(df, f"pfs_months ~ treatment_tamoxifen * er_positive + {adj_terms}")
b_int, p_int = grab(m13a, "treatment_tamoxifen:er_positive")
b_main, p_main = grab(m13a, "treatment_tamoxifen")
# stratified
def strat_effect(df, trt, mask):
    sub = df.loc[mask]
    m = ols_formula(sub, f"pfs_months ~ {trt} + {adj_terms}")
    b, p = grab(m, trt)
    return b, p, len(sub)
b_er1, p_er1, n_er1 = strat_effect(df, "treatment_tamoxifen", df["er_positive"] == 1)
b_er0, p_er0, n_er0 = strat_effect(df, "treatment_tamoxifen", df["er_positive"] == 0)
m13b = ols_formula(df, f"pfs_months ~ treatment_tamoxifen * pr_positive + {adj_terms}")
b_int2, p_int2 = grab(m13b, "treatment_tamoxifen:pr_positive")
analyses = [
    {"hypothesis_ids": ["h13.1"], "code": "OLS pfs ~ tamoxifen*er + adj",
     "result_summary": f"Interaction tamoxifen:er_positive = {b_int:.4f} (p={p_int:.3g}); main tamoxifen (in ER-) = {b_main:.4f} (p={p_main:.3g}).",
     "p_value": fmt_p(p_int), "effect_estimate": fmt_e(b_int),
     "significant": bool(p_int < 0.05)},
    {"hypothesis_ids": ["h13.2"], "code": "stratified OLS",
     "result_summary": f"In ER+ (n={n_er1}): tamoxifen effect = {b_er1:.4f} (p={p_er1:.3g}); In ER- (n={n_er0}): tamoxifen effect = {b_er0:.4f} (p={p_er0:.3g}).",
     "p_value": fmt_p(p_er1), "effect_estimate": fmt_e(b_er1),
     "significant": bool(p_er1 < 0.05)},
    {"hypothesis_ids": ["h13.3"], "code": "OLS pfs ~ tamoxifen*pr + adj",
     "result_summary": f"Interaction tamoxifen:pr_positive = {b_int2:.4f} (p={p_int2:.3g}).",
     "p_value": fmt_p(p_int2), "effect_estimate": fmt_e(b_int2),
     "significant": bool(p_int2 < 0.05)},
]
add_iter(13, hyps, analyses)

# =====================================================================
# Iteration 14: trastuzumab × HER2
# =====================================================================
hyps = [
    {"id": "h14.1", "text": "treatment_trastuzumab × her2_positive interaction is positive: trastuzumab effect on pfs_months is larger in HER2-positive patients.", "kind": "novel"},
    {"id": "h14.2", "text": "Among HER2-positive patients, treatment_trastuzumab has a positive effect on pfs_months; among HER2-negative patients, the effect is null or negative.", "kind": "refined"},
    {"id": "h14.3", "text": "treatment_trastuzumab × her2_low interaction: trastuzumab is not expected to benefit HER2-low patients (interaction with her2_low ≈ 0).", "kind": "novel"},
]
m14 = ols_formula(df, f"pfs_months ~ treatment_trastuzumab * her2_positive + {adj_terms}")
b_int, p_int = grab(m14, "treatment_trastuzumab:her2_positive")
b_h1, p_h1, n_h1 = strat_effect(df, "treatment_trastuzumab", df["her2_positive"] == 1)
b_h0, p_h0, n_h0 = strat_effect(df, "treatment_trastuzumab", df["her2_positive"] == 0)
m14b = ols_formula(df, f"pfs_months ~ treatment_trastuzumab * her2_low + {adj_terms}")
b_int2, p_int2 = grab(m14b, "treatment_trastuzumab:her2_low")
analyses = [
    {"hypothesis_ids": ["h14.1"], "code": "OLS pfs ~ trastuzumab*her2_pos + adj",
     "result_summary": f"Interaction trastuzumab:her2_positive = {b_int:.4f} (p={p_int:.3g}).",
     "p_value": fmt_p(p_int), "effect_estimate": fmt_e(b_int),
     "significant": bool(p_int < 0.05)},
    {"hypothesis_ids": ["h14.2"], "code": "stratified OLS",
     "result_summary": f"In HER2+ (n={n_h1}): trastuzumab effect = {b_h1:.4f} (p={p_h1:.3g}); in HER2- (n={n_h0}): {b_h0:.4f} (p={p_h0:.3g}).",
     "p_value": fmt_p(p_h1), "effect_estimate": fmt_e(b_h1),
     "significant": bool(p_h1 < 0.05)},
    {"hypothesis_ids": ["h14.3"], "code": "OLS pfs ~ trastuzumab*her2_low + adj",
     "result_summary": f"Interaction trastuzumab:her2_low = {b_int2:.4f} (p={p_int2:.3g}).",
     "p_value": fmt_p(p_int2), "effect_estimate": fmt_e(b_int2),
     "significant": bool(p_int2 < 0.05)},
]
add_iter(14, hyps, analyses)

# =====================================================================
# Iteration 15: olaparib × BRCA1/BRCA2
# =====================================================================
hyps = [
    {"id": "h15.1", "text": "treatment_olaparib × brca1_mutation interaction is positive: olaparib prolongs pfs_months selectively in BRCA1-mutated patients.", "kind": "novel"},
    {"id": "h15.2", "text": "treatment_olaparib × brca2_mutation interaction is positive: olaparib prolongs pfs_months selectively in BRCA2-mutated patients.", "kind": "novel"},
    {"id": "h15.3", "text": "Among patients carrying any BRCA1 or BRCA2 mutation, treatment_olaparib has a positive effect on pfs_months; among non-carriers the effect is null or negative.", "kind": "refined"},
]
m15a = ols_formula(df, f"pfs_months ~ treatment_olaparib * brca1_mutation + {adj_terms}")
b_int_b1, p_int_b1 = grab(m15a, "treatment_olaparib:brca1_mutation")
m15b = ols_formula(df, f"pfs_months ~ treatment_olaparib * brca2_mutation + {adj_terms}")
b_int_b2, p_int_b2 = grab(m15b, "treatment_olaparib:brca2_mutation")
df = df.copy()
df["brca_any"] = ((df["brca1_mutation"] == 1) | (df["brca2_mutation"] == 1)).astype(int)
m15c = ols_formula(df, f"pfs_months ~ treatment_olaparib * brca_any + {adj_terms}")
b_int_any, p_int_any = grab(m15c, "treatment_olaparib:brca_any")
b_carr, p_carr, n_carr = strat_effect(df, "treatment_olaparib", df["brca_any"] == 1)
b_nonc, p_nonc, n_nonc = strat_effect(df, "treatment_olaparib", df["brca_any"] == 0)
analyses = [
    {"hypothesis_ids": ["h15.1"], "code": "OLS pfs ~ olaparib*brca1 + adj",
     "result_summary": f"Interaction olaparib:brca1 = {b_int_b1:.4f} (p={p_int_b1:.3g}).",
     "p_value": fmt_p(p_int_b1), "effect_estimate": fmt_e(b_int_b1),
     "significant": bool(p_int_b1 < 0.05)},
    {"hypothesis_ids": ["h15.2"], "code": "OLS pfs ~ olaparib*brca2 + adj",
     "result_summary": f"Interaction olaparib:brca2 = {b_int_b2:.4f} (p={p_int_b2:.3g}).",
     "p_value": fmt_p(p_int_b2), "effect_estimate": fmt_e(b_int_b2),
     "significant": bool(p_int_b2 < 0.05)},
    {"hypothesis_ids": ["h15.3"], "code": "OLS pfs ~ olaparib*brca_any + adj, then stratified",
     "result_summary": f"Interaction olaparib:brca_any = {b_int_any:.4f} (p={p_int_any:.3g}). In BRCA1/2 carriers (n={n_carr}): olaparib effect = {b_carr:.4f} (p={p_carr:.3g}); in non-carriers (n={n_nonc}): {b_nonc:.4f} (p={p_nonc:.3g}).",
     "p_value": fmt_p(p_int_any), "effect_estimate": fmt_e(b_int_any),
     "significant": bool(p_int_any < 0.05)},
]
add_iter(15, hyps, analyses)

# =====================================================================
# Iteration 16: palbociclib × ER, palbociclib × her2_positive
# =====================================================================
hyps = [
    {"id": "h16.1", "text": "treatment_palbociclib × er_positive interaction is positive: palbociclib prolongs pfs_months preferentially in ER-positive (HR+) patients.", "kind": "novel"},
    {"id": "h16.2", "text": "treatment_palbociclib × her2_positive interaction is negative: palbociclib has reduced or null effect in HER2-positive patients.", "kind": "novel"},
    {"id": "h16.3", "text": "In ER-positive AND HER2-negative patients (i.e., HR+/HER2-), treatment_palbociclib has a positive effect on pfs_months.", "kind": "refined"},
]
m16a = ols_formula(df, f"pfs_months ~ treatment_palbociclib * er_positive + {adj_terms}")
b_int_e, p_int_e = grab(m16a, "treatment_palbociclib:er_positive")
m16b = ols_formula(df, f"pfs_months ~ treatment_palbociclib * her2_positive + {adj_terms}")
b_int_h, p_int_h = grab(m16b, "treatment_palbociclib:her2_positive")
mask_hrp_h2n = (df["er_positive"] == 1) & (df["her2_positive"] == 0)
b_hrh, p_hrh, n_hrh = strat_effect(df, "treatment_palbociclib", mask_hrp_h2n)
analyses = [
    {"hypothesis_ids": ["h16.1"], "code": "OLS pfs ~ palbociclib*er_pos + adj",
     "result_summary": f"Interaction palbociclib:er_positive = {b_int_e:.4f} (p={p_int_e:.3g}).",
     "p_value": fmt_p(p_int_e), "effect_estimate": fmt_e(b_int_e),
     "significant": bool(p_int_e < 0.05)},
    {"hypothesis_ids": ["h16.2"], "code": "OLS pfs ~ palbociclib*her2_pos + adj",
     "result_summary": f"Interaction palbociclib:her2_positive = {b_int_h:.4f} (p={p_int_h:.3g}).",
     "p_value": fmt_p(p_int_h), "effect_estimate": fmt_e(b_int_h),
     "significant": bool(p_int_h < 0.05)},
    {"hypothesis_ids": ["h16.3"], "code": "stratified OLS in ER+/HER2-",
     "result_summary": f"Among HR+/HER2- (n={n_hrh}): palbociclib adjusted effect = {b_hrh:.4f} months (p={p_hrh:.3g}).",
     "p_value": fmt_p(p_hrh), "effect_estimate": fmt_e(b_hrh),
     "significant": bool(p_hrh < 0.05)},
]
add_iter(16, hyps, analyses)

# =====================================================================
# Iteration 17: pembrolizumab × ER (TNBC enrichment proxy)
# =====================================================================
df["tnbc"] = ((df["er_positive"] == 0) & (df["pr_positive"] == 0) & (df["her2_positive"] == 0)).astype(int)
hyps = [
    {"id": "h17.1", "text": "treatment_pembrolizumab × er_positive interaction: pembrolizumab effect is more favorable in ER-negative patients (negative interaction with er_positive).", "kind": "novel"},
    {"id": "h17.2", "text": "treatment_pembrolizumab has a positive effect on pfs_months in triple-negative breast cancer (er_positive=0 AND pr_positive=0 AND her2_positive=0).", "kind": "novel"},
    {"id": "h17.3", "text": "treatment_sacituzumab_govitecan has a positive effect on pfs_months in triple-negative breast cancer.", "kind": "novel"},
]
m17a = ols_formula(df, f"pfs_months ~ treatment_pembrolizumab * er_positive + {adj_terms}")
b_pe_er, p_pe_er = grab(m17a, "treatment_pembrolizumab:er_positive")
m17b = ols_formula(df, f"pfs_months ~ treatment_pembrolizumab + {adj_terms}", )
b_pe_tnbc, p_pe_tnbc, n_pe_tnbc = strat_effect(df, "treatment_pembrolizumab", df["tnbc"] == 1)
b_sg_tnbc, p_sg_tnbc, n_sg_tnbc = strat_effect(df, "treatment_sacituzumab_govitecan", df["tnbc"] == 1)
analyses = [
    {"hypothesis_ids": ["h17.1"], "code": "OLS pfs ~ pembro*er_pos + adj",
     "result_summary": f"Interaction pembrolizumab:er_positive = {b_pe_er:.4f} (p={p_pe_er:.3g}).",
     "p_value": fmt_p(p_pe_er), "effect_estimate": fmt_e(b_pe_er),
     "significant": bool(p_pe_er < 0.05)},
    {"hypothesis_ids": ["h17.2"], "code": "stratified OLS in TNBC",
     "result_summary": f"Among TNBC (n={n_pe_tnbc}): pembrolizumab adjusted effect = {b_pe_tnbc:.4f} (p={p_pe_tnbc:.3g}).",
     "p_value": fmt_p(p_pe_tnbc), "effect_estimate": fmt_e(b_pe_tnbc),
     "significant": bool(p_pe_tnbc < 0.05)},
    {"hypothesis_ids": ["h17.3"], "code": "stratified OLS in TNBC",
     "result_summary": f"Among TNBC (n={n_sg_tnbc}): sacituzumab adjusted effect = {b_sg_tnbc:.4f} (p={p_sg_tnbc:.3g}).",
     "p_value": fmt_p(p_sg_tnbc), "effect_estimate": fmt_e(b_sg_tnbc),
     "significant": bool(p_sg_tnbc < 0.05)},
]
add_iter(17, hyps, analyses)

# =====================================================================
# Iteration 18: heterogeneity screen — interaction of every treatment with each binary feature
# =====================================================================
mod_features = ["sex_female", "stage_iv", "has_brain_mets", "node_positive", "postmenopausal",
                "er_positive", "pr_positive", "her2_positive", "her2_low",
                "brca1_mutation", "brca2_mutation", "pik3ca_mutation"]
hyps = [{"id": "h18.0", "text": "Across all six treatments and 12 binary modifiers, at least one significant treatment×modifier interaction on pfs_months exists beyond the obvious receptor matches.", "kind": "novel"}]
analyses = []
inter_records = []
for t in TRT_COLS:
    for f in mod_features:
        try:
            m = ols_formula(df, f"pfs_months ~ {t} * {f} + {adj_terms}")
            term = f"{t}:{f}"
            b, p = float(m.params[term]), float(m.pvalues[term])
            inter_records.append((t, f, b, p))
        except Exception:
            continue
inter_records.sort(key=lambda r: r[3])
top = inter_records[:8]
analyses.append({
    "hypothesis_ids": ["h18.0"],
    "code": "Loop OLS pfs ~ trt*feat + adj for trt in TRT_COLS, feat in 12 binary features",
    "result_summary": "Top-8 smallest interaction p-values:\n" + "\n".join([f"  {t}*{f}: beta={b:+.3f} p={p:.3g}" for t, f, b, p in top]),
    "p_value": fmt_p(top[0][3]) if top else None,
    "effect_estimate": fmt_e(top[0][2]) if top else None,
    "significant": bool(top[0][3] < 0.05) if top else None,
})
# Save inter_records
results["interaction_screen"] = [{"treatment": t, "feature": f, "beta": float(b), "p": float(p)} for t, f, b, p in inter_records]
add_iter(18, hyps, analyses)

# =====================================================================
# Iteration 19: heterogeneity by ECOG (a continuous-ish modifier)
# =====================================================================
hyps = [
    {"id": "h19.1", "text": "Treatment effects are blunted in patients with ecog_ps>=2: every treatment × ecog_ps interaction is non-positive (treatment effect smaller as ECOG rises).", "kind": "novel"},
    {"id": "h19.2", "text": "treatment_palbociclib's PFS benefit shrinks or disappears in patients with ecog_ps==2.", "kind": "refined"},
]
analyses = []
for t in TRT_COLS:
    m = ols_formula(df, f"pfs_months ~ {t} * ecog_ps + {adj_terms}")
    term = f"{t}:ecog_ps"
    b, p = float(m.params[term]), float(m.pvalues[term])
    analyses.append({
        "hypothesis_ids": ["h19.1"],
        "code": f"OLS pfs ~ {t}*ecog_ps + adj",
        "result_summary": f"Interaction {t}:ecog_ps = {b:+.4f} (p={p:.3g}).",
        "p_value": fmt_p(p), "effect_estimate": fmt_e(b),
        "significant": bool(p < 0.05),
    })
b_pal_e0, p_pal_e0, n_pal_e0 = strat_effect(df, "treatment_palbociclib", df["ecog_ps"] == 0)
b_pal_e2, p_pal_e2, n_pal_e2 = strat_effect(df, "treatment_palbociclib", df["ecog_ps"] == 2)
analyses.append({
    "hypothesis_ids": ["h19.2"],
    "code": "stratified OLS by ecog_ps",
    "result_summary": f"Palbociclib effect at ECOG=0 (n={n_pal_e0}): {b_pal_e0:+.4f} (p={p_pal_e0:.3g}); ECOG=2 (n={n_pal_e2}): {b_pal_e2:+.4f} (p={p_pal_e2:.3g}).",
    "p_value": fmt_p(p_pal_e2), "effect_estimate": fmt_e(b_pal_e2),
    "significant": bool(p_pal_e2 < 0.05),
})
add_iter(19, hyps, analyses)

# =====================================================================
# Iteration 20: heterogeneity by visceral burden (brain mets, stage_iv) and labs
# =====================================================================
hyps = [
    {"id": "h20.1", "text": "treatment_trastuzumab benefit (the trastuzumab × her2_positive effect) is preserved even in patients with brain mets (no negative trastuzumab × has_brain_mets interaction restricted to HER2+).", "kind": "novel"},
    {"id": "h20.2", "text": "treatment_olaparib benefit (in BRCA carriers) is attenuated by high LDH (negative olaparib × ldh_u_l interaction within BRCA carriers).", "kind": "novel"},
    {"id": "h20.3", "text": "Among ER-positive AND HER2-negative AND ECOG<=1 patients, treatment_palbociclib produces a clear positive PFS effect.", "kind": "refined"},
]
mask_h2p = df["her2_positive"] == 1
m20a = ols_formula(df.loc[mask_h2p].copy(), f"pfs_months ~ treatment_trastuzumab * has_brain_mets + {adj_terms}")
b_tr_bm, p_tr_bm = grab(m20a, "treatment_trastuzumab:has_brain_mets")
mask_brca = df["brca_any"] == 1
m20b = ols_formula(df.loc[mask_brca].copy(), f"pfs_months ~ treatment_olaparib * ldh_u_l + {adj_terms}")
b_ol_l, p_ol_l = grab(m20b, "treatment_olaparib:ldh_u_l")
mask_hrh_e = mask_hrp_h2n & (df["ecog_ps"] <= 1)
b_pal_g, p_pal_g, n_pal_g = strat_effect(df, "treatment_palbociclib", mask_hrh_e)
analyses = [
    {"hypothesis_ids": ["h20.1"], "code": "OLS in HER2+ subset",
     "result_summary": f"In HER2+ (n={int(mask_h2p.sum())}): trastuzumab:has_brain_mets = {b_tr_bm:+.4f} (p={p_tr_bm:.3g}).",
     "p_value": fmt_p(p_tr_bm), "effect_estimate": fmt_e(b_tr_bm),
     "significant": bool(p_tr_bm < 0.05)},
    {"hypothesis_ids": ["h20.2"], "code": "OLS in BRCA carriers",
     "result_summary": f"Among BRCA1/2 carriers (n={int(mask_brca.sum())}): olaparib:ldh_u_l = {b_ol_l:+.6f} (p={p_ol_l:.3g}).",
     "p_value": fmt_p(p_ol_l), "effect_estimate": fmt_e(b_ol_l),
     "significant": bool(p_ol_l < 0.05)},
    {"hypothesis_ids": ["h20.3"], "code": "stratified OLS in HR+/HER2-/ECOG<=1",
     "result_summary": f"Among HR+/HER2- with ECOG<=1 (n={n_pal_g}): palbociclib effect = {b_pal_g:+.4f} (p={p_pal_g:.3g}).",
     "p_value": fmt_p(p_pal_g), "effect_estimate": fmt_e(b_pal_g),
     "significant": bool(p_pal_g < 0.05)},
]
add_iter(20, hyps, analyses)

# =====================================================================
# Iteration 21: subgroup discovery — exhaustive 2-feature subgroups for the largest treatment effects
# =====================================================================
hyps = [
    {"id": "h21.1", "text": "For each treatment, the patient subgroup with the largest adjusted PFS benefit is defined by 1-2 binary biomarkers that match canonical breast-cancer biology (e.g., olaparib in BRCA carriers, trastuzumab in HER2+, tamoxifen/palbociclib in ER+).", "kind": "novel"},
]
analyses = []
sub_features = ["er_positive", "pr_positive", "her2_positive", "her2_low",
                "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
                "node_positive", "postmenopausal", "stage_iv", "has_brain_mets"]

best_records = {}
for t in TRT_COLS:
    rows = []
    # 1-feature
    for f in sub_features:
        for v in (0, 1):
            mask = df[f] == v
            if mask.sum() < 200:
                continue
            try:
                b, p, n = strat_effect(df, t, mask)
                rows.append((f"{f}={v}", b, p, n))
            except Exception:
                pass
    # 2-feature combinations (limited to top biomarkers to keep runtime bounded)
    pri = ["er_positive", "her2_positive", "her2_low", "brca1_mutation", "brca2_mutation",
           "pik3ca_mutation", "stage_iv", "has_brain_mets", "pr_positive"]
    for i in range(len(pri)):
        for j in range(i+1, len(pri)):
            f1, f2 = pri[i], pri[j]
            for v1 in (0, 1):
                for v2 in (0, 1):
                    mask = (df[f1] == v1) & (df[f2] == v2)
                    if mask.sum() < 300:
                        continue
                    try:
                        b, p, n = strat_effect(df, t, mask)
                        rows.append((f"{f1}={v1} & {f2}={v2}", b, p, n))
                    except Exception:
                        pass
    rows.sort(key=lambda r: r[1], reverse=True)
    best = rows[:5]
    best_records[t] = best
    summary = "; ".join([f"[{lbl} (n={n}): {b:+.3f} months, p={p:.3g}]" for lbl, b, p, n in best])
    analyses.append({
        "hypothesis_ids": ["h21.1"],
        "code": f"exhaustive 1-2 feature subgroup search; adjusted effect of {t}",
        "result_summary": f"Top-5 subgroups for {t} ranked by adjusted PFS benefit: {summary}",
        "p_value": fmt_p(best[0][2]) if best else None,
        "effect_estimate": fmt_e(best[0][1]) if best else None,
        "significant": bool(best[0][2] < 0.05) if best else None,
    })
add_iter(21, hyps, analyses)

# =====================================================================
# Iteration 22: tree-based subgroup discovery (causal-style: residuals after baseline)
# =====================================================================
hyps = [
    {"id": "h22.1", "text": "Greedy 3-feature conjunctions search recovers, for each treatment, a small subgroup definition where the adjusted PFS benefit is maximal — matching canonical biomarker-defined subgroups.", "kind": "novel"},
]
analyses = []
search_features = ["er_positive", "pr_positive", "her2_positive", "her2_low",
                   "brca1_mutation", "brca2_mutation", "pik3ca_mutation",
                   "node_positive", "postmenopausal", "stage_iv", "has_brain_mets",
                   "sex_female", "brca_any"]

def best_conjunction(t, max_depth=3, min_n=300):
    """Greedy search: extend the conjunction with the literal that most increases the adjusted effect."""
    current_mask = pd.Series(True, index=df.index)
    current_label = []
    history = []
    for depth in range(max_depth):
        best = None
        for f in search_features:
            for v in (0, 1):
                lit = (f, v)
                if any(l[0] == f for l in current_label):
                    continue
                m = current_mask & (df[f] == v)
                if m.sum() < min_n:
                    continue
                try:
                    b, p, n = strat_effect(df, t, m)
                except Exception:
                    continue
                if best is None or b > best[0]:
                    best = (b, p, n, lit, m)
        if best is None:
            break
        b, p, n, lit, m = best
        current_label.append(lit)
        current_mask = m
        history.append((current_label[:], b, p, n))
    return history

for t in TRT_COLS:
    hist = best_conjunction(t)
    desc = "; ".join([f"depth-{i+1} {' & '.join([f'{f}={v}' for f, v in lab])} (n={n}): {b:+.3f} (p={p:.3g})"
                     for i, (lab, b, p, n) in enumerate(hist)])
    final = hist[-1] if hist else None
    analyses.append({
        "hypothesis_ids": ["h22.1"],
        "code": f"Greedy conjunction subgroup search for {t}",
        "result_summary": f"Greedy subgroup path for {t}: {desc}",
        "p_value": fmt_p(final[2]) if final else None,
        "effect_estimate": fmt_e(final[1]) if final else None,
        "significant": bool(final[2] < 0.05) if final else None,
    })
add_iter(22, hyps, analyses)

# =====================================================================
# Iteration 23: confirm full subgroup definitions with suppressing variables
# =====================================================================
hyps = [
    {"id": "h23.1", "text": "treatment_olaparib produces a positive PFS effect specifically in patients carrying brca1_mutation=1 OR brca2_mutation=1; in non-carriers the adjusted effect is null.", "kind": "refined"},
    {"id": "h23.2", "text": "treatment_trastuzumab produces a positive PFS effect in HER2-positive (her2_positive=1) patients only; in HER2-negative patients the effect is null or negative.", "kind": "refined"},
    {"id": "h23.3", "text": "treatment_palbociclib produces a positive PFS effect specifically in ER-positive AND HER2-negative patients; in HER2-positive or ER-negative patients the effect is null or negative.", "kind": "refined"},
    {"id": "h23.4", "text": "treatment_tamoxifen produces a positive PFS effect specifically in ER-positive patients; in ER-negative patients the effect is null or negative.", "kind": "refined"},
    {"id": "h23.5", "text": "treatment_sacituzumab_govitecan produces a positive PFS effect in ER-negative AND HER2-negative (i.e., TNBC-leaning) patients.", "kind": "refined"},
    {"id": "h23.6", "text": "treatment_pembrolizumab produces a positive PFS effect in ER-negative AND HER2-negative (TNBC-leaning) patients; in ER-positive disease the effect is null or negative.", "kind": "refined"},
]
analyses = []
def confirm(t, label, mask_in, mask_out):
    b_in, p_in, n_in = strat_effect(df, t, mask_in)
    b_out, p_out, n_out = strat_effect(df, t, mask_out)
    return {
        "code": f"stratified OLS, {t}",
        "result_summary": f"{label}: subgroup effect = {b_in:+.4f} (p={p_in:.3g}, n={n_in}); complement effect = {b_out:+.4f} (p={p_out:.3g}, n={n_out}).",
        "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in),
        "significant": bool(p_in < 0.05),
    }
mask_brca_pos = df["brca_any"] == 1
res = confirm("treatment_olaparib", "olaparib in BRCA1/2-carriers vs non-carriers", mask_brca_pos, ~mask_brca_pos)
res["hypothesis_ids"] = ["h23.1"]; analyses.append(res)
mask_h2p = df["her2_positive"] == 1
res = confirm("treatment_trastuzumab", "trastuzumab in HER2+ vs HER2-", mask_h2p, ~mask_h2p)
res["hypothesis_ids"] = ["h23.2"]; analyses.append(res)
mask_pal = (df["er_positive"] == 1) & (df["her2_positive"] == 0)
res = confirm("treatment_palbociclib", "palbociclib in ER+/HER2- vs others", mask_pal, ~mask_pal)
res["hypothesis_ids"] = ["h23.3"]; analyses.append(res)
mask_er = df["er_positive"] == 1
res = confirm("treatment_tamoxifen", "tamoxifen in ER+ vs ER-", mask_er, ~mask_er)
res["hypothesis_ids"] = ["h23.4"]; analyses.append(res)
mask_dn = (df["er_positive"] == 0) & (df["her2_positive"] == 0)
res = confirm("treatment_sacituzumab_govitecan", "sacituzumab in ER-/HER2- vs others", mask_dn, ~mask_dn)
res["hypothesis_ids"] = ["h23.5"]; analyses.append(res)
res = confirm("treatment_pembrolizumab", "pembrolizumab in ER-/HER2- vs others", mask_dn, ~mask_dn)
res["hypothesis_ids"] = ["h23.6"]; analyses.append(res)
add_iter(23, hyps, analyses)

# =====================================================================
# Iteration 24: include all interactions in one joint model (each treatment x its primary biomarker)
# =====================================================================
hyps = [
    {"id": "h24.1", "text": "In a joint multivariable model that includes all six treatments and the canonical treatment-by-biomarker interactions, each canonical interaction is positive and significant: tamoxifen×er_positive, palbociclib×er_positive (with negative palbociclib×her2_positive), trastuzumab×her2_positive, olaparib×brca_any, sacituzumab×(er_neg & her2_neg), pembrolizumab×(er_neg & her2_neg).", "kind": "novel"},
]
df["tn_proxy"] = ((df["er_positive"] == 0) & (df["her2_positive"] == 0)).astype(int)
formula = (
    "pfs_months ~ "
    "treatment_tamoxifen*er_positive + "
    "treatment_palbociclib*er_positive + treatment_palbociclib:her2_positive + "
    "treatment_trastuzumab*her2_positive + "
    "treatment_olaparib*brca_any + "
    "treatment_sacituzumab_govitecan*tn_proxy + "
    "treatment_pembrolizumab*tn_proxy + "
    + adj_terms
)
m_joint = ols_formula(df, formula)
analyses = []
for term in [
    "treatment_tamoxifen:er_positive",
    "treatment_palbociclib:er_positive",
    "treatment_palbociclib:her2_positive",
    "treatment_trastuzumab:her2_positive",
    "treatment_olaparib:brca_any",
    "treatment_sacituzumab_govitecan:tn_proxy",
    "treatment_pembrolizumab:tn_proxy",
]:
    b, p = float(m_joint.params[term]), float(m_joint.pvalues[term])
    analyses.append({
        "hypothesis_ids": ["h24.1"],
        "code": "joint OLS with all canonical interactions",
        "result_summary": f"{term} = {b:+.4f} (p={p:.3g}).",
        "p_value": fmt_p(p), "effect_estimate": fmt_e(b),
        "significant": bool(p < 0.05),
    })
add_iter(24, hyps, analyses)

# =====================================================================
# Iteration 25: final best-supported subgroup hypotheses (with suppressing variables)
# =====================================================================
hyps = [
    {"id": "h25.1", "text": "FINAL: treatment_olaparib increases pfs_months specifically in patients with brca1_mutation=1 OR brca2_mutation=1; in BRCA1/2 wild-type patients the adjusted effect is not positive.", "kind": "refined"},
    {"id": "h25.2", "text": "FINAL: treatment_trastuzumab increases pfs_months specifically in her2_positive=1 patients; in her2_positive=0 the adjusted effect is not positive.", "kind": "refined"},
    {"id": "h25.3", "text": "FINAL: treatment_palbociclib increases pfs_months specifically in ER-positive patients without HER2 amplification (er_positive=1 AND her2_positive=0); HER2-positivity suppresses any palbociclib benefit.", "kind": "refined"},
    {"id": "h25.4", "text": "FINAL: treatment_tamoxifen increases pfs_months specifically in er_positive=1 patients; in er_positive=0 the adjusted effect is not positive.", "kind": "refined"},
    {"id": "h25.5", "text": "FINAL: treatment_sacituzumab_govitecan increases pfs_months specifically in ER-negative AND HER2-negative patients (er_positive=0 AND her2_positive=0); ER-positivity suppresses any benefit.", "kind": "refined"},
    {"id": "h25.6", "text": "FINAL: treatment_pembrolizumab increases pfs_months specifically in ER-negative AND HER2-negative patients (er_positive=0 AND her2_positive=0); ER-positivity suppresses any benefit.", "kind": "refined"},
]
analyses = []
# Re-run final stratified estimates explicitly mapped to final hypotheses
mask_brca_pos = df["brca_any"] == 1
b_in, p_in, n_in = strat_effect(df, "treatment_olaparib", mask_brca_pos)
b_out, p_out, n_out = strat_effect(df, "treatment_olaparib", ~mask_brca_pos)
analyses.append({"hypothesis_ids": ["h25.1"], "code": "stratified OLS",
    "result_summary": f"Olaparib in BRCA1/2 carriers (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); non-carriers (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

b_in, p_in, n_in = strat_effect(df, "treatment_trastuzumab", df["her2_positive"] == 1)
b_out, p_out, n_out = strat_effect(df, "treatment_trastuzumab", df["her2_positive"] == 0)
analyses.append({"hypothesis_ids": ["h25.2"], "code": "stratified OLS",
    "result_summary": f"Trastuzumab in HER2+ (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); HER2- (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

mask_pal_pos = (df["er_positive"] == 1) & (df["her2_positive"] == 0)
b_in, p_in, n_in = strat_effect(df, "treatment_palbociclib", mask_pal_pos)
b_out, p_out, n_out = strat_effect(df, "treatment_palbociclib", ~mask_pal_pos)
analyses.append({"hypothesis_ids": ["h25.3"], "code": "stratified OLS",
    "result_summary": f"Palbociclib in ER+/HER2- (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); complement (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

b_in, p_in, n_in = strat_effect(df, "treatment_tamoxifen", df["er_positive"] == 1)
b_out, p_out, n_out = strat_effect(df, "treatment_tamoxifen", df["er_positive"] == 0)
analyses.append({"hypothesis_ids": ["h25.4"], "code": "stratified OLS",
    "result_summary": f"Tamoxifen in ER+ (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); ER- (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

mask_dn = (df["er_positive"] == 0) & (df["her2_positive"] == 0)
b_in, p_in, n_in = strat_effect(df, "treatment_sacituzumab_govitecan", mask_dn)
b_out, p_out, n_out = strat_effect(df, "treatment_sacituzumab_govitecan", ~mask_dn)
analyses.append({"hypothesis_ids": ["h25.5"], "code": "stratified OLS",
    "result_summary": f"Sacituzumab in ER-/HER2- (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); complement (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

b_in, p_in, n_in = strat_effect(df, "treatment_pembrolizumab", mask_dn)
b_out, p_out, n_out = strat_effect(df, "treatment_pembrolizumab", ~mask_dn)
analyses.append({"hypothesis_ids": ["h25.6"], "code": "stratified OLS",
    "result_summary": f"Pembrolizumab in ER-/HER2- (n={n_in}): {b_in:+.4f} (p={p_in:.3g}); complement (n={n_out}): {b_out:+.4f} (p={p_out:.3g}).",
    "p_value": fmt_p(p_in), "effect_estimate": fmt_e(b_in), "significant": bool(p_in < 0.05)})

add_iter(25, hyps, analyses)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done. Iterations:", len(results["iterations"]))
