"""Iterative analysis of ds001_prostate. Outputs transcript.json and analysis_summary.txt."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
OUT = "pfs_months"

iterations = []


def add_iter(idx, hypotheses, analyses):
    iterations.append({
        "index": idx,
        "proposed_hypotheses": hypotheses,
        "analyses": analyses,
    })


def ttest(g1, g0, label1="grp1", label0="grp0"):
    t = stats.ttest_ind(g1, g0, equal_var=False)
    return {
        "n1": int(len(g1)),
        "n0": int(len(g0)),
        "mean1": float(np.mean(g1)),
        "mean0": float(np.mean(g0)),
        "diff": float(np.mean(g1) - np.mean(g0)),
        "t": float(t.statistic),
        "p": float(t.pvalue),
    }


def ols(formula):
    m = smf.ols(formula, data=df).fit()
    return m


def coef_row(m, name):
    return {
        "coef": float(m.params[name]),
        "p": float(m.pvalues[name]),
        "se": float(m.bse[name]),
    }


# ============================================================
# ITERATION 1: Main effects of each of the 6 treatments on PFS
# ============================================================
hyps = [
    {"id": "h1.1", "text": "Patients receiving treatment_enzalutamide have different mean pfs_months than those not receiving it.", "kind": "novel"},
    {"id": "h1.2", "text": "Patients receiving treatment_abiraterone have different mean pfs_months than those not receiving it.", "kind": "novel"},
    {"id": "h1.3", "text": "Patients receiving treatment_docetaxel have different mean pfs_months than those not receiving it.", "kind": "novel"},
    {"id": "h1.4", "text": "Patients receiving treatment_olaparib have different mean pfs_months than those not receiving it.", "kind": "novel"},
    {"id": "h1.5", "text": "Patients receiving treatment_lu177_psma have different mean pfs_months than those not receiving it.", "kind": "novel"},
    {"id": "h1.6", "text": "Patients receiving treatment_pembrolizumab have different mean pfs_months than those not receiving it.", "kind": "novel"},
]
analyses = []
for tx, hid in [
    ("treatment_enzalutamide", "h1.1"),
    ("treatment_abiraterone", "h1.2"),
    ("treatment_docetaxel", "h1.3"),
    ("treatment_olaparib", "h1.4"),
    ("treatment_lu177_psma", "h1.5"),
    ("treatment_pembrolizumab", "h1.6"),
]:
    r = ttest(df.loc[df[tx] == 1, OUT], df.loc[df[tx] == 0, OUT])
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{tx}']==1,'pfs_months'], df.loc[df['{tx}']==0,'pfs_months'])",
        "result_summary": f"Mean pfs_months {r['mean1']:.3f} on {tx} (n={r['n1']}) vs {r['mean0']:.3f} off (n={r['n0']}); diff={r['diff']:.3f}, t-test p={r['p']:.3g}.",
        "p_value": r["p"],
        "effect_estimate": r["diff"],
        "significant": bool(r["p"] < 0.05),
    })
add_iter(1, hyps, analyses)

# ============================================================
# ITERATION 2: Disease severity main effects
# ============================================================
hyps = [
    {"id": "h2.1", "text": "Higher ecog_ps is associated with shorter pfs_months (negative slope in OLS).", "kind": "novel"},
    {"id": "h2.2", "text": "Patients with mcrpc=1 have shorter mean pfs_months than mcrpc=0.", "kind": "novel"},
    {"id": "h2.3", "text": "Patients with visceral_mets=1 have shorter mean pfs_months than visceral_mets=0.", "kind": "novel"},
    {"id": "h2.4", "text": "Patients with bone_mets=1 have shorter mean pfs_months than bone_mets=0.", "kind": "novel"},
    {"id": "h2.5", "text": "Higher gleason_score is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.6", "text": "Higher log10(psa_ng_ml) is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
m = ols("pfs_months ~ ecog_ps")
c = coef_row(m, "ecog_ps")
analyses.append({"hypothesis_ids": ["h2.1"], "code": "smf.ols('pfs_months ~ ecog_ps', df).fit()",
                 "result_summary": f"OLS slope of pfs_months on ecog_ps = {c['coef']:.3f} months/unit (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
for hid, var in [("h2.2", "mcrpc"), ("h2.3", "visceral_mets"), ("h2.4", "bone_mets")]:
    r = ttest(df.loc[df[var] == 1, OUT], df.loc[df[var] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
                     "result_summary": f"Mean pfs_months {r['mean1']:.3f} with {var}=1 vs {r['mean0']:.3f} with {var}=0 (diff={r['diff']:.3f}, p={r['p']:.3g}).",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
m = ols("pfs_months ~ gleason_score")
c = coef_row(m, "gleason_score")
analyses.append({"hypothesis_ids": ["h2.5"], "code": "smf.ols('pfs_months ~ gleason_score', df).fit()",
                 "result_summary": f"OLS slope of pfs_months on gleason_score = {c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
df["log10_psa"] = np.log10(df["psa_ng_ml"].clip(lower=0.01))
m = ols("pfs_months ~ log10_psa")
c = coef_row(m, "log10_psa")
analyses.append({"hypothesis_ids": ["h2.6"], "code": "smf.ols('pfs_months ~ log10_psa', df).fit()",
                 "result_summary": f"OLS slope of pfs_months on log10(psa) = {c['coef']:.3f} months/log10-unit (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(2, hyps, analyses)

# ============================================================
# ITERATION 3: Lab and inflammation markers
# ============================================================
hyps = [
    {"id": "h3.1", "text": "Higher albumin_g_dl is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h3.2", "text": "Higher ldh_u_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.3", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h3.4", "text": "Higher crp_mg_l is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.5", "text": "Higher nlr (neutrophil-to-lymphocyte ratio) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h3.6", "text": "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
for hid, var in [("h3.1", "albumin_g_dl"), ("h3.2", "ldh_u_l"), ("h3.3", "hemoglobin_g_dl"),
                 ("h3.4", "crp_mg_l"), ("h3.5", "nlr"), ("h3.6", "alkaline_phosphatase_u_l")]:
    m = ols(f"pfs_months ~ {var}")
    c = coef_row(m, var)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('pfs_months ~ {var}', df).fit()",
                     "result_summary": f"OLS slope on {var} = {c['coef']:.4g} months per unit (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(3, hyps, analyses)

# ============================================================
# ITERATION 4: Biomarker main effects
# ============================================================
hyps = [
    {"id": "h4.1", "text": "Patients with brca2_mutation=1 have different mean pfs_months than brca2_mutation=0.", "kind": "novel"},
    {"id": "h4.2", "text": "Patients with ar_v7_positive=1 have shorter mean pfs_months than ar_v7_positive=0.", "kind": "novel"},
    {"id": "h4.3", "text": "Patients with msi_high=1 have different mean pfs_months than msi_high=0.", "kind": "novel"},
    {"id": "h4.4", "text": "Patients with psma_high=1 have different mean pfs_months than psma_high=0.", "kind": "novel"},
    {"id": "h4.5", "text": "Patients with tp53_mutation=1 have shorter mean pfs_months than tp53_mutation=0.", "kind": "novel"},
    {"id": "h4.6", "text": "Patients with pten_loss=1 have shorter mean pfs_months than pten_loss=0.", "kind": "novel"},
]
analyses = []
for hid, var in [("h4.1", "brca2_mutation"), ("h4.2", "ar_v7_positive"), ("h4.3", "msi_high"),
                 ("h4.4", "psma_high"), ("h4.5", "tp53_mutation"), ("h4.6", "pten_loss")]:
    r = ttest(df.loc[df[var] == 1, OUT], df.loc[df[var] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
                     "result_summary": f"Mean pfs_months {r['mean1']:.3f} with {var}=1 (n={r['n1']}) vs {r['mean0']:.3f} (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(4, hyps, analyses)

# ============================================================
# ITERATION 5: Targeted-therapy biomarker x treatment interactions
# ============================================================
hyps = [
    {"id": "h5.1", "text": "BRCA2-mutant patients receiving treatment_olaparib have longer pfs_months than BRCA2-mutant patients not receiving olaparib (positive interaction brca2_mutation:treatment_olaparib).", "kind": "novel"},
    {"id": "h5.2", "text": "PSMA-high patients receiving treatment_lu177_psma have longer pfs_months than PSMA-high patients not receiving Lu177-PSMA (positive interaction psma_high:treatment_lu177_psma).", "kind": "novel"},
    {"id": "h5.3", "text": "MSI-high patients receiving treatment_pembrolizumab have longer pfs_months than MSI-high patients not receiving pembrolizumab (positive interaction msi_high:treatment_pembrolizumab).", "kind": "novel"},
    {"id": "h5.4", "text": "AR-V7 positive patients receiving treatment_enzalutamide have shorter pfs_months than AR-V7 negative patients receiving enzalutamide (negative interaction ar_v7_positive:treatment_enzalutamide).", "kind": "novel"},
]
analyses = []
for hid, biomk, tx in [
    ("h5.1", "brca2_mutation", "treatment_olaparib"),
    ("h5.2", "psma_high", "treatment_lu177_psma"),
    ("h5.3", "msi_high", "treatment_pembrolizumab"),
    ("h5.4", "ar_v7_positive", "treatment_enzalutamide"),
]:
    formula = f"pfs_months ~ {biomk} * {tx}"
    m = ols(formula)
    iname = f"{biomk}:{tx}"
    c = coef_row(m, iname)
    # Also stratified means for narrative
    g11 = df.loc[(df[biomk] == 1) & (df[tx] == 1), OUT].mean()
    g10 = df.loc[(df[biomk] == 1) & (df[tx] == 0), OUT].mean()
    g01 = df.loc[(df[biomk] == 0) & (df[tx] == 1), OUT].mean()
    g00 = df.loc[(df[biomk] == 0) & (df[tx] == 0), OUT].mean()
    analyses.append({
        "hypothesis_ids": [hid],
        "code": f"smf.ols('{formula}', df).fit()",
        "result_summary": f"Interaction {iname} coef={c['coef']:.3f} (p={c['p']:.3g}). Means by ({biomk},{tx}): (1,1)={g11:.2f}, (1,0)={g10:.2f}, (0,1)={g01:.2f}, (0,0)={g00:.2f}.",
        "p_value": c["p"],
        "effect_estimate": c["coef"],
        "significant": bool(c["p"] < 0.05),
    })
add_iter(5, hyps, analyses)

# ============================================================
# ITERATION 6: Symptom burden main effects
# ============================================================
hyps = [
    {"id": "h6.1", "text": "Higher fatigue_grade is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.2", "text": "Higher pain_nrs is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.3", "text": "Higher dyspnea_grade is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.4", "text": "Higher appetite_loss_grade is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h6.5", "text": "Higher weight_loss_pct_6mo is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
for hid, var in [("h6.1", "fatigue_grade"), ("h6.2", "pain_nrs"), ("h6.3", "dyspnea_grade"),
                 ("h6.4", "appetite_loss_grade"), ("h6.5", "weight_loss_pct_6mo")]:
    m = ols(f"pfs_months ~ {var}")
    c = coef_row(m, var)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('pfs_months ~ {var}', df).fit()",
                     "result_summary": f"OLS slope on {var} = {c['coef']:.4g} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(6, hyps, analyses)

# ============================================================
# ITERATION 7: Demographic / SES main effects
# ============================================================
hyps = [
    {"id": "h7.1", "text": "Older age_years is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h7.2", "text": "Patients with rural_residence=1 have shorter pfs_months than urban-residence patients.", "kind": "novel"},
    {"id": "h7.3", "text": "Mean pfs_months differs across race_ethnicity categories (ANOVA).", "kind": "novel"},
    {"id": "h7.4", "text": "Mean pfs_months differs across insurance_type categories (ANOVA).", "kind": "novel"},
    {"id": "h7.5", "text": "Higher smoking_pack_years is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h7.6", "text": "More years of education_years is associated with longer pfs_months.", "kind": "novel"},
]
analyses = []
m = ols("pfs_months ~ age_years")
c = coef_row(m, "age_years")
analyses.append({"hypothesis_ids": ["h7.1"], "code": "smf.ols('pfs_months ~ age_years', df).fit()",
                 "result_summary": f"OLS slope of pfs_months on age_years = {c['coef']:.4g} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
r = ttest(df.loc[df["rural_residence"] == 1, OUT], df.loc[df["rural_residence"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h7.2"],
                 "code": "stats.ttest_ind(df.loc[df['rural_residence']==1,'pfs_months'], df.loc[df['rural_residence']==0,'pfs_months'])",
                 "result_summary": f"Mean pfs_months rural={r['mean1']:.3f} vs urban={r['mean0']:.3f} (diff={r['diff']:.3f}, p={r['p']:.3g}).",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
groups = [df.loc[df["race_ethnicity"] == g, OUT].values for g in df["race_ethnicity"].unique()]
F, p = stats.f_oneway(*groups)
means = df.groupby("race_ethnicity")[OUT].mean().to_dict()
overall = df[OUT].mean()
max_dev = max(means.values()) - min(means.values())
analyses.append({"hypothesis_ids": ["h7.3"],
                 "code": "stats.f_oneway(*[df.loc[df.race_ethnicity==g,'pfs_months'] for g in df.race_ethnicity.unique()])",
                 "result_summary": f"ANOVA F={F:.3f} p={p:.3g}; means by race: {means}.",
                 "p_value": float(p), "effect_estimate": float(max_dev), "significant": bool(p < 0.05)})
groups = [df.loc[df["insurance_type"] == g, OUT].values for g in df["insurance_type"].unique()]
F, p = stats.f_oneway(*groups)
means = df.groupby("insurance_type")[OUT].mean().to_dict()
max_dev = max(means.values()) - min(means.values())
analyses.append({"hypothesis_ids": ["h7.4"],
                 "code": "stats.f_oneway(*[df.loc[df.insurance_type==g,'pfs_months'] for g in df.insurance_type.unique()])",
                 "result_summary": f"ANOVA F={F:.3f} p={p:.3g}; means by insurance: {means}.",
                 "p_value": float(p), "effect_estimate": float(max_dev), "significant": bool(p < 0.05)})
m = ols("pfs_months ~ smoking_pack_years")
c = coef_row(m, "smoking_pack_years")
analyses.append({"hypothesis_ids": ["h7.5"], "code": "smf.ols('pfs_months ~ smoking_pack_years', df).fit()",
                 "result_summary": f"OLS slope on smoking_pack_years = {c['coef']:.4g} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
m = ols("pfs_months ~ education_years")
c = coef_row(m, "education_years")
analyses.append({"hypothesis_ids": ["h7.6"], "code": "smf.ols('pfs_months ~ education_years', df).fit()",
                 "result_summary": f"OLS slope on education_years = {c['coef']:.4g} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(7, hyps, analyses)

# ============================================================
# ITERATION 8: Prior therapy & comorbidity effects
# ============================================================
hyps = [
    {"id": "h8.1", "text": "More prior_lines_of_therapy is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h8.2", "text": "Patients with prior_chemotherapy=1 have shorter pfs_months than chemo-naive patients.", "kind": "novel"},
    {"id": "h8.3", "text": "Patients with chronic_kidney_disease=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h8.4", "text": "Patients with heart_failure=1 have shorter pfs_months than those without.", "kind": "novel"},
    {"id": "h8.5", "text": "Patients with diabetes_mellitus=1 have different pfs_months than those without.", "kind": "novel"},
    {"id": "h8.6", "text": "More years_since_diagnosis is associated with longer pfs_months (selection by survival).", "kind": "novel"},
]
analyses = []
m = ols("pfs_months ~ prior_lines_of_therapy")
c = coef_row(m, "prior_lines_of_therapy")
analyses.append({"hypothesis_ids": ["h8.1"], "code": "smf.ols('pfs_months ~ prior_lines_of_therapy', df).fit()",
                 "result_summary": f"OLS slope on prior_lines_of_therapy = {c['coef']:.4g} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
for hid, var in [("h8.2", "prior_chemotherapy"), ("h8.3", "chronic_kidney_disease"),
                 ("h8.4", "heart_failure"), ("h8.5", "diabetes_mellitus")]:
    r = ttest(df.loc[df[var] == 1, OUT], df.loc[df[var] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"stats.ttest_ind(df.loc[df['{var}']==1,'pfs_months'], df.loc[df['{var}']==0,'pfs_months'])",
                     "result_summary": f"Mean pfs_months {r['mean1']:.3f} with {var}=1 (n={r['n1']}) vs {r['mean0']:.3f} (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
m = ols("pfs_months ~ years_since_diagnosis")
c = coef_row(m, "years_since_diagnosis")
analyses.append({"hypothesis_ids": ["h8.6"], "code": "smf.ols('pfs_months ~ years_since_diagnosis', df).fit()",
                 "result_summary": f"OLS slope on years_since_diagnosis = {c['coef']:.4g} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(8, hyps, analyses)

# ============================================================
# ITERATION 9: Multivariable adjusted treatment effects
# ============================================================
hyps = [
    {"id": "h9.1", "text": "After adjusting for ecog_ps, age_years, mcrpc, visceral_mets, ldh_u_l, albumin_g_dl, hemoglobin_g_dl and prior_lines_of_therapy, treatment_docetaxel remains independently associated with pfs_months.", "kind": "novel"},
    {"id": "h9.2", "text": "After the same adjustments, treatment_olaparib remains independently associated with pfs_months.", "kind": "novel"},
    {"id": "h9.3", "text": "After the same adjustments, treatment_lu177_psma remains independently associated with pfs_months.", "kind": "novel"},
    {"id": "h9.4", "text": "After the same adjustments, treatment_pembrolizumab remains independently associated with pfs_months.", "kind": "novel"},
]
adj = "ecog_ps + age_years + mcrpc + visceral_mets + ldh_u_l + albumin_g_dl + hemoglobin_g_dl + prior_lines_of_therapy"
m = ols(f"pfs_months ~ treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab + {adj}")
analyses = []
for hid, tx in [("h9.1", "treatment_docetaxel"), ("h9.2", "treatment_olaparib"),
                ("h9.3", "treatment_lu177_psma"), ("h9.4", "treatment_pembrolizumab")]:
    c = coef_row(m, tx)
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"smf.ols('pfs_months ~ all_tx + {adj}', df).fit()",
                     "result_summary": f"Adjusted coef for {tx} = {c['coef']:.3f} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(9, hyps, analyses)

# ============================================================
# ITERATION 10: Refined biomarker interactions, adjusted
# ============================================================
hyps = [
    {"id": "h10.1", "text": "After adjusting for ecog_ps, age_years, mcrpc, visceral_mets, the interaction brca2_mutation:treatment_olaparib remains positive and significant.", "kind": "refined"},
    {"id": "h10.2", "text": "After the same adjustments, the interaction psma_high:treatment_lu177_psma remains positive and significant.", "kind": "refined"},
    {"id": "h10.3", "text": "After the same adjustments, the interaction msi_high:treatment_pembrolizumab remains positive and significant.", "kind": "refined"},
    {"id": "h10.4", "text": "After the same adjustments, the interaction ar_v7_positive:treatment_enzalutamide remains negative and significant.", "kind": "refined"},
]
adj = "ecog_ps + age_years + mcrpc + visceral_mets"
analyses = []
for hid, biomk, tx in [
    ("h10.1", "brca2_mutation", "treatment_olaparib"),
    ("h10.2", "psma_high", "treatment_lu177_psma"),
    ("h10.3", "msi_high", "treatment_pembrolizumab"),
    ("h10.4", "ar_v7_positive", "treatment_enzalutamide"),
]:
    formula = f"pfs_months ~ {biomk} * {tx} + {adj}"
    m = ols(formula)
    iname = f"{biomk}:{tx}"
    c = coef_row(m, iname)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('{formula}', df).fit()",
                     "result_summary": f"Adjusted interaction {iname} coef={c['coef']:.3f} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(10, hyps, analyses)

# ============================================================
# ITERATION 11: Subgroup means in matched biomarker-treatment cells
# ============================================================
hyps = [
    {"id": "h11.1", "text": "Among brca2_mutation=1 patients, mean pfs_months is higher in those receiving treatment_olaparib than those not.", "kind": "refined"},
    {"id": "h11.2", "text": "Among psma_high=1 patients, mean pfs_months is higher in those receiving treatment_lu177_psma than those not.", "kind": "refined"},
    {"id": "h11.3", "text": "Among msi_high=1 patients, mean pfs_months is higher in those receiving treatment_pembrolizumab than those not.", "kind": "refined"},
    {"id": "h11.4", "text": "Among ar_v7_positive=1 patients, mean pfs_months is lower in those receiving treatment_enzalutamide than those not.", "kind": "refined"},
]
analyses = []
for hid, biomk, tx in [
    ("h11.1", "brca2_mutation", "treatment_olaparib"),
    ("h11.2", "psma_high", "treatment_lu177_psma"),
    ("h11.3", "msi_high", "treatment_pembrolizumab"),
    ("h11.4", "ar_v7_positive", "treatment_enzalutamide"),
]:
    sub = df[df[biomk] == 1]
    r = ttest(sub.loc[sub[tx] == 1, OUT], sub.loc[sub[tx] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"sub=df[df['{biomk}']==1]; stats.ttest_ind(sub.loc[sub['{tx}']==1,'pfs_months'], sub.loc[sub['{tx}']==0,'pfs_months'])",
                     "result_summary": f"In {biomk}=1 (N={r['n1']+r['n0']}): mean {r['mean1']:.3f} on {tx} (n={r['n1']}) vs {r['mean0']:.3f} off (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(11, hyps, analyses)

# ============================================================
# ITERATION 12: Effect of docetaxel by visceral_mets subgroup
# ============================================================
hyps = [
    {"id": "h12.1", "text": "Among visceral_mets=1 patients, treatment_docetaxel is associated with longer mean pfs_months than no docetaxel.", "kind": "novel"},
    {"id": "h12.2", "text": "Among visceral_mets=0 patients, treatment_docetaxel is associated with different mean pfs_months than no docetaxel.", "kind": "novel"},
    {"id": "h12.3", "text": "There is a significant visceral_mets:treatment_docetaxel interaction effect on pfs_months.", "kind": "novel"},
]
analyses = []
sub = df[df["visceral_mets"] == 1]
r = ttest(sub.loc[sub["treatment_docetaxel"] == 1, OUT], sub.loc[sub["treatment_docetaxel"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h12.1"],
                 "code": "sub=df[df['visceral_mets']==1]; ttest docetaxel groups",
                 "result_summary": f"Visceral_mets=1: docetaxel mean {r['mean1']:.3f} vs {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
sub = df[df["visceral_mets"] == 0]
r = ttest(sub.loc[sub["treatment_docetaxel"] == 1, OUT], sub.loc[sub["treatment_docetaxel"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h12.2"],
                 "code": "sub=df[df['visceral_mets']==0]; ttest docetaxel groups",
                 "result_summary": f"Visceral_mets=0: docetaxel mean {r['mean1']:.3f} vs {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
m = ols("pfs_months ~ visceral_mets * treatment_docetaxel")
c = coef_row(m, "visceral_mets:treatment_docetaxel")
analyses.append({"hypothesis_ids": ["h12.3"],
                 "code": "smf.ols('pfs_months ~ visceral_mets * treatment_docetaxel', df).fit()",
                 "result_summary": f"Interaction visceral_mets:treatment_docetaxel coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(12, hyps, analyses)

# ============================================================
# ITERATION 13: ECOG-PS interactions with treatment effectiveness
# ============================================================
hyps = [
    {"id": "h13.1", "text": "ecog_ps:treatment_docetaxel interaction is significant — docetaxel benefit/harm differs by performance status.", "kind": "novel"},
    {"id": "h13.2", "text": "Patients with ecog_ps>=2 have markedly shorter pfs_months than ecog_ps<=1 in the overall cohort.", "kind": "refined"},
    {"id": "h13.3", "text": "Among ecog_ps=0 patients, treatment_abiraterone is associated with longer pfs_months than no abiraterone.", "kind": "novel"},
]
analyses = []
m = ols("pfs_months ~ ecog_ps * treatment_docetaxel")
c = coef_row(m, "ecog_ps:treatment_docetaxel")
analyses.append({"hypothesis_ids": ["h13.1"],
                 "code": "smf.ols('pfs_months ~ ecog_ps * treatment_docetaxel', df).fit()",
                 "result_summary": f"Interaction ecog_ps:treatment_docetaxel coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
g_hi = df.loc[df["ecog_ps"] >= 2, OUT]
g_lo = df.loc[df["ecog_ps"] <= 1, OUT]
r = ttest(g_hi, g_lo)
analyses.append({"hypothesis_ids": ["h13.2"],
                 "code": "ttest ecog_ps>=2 vs ecog_ps<=1",
                 "result_summary": f"ecog>=2 mean {r['mean1']:.3f} (n={r['n1']}) vs ecog<=1 mean {r['mean0']:.3f} (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
sub = df[df["ecog_ps"] == 0]
r = ttest(sub.loc[sub["treatment_abiraterone"] == 1, OUT], sub.loc[sub["treatment_abiraterone"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h13.3"],
                 "code": "sub=df[df['ecog_ps']==0]; ttest abiraterone groups",
                 "result_summary": f"ecog_ps=0 (n={r['n0']+r['n1']}): abiraterone mean {r['mean1']:.3f} vs {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(13, hyps, analyses)

# ============================================================
# ITERATION 14: SNP screen vs PFS
# ============================================================
snps = [c for c in df.columns if c.startswith("snp_rs")]
hyps = [{"id": "h14.0", "text": f"At least one of the {len(snps)} SNP genotype features is associated with pfs_months at p<0.05 in univariable OLS (false-positive screening at alpha=0.05).", "kind": "novel"}]
analyses = []
results = []
for s in snps:
    m = ols(f"pfs_months ~ {s}")
    c = coef_row(m, s)
    results.append((s, c["coef"], c["p"]))
results.sort(key=lambda x: x[2])
nsig = sum(1 for _, _, p in results if p < 0.05)
top = results[:5]
analyses.append({"hypothesis_ids": ["h14.0"],
                 "code": "for each snp: smf.ols('pfs_months ~ snp', df).fit()",
                 "result_summary": f"{nsig}/{len(snps)} SNPs reach p<0.05 univariably (Bonferroni threshold {0.05/len(snps):.2g}). Top: " + ", ".join(f"{s} coef={b:.3f} p={p:.3g}" for s, b, p in top),
                 "p_value": float(top[0][2]), "effect_estimate": float(top[0][1]),
                 "significant": bool(top[0][2] < 0.05 / len(snps))})
add_iter(14, hyps, analyses)

# ============================================================
# ITERATION 15: Refined — three-way subgroup checks of biomarker matched therapy
# ============================================================
hyps = [
    {"id": "h15.1", "text": "Among brca2_mutation=0 patients, treatment_olaparib has no benefit (or different effect) on pfs_months compared to no olaparib.", "kind": "refined"},
    {"id": "h15.2", "text": "Among psma_high=0 patients, treatment_lu177_psma has no benefit (or different effect) on pfs_months compared to no lu177_psma.", "kind": "refined"},
    {"id": "h15.3", "text": "Among msi_high=0 patients, treatment_pembrolizumab is not beneficial (or different) for pfs_months.", "kind": "refined"},
    {"id": "h15.4", "text": "Among ar_v7_positive=0 patients, treatment_enzalutamide is associated with longer pfs_months compared to no enzalutamide.", "kind": "refined"},
]
analyses = []
for hid, biomk, tx in [
    ("h15.1", "brca2_mutation", "treatment_olaparib"),
    ("h15.2", "psma_high", "treatment_lu177_psma"),
    ("h15.3", "msi_high", "treatment_pembrolizumab"),
    ("h15.4", "ar_v7_positive", "treatment_enzalutamide"),
]:
    sub = df[df[biomk] == 0]
    r = ttest(sub.loc[sub[tx] == 1, OUT], sub.loc[sub[tx] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"sub=df[df['{biomk}']==0]; ttest {tx} groups",
                     "result_summary": f"In {biomk}=0 (N={r['n1']+r['n0']}): mean {r['mean1']:.3f} on {tx} (n={r['n1']}) vs {r['mean0']:.3f} off (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(15, hyps, analyses)

# ============================================================
# ITERATION 16: Symptom interactions with treatment
# ============================================================
hyps = [
    {"id": "h16.1", "text": "After adjusting for ecog_ps, mcrpc and visceral_mets, pain_nrs remains negatively associated with pfs_months.", "kind": "refined"},
    {"id": "h16.2", "text": "After the same adjustments, weight_loss_pct_6mo remains negatively associated with pfs_months.", "kind": "refined"},
    {"id": "h16.3", "text": "After the same adjustments, fatigue_grade remains negatively associated with pfs_months.", "kind": "refined"},
]
adj = "ecog_ps + mcrpc + visceral_mets"
analyses = []
for hid, var in [("h16.1", "pain_nrs"), ("h16.2", "weight_loss_pct_6mo"), ("h16.3", "fatigue_grade")]:
    m = ols(f"pfs_months ~ {var} + {adj}")
    c = coef_row(m, var)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('pfs_months ~ {var} + {adj}', df).fit()",
                     "result_summary": f"Adjusted coef on {var} = {c['coef']:.4g} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(16, hyps, analyses)

# ============================================================
# ITERATION 17: Race & insurance disparities adjusted
# ============================================================
hyps = [
    {"id": "h17.1", "text": "After adjusting for ecog_ps, age_years, mcrpc, visceral_mets and prior_lines_of_therapy, mean pfs_months still differs across race_ethnicity categories (joint F-test p<0.05).", "kind": "refined"},
    {"id": "h17.2", "text": "After the same adjustments, mean pfs_months still differs across insurance_type categories (joint F-test p<0.05).", "kind": "refined"},
    {"id": "h17.3", "text": "After the same adjustments, rural_residence remains associated with shorter pfs_months.", "kind": "refined"},
]
adj = "ecog_ps + age_years + mcrpc + visceral_mets + prior_lines_of_therapy"
analyses = []
m_full = ols(f"pfs_months ~ C(race_ethnicity) + {adj}")
m_red = ols(f"pfs_months ~ {adj}")
ftest = m_full.compare_f_test(m_red)
F, p, _ = ftest
race_means = df.groupby("race_ethnicity")[OUT].mean().to_dict()
analyses.append({"hypothesis_ids": ["h17.1"],
                 "code": f"compare_f_test full vs reduced model with C(race_ethnicity)",
                 "result_summary": f"Adjusted joint F-test for race_ethnicity F={F:.3f} p={p:.3g}; raw means: {race_means}.",
                 "p_value": float(p), "effect_estimate": float(max(race_means.values()) - min(race_means.values())),
                 "significant": bool(p < 0.05)})
m_full = ols(f"pfs_months ~ C(insurance_type) + {adj}")
m_red = ols(f"pfs_months ~ {adj}")
F, p, _ = m_full.compare_f_test(m_red)
ins_means = df.groupby("insurance_type")[OUT].mean().to_dict()
analyses.append({"hypothesis_ids": ["h17.2"],
                 "code": "compare_f_test full vs reduced model with C(insurance_type)",
                 "result_summary": f"Adjusted joint F-test for insurance_type F={F:.3f} p={p:.3g}; raw means: {ins_means}.",
                 "p_value": float(p), "effect_estimate": float(max(ins_means.values()) - min(ins_means.values())),
                 "significant": bool(p < 0.05)})
m = ols(f"pfs_months ~ rural_residence + {adj}")
c = coef_row(m, "rural_residence")
analyses.append({"hypothesis_ids": ["h17.3"],
                 "code": f"smf.ols('pfs_months ~ rural_residence + {adj}', df).fit()",
                 "result_summary": f"Adjusted coef rural_residence = {c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(17, hyps, analyses)

# ============================================================
# ITERATION 18: TP53 and PTEN modify treatment effects
# ============================================================
hyps = [
    {"id": "h18.1", "text": "Interaction tp53_mutation:treatment_docetaxel is non-zero (modifies docetaxel effect on pfs_months).", "kind": "novel"},
    {"id": "h18.2", "text": "Interaction pten_loss:treatment_abiraterone is non-zero (modifies abiraterone effect on pfs_months).", "kind": "novel"},
    {"id": "h18.3", "text": "Interaction tp53_mutation:treatment_olaparib is non-zero.", "kind": "novel"},
]
analyses = []
for hid, biomk, tx in [
    ("h18.1", "tp53_mutation", "treatment_docetaxel"),
    ("h18.2", "pten_loss", "treatment_abiraterone"),
    ("h18.3", "tp53_mutation", "treatment_olaparib"),
]:
    formula = f"pfs_months ~ {biomk} * {tx}"
    m = ols(formula)
    iname = f"{biomk}:{tx}"
    c = coef_row(m, iname)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('{formula}', df).fit()",
                     "result_summary": f"Interaction {iname} coef={c['coef']:.3f} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(18, hyps, analyses)

# ============================================================
# ITERATION 19: BMI and vitals
# ============================================================
hyps = [
    {"id": "h19.1", "text": "Higher bmi is associated with longer pfs_months.", "kind": "novel"},
    {"id": "h19.2", "text": "Higher heart_rate_bpm is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h19.3", "text": "Lower spo2_pct is associated with shorter pfs_months (positive slope).", "kind": "novel"},
    {"id": "h19.4", "text": "Higher systolic_bp_mmhg is associated with longer pfs_months.", "kind": "novel"},
]
analyses = []
for hid, var in [("h19.1", "bmi"), ("h19.2", "heart_rate_bpm"),
                 ("h19.3", "spo2_pct"), ("h19.4", "systolic_bp_mmhg")]:
    m = ols(f"pfs_months ~ {var}")
    c = coef_row(m, var)
    analyses.append({"hypothesis_ids": [hid], "code": f"smf.ols('pfs_months ~ {var}', df).fit()",
                     "result_summary": f"OLS slope on {var} = {c['coef']:.4g} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(19, hyps, analyses)

# ============================================================
# ITERATION 20: Site of metastasis
# ============================================================
hyps = [
    {"id": "h20.1", "text": "liver_mets=1 is associated with shorter pfs_months than liver_mets=0.", "kind": "novel"},
    {"id": "h20.2", "text": "adrenal_mets=1 is associated with shorter pfs_months than adrenal_mets=0.", "kind": "novel"},
    {"id": "h20.3", "text": "pleural_effusion=1 is associated with shorter pfs_months than pleural_effusion=0.", "kind": "novel"},
]
analyses = []
for hid, var in [("h20.1", "liver_mets"), ("h20.2", "adrenal_mets"), ("h20.3", "pleural_effusion")]:
    r = ttest(df.loc[df[var] == 1, OUT], df.loc[df[var] == 0, OUT])
    analyses.append({"hypothesis_ids": [hid],
                     "code": f"ttest pfs by {var}",
                     "result_summary": f"Mean pfs {r['mean1']:.3f} ({var}=1, n={r['n1']}) vs {r['mean0']:.3f} (n={r['n0']}); diff={r['diff']:.3f}, p={r['p']:.3g}.",
                     "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(20, hyps, analyses)

# ============================================================
# ITERATION 21: Three-way visceral×ECOG×treatment_docetaxel
# ============================================================
hyps = [
    {"id": "h21.1", "text": "Among ecog_ps>=2 visceral_mets=1 patients, treatment_docetaxel still has the same direction of effect on pfs_months as in the overall cohort.", "kind": "refined"},
    {"id": "h21.2", "text": "The three-way interaction ecog_ps:visceral_mets:treatment_docetaxel has p<0.05 in OLS.", "kind": "novel"},
]
analyses = []
sub = df[(df["ecog_ps"] >= 2) & (df["visceral_mets"] == 1)]
r = ttest(sub.loc[sub["treatment_docetaxel"] == 1, OUT], sub.loc[sub["treatment_docetaxel"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h21.1"],
                 "code": "subset ecog>=2 & visceral_mets==1; ttest docetaxel groups",
                 "result_summary": f"ECOG>=2 & visceral_mets=1 (n={r['n1']+r['n0']}): docetaxel mean {r['mean1']:.3f} vs {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
m = ols("pfs_months ~ ecog_ps * visceral_mets * treatment_docetaxel")
c = coef_row(m, "ecog_ps:visceral_mets:treatment_docetaxel")
analyses.append({"hypothesis_ids": ["h21.2"],
                 "code": "smf.ols('pfs_months ~ ecog_ps * visceral_mets * treatment_docetaxel', df).fit()",
                 "result_summary": f"3-way interaction coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(21, hyps, analyses)

# ============================================================
# ITERATION 22: AR-V7 also impacts abiraterone (related mechanism)
# ============================================================
hyps = [
    {"id": "h22.1", "text": "Interaction ar_v7_positive:treatment_abiraterone is negative (AR-V7+ patients derive less benefit from abiraterone).", "kind": "novel"},
    {"id": "h22.2", "text": "Among ar_v7_positive=1 patients, mean pfs_months on treatment_abiraterone is shorter than off.", "kind": "novel"},
    {"id": "h22.3", "text": "Among ar_v7_positive=0 patients, mean pfs_months on treatment_abiraterone is longer than off.", "kind": "novel"},
]
analyses = []
m = ols("pfs_months ~ ar_v7_positive * treatment_abiraterone")
c = coef_row(m, "ar_v7_positive:treatment_abiraterone")
analyses.append({"hypothesis_ids": ["h22.1"],
                 "code": "smf.ols('pfs_months ~ ar_v7_positive * treatment_abiraterone', df).fit()",
                 "result_summary": f"Interaction ar_v7_positive:treatment_abiraterone coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
sub = df[df["ar_v7_positive"] == 1]
r = ttest(sub.loc[sub["treatment_abiraterone"] == 1, OUT], sub.loc[sub["treatment_abiraterone"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h22.2"],
                 "code": "sub=df[df['ar_v7_positive']==1]; ttest abiraterone groups",
                 "result_summary": f"AR-V7+ (n={r['n1']+r['n0']}): abi mean {r['mean1']:.3f} vs no-abi {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
sub = df[df["ar_v7_positive"] == 0]
r = ttest(sub.loc[sub["treatment_abiraterone"] == 1, OUT], sub.loc[sub["treatment_abiraterone"] == 0, OUT])
analyses.append({"hypothesis_ids": ["h22.3"],
                 "code": "sub=df[df['ar_v7_positive']==0]; ttest abiraterone groups",
                 "result_summary": f"AR-V7- (n={r['n1']+r['n0']}): abi mean {r['mean1']:.3f} vs no-abi {r['mean0']:.3f}, diff={r['diff']:.3f}, p={r['p']:.3g}.",
                 "p_value": r["p"], "effect_estimate": r["diff"], "significant": bool(r["p"] < 0.05)})
add_iter(22, hyps, analyses)

# ============================================================
# ITERATION 23: Comprehensive multivariable model with biomarker matched-therapy interactions
# ============================================================
hyps = [
    {"id": "h23.1", "text": "In a single OLS containing ecog_ps, age_years, mcrpc, visceral_mets, ldh_u_l, albumin_g_dl, hemoglobin_g_dl, prior_lines_of_therapy, all six treatments, and biomarker:matched-therapy interactions (brca2_mutation:treatment_olaparib, psma_high:treatment_lu177_psma, msi_high:treatment_pembrolizumab, ar_v7_positive:treatment_enzalutamide), the brca2_mutation:treatment_olaparib interaction is positive and significant.", "kind": "refined"},
    {"id": "h23.2", "text": "In the same comprehensive model, the psma_high:treatment_lu177_psma interaction is positive and significant.", "kind": "refined"},
    {"id": "h23.3", "text": "In the same comprehensive model, the msi_high:treatment_pembrolizumab interaction is positive and significant.", "kind": "refined"},
    {"id": "h23.4", "text": "In the same comprehensive model, the ar_v7_positive:treatment_enzalutamide interaction is negative and significant.", "kind": "refined"},
]
formula = ("pfs_months ~ ecog_ps + age_years + mcrpc + visceral_mets + ldh_u_l + albumin_g_dl + "
           "hemoglobin_g_dl + prior_lines_of_therapy + treatment_enzalutamide + treatment_abiraterone + "
           "treatment_docetaxel + treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab + "
           "brca2_mutation + psma_high + msi_high + ar_v7_positive + "
           "brca2_mutation:treatment_olaparib + psma_high:treatment_lu177_psma + "
           "msi_high:treatment_pembrolizumab + ar_v7_positive:treatment_enzalutamide")
m_big = ols(formula)
analyses = []
for hid, name in [
    ("h23.1", "brca2_mutation:treatment_olaparib"),
    ("h23.2", "psma_high:treatment_lu177_psma"),
    ("h23.3", "msi_high:treatment_pembrolizumab"),
    ("h23.4", "ar_v7_positive:treatment_enzalutamide"),
]:
    c = coef_row(m_big, name)
    analyses.append({"hypothesis_ids": [hid], "code": "comprehensive multivariable OLS with interactions",
                     "result_summary": f"Adjusted interaction {name} coef={c['coef']:.3f} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(23, hyps, analyses)

# ============================================================
# ITERATION 24: Interaction of ECOG with biomarker therapy
# ============================================================
hyps = [
    {"id": "h24.1", "text": "Among ecog_ps==0, the brca2_mutation:treatment_olaparib interaction is positive and significant.", "kind": "refined"},
    {"id": "h24.2", "text": "Among ecog_ps>=2, the brca2_mutation:treatment_olaparib interaction is still positive but smaller (modulated by performance status).", "kind": "refined"},
    {"id": "h24.3", "text": "Three-way interaction ecog_ps:brca2_mutation:treatment_olaparib has p<0.05.", "kind": "novel"},
]
analyses = []
sub = df[df["ecog_ps"] == 0]
m = smf.ols("pfs_months ~ brca2_mutation * treatment_olaparib", data=sub).fit()
c = {"coef": float(m.params["brca2_mutation:treatment_olaparib"]),
     "p": float(m.pvalues["brca2_mutation:treatment_olaparib"])}
analyses.append({"hypothesis_ids": ["h24.1"],
                 "code": "sub=df[df['ecog_ps']==0]; smf.ols('pfs_months ~ brca2_mutation * treatment_olaparib', sub).fit()",
                 "result_summary": f"ECOG=0 stratum: interaction brca2_mutation:treatment_olaparib coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
sub = df[df["ecog_ps"] >= 2]
if len(sub) > 50:
    m = smf.ols("pfs_months ~ brca2_mutation * treatment_olaparib", data=sub).fit()
    c = {"coef": float(m.params["brca2_mutation:treatment_olaparib"]),
         "p": float(m.pvalues["brca2_mutation:treatment_olaparib"])}
else:
    c = {"coef": float("nan"), "p": float("nan")}
analyses.append({"hypothesis_ids": ["h24.2"],
                 "code": "sub=df[df['ecog_ps']>=2]; smf.ols('pfs_months ~ brca2_mutation * treatment_olaparib', sub).fit()",
                 "result_summary": f"ECOG>=2 stratum (n={len(sub)}): interaction coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05) if c['p'] == c['p'] else False})
m = ols("pfs_months ~ ecog_ps * brca2_mutation * treatment_olaparib")
c = coef_row(m, "ecog_ps:brca2_mutation:treatment_olaparib")
analyses.append({"hypothesis_ids": ["h24.3"],
                 "code": "smf.ols('pfs_months ~ ecog_ps * brca2_mutation * treatment_olaparib', df).fit()",
                 "result_summary": f"3-way interaction coef={c['coef']:.3f} (p={c['p']:.3g}).",
                 "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
add_iter(24, hyps, analyses)

# ============================================================
# ITERATION 25: Final summary checks
# ============================================================
hyps = [
    {"id": "h25.1", "text": "After adjusting for ecog_ps, age_years, mcrpc, visceral_mets, ldh_u_l, albumin_g_dl, hemoglobin_g_dl, prior_lines_of_therapy and the four biomarker:matched-treatment interactions, the main effect of treatment_olaparib (off-target use, when brca2_mutation=0) on pfs_months is null or negative.", "kind": "refined"},
    {"id": "h25.2", "text": "In the comprehensive multivariable model the main effect of treatment_lu177_psma (in psma_high=0) on pfs_months is null or negative.", "kind": "refined"},
    {"id": "h25.3", "text": "In the comprehensive multivariable model the main effect of treatment_pembrolizumab (in msi_high=0) on pfs_months is null or negative.", "kind": "refined"},
    {"id": "h25.4", "text": "Across the comprehensive model, the four biomarker:matched-treatment interactions remain the dominant treatment-effect drivers; main treatment terms alone are small.", "kind": "refined"},
]
analyses = []
# Reuse m_big from iteration 23
for hid, tx in [("h25.1", "treatment_olaparib"), ("h25.2", "treatment_lu177_psma"),
                ("h25.3", "treatment_pembrolizumab")]:
    c = coef_row(m_big, tx)
    analyses.append({"hypothesis_ids": [hid], "code": "comprehensive multivariable OLS, main-effect coef when interaction term present",
                     "result_summary": f"In comprehensive model, main effect of {tx} (i.e. effect when its biomarker=0) coef={c['coef']:.3f} (p={c['p']:.3g}).",
                     "p_value": c["p"], "effect_estimate": c["coef"], "significant": bool(c["p"] < 0.05)})
# Compare magnitudes of interaction vs main treatment terms
tx_names = ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
            "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]
inter_names = ["brca2_mutation:treatment_olaparib", "psma_high:treatment_lu177_psma",
               "msi_high:treatment_pembrolizumab", "ar_v7_positive:treatment_enzalutamide"]
mean_main = float(np.mean([abs(m_big.params[n]) for n in tx_names]))
mean_inter = float(np.mean([abs(m_big.params[n]) for n in inter_names]))
analyses.append({"hypothesis_ids": ["h25.4"], "code": "compare |coef| means main vs biomarker-treatment interaction terms",
                 "result_summary": f"Mean |coef| main treatment terms = {mean_main:.3f}; mean |coef| biomarker:treatment interaction terms = {mean_inter:.3f}. Ratio interaction/main = {mean_inter / mean_main:.2f}.",
                 "p_value": None, "effect_estimate": float(mean_inter - mean_main),
                 "significant": bool(mean_inter > mean_main)})
add_iter(25, hyps, analyses)

# ============================================================
# Write transcript
# ============================================================
transcript = {
    "dataset_id": "ds001_prostate",
    "model_id": "claude-opus-4-7",
    "harness_id": "claude-code@iterative-eda-2026.04",
    "max_iterations": 25,
    "iterations": iterations,
}
with open("transcript.json", "w") as f:
    json.dump(transcript, f, indent=2)

print(f"Wrote transcript.json with {len(iterations)} iterations.")

# Print key results for narrative composition
print("=" * 60)
for it in iterations:
    print(f"\n--- Iteration {it['index']} ---")
    for a in it["analyses"]:
        print(f"  [{','.join(a['hypothesis_ids'])}] {a['result_summary']}")
