"""Run 10-iteration analysis on ds001_prostate dataset.

Produces all_results.json (raw analysis records) which is then assembled
into transcript.json and analysis_summary.txt.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")

OUT = {"iterations": []}


def add_iter(idx, hyps, analyses):
    OUT["iterations"].append(
        {"index": idx, "proposed_hypotheses": hyps, "analyses": analyses}
    )


def t_test(col, group_col, label_treated="1", label_control="0"):
    """Welch t-test of `col` between rows where group_col==1 and ==0.
    Returns dict with effect (mean diff treated - control), p, sig.
    """
    a = df.loc[df[group_col] == 1, col]
    b = df.loc[df[group_col] == 0, col]
    res = stats.ttest_ind(a, b, equal_var=False)
    eff = float(a.mean() - b.mean())
    return {
        "effect_estimate": eff,
        "p_value": float(res.pvalue),
        "significant": bool(res.pvalue < 0.05),
        "result_summary": (
            f"Mean {col}: {a.mean():.3f} when {group_col}=1 (n={len(a)}) vs "
            f"{b.mean():.3f} when {group_col}=0 (n={len(b)}); "
            f"Welch t p={res.pvalue:.3g}; mean diff={eff:.3f}"
        ),
    }


def lin_reg(formula, focus_term):
    """Fit OLS, return signed coefficient for focus_term plus p value."""
    m = smf.ols(formula, data=df).fit()
    coef = float(m.params[focus_term])
    p = float(m.pvalues[focus_term])
    return {
        "effect_estimate": coef,
        "p_value": p,
        "significant": bool(p < 0.05),
        "result_summary": (
            f"OLS '{formula}': beta[{focus_term}]={coef:.4f} "
            f"(SE={m.bse[focus_term]:.4f}), p={p:.3g}"
        ),
    }


def interaction(treat, biomarker):
    """Test treat * biomarker interaction in OLS on pfs_months.
    Reports the interaction coefficient and the on-treatment effect within
    biomarker-positive vs biomarker-negative subsets.
    """
    f = f"pfs_months ~ {treat} * {biomarker}"
    m = smf.ols(f, data=df).fit()
    inter_term = f"{treat}:{biomarker}"
    coef = float(m.params[inter_term])
    p = float(m.pvalues[inter_term])
    # subgroup mean diffs
    pos = df[df[biomarker] == 1]
    neg = df[df[biomarker] == 0]
    eff_pos = (
        pos.loc[pos[treat] == 1, "pfs_months"].mean()
        - pos.loc[pos[treat] == 0, "pfs_months"].mean()
    )
    eff_neg = (
        neg.loc[neg[treat] == 1, "pfs_months"].mean()
        - neg.loc[neg[treat] == 0, "pfs_months"].mean()
    )
    return {
        "effect_estimate": coef,
        "p_value": p,
        "significant": bool(p < 0.05),
        "result_summary": (
            f"Interaction {treat}:{biomarker} beta={coef:.4f}, p={p:.3g}. "
            f"Effect of {treat} in {biomarker}=1: {eff_pos:.3f} months; "
            f"in {biomarker}=0: {eff_neg:.3f} months."
        ),
    }


# ----------------------------------------------------------------------------
# Iteration 1: Main effects of each of the 6 listed treatments on pfs_months
# ----------------------------------------------------------------------------
hyps1 = [
    {"id": "h1.1", "text": "Patients receiving treatment_enzalutamide have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1.2", "text": "Patients receiving treatment_abiraterone have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1.3", "text": "Patients receiving treatment_docetaxel have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1.4", "text": "Patients receiving treatment_olaparib have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1.5", "text": "Patients receiving treatment_lu177_psma have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
    {"id": "h1.6", "text": "Patients receiving treatment_pembrolizumab have longer mean pfs_months than patients not receiving it.", "kind": "novel"},
]
analyses1 = []
for hid, treat in zip(
    ["h1.1","h1.2","h1.3","h1.4","h1.5","h1.6"],
    ["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel","treatment_olaparib","treatment_lu177_psma","treatment_pembrolizumab"],
):
    r = t_test("pfs_months", treat)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"stats.ttest_ind(df.loc[df['{treat}']==1,'pfs_months'], df.loc[df['{treat}']==0,'pfs_months'], equal_var=False)"
    analyses1.append(r)
add_iter(1, hyps1, analyses1)


# ----------------------------------------------------------------------------
# Iteration 2: Classic prognostic features (disease state, sites of disease,
# tumor markers, gleason)
# ----------------------------------------------------------------------------
hyps2 = [
    {"id": "h2.1", "text": "Higher ecog_ps is associated with shorter pfs_months (negative linear association).", "kind": "novel"},
    {"id": "h2.2", "text": "Patients with mcrpc=1 have shorter mean pfs_months than patients with mcrpc=0.", "kind": "novel"},
    {"id": "h2.3", "text": "Patients with visceral_mets=1 have shorter mean pfs_months than those without.", "kind": "novel"},
    {"id": "h2.4", "text": "Patients with liver_mets=1 have shorter mean pfs_months than those without.", "kind": "novel"},
    {"id": "h2.5", "text": "Patients with bone_mets=1 have shorter mean pfs_months than those without.", "kind": "novel"},
    {"id": "h2.6", "text": "Higher psa_ng_ml (log-transformed) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.7", "text": "Higher gleason_score is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h2.8", "text": "Older age_years is associated with shorter pfs_months.", "kind": "novel"},
]
analyses2 = []
# ECOG: linear regression
analyses2.append({**lin_reg("pfs_months ~ ecog_ps", "ecog_ps"),
                  "hypothesis_ids": ["h2.1"],
                  "code": "smf.ols('pfs_months ~ ecog_ps', df).fit()"})
for hid, col in [("h2.2","mcrpc"),("h2.3","visceral_mets"),("h2.4","liver_mets"),("h2.5","bone_mets")]:
    r = t_test("pfs_months", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"stats.ttest_ind on pfs_months by {col}"
    analyses2.append(r)
# log PSA
df["log_psa"] = np.log1p(df["psa_ng_ml"])
analyses2.append({**lin_reg("pfs_months ~ log_psa", "log_psa"),
                  "hypothesis_ids": ["h2.6"],
                  "code": "smf.ols('pfs_months ~ log_psa', df).fit()"})
analyses2.append({**lin_reg("pfs_months ~ gleason_score", "gleason_score"),
                  "hypothesis_ids": ["h2.7"],
                  "code": "smf.ols('pfs_months ~ gleason_score', df).fit()"})
analyses2.append({**lin_reg("pfs_months ~ age_years", "age_years"),
                  "hypothesis_ids": ["h2.8"],
                  "code": "smf.ols('pfs_months ~ age_years', df).fit()"})
add_iter(2, hyps2, analyses2)


# ----------------------------------------------------------------------------
# Iteration 3: Biomarker main effects (BRCA2, AR-V7, MSI-high, PSMA-high,
# TP53, PTEN loss)
# ----------------------------------------------------------------------------
hyps3 = [
    {"id":"h3.1","text":"Patients with brca2_mutation=1 have different mean pfs_months than those without (predicted shorter overall, but PARP-targetable).","kind":"novel"},
    {"id":"h3.2","text":"Patients with ar_v7_positive=1 have shorter mean pfs_months than those without (associated with AR-targeted resistance).","kind":"novel"},
    {"id":"h3.3","text":"Patients with msi_high=1 have different mean pfs_months than those without (overall effect uncertain; immunotherapy responders).","kind":"novel"},
    {"id":"h3.4","text":"Patients with psma_high=1 have different mean pfs_months than those without (theranostic-eligible subset).","kind":"novel"},
    {"id":"h3.5","text":"Patients with tp53_mutation=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h3.6","text":"Patients with pten_loss=1 have shorter mean pfs_months than those without.","kind":"novel"},
]
analyses3 = []
for hid, col in [("h3.1","brca2_mutation"),("h3.2","ar_v7_positive"),("h3.3","msi_high"),
                 ("h3.4","psma_high"),("h3.5","tp53_mutation"),("h3.6","pten_loss")]:
    r = t_test("pfs_months", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"stats.ttest_ind on pfs_months by {col}"
    analyses3.append(r)
add_iter(3, hyps3, analyses3)


# ----------------------------------------------------------------------------
# Iteration 4: Treatment x biomarker INTERACTIONS (the precision-oncology
# hypotheses): BRCA2xolaparib, AR-V7xAR-inhibitors, MSIxpembrolizumab,
# PSMAxLu177-PSMA
# ----------------------------------------------------------------------------
hyps4 = [
    {"id":"h4.1","text":"The benefit of treatment_olaparib on pfs_months is larger in brca2_mutation=1 patients than in brca2_mutation=0 patients (positive treatment x biomarker interaction).","kind":"novel"},
    {"id":"h4.2","text":"The benefit of treatment_enzalutamide on pfs_months is smaller (or harmful) in ar_v7_positive=1 patients than in ar_v7_positive=0 (negative interaction).","kind":"novel"},
    {"id":"h4.3","text":"The benefit of treatment_abiraterone on pfs_months is smaller in ar_v7_positive=1 patients than in ar_v7_positive=0 patients (negative interaction).","kind":"novel"},
    {"id":"h4.4","text":"The benefit of treatment_pembrolizumab on pfs_months is larger in msi_high=1 patients than in msi_high=0 patients (positive interaction).","kind":"novel"},
    {"id":"h4.5","text":"The benefit of treatment_lu177_psma on pfs_months is larger in psma_high=1 patients than in psma_high=0 patients (positive interaction).","kind":"novel"},
]
analyses4 = []
for hid, treat, bio in [
    ("h4.1","treatment_olaparib","brca2_mutation"),
    ("h4.2","treatment_enzalutamide","ar_v7_positive"),
    ("h4.3","treatment_abiraterone","ar_v7_positive"),
    ("h4.4","treatment_pembrolizumab","msi_high"),
    ("h4.5","treatment_lu177_psma","psma_high"),
]:
    r = interaction(treat, bio)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"smf.ols('pfs_months ~ {treat} * {bio}', df).fit()"
    analyses4.append(r)
add_iter(4, hyps4, analyses4)


# ----------------------------------------------------------------------------
# Iteration 5: Lab-based prognostic factors. Hypotheses informed by
# iteration 2 (ECOG, mets are strongly prognostic). Now check albumin, LDH,
# hemoglobin, ALP, NLR, CRP.
# ----------------------------------------------------------------------------
hyps5 = [
    {"id":"h5.1","text":"Higher albumin_g_dl is associated with longer pfs_months (positive linear association).","kind":"novel"},
    {"id":"h5.2","text":"Higher ldh_u_l is associated with shorter pfs_months (negative linear association).","kind":"novel"},
    {"id":"h5.3","text":"Higher hemoglobin_g_dl is associated with longer pfs_months.","kind":"novel"},
    {"id":"h5.4","text":"Higher alkaline_phosphatase_u_l is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h5.5","text":"Higher nlr (neutrophil-lymphocyte ratio) is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h5.6","text":"Higher crp_mg_l is associated with shorter pfs_months.","kind":"novel"},
]
analyses5 = []
for hid, col in [
    ("h5.1","albumin_g_dl"),
    ("h5.2","ldh_u_l"),
    ("h5.3","hemoglobin_g_dl"),
    ("h5.4","alkaline_phosphatase_u_l"),
    ("h5.5","nlr"),
    ("h5.6","crp_mg_l"),
]:
    r = lin_reg(f"pfs_months ~ {col}", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"smf.ols('pfs_months ~ {col}', df).fit()"
    analyses5.append(r)
add_iter(5, hyps5, analyses5)


# ----------------------------------------------------------------------------
# Iteration 6: Symptom burden and weight loss as prognostic factors
# ----------------------------------------------------------------------------
hyps6 = [
    {"id":"h6.1","text":"Higher fatigue_grade is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h6.2","text":"Higher pain_nrs is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h6.3","text":"Higher dyspnea_grade is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h6.4","text":"Higher appetite_loss_grade is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h6.5","text":"Greater weight_loss_pct_6mo is associated with shorter pfs_months.","kind":"novel"},
]
analyses6 = []
for hid, col in [
    ("h6.1","fatigue_grade"),
    ("h6.2","pain_nrs"),
    ("h6.3","dyspnea_grade"),
    ("h6.4","appetite_loss_grade"),
    ("h6.5","weight_loss_pct_6mo"),
]:
    r = lin_reg(f"pfs_months ~ {col}", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"smf.ols('pfs_months ~ {col}', df).fit()"
    analyses6.append(r)
add_iter(6, hyps6, analyses6)


# ----------------------------------------------------------------------------
# Iteration 7: Comorbidities and prior therapies as prognostic factors
# ----------------------------------------------------------------------------
hyps7 = [
    {"id":"h7.1","text":"Patients with chronic_kidney_disease=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h7.2","text":"Patients with heart_failure=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h7.3","text":"Patients with diabetes_mellitus=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h7.4","text":"Greater prior_lines_of_therapy is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h7.5","text":"Patients with prior_chemotherapy=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h7.6","text":"Greater years_since_diagnosis is associated with shorter pfs_months.","kind":"novel"},
]
analyses7 = []
for hid, col in [("h7.1","chronic_kidney_disease"),("h7.2","heart_failure"),("h7.3","diabetes_mellitus"),("h7.5","prior_chemotherapy")]:
    r = t_test("pfs_months", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"stats.ttest_ind on pfs_months by {col}"
    analyses7.append(r)
for hid, col in [("h7.4","prior_lines_of_therapy"),("h7.6","years_since_diagnosis")]:
    r = lin_reg(f"pfs_months ~ {col}", col)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"smf.ols('pfs_months ~ {col}', df).fit()"
    analyses7.append(r)
add_iter(7, hyps7, analyses7)


# ----------------------------------------------------------------------------
# Iteration 8: Demographics & social determinants
# ----------------------------------------------------------------------------
hyps8 = [
    {"id":"h8.1","text":"Mean pfs_months differs across race_ethnicity categories (overall ANOVA).","kind":"novel"},
    {"id":"h8.2","text":"Mean pfs_months differs across insurance_type categories (overall ANOVA).","kind":"novel"},
    {"id":"h8.3","text":"Patients with rural_residence=1 have shorter mean pfs_months than those without.","kind":"novel"},
    {"id":"h8.4","text":"Higher smoking_pack_years is associated with shorter pfs_months.","kind":"novel"},
    {"id":"h8.5","text":"More education_years is associated with longer pfs_months.","kind":"novel"},
    {"id":"h8.6","text":"Patients with insurance_type='uninsured' have shorter mean pfs_months than patients with insurance_type='private'.","kind":"novel"},
]
analyses8 = []
# ANOVA F test for race via one-way ANOVA on group samples
race_groups = [g["pfs_months"].values for _, g in df.groupby("race_ethnicity")]
f_race = stats.f_oneway(*race_groups)
race_means = df.groupby("race_ethnicity")["pfs_months"].mean().to_dict()
analyses8.append({
    "hypothesis_ids":["h8.1"],
    "code":"stats.f_oneway across race_ethnicity groups",
    "result_summary": f"One-way ANOVA pfs_months by race_ethnicity; group means {race_means}; F={float(f_race.statistic):.3f}, p={float(f_race.pvalue):.3g}",
    "p_value": float(f_race.pvalue),
    "effect_estimate": float(max(race_means.values()) - min(race_means.values())),
    "significant": bool(float(f_race.pvalue) < 0.05),
})
ins_groups = [g["pfs_months"].values for _, g in df.groupby("insurance_type")]
f_ins = stats.f_oneway(*ins_groups)
ins_means = df.groupby("insurance_type")["pfs_months"].mean().to_dict()
analyses8.append({
    "hypothesis_ids":["h8.2"],
    "code":"stats.f_oneway across insurance_type groups",
    "result_summary": f"One-way ANOVA pfs_months by insurance_type; group means {ins_means}; F={float(f_ins.statistic):.3f}, p={float(f_ins.pvalue):.3g}",
    "p_value": float(f_ins.pvalue),
    "effect_estimate": float(max(ins_means.values()) - min(ins_means.values())),
    "significant": bool(float(f_ins.pvalue) < 0.05),
})
r = t_test("pfs_months","rural_residence")
r["hypothesis_ids"] = ["h8.3"]
r["code"] = "stats.ttest_ind by rural_residence"
analyses8.append(r)
analyses8.append({**lin_reg("pfs_months ~ smoking_pack_years","smoking_pack_years"),
                  "hypothesis_ids":["h8.4"],
                  "code":"smf.ols('pfs_months ~ smoking_pack_years', df).fit()"})
analyses8.append({**lin_reg("pfs_months ~ education_years","education_years"),
                  "hypothesis_ids":["h8.5"],
                  "code":"smf.ols('pfs_months ~ education_years', df).fit()"})
# uninsured vs private
sub = df[df["insurance_type"].isin(["uninsured","private"])].copy()
a = sub.loc[sub["insurance_type"]=="uninsured","pfs_months"]
b = sub.loc[sub["insurance_type"]=="private","pfs_months"]
res = stats.ttest_ind(a, b, equal_var=False)
analyses8.append({
    "hypothesis_ids":["h8.6"],
    "code":"stats.ttest_ind(uninsured vs private)",
    "result_summary": f"Uninsured mean={a.mean():.3f} (n={len(a)}) vs private mean={b.mean():.3f} (n={len(b)}); diff={a.mean()-b.mean():.3f}; Welch p={float(res.pvalue):.3g}",
    "p_value": float(res.pvalue),
    "effect_estimate": float(a.mean()-b.mean()),
    "significant": bool(float(res.pvalue)<0.05),
})
add_iter(8, hyps8, analyses8)


# ----------------------------------------------------------------------------
# Iteration 9: SNPs - screen each SNP for main effect on pfs_months.
# Report the strongest effect found and a few specific candidates.
# ----------------------------------------------------------------------------
snp_cols = [c for c in df.columns if c.startswith("snp_")]
hyps9 = [
    {"id":"h9.1","text":"Among the 25 SNP variants in the dataset, at least one is associated with pfs_months at the Bonferroni-corrected significance threshold (overall screen).","kind":"novel"},
    {"id":"h9.2","text":"snp_rs1045642 (ABCB1) is associated with pfs_months.","kind":"novel"},
    {"id":"h9.3","text":"snp_rs429358 (APOE) is associated with pfs_months.","kind":"novel"},
    {"id":"h9.4","text":"snp_rs1800629 (TNF-alpha) is associated with pfs_months.","kind":"novel"},
]
analyses9 = []
snp_results = []
for c in snp_cols:
    res = lin_reg(f"pfs_months ~ {c}", c)
    snp_results.append((c, res["effect_estimate"], res["p_value"]))
# overall screen result: how many significant after correction
ps = np.array([r[2] for r in snp_results])
n_sig_uncorr = int((ps < 0.05).sum())
n_sig_bonf = int((ps < 0.05 / len(ps)).sum())
min_p = float(ps.min())
best_snp = snp_results[int(np.argmin(ps))]
analyses9.append({
    "hypothesis_ids":["h9.1"],
    "code":"loop OLS pfs_months~snp_X for all 30 SNPs",
    "result_summary": (
        f"Of {len(snp_cols)} SNPs, {n_sig_uncorr} reached p<0.05 uncorrected; "
        f"{n_sig_bonf} survived Bonferroni (p<{0.05/len(snp_cols):.3g}); "
        f"smallest p was {min_p:.3g} for {best_snp[0]} (beta={best_snp[1]:.4f})."
    ),
    "p_value": min_p,
    "effect_estimate": float(best_snp[1]),
    "significant": bool(n_sig_bonf > 0),
})
for hid, c in [("h9.2","snp_rs1045642"),("h9.3","snp_rs429358"),("h9.4","snp_rs1800629")]:
    r = lin_reg(f"pfs_months ~ {c}", c)
    r["hypothesis_ids"] = [hid]
    r["code"] = f"smf.ols('pfs_months ~ {c}', df).fit()"
    analyses9.append(r)
add_iter(9, hyps9, analyses9)


# ----------------------------------------------------------------------------
# Iteration 10: Multivariable model + key refinements:
#  - Joint OLS adjusting for ECOG, mCRPC, visceral mets, log PSA,
#    albumin, hemoglobin, LDH, NLR
#  - Refined precision interactions: do they hold after adjusting for ECOG?
#  - Insurance disparity: does it persist after controlling for disease state?
# ----------------------------------------------------------------------------
hyps10 = [
    {"id":"h10.1","text":"In a multivariable OLS adjusting for ecog_ps, mcrpc, visceral_mets, log_psa, albumin_g_dl, hemoglobin_g_dl, ldh_u_l, nlr, ecog_ps remains independently associated with shorter pfs_months.","kind":"refined"},
    {"id":"h10.2","text":"In the same multivariable model, albumin_g_dl is independently associated with longer pfs_months.","kind":"refined"},
    {"id":"h10.3","text":"The benefit of treatment_olaparib in brca2_mutation=1 patients vs brca2_mutation=0 patients (interaction effect on pfs_months) persists after adjusting for ecog_ps and mcrpc.","kind":"refined"},
    {"id":"h10.4","text":"The benefit of treatment_pembrolizumab in msi_high=1 vs msi_high=0 patients (interaction) persists after adjusting for ecog_ps and mcrpc.","kind":"refined"},
    {"id":"h10.5","text":"The insurance_type disparity (uninsured vs private patients) in pfs_months persists after adjusting for ecog_ps, mcrpc, visceral_mets, and log_psa.","kind":"refined"},
    {"id":"h10.6","text":"After multivariable adjustment, treatment_docetaxel remains negatively associated with pfs_months (consistent with selection of sicker patients for docetaxel rather than direct harm).","kind":"refined"},
]
analyses10 = []
# Multivariable model
formula_full = (
    "pfs_months ~ ecog_ps + mcrpc + visceral_mets + log_psa + albumin_g_dl "
    "+ hemoglobin_g_dl + ldh_u_l + nlr + treatment_enzalutamide + treatment_abiraterone "
    "+ treatment_docetaxel + treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab"
)
m_full = smf.ols(formula_full, df).fit()
analyses10.append({
    "hypothesis_ids":["h10.1"],
    "code": f"smf.ols('{formula_full}', df).fit()",
    "result_summary": (
        f"Adjusted ecog_ps beta={m_full.params['ecog_ps']:.4f} "
        f"(SE={m_full.bse['ecog_ps']:.4f}), p={m_full.pvalues['ecog_ps']:.3g}; "
        f"R^2={m_full.rsquared:.3f}"
    ),
    "p_value": float(m_full.pvalues["ecog_ps"]),
    "effect_estimate": float(m_full.params["ecog_ps"]),
    "significant": bool(m_full.pvalues["ecog_ps"]<0.05),
})
analyses10.append({
    "hypothesis_ids":["h10.2"],
    "code": f"smf.ols('{formula_full}', df).fit()",
    "result_summary": (
        f"Adjusted albumin_g_dl beta={m_full.params['albumin_g_dl']:.4f} "
        f"(SE={m_full.bse['albumin_g_dl']:.4f}), p={m_full.pvalues['albumin_g_dl']:.3g}"
    ),
    "p_value": float(m_full.pvalues["albumin_g_dl"]),
    "effect_estimate": float(m_full.params["albumin_g_dl"]),
    "significant": bool(m_full.pvalues["albumin_g_dl"]<0.05),
})
# Adjusted interactions
f1 = "pfs_months ~ treatment_olaparib * brca2_mutation + ecog_ps + mcrpc"
m1 = smf.ols(f1, df).fit()
analyses10.append({
    "hypothesis_ids":["h10.3"],
    "code": f"smf.ols('{f1}', df).fit()",
    "result_summary": (
        f"Adjusted olaparib:brca2 interaction beta={m1.params['treatment_olaparib:brca2_mutation']:.4f}, "
        f"p={m1.pvalues['treatment_olaparib:brca2_mutation']:.3g}"
    ),
    "p_value": float(m1.pvalues["treatment_olaparib:brca2_mutation"]),
    "effect_estimate": float(m1.params["treatment_olaparib:brca2_mutation"]),
    "significant": bool(m1.pvalues["treatment_olaparib:brca2_mutation"]<0.05),
})
f2 = "pfs_months ~ treatment_pembrolizumab * msi_high + ecog_ps + mcrpc"
m2 = smf.ols(f2, df).fit()
analyses10.append({
    "hypothesis_ids":["h10.4"],
    "code": f"smf.ols('{f2}', df).fit()",
    "result_summary": (
        f"Adjusted pembrolizumab:msi_high interaction beta={m2.params['treatment_pembrolizumab:msi_high']:.4f}, "
        f"p={m2.pvalues['treatment_pembrolizumab:msi_high']:.3g}"
    ),
    "p_value": float(m2.pvalues["treatment_pembrolizumab:msi_high"]),
    "effect_estimate": float(m2.params["treatment_pembrolizumab:msi_high"]),
    "significant": bool(m2.pvalues["treatment_pembrolizumab:msi_high"]<0.05),
})
# Insurance disparity adjusted
sub = df[df["insurance_type"].isin(["uninsured","private"])].copy()
sub["uninsured"] = (sub["insurance_type"]=="uninsured").astype(int)
f3 = "pfs_months ~ uninsured + ecog_ps + mcrpc + visceral_mets + log_psa"
m3 = smf.ols(f3, sub).fit()
analyses10.append({
    "hypothesis_ids":["h10.5"],
    "code": f"smf.ols('{f3}', sub).fit()  # sub = uninsured/private only",
    "result_summary": (
        f"Adjusted uninsured (vs private) beta={m3.params['uninsured']:.4f} "
        f"(SE={m3.bse['uninsured']:.4f}), p={m3.pvalues['uninsured']:.3g}"
    ),
    "p_value": float(m3.pvalues["uninsured"]),
    "effect_estimate": float(m3.params["uninsured"]),
    "significant": bool(m3.pvalues["uninsured"]<0.05),
})
# docetaxel adjusted
analyses10.append({
    "hypothesis_ids":["h10.6"],
    "code": f"smf.ols('{formula_full}', df).fit()",
    "result_summary": (
        f"Adjusted treatment_docetaxel beta={m_full.params['treatment_docetaxel']:.4f} "
        f"(SE={m_full.bse['treatment_docetaxel']:.4f}), p={m_full.pvalues['treatment_docetaxel']:.3g}"
    ),
    "p_value": float(m_full.pvalues["treatment_docetaxel"]),
    "effect_estimate": float(m_full.params["treatment_docetaxel"]),
    "significant": bool(m_full.pvalues["treatment_docetaxel"]<0.05),
})
add_iter(10, hyps10, analyses10)


with open("all_results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)

# Print compact summary
for it in OUT["iterations"]:
    print(f"=== Iteration {it['index']} ===")
    for a in it["analyses"]:
        sig = "*" if a.get("significant") else " "
        print(f"  [{sig}] {','.join(a['hypothesis_ids'])}: eff={a.get('effect_estimate')!r} p={a.get('p_value')!r}")
        print(f"      {a['result_summary']}")
