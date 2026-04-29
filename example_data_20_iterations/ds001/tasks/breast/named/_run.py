"""Comprehensive analysis of ds001_breast cohort. Emits results to results_out.json."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
out = []  # list of {iter, hypotheses:[{id,text,kind}], analyses:[{hids,summary,p,eff,sig,code}]}


def add(idx, hyps, analyses):
    out.append({"index": idx, "hypotheses": hyps, "analyses": analyses})


def ttest(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(np.mean(a) - np.mean(b)), float(p)


def ols_coef(formula, data=df):
    m = smf.ols(formula, data=data).fit()
    return m


# ---------- Iteration 1: baseline treatment main effects on PFS ----------
hyps = [
    {"id": "h1_palbo", "text": "Patients receiving treatment_palbociclib have longer mean pfs_months than those not receiving palbociclib.", "kind": "novel"},
    {"id": "h1_tras", "text": "Patients receiving treatment_trastuzumab have longer mean pfs_months than those not receiving trastuzumab.", "kind": "novel"},
    {"id": "h1_pembro", "text": "Patients receiving treatment_pembrolizumab have longer mean pfs_months than those not receiving pembrolizumab.", "kind": "novel"},
    {"id": "h1_olap", "text": "Patients receiving treatment_olaparib have longer mean pfs_months than those not receiving olaparib.", "kind": "novel"},
    {"id": "h1_tam", "text": "Patients receiving treatment_tamoxifen have longer mean pfs_months than those not receiving tamoxifen.", "kind": "novel"},
    {"id": "h1_sg", "text": "Patients receiving treatment_sacituzumab_govitecan have longer mean pfs_months than those not receiving it.", "kind": "novel"},
]
analyses = []
for tcol, hid in [("treatment_palbociclib", "h1_palbo"), ("treatment_trastuzumab", "h1_tras"),
                  ("treatment_pembrolizumab", "h1_pembro"), ("treatment_olaparib", "h1_olap"),
                  ("treatment_tamoxifen", "h1_tam"), ("treatment_sacituzumab_govitecan", "h1_sg")]:
    eff, p = ttest(df.loc[df[tcol] == 1, "pfs_months"], df.loc[df[tcol] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{tcol}']==1,'pfs_months'], df.loc[df['{tcol}']==0,'pfs_months'])",
        "summary": f"Mean PFS on {tcol}={df.loc[df[tcol]==1,'pfs_months'].mean():.2f} mo vs off={df.loc[df[tcol]==0,'pfs_months'].mean():.2f} mo (Welch t-test diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(1, hyps, analyses)

# ---------- Iteration 2: ER, PR, HER2 status main effects on PFS ----------
hyps = [
    {"id": "h2_er", "text": "ER-positive (er_positive=1) patients have longer mean pfs_months than ER-negative patients.", "kind": "novel"},
    {"id": "h2_pr", "text": "PR-positive (pr_positive=1) patients have longer mean pfs_months than PR-negative patients.", "kind": "novel"},
    {"id": "h2_her2", "text": "HER2-positive (her2_positive=1) patients have shorter mean pfs_months than HER2-negative patients (worse prognosis without HER2-directed therapy).", "kind": "novel"},
    {"id": "h2_her2low", "text": "HER2-low (her2_low=1) patients have different mean pfs_months than HER2-not-low patients.", "kind": "novel"},
]
analyses = []
for col, hid in [("er_positive", "h2_er"), ("pr_positive", "h2_pr"), ("her2_positive", "h2_her2"), ("her2_low", "h2_her2low")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{col}']==1,'pfs_months'], df.loc[df['{col}']==0,'pfs_months'])",
        "summary": f"Mean PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(2, hyps, analyses)

# ---------- Iteration 3: stage_iv, has_brain_mets, ECOG, age ----------
hyps = [
    {"id": "h3_stage", "text": "Stage IV (stage_iv=1) patients have shorter mean pfs_months than non-stage-IV patients.", "kind": "novel"},
    {"id": "h3_brain", "text": "Patients with brain metastases (has_brain_mets=1) have shorter mean pfs_months than those without.", "kind": "novel"},
    {"id": "h3_ecog", "text": "Higher ECOG performance status (ecog_ps) is associated with shorter pfs_months (negative slope in linear regression).", "kind": "novel"},
    {"id": "h3_age", "text": "Older age_years is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
analyses = []
for col, hid in [("stage_iv", "h3_stage"), ("has_brain_mets", "h3_brain")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"stats.ttest_ind(df.loc[df['{col}']==1,'pfs_months'], df.loc[df['{col}']==0,'pfs_months'])",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
m = ols_coef("pfs_months ~ ecog_ps")
analyses.append({"hids": ["h3_ecog"], "code": "smf.ols('pfs_months ~ ecog_ps').fit()",
                 "summary": f"OLS slope of pfs_months on ecog_ps = {m.params['ecog_ps']:.3f} mo per ECOG point (p={m.pvalues['ecog_ps']:.3g}).",
                 "p": float(m.pvalues['ecog_ps']), "eff": float(m.params['ecog_ps']), "sig": float(m.pvalues['ecog_ps']) < 0.05})
m = ols_coef("pfs_months ~ age_years")
analyses.append({"hids": ["h3_age"], "code": "smf.ols('pfs_months ~ age_years').fit()",
                 "summary": f"OLS slope of pfs_months on age_years = {m.params['age_years']:.4f} mo per year (p={m.pvalues['age_years']:.3g}).",
                 "p": float(m.pvalues['age_years']), "eff": float(m.params['age_years']), "sig": float(m.pvalues['age_years']) < 0.05})
add(3, hyps, analyses)

# ---------- Iteration 4: trastuzumab x HER2 interaction ----------
hyps = [
    {"id": "h4_tras_her2", "text": "Trastuzumab benefit (longer pfs_months) is larger in HER2-positive patients than in HER2-negative patients (positive treatment_trastuzumab × her2_positive interaction).", "kind": "novel"},
    {"id": "h4_tras_in_her2pos", "text": "Within her2_positive=1 patients, treatment_trastuzumab=1 yields longer pfs_months than treatment_trastuzumab=0.", "kind": "novel"},
]
m = ols_coef("pfs_months ~ treatment_trastuzumab * her2_positive")
inter = "treatment_trastuzumab:her2_positive"
analyses = [{
    "hids": ["h4_tras_her2"],
    "code": "smf.ols('pfs_months ~ treatment_trastuzumab * her2_positive').fit()",
    "summary": f"Interaction coef trastuzumab×her2_positive = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Trastuzumab main = {m.params['treatment_trastuzumab']:.3f}, HER2pos main = {m.params['her2_positive']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
sub = df[df.her2_positive == 1]
eff, p = ttest(sub.loc[sub.treatment_trastuzumab == 1, "pfs_months"], sub.loc[sub.treatment_trastuzumab == 0, "pfs_months"])
analyses.append({
    "hids": ["h4_tras_in_her2pos"],
    "code": "df[df.her2_positive==1].groupby('treatment_trastuzumab').pfs_months.mean()",
    "summary": f"In HER2+ (n={len(sub)}): trastuzumab=1 PFS={sub.loc[sub.treatment_trastuzumab==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_trastuzumab==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
    "p": p, "eff": eff, "sig": p < 0.05,
})
add(4, hyps, analyses)

# ---------- Iteration 5: tamoxifen x ER interaction ----------
hyps = [
    {"id": "h5_tam_er", "text": "Tamoxifen benefit on pfs_months is larger in ER-positive patients than in ER-negative patients (positive treatment_tamoxifen × er_positive interaction).", "kind": "novel"},
    {"id": "h5_tam_in_erpos", "text": "Within er_positive=1 patients, treatment_tamoxifen=1 yields longer pfs_months than treatment_tamoxifen=0.", "kind": "novel"},
    {"id": "h5_tam_in_erneg", "text": "Within er_positive=0 patients, treatment_tamoxifen has no benefit (or harm) on pfs_months.", "kind": "novel"},
]
m = ols_coef("pfs_months ~ treatment_tamoxifen * er_positive")
inter = "treatment_tamoxifen:er_positive"
analyses = [{
    "hids": ["h5_tam_er"],
    "code": "smf.ols('pfs_months ~ treatment_tamoxifen * er_positive').fit()",
    "summary": f"Interaction tamoxifen×er_positive = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Tamoxifen main = {m.params['treatment_tamoxifen']:.3f}, ER main = {m.params['er_positive']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
for sub_label, sub_q, hid in [("ER+", df.er_positive == 1, "h5_tam_in_erpos"), ("ER-", df.er_positive == 0, "h5_tam_in_erneg")]:
    sub = df[sub_q]
    eff, p = ttest(sub.loc[sub.treatment_tamoxifen == 1, "pfs_months"], sub.loc[sub.treatment_tamoxifen == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"# stratum {sub_label}",
        "summary": f"In {sub_label} (n={len(sub)}): tamoxifen=1 PFS={sub.loc[sub.treatment_tamoxifen==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_tamoxifen==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(5, hyps, analyses)

# ---------- Iteration 6: olaparib x BRCA interaction ----------
hyps = [
    {"id": "h6_olap_brca", "text": "Olaparib benefit on pfs_months is larger in BRCA1- or BRCA2-mutated patients than in BRCA wild-type (positive treatment_olaparib × brca_any interaction).", "kind": "novel"},
    {"id": "h6_olap_in_brca", "text": "Within BRCA1 or BRCA2 mutation carriers, treatment_olaparib=1 yields longer pfs_months than treatment_olaparib=0.", "kind": "novel"},
    {"id": "h6_olap_in_wt", "text": "Within BRCA wild-type patients, treatment_olaparib has minimal effect on pfs_months.", "kind": "novel"},
]
df["brca_any"] = ((df.brca1_mutation == 1) | (df.brca2_mutation == 1)).astype(int)
m = ols_coef("pfs_months ~ treatment_olaparib * brca_any")
inter = "treatment_olaparib:brca_any"
analyses = [{
    "hids": ["h6_olap_brca"],
    "code": "df['brca_any'] = (df.brca1_mutation|df.brca2_mutation).astype(int); smf.ols('pfs_months ~ treatment_olaparib * brca_any').fit()",
    "summary": f"Interaction olaparib×brca_any = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Olaparib main = {m.params['treatment_olaparib']:.3f}, BRCA main = {m.params['brca_any']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
for sub_label, sub_q, hid in [("BRCA+", df.brca_any == 1, "h6_olap_in_brca"), ("BRCA-WT", df.brca_any == 0, "h6_olap_in_wt")]:
    sub = df[sub_q]
    eff, p = ttest(sub.loc[sub.treatment_olaparib == 1, "pfs_months"], sub.loc[sub.treatment_olaparib == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"# stratum {sub_label}",
        "summary": f"In {sub_label} (n={len(sub)}): olaparib=1 PFS={sub.loc[sub.treatment_olaparib==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_olaparib==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(6, hyps, analyses)

# ---------- Iteration 7: pembrolizumab x msi_high interaction ----------
hyps = [
    {"id": "h7_pembro_msi", "text": "Pembrolizumab benefit on pfs_months is larger in MSI-high patients than in MSI-stable patients (positive treatment_pembrolizumab × msi_high interaction).", "kind": "novel"},
    {"id": "h7_pembro_in_msi", "text": "Within msi_high=1 patients, treatment_pembrolizumab=1 yields longer pfs_months than treatment_pembrolizumab=0.", "kind": "novel"},
    {"id": "h7_pembro_in_mss", "text": "Within msi_high=0 patients, treatment_pembrolizumab has minimal effect on pfs_months.", "kind": "novel"},
]
m = ols_coef("pfs_months ~ treatment_pembrolizumab * msi_high")
inter = "treatment_pembrolizumab:msi_high"
analyses = [{
    "hids": ["h7_pembro_msi"],
    "code": "smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high').fit()",
    "summary": f"Interaction pembro×msi_high = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}).",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
for sub_label, sub_q, hid in [("MSI-H", df.msi_high == 1, "h7_pembro_in_msi"), ("MSS", df.msi_high == 0, "h7_pembro_in_mss")]:
    sub = df[sub_q]
    if sub.treatment_pembrolizumab.sum() > 0 and (sub.treatment_pembrolizumab == 0).sum() > 0:
        eff, p = ttest(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"], sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"])
    else:
        eff, p = np.nan, np.nan
    analyses.append({
        "hids": [hid],
        "code": f"# stratum {sub_label}",
        "summary": f"In {sub_label} (n={len(sub)}): pembro=1 PFS={sub.loc[sub.treatment_pembrolizumab==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_pembrolizumab==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": float(p) if not np.isnan(p) else None, "eff": float(eff) if not np.isnan(eff) else None, "sig": (p < 0.05) if not np.isnan(p) else None,
    })
add(7, hyps, analyses)

# ---------- Iteration 8: palbociclib x ER interaction ----------
hyps = [
    {"id": "h8_palbo_er", "text": "Palbociclib benefit on pfs_months is larger in ER-positive patients than in ER-negative patients (positive treatment_palbociclib × er_positive interaction).", "kind": "novel"},
    {"id": "h8_palbo_in_erpos", "text": "Within er_positive=1, treatment_palbociclib=1 yields longer pfs_months than =0.", "kind": "novel"},
    {"id": "h8_palbo_in_erneg", "text": "Within er_positive=0, treatment_palbociclib has minimal benefit on pfs_months.", "kind": "novel"},
]
m = ols_coef("pfs_months ~ treatment_palbociclib * er_positive")
inter = "treatment_palbociclib:er_positive"
analyses = [{
    "hids": ["h8_palbo_er"],
    "code": "smf.ols('pfs_months ~ treatment_palbociclib * er_positive').fit()",
    "summary": f"Interaction palbo×er_positive = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Palbo main = {m.params['treatment_palbociclib']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
for sub_label, sub_q, hid in [("ER+", df.er_positive == 1, "h8_palbo_in_erpos"), ("ER-", df.er_positive == 0, "h8_palbo_in_erneg")]:
    sub = df[sub_q]
    eff, p = ttest(sub.loc[sub.treatment_palbociclib == 1, "pfs_months"], sub.loc[sub.treatment_palbociclib == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"# stratum {sub_label}",
        "summary": f"In {sub_label} (n={len(sub)}): palbo=1 PFS={sub.loc[sub.treatment_palbociclib==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_palbociclib==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(8, hyps, analyses)

# ---------- Iteration 9: sacituzumab govitecan effect; subgroup by HER2 / ER ----------
hyps = [
    {"id": "h9_sg_tnbc", "text": "Sacituzumab govitecan benefit is larger in triple-negative (ER-/PR-/HER2-) patients than in others (positive interaction with TNBC indicator).", "kind": "novel"},
    {"id": "h9_sg_her2low", "text": "Sacituzumab govitecan benefit on pfs_months differs by her2_low status (interaction treatment_sacituzumab_govitecan × her2_low).", "kind": "novel"},
]
df["tnbc"] = ((df.er_positive == 0) & (df.pr_positive == 0) & (df.her2_positive == 0)).astype(int)
m = ols_coef("pfs_months ~ treatment_sacituzumab_govitecan * tnbc")
inter = "treatment_sacituzumab_govitecan:tnbc"
analyses = [{
    "hids": ["h9_sg_tnbc"],
    "code": "df['tnbc']=((df.er_positive==0)&(df.pr_positive==0)&(df.her2_positive==0)).astype(int); smf.ols('pfs_months ~ treatment_sacituzumab_govitecan * tnbc').fit()",
    "summary": f"Interaction sacit×tnbc = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Sacit main = {m.params['treatment_sacituzumab_govitecan']:.3f}, TNBC main = {m.params['tnbc']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
m2 = ols_coef("pfs_months ~ treatment_sacituzumab_govitecan * her2_low")
inter2 = "treatment_sacituzumab_govitecan:her2_low"
analyses.append({
    "hids": ["h9_sg_her2low"],
    "code": "smf.ols('pfs_months ~ treatment_sacituzumab_govitecan * her2_low').fit()",
    "summary": f"Interaction sacit×her2_low = {m2.params[inter2]:.3f} (p={m2.pvalues[inter2]:.3g}).",
    "p": float(m2.pvalues[inter2]), "eff": float(m2.params[inter2]), "sig": float(m2.pvalues[inter2]) < 0.05,
})
add(9, hyps, analyses)

# ---------- Iteration 10: lab prognostic markers ----------
hyps = [
    {"id": "h10_alb", "text": "Higher albumin_g_dl is associated with longer pfs_months (positive slope in OLS).", "kind": "novel"},
    {"id": "h10_ldh", "text": "Higher ldh_u_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h10_nlr", "text": "Higher nlr is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h10_crp", "text": "Higher crp_mg_l is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h10_wt", "text": "Greater weight_loss_pct_6mo is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h10_hgb", "text": "Higher hemoglobin_g_dl is associated with longer pfs_months (positive slope).", "kind": "novel"},
]
analyses = []
for col, hid in [("albumin_g_dl", "h10_alb"), ("ldh_u_l", "h10_ldh"), ("nlr", "h10_nlr"),
                 ("crp_mg_l", "h10_crp"), ("weight_loss_pct_6mo", "h10_wt"), ("hemoglobin_g_dl", "h10_hgb")]:
    m = ols_coef(f"pfs_months ~ {col}")
    analyses.append({
        "hids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}').fit()",
        "summary": f"OLS slope of pfs_months on {col} = {m.params[col]:.5f} (p={m.pvalues[col]:.3g}).",
        "p": float(m.pvalues[col]), "eff": float(m.params[col]), "sig": float(m.pvalues[col]) < 0.05,
    })
add(10, hyps, analyses)

# ---------- Iteration 11: race/ethnicity disparity ----------
hyps = [
    {"id": "h11_race", "text": "Mean pfs_months differs across race_ethnicity categories (omnibus ANOVA).", "kind": "novel"},
    {"id": "h11_blackvw", "text": "Black patients (race_ethnicity=='black') have shorter mean pfs_months than white patients.", "kind": "novel"},
    {"id": "h11_hispanic", "text": "Hispanic patients have shorter mean pfs_months than white patients.", "kind": "novel"},
]
groups = [df.loc[df.race_ethnicity == r, "pfs_months"].values for r in df.race_ethnicity.unique()]
F, p_anova = stats.f_oneway(*groups)
analyses = [{
    "hids": ["h11_race"],
    "code": "stats.f_oneway over race_ethnicity groups",
    "summary": f"ANOVA across race_ethnicity F={F:.3f} p={p_anova:.3g}. Means: " + ", ".join(f"{r}={df.loc[df.race_ethnicity==r,'pfs_months'].mean():.2f}" for r in df.race_ethnicity.unique()),
    "p": float(p_anova), "eff": float(df.loc[df.race_ethnicity == "black", "pfs_months"].mean() - df.loc[df.race_ethnicity == "white", "pfs_months"].mean()),
    "sig": p_anova < 0.05,
}]
for r, hid in [("black", "h11_blackvw"), ("hispanic", "h11_hispanic")]:
    a = df.loc[df.race_ethnicity == r, "pfs_months"]
    b = df.loc[df.race_ethnicity == "white", "pfs_months"]
    eff, p = ttest(a, b)
    analyses.append({
        "hids": [hid],
        "code": f"# {r} vs white t-test",
        "summary": f"PFS {r}: {a.mean():.2f} (n={len(a)}) vs white: {b.mean():.2f} (n={len(b)}) (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(11, hyps, analyses)

# ---------- Iteration 12: insurance and rural residence ----------
hyps = [
    {"id": "h12_ins", "text": "Mean pfs_months differs across insurance_type categories (omnibus ANOVA).", "kind": "novel"},
    {"id": "h12_unins", "text": "Uninsured patients have shorter mean pfs_months than privately insured patients.", "kind": "novel"},
    {"id": "h12_medicaid", "text": "Medicaid patients have shorter mean pfs_months than privately insured patients.", "kind": "novel"},
    {"id": "h12_rural", "text": "Rural residence (rural_residence=1) is associated with shorter pfs_months than urban residence.", "kind": "novel"},
]
groups = [df.loc[df.insurance_type == i, "pfs_months"].values for i in df.insurance_type.unique()]
F, p_anova = stats.f_oneway(*groups)
analyses = [{
    "hids": ["h12_ins"],
    "code": "stats.f_oneway over insurance_type",
    "summary": f"ANOVA across insurance_type F={F:.3f} p={p_anova:.3g}. Means: " + ", ".join(f"{i}={df.loc[df.insurance_type==i,'pfs_months'].mean():.2f}" for i in df.insurance_type.unique()),
    "p": float(p_anova), "eff": float(df.loc[df.insurance_type == "uninsured", "pfs_months"].mean() - df.loc[df.insurance_type == "private", "pfs_months"].mean()),
    "sig": p_anova < 0.05,
}]
for ins, hid in [("uninsured", "h12_unins"), ("medicaid", "h12_medicaid")]:
    a = df.loc[df.insurance_type == ins, "pfs_months"]
    b = df.loc[df.insurance_type == "private", "pfs_months"]
    eff, p = ttest(a, b)
    analyses.append({
        "hids": [hid],
        "code": f"# {ins} vs private",
        "summary": f"PFS {ins}: {a.mean():.2f} (n={len(a)}) vs private: {b.mean():.2f} (n={len(b)}) (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
eff, p = ttest(df.loc[df.rural_residence == 1, "pfs_months"], df.loc[df.rural_residence == 0, "pfs_months"])
analyses.append({
    "hids": ["h12_rural"],
    "code": "rural vs urban t-test",
    "summary": f"PFS rural: {df.loc[df.rural_residence==1,'pfs_months'].mean():.2f} vs urban: {df.loc[df.rural_residence==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
    "p": p, "eff": eff, "sig": p < 0.05,
})
add(12, hyps, analyses)

# ---------- Iteration 13: visceral metastases ----------
hyps = [
    {"id": "h13_liver", "text": "Liver metastases (liver_mets=1) is associated with shorter pfs_months than no liver mets.", "kind": "novel"},
    {"id": "h13_bone", "text": "Bone metastases (bone_mets=1) is associated with shorter pfs_months than no bone mets.", "kind": "novel"},
    {"id": "h13_pleural", "text": "Pleural effusion (pleural_effusion=1) is associated with shorter pfs_months than no effusion.", "kind": "novel"},
    {"id": "h13_adrenal", "text": "Adrenal metastases (adrenal_mets=1) is associated with shorter pfs_months than no adrenal mets.", "kind": "novel"},
]
analyses = []
for col, hid in [("liver_mets", "h13_liver"), ("bone_mets", "h13_bone"), ("pleural_effusion", "h13_pleural"), ("adrenal_mets", "h13_adrenal")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"t-test {col}",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} (n={int(df[col].sum())}) vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(13, hyps, analyses)

# ---------- Iteration 14: tumor biology - Ki67, tumor size, tp53 ----------
hyps = [
    {"id": "h14_ki67", "text": "Higher ki67_pct (proliferation index) is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h14_tsize", "text": "Larger tumor_size_cm is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h14_tp53", "text": "tp53_mutation=1 is associated with shorter pfs_months than tp53 wild-type.", "kind": "novel"},
    {"id": "h14_pten", "text": "pten_loss=1 is associated with shorter pfs_months than no pten loss.", "kind": "novel"},
    {"id": "h14_pik3ca", "text": "pik3ca_mutation=1 is associated with different pfs_months than pik3ca wild-type.", "kind": "novel"},
]
analyses = []
for col, hid in [("ki67_pct", "h14_ki67"), ("tumor_size_cm", "h14_tsize")]:
    m = ols_coef(f"pfs_months ~ {col}")
    analyses.append({
        "hids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}').fit()",
        "summary": f"OLS slope on {col} = {m.params[col]:.5f} (p={m.pvalues[col]:.3g}).",
        "p": float(m.pvalues[col]), "eff": float(m.params[col]), "sig": float(m.pvalues[col]) < 0.05,
    })
for col, hid in [("tp53_mutation", "h14_tp53"), ("pten_loss", "h14_pten"), ("pik3ca_mutation", "h14_pik3ca")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"t-test {col}",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(14, hyps, analyses)

# ---------- Iteration 15: symptom grades ----------
hyps = [
    {"id": "h15_fatigue", "text": "Higher fatigue_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h15_pain", "text": "Higher pain_nrs is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h15_dyspnea", "text": "Higher dyspnea_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h15_appetite", "text": "Higher appetite_loss_grade is associated with shorter pfs_months (negative slope).", "kind": "novel"},
]
analyses = []
for col, hid in [("fatigue_grade", "h15_fatigue"), ("pain_nrs", "h15_pain"), ("dyspnea_grade", "h15_dyspnea"), ("appetite_loss_grade", "h15_appetite")]:
    m = ols_coef(f"pfs_months ~ {col}")
    analyses.append({
        "hids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}').fit()",
        "summary": f"OLS slope on {col} = {m.params[col]:.5f} (p={m.pvalues[col]:.3g}).",
        "p": float(m.pvalues[col]), "eff": float(m.params[col]), "sig": float(m.pvalues[col]) < 0.05,
    })
add(15, hyps, analyses)

# ---------- Iteration 16: prior therapy and lines of therapy ----------
hyps = [
    {"id": "h16_lines", "text": "Higher prior_lines_of_therapy is associated with shorter pfs_months (negative slope).", "kind": "novel"},
    {"id": "h16_priorchemo", "text": "Prior chemotherapy (prior_chemotherapy=1) is associated with shorter pfs_months than no prior chemo.", "kind": "novel"},
    {"id": "h16_priorrad", "text": "Prior radiation (prior_radiation=1) is associated with different pfs_months than no prior radiation.", "kind": "novel"},
    {"id": "h16_priorimm", "text": "Prior immunotherapy is associated with different pfs_months.", "kind": "novel"},
]
analyses = []
m = ols_coef("pfs_months ~ prior_lines_of_therapy")
analyses.append({
    "hids": ["h16_lines"],
    "code": "smf.ols('pfs_months ~ prior_lines_of_therapy').fit()",
    "summary": f"OLS slope on prior_lines_of_therapy = {m.params['prior_lines_of_therapy']:.4f} (p={m.pvalues['prior_lines_of_therapy']:.3g}).",
    "p": float(m.pvalues['prior_lines_of_therapy']), "eff": float(m.params['prior_lines_of_therapy']), "sig": float(m.pvalues['prior_lines_of_therapy']) < 0.05,
})
for col, hid in [("prior_chemotherapy", "h16_priorchemo"), ("prior_radiation", "h16_priorrad"), ("prior_immunotherapy", "h16_priorimm")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"t-test {col}",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
add(16, hyps, analyses)

# ---------- Iteration 17: pembrolizumab x tnbc / her2_low / pdl1-proxy ----------
hyps = [
    {"id": "h17_pembro_tnbc", "text": "Pembrolizumab benefit on pfs_months is larger in TNBC patients than non-TNBC (positive treatment_pembrolizumab × tnbc interaction).", "kind": "novel"},
    {"id": "h17_pembro_in_tnbc", "text": "Within TNBC, treatment_pembrolizumab=1 yields longer pfs_months than =0.", "kind": "novel"},
]
m = ols_coef("pfs_months ~ treatment_pembrolizumab * tnbc")
inter = "treatment_pembrolizumab:tnbc"
analyses = [{
    "hids": ["h17_pembro_tnbc"],
    "code": "smf.ols('pfs_months ~ treatment_pembrolizumab * tnbc').fit()",
    "summary": f"Interaction pembro×tnbc = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Pembro main = {m.params['treatment_pembrolizumab']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
}]
sub = df[df.tnbc == 1]
eff, p = ttest(sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"], sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"])
analyses.append({
    "hids": ["h17_pembro_in_tnbc"],
    "code": "df[df.tnbc==1].groupby('treatment_pembrolizumab').pfs_months.mean()",
    "summary": f"In TNBC (n={len(sub)}): pembro=1 PFS={sub.loc[sub.treatment_pembrolizumab==1,'pfs_months'].mean():.2f} vs =0 PFS={sub.loc[sub.treatment_pembrolizumab==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
    "p": p, "eff": eff, "sig": p < 0.05,
})
add(17, hyps, analyses)

# ---------- Iteration 18: comorbidity burden / organ function ----------
hyps = [
    {"id": "h18_ckd", "text": "Chronic kidney disease (chronic_kidney_disease=1) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h18_hf", "text": "Heart failure (heart_failure=1) is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h18_dm", "text": "Diabetes mellitus is associated with different pfs_months.", "kind": "novel"},
    {"id": "h18_creat", "text": "Higher creatinine_mg_dl is associated with shorter pfs_months.", "kind": "novel"},
    {"id": "h18_bili", "text": "Higher total_bilirubin_mg_dl is associated with shorter pfs_months.", "kind": "novel"},
]
analyses = []
for col, hid in [("chronic_kidney_disease", "h18_ckd"), ("heart_failure", "h18_hf"), ("diabetes_mellitus", "h18_dm")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"t-test {col}",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
for col, hid in [("creatinine_mg_dl", "h18_creat"), ("total_bilirubin_mg_dl", "h18_bili")]:
    m = ols_coef(f"pfs_months ~ {col}")
    analyses.append({
        "hids": [hid],
        "code": f"smf.ols('pfs_months ~ {col}').fit()",
        "summary": f"OLS slope on {col} = {m.params[col]:.4f} (p={m.pvalues[col]:.3g}).",
        "p": float(m.pvalues[col]), "eff": float(m.params[col]), "sig": float(m.pvalues[col]) < 0.05,
    })
add(18, hyps, analyses)

# ---------- Iteration 19: multivariable regression (key prognostic + treatments) ----------
hyps = [
    {"id": "h19_mv", "text": "After adjusting for age, ECOG, stage, ER, HER2, ki67, albumin, LDH, NLR, the main effect of treatment_palbociclib on pfs_months remains positive and significant.", "kind": "novel"},
    {"id": "h19_mv_ecog", "text": "After multivariable adjustment, ecog_ps retains a negative coefficient on pfs_months.", "kind": "novel"},
    {"id": "h19_mv_alb", "text": "After multivariable adjustment, albumin_g_dl retains a positive coefficient on pfs_months.", "kind": "novel"},
]
covars = ("age_years + ecog_ps + stage_iv + has_brain_mets + er_positive + pr_positive + her2_positive + her2_low + "
          "ki67_pct + tumor_size_cm + albumin_g_dl + ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo + hemoglobin_g_dl + "
          "tp53_mutation + pten_loss + brca_any + msi_high + tnbc + liver_mets + bone_mets + pleural_effusion + "
          "treatment_tamoxifen + treatment_palbociclib + treatment_trastuzumab + treatment_olaparib + "
          "treatment_sacituzumab_govitecan + treatment_pembrolizumab + prior_lines_of_therapy + fatigue_grade + pain_nrs")
m = ols_coef(f"pfs_months ~ {covars}")
analyses = [{
    "hids": ["h19_mv"],
    "code": f"smf.ols('pfs_months ~ {covars}').fit()  # multivariable",
    "summary": f"Multivariable OLS adj coef treatment_palbociclib = {m.params['treatment_palbociclib']:.3f} (p={m.pvalues['treatment_palbociclib']:.3g}); R² = {m.rsquared:.3f}.",
    "p": float(m.pvalues["treatment_palbociclib"]), "eff": float(m.params["treatment_palbociclib"]), "sig": float(m.pvalues["treatment_palbociclib"]) < 0.05,
}, {
    "hids": ["h19_mv_ecog"],
    "code": "same MV model",
    "summary": f"MV adj coef ecog_ps = {m.params['ecog_ps']:.3f} (p={m.pvalues['ecog_ps']:.3g}).",
    "p": float(m.pvalues["ecog_ps"]), "eff": float(m.params["ecog_ps"]), "sig": float(m.pvalues["ecog_ps"]) < 0.05,
}, {
    "hids": ["h19_mv_alb"],
    "code": "same MV model",
    "summary": f"MV adj coef albumin_g_dl = {m.params['albumin_g_dl']:.3f} (p={m.pvalues['albumin_g_dl']:.3g}).",
    "p": float(m.pvalues["albumin_g_dl"]), "eff": float(m.params["albumin_g_dl"]), "sig": float(m.pvalues["albumin_g_dl"]) < 0.05,
}]
add(19, hyps, analyses)
mv_model = m  # save for later

# ---------- Iteration 20: refined - adjusted treatment effects (other agents) ----------
hyps = [
    {"id": "h20_tras_adj", "text": "After multivariable adjustment, treatment_trastuzumab coefficient on pfs_months is non-significant (HER2 status absorbs the effect).", "kind": "refined"},
    {"id": "h20_olap_adj", "text": "After multivariable adjustment, treatment_olaparib coefficient on pfs_months is non-significant overall (effect concentrated in BRCA+).", "kind": "refined"},
    {"id": "h20_pembro_adj", "text": "After multivariable adjustment, treatment_pembrolizumab coefficient on pfs_months is non-significant overall.", "kind": "refined"},
    {"id": "h20_sg_adj", "text": "After multivariable adjustment, treatment_sacituzumab_govitecan coefficient on pfs_months is non-significant overall.", "kind": "refined"},
    {"id": "h20_tam_adj", "text": "After multivariable adjustment, treatment_tamoxifen coefficient on pfs_months is non-significant overall.", "kind": "refined"},
]
analyses = []
for tcol, hid in [("treatment_trastuzumab", "h20_tras_adj"), ("treatment_olaparib", "h20_olap_adj"),
                   ("treatment_pembrolizumab", "h20_pembro_adj"), ("treatment_sacituzumab_govitecan", "h20_sg_adj"),
                   ("treatment_tamoxifen", "h20_tam_adj")]:
    analyses.append({
        "hids": [hid],
        "code": "MV model coef",
        "summary": f"MV adj coef {tcol} = {mv_model.params[tcol]:.3f} (p={mv_model.pvalues[tcol]:.3g}).",
        "p": float(mv_model.pvalues[tcol]), "eff": float(mv_model.params[tcol]), "sig": float(mv_model.pvalues[tcol]) < 0.05,
    })
add(20, hyps, analyses)

# ---------- Iteration 21: SNP scan vs PFS (omnibus + a few) ----------
snps = [c for c in df.columns if c.startswith("snp_")]
hyps = [
    {"id": "h21_snp_any", "text": "At least one SNP carrier-status indicator (snp_*) is associated with pfs_months at p<0.05 in unadjusted t-tests after multiple-testing context.", "kind": "novel"},
    {"id": "h21_snp_min", "text": "The minimum unadjusted t-test p-value across all snp_* carrier indicators (24 tests) survives a Bonferroni correction at α=0.05.", "kind": "novel"},
]
ps = []
for s in snps:
    a = df.loc[df[s] == 1, "pfs_months"]
    b = df.loc[df[s] == 0, "pfs_months"]
    if len(a) > 30 and len(b) > 30:
        _, p = stats.ttest_ind(a, b, equal_var=False)
        ps.append((s, p, a.mean() - b.mean()))
ps_sorted = sorted(ps, key=lambda x: x[1])
n_sig_raw = sum(1 for _, p, _ in ps if p < 0.05)
min_s, min_p, min_eff = ps_sorted[0]
bonf_alpha = 0.05 / len(ps)
analyses = [{
    "hids": ["h21_snp_any"],
    "code": "for snp in snps: t-test pfs_months by carrier",
    "summary": f"Tested {len(ps)} SNPs unadjusted; {n_sig_raw} reached p<0.05 (~{0.05*len(ps):.1f} expected by chance).",
    "p": None, "eff": float(n_sig_raw - 0.05 * len(ps)), "sig": n_sig_raw > 2 * 0.05 * len(ps),
}, {
    "hids": ["h21_snp_min"],
    "code": "min p-value across SNP t-tests",
    "summary": f"Smallest SNP p-value = {min_p:.3g} ({min_s}, mean diff={min_eff:.3f} mo); Bonferroni α={bonf_alpha:.4f}.",
    "p": float(min_p), "eff": float(min_eff), "sig": min_p < bonf_alpha,
}]
add(21, hyps, analyses)

# ---------- Iteration 22: postmenopausal + tamoxifen interaction; node_positive main; postmenopausal main ----------
hyps = [
    {"id": "h22_postmeno", "text": "Postmenopausal patients have different mean pfs_months than premenopausal patients.", "kind": "novel"},
    {"id": "h22_node", "text": "Node-positive patients have shorter pfs_months than node-negative patients.", "kind": "novel"},
    {"id": "h22_tam_postmeno", "text": "Tamoxifen treatment effect on pfs_months differs by postmenopausal status (interaction).", "kind": "novel"},
]
analyses = []
for col, hid in [("postmenopausal", "h22_postmeno"), ("node_positive", "h22_node")]:
    eff, p = ttest(df.loc[df[col] == 1, "pfs_months"], df.loc[df[col] == 0, "pfs_months"])
    analyses.append({
        "hids": [hid],
        "code": f"t-test {col}",
        "summary": f"PFS {col}=1: {df.loc[df[col]==1,'pfs_months'].mean():.2f} vs =0: {df.loc[df[col]==0,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": p, "eff": eff, "sig": p < 0.05,
    })
m = ols_coef("pfs_months ~ treatment_tamoxifen * postmenopausal")
inter = "treatment_tamoxifen:postmenopausal"
analyses.append({
    "hids": ["h22_tam_postmeno"],
    "code": "smf.ols('pfs_months ~ treatment_tamoxifen * postmenopausal').fit()",
    "summary": f"Interaction tamox×postmeno = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}).",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
})
add(22, hyps, analyses)

# ---------- Iteration 23: combination treatment effects (palbo + tamoxifen, palbo + ER) ----------
hyps = [
    {"id": "h23_combo", "text": "Patients receiving both treatment_palbociclib and treatment_tamoxifen have longer pfs_months than those receiving palbociclib alone (additive/synergistic combination).", "kind": "novel"},
    {"id": "h23_combo_inter", "text": "There is a positive interaction between treatment_palbociclib and treatment_tamoxifen on pfs_months.", "kind": "novel"},
    {"id": "h23_palbo_pik3ca", "text": "Palbociclib benefit is larger in pik3ca_mutation=1 patients than wild-type (positive interaction).", "kind": "novel"},
]
both = (df.treatment_palbociclib == 1) & (df.treatment_tamoxifen == 1)
palbo_only = (df.treatment_palbociclib == 1) & (df.treatment_tamoxifen == 0)
eff, p = ttest(df.loc[both, "pfs_months"], df.loc[palbo_only, "pfs_months"])
analyses = [{
    "hids": ["h23_combo"],
    "code": "compare both vs palbo-only",
    "summary": f"PFS palbo+tam (n={int(both.sum())})={df.loc[both,'pfs_months'].mean():.2f} vs palbo-only (n={int(palbo_only.sum())})={df.loc[palbo_only,'pfs_months'].mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
    "p": p, "eff": eff, "sig": p < 0.05,
}]
m = ols_coef("pfs_months ~ treatment_palbociclib * treatment_tamoxifen")
inter = "treatment_palbociclib:treatment_tamoxifen"
analyses.append({
    "hids": ["h23_combo_inter"],
    "code": "smf.ols('pfs_months ~ treatment_palbociclib * treatment_tamoxifen').fit()",
    "summary": f"Interaction palbo×tam = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}). Palbo main = {m.params['treatment_palbociclib']:.3f}, Tam main = {m.params['treatment_tamoxifen']:.3f}.",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
})
m = ols_coef("pfs_months ~ treatment_palbociclib * pik3ca_mutation")
inter = "treatment_palbociclib:pik3ca_mutation"
analyses.append({
    "hids": ["h23_palbo_pik3ca"],
    "code": "smf.ols('pfs_months ~ treatment_palbociclib * pik3ca_mutation').fit()",
    "summary": f"Interaction palbo×pik3ca = {m.params[inter]:.3f} (p={m.pvalues[inter]:.3g}).",
    "p": float(m.pvalues[inter]), "eff": float(m.params[inter]), "sig": float(m.pvalues[inter]) < 0.05,
})
add(23, hyps, analyses)

# ---------- Iteration 24: refined - confirmed biomarker-treatment matching summary ----------
# Use a single MV model that includes the four matched interactions
hyps = [
    {"id": "h24_joint_match", "text": "In a model with all four matched biomarker×drug interactions (trastuzumab×her2_positive, tamoxifen×er_positive, olaparib×brca_any, pembrolizumab×msi_high) plus key covariates, each interaction coefficient on pfs_months remains positive.", "kind": "refined"},
    {"id": "h24_pembro_msi_adj", "text": "After adjustment, treatment_pembrolizumab × msi_high interaction coefficient on pfs_months is positive and significant.", "kind": "refined"},
    {"id": "h24_palbo_er_adj", "text": "After adjustment, treatment_palbociclib × er_positive interaction coefficient on pfs_months is positive and significant.", "kind": "refined"},
]
form = ("pfs_months ~ age_years + ecog_ps + stage_iv + ki67_pct + albumin_g_dl + ldh_u_l + nlr + "
        "treatment_trastuzumab*her2_positive + treatment_tamoxifen*er_positive + "
        "treatment_olaparib*brca_any + treatment_pembrolizumab*msi_high + "
        "treatment_palbociclib*er_positive + treatment_sacituzumab_govitecan*tnbc")
m = ols_coef(form)
key_inters = {
    "trastuzumab×her2_positive": "treatment_trastuzumab:her2_positive",
    "tamoxifen×er_positive": "treatment_tamoxifen:er_positive",
    "olaparib×brca_any": "treatment_olaparib:brca_any",
    "pembrolizumab×msi_high": "treatment_pembrolizumab:msi_high",
    "palbociclib×er_positive": "treatment_palbociclib:er_positive",
    "sacit×tnbc": "treatment_sacituzumab_govitecan:tnbc",
}
joint_summary = "Adjusted interaction coefs: " + "; ".join(f"{k}={m.params[v]:.3f} (p={m.pvalues[v]:.3g})" for k, v in key_inters.items())
analyses = [{
    "hids": ["h24_joint_match"],
    "code": f"smf.ols('{form}').fit()",
    "summary": joint_summary,
    "p": float(min(m.pvalues[v] for v in key_inters.values())),
    "eff": float(np.mean([m.params[v] for v in key_inters.values()])),
    "sig": all(m.pvalues[v] < 0.05 and m.params[v] > 0 for v in key_inters.values()),
}, {
    "hids": ["h24_pembro_msi_adj"],
    "code": "same joint MV model",
    "summary": f"Adjusted pembro×msi_high coef = {m.params['treatment_pembrolizumab:msi_high']:.3f} (p={m.pvalues['treatment_pembrolizumab:msi_high']:.3g}).",
    "p": float(m.pvalues["treatment_pembrolizumab:msi_high"]), "eff": float(m.params["treatment_pembrolizumab:msi_high"]),
    "sig": float(m.pvalues["treatment_pembrolizumab:msi_high"]) < 0.05 and m.params["treatment_pembrolizumab:msi_high"] > 0,
}, {
    "hids": ["h24_palbo_er_adj"],
    "code": "same joint MV model",
    "summary": f"Adjusted palbo×er_positive coef = {m.params['treatment_palbociclib:er_positive']:.3f} (p={m.pvalues['treatment_palbociclib:er_positive']:.3g}).",
    "p": float(m.pvalues["treatment_palbociclib:er_positive"]), "eff": float(m.params["treatment_palbociclib:er_positive"]),
    "sig": float(m.pvalues["treatment_palbociclib:er_positive"]) < 0.05 and m.params["treatment_palbociclib:er_positive"] > 0,
}]
add(24, hyps, analyses)

# ---------- Iteration 25: refined synthesis - cohort means by biomarker-matched vs unmatched treatment ----------
hyps = [
    {"id": "h25_match_palbo_erpos", "text": "Among ER-positive patients, mean pfs_months is higher for those receiving palbociclib than for those not receiving palbociclib (signal concentrated in matched subgroup).", "kind": "refined"},
    {"id": "h25_match_olap_brca", "text": "Among BRCA1/2 carriers, mean pfs_months is higher for those receiving olaparib than for those not receiving olaparib.", "kind": "refined"},
    {"id": "h25_match_tras_her2", "text": "Among HER2+ patients, mean pfs_months is higher for those receiving trastuzumab than for those not receiving trastuzumab.", "kind": "refined"},
    {"id": "h25_match_pembro_msi", "text": "Among MSI-high patients, mean pfs_months is higher for those receiving pembrolizumab than for those not receiving pembrolizumab.", "kind": "refined"},
    {"id": "h25_match_tam_er", "text": "Among ER-positive patients, mean pfs_months is higher for those receiving tamoxifen than for those not receiving tamoxifen.", "kind": "refined"},
]
analyses = []
matches = [("er_positive", "treatment_palbociclib", "h25_match_palbo_erpos"),
           ("brca_any", "treatment_olaparib", "h25_match_olap_brca"),
           ("her2_positive", "treatment_trastuzumab", "h25_match_tras_her2"),
           ("msi_high", "treatment_pembrolizumab", "h25_match_pembro_msi"),
           ("er_positive", "treatment_tamoxifen", "h25_match_tam_er")]
for marker, drug, hid in matches:
    sub = df[df[marker] == 1]
    a = sub.loc[sub[drug] == 1, "pfs_months"]
    b = sub.loc[sub[drug] == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        eff, p = ttest(a, b)
    else:
        eff, p = np.nan, np.nan
    analyses.append({
        "hids": [hid],
        "code": f"# subgroup {marker}=1, t-test by {drug}",
        "summary": f"Within {marker}=1 (n={len(sub)}): {drug}=1 (n={len(a)}) PFS={a.mean():.2f} vs =0 (n={len(b)}) PFS={b.mean():.2f} (diff={eff:.3f}, p={p:.3g}).",
        "p": float(p) if not np.isnan(p) else None, "eff": float(eff) if not np.isnan(eff) else None,
        "sig": (p < 0.05) if not np.isnan(p) else None,
    })
add(25, hyps, analyses)

# Save
with open("results_out.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("Wrote results_out.json with", len(out), "iterations")
for o in out:
    print(f"Iter {o['index']}: {len(o['hypotheses'])} hyps, {len(o['analyses'])} analyses")
