"""
Iterative hypothesis-test analysis for ds001_breast.
Outcome: pfs_months (continuous).
Up to 25 iterations of (propose hypothesis -> test -> store).

Stores results as a flat list of dicts (one per analysis), then post-processes
into transcript.json grouped by iteration.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_parquet("dataset.parquet")
N = len(df)

records = []  # flat list, will be grouped into iterations later


def add(iteration, hid, htext, kind, code, summary, p, effect, sig=None):
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    records.append({
        "iteration": iteration,
        "hid": hid,
        "htext": htext,
        "kind": kind,
        "code": code,
        "summary": summary,
        "p": None if p is None or (isinstance(p, float) and np.isnan(p)) else float(p),
        "effect": None if effect is None or (isinstance(effect, float) and np.isnan(effect)) else float(effect),
        "sig": sig,
    })


def t_test_pfs_by_binary(col):
    """Welch t-test of pfs_months grouped by binary col. Returns (effect, p). Effect = mean(pos)-mean(neg)."""
    g1 = df.loc[df[col] == 1, "pfs_months"]
    g0 = df.loc[df[col] == 0, "pfs_months"]
    res = stats.ttest_ind(g1, g0, equal_var=False)
    return float(g1.mean() - g0.mean()), float(res.pvalue), float(g1.mean()), float(g0.mean()), len(g1), len(g0)


def ols_summary(formula):
    m = ols(formula, data=df).fit()
    return m


# ============================================================================
# ITERATION 1: Main effects of each systemic therapy on PFS
# ============================================================================
it = 1
hyps_meta = [
    ("h1_1", "Patients receiving treatment_trastuzumab have higher mean pfs_months than those not receiving it."),
    ("h1_2", "Patients receiving treatment_tamoxifen have higher mean pfs_months than those not receiving it."),
    ("h1_3", "Patients receiving treatment_palbociclib have higher mean pfs_months than those not receiving it."),
    ("h1_4", "Patients receiving treatment_olaparib have higher mean pfs_months than those not receiving it."),
    ("h1_5", "Patients receiving treatment_sacituzumab_govitecan have higher mean pfs_months than those not receiving it."),
    ("h1_6", "Patients receiving treatment_pembrolizumab have higher mean pfs_months than those not receiving it."),
]
for hid, htext in hyps_meta:
    col = "treatment_" + htext.split("treatment_")[1].split(" ")[0]
    eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary(col)
    add(it, hid, htext, "novel",
        f"stats.ttest_ind(df.loc[df['{col}']==1,'pfs_months'], df.loc[df['{col}']==0,'pfs_months'], equal_var=False)",
        f"Mean PFS = {m1:.2f} mo on {col} (n={n1}) vs {m0:.2f} off (n={n0}); diff = {eff:+.2f}, Welch t-test p = {p:.3g}.",
        p, eff)

# ============================================================================
# ITERATION 2: Biomarker main effects on PFS
# ============================================================================
it = 2
biom = [
    ("h2_1", "er_positive", "ER-positive patients have different mean pfs_months than ER-negative patients."),
    ("h2_2", "pr_positive", "PR-positive patients have different mean pfs_months than PR-negative patients."),
    ("h2_3", "her2_positive", "HER2-positive patients have different mean pfs_months than HER2-negative patients."),
    ("h2_4", "her2_low", "HER2-low patients have different mean pfs_months than non-HER2-low patients."),
    ("h2_5", "brca1_mutation", "BRCA1-mutated patients have different mean pfs_months than BRCA1-wildtype patients."),
    ("h2_6", "brca2_mutation", "BRCA2-mutated patients have different mean pfs_months than BRCA2-wildtype patients."),
    ("h2_7", "pik3ca_mutation", "PIK3CA-mutated patients have different mean pfs_months than PIK3CA-wildtype patients."),
    ("h2_8", "tp53_mutation", "TP53-mutated patients have different mean pfs_months than TP53-wildtype patients."),
    ("h2_9", "msi_high", "MSI-high patients have different mean pfs_months than MSI-stable patients."),
]
for hid, col, htext in biom:
    eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary(col)
    add(it, hid, htext, "novel",
        f"stats.ttest_ind by {col}",
        f"Mean PFS = {m1:.2f} mo {col}=1 (n={n1}) vs {m0:.2f} =0 (n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
        p, eff)

# ============================================================================
# ITERATION 3: Disease burden / staging features
# ============================================================================
it = 3
burden = [
    ("h3_1", "stage_iv", "Stage-IV patients have lower mean pfs_months than non-stage-IV patients."),
    ("h3_2", "has_brain_mets", "Patients with brain metastases have lower mean pfs_months than those without."),
    ("h3_3", "liver_mets", "Patients with liver metastases have lower mean pfs_months than those without."),
    ("h3_4", "bone_mets", "Patients with bone metastases have lower mean pfs_months than those without."),
    ("h3_5", "node_positive", "Node-positive patients have lower mean pfs_months than node-negative patients."),
    ("h3_6", "pleural_effusion", "Patients with pleural effusion have lower mean pfs_months than those without."),
]
for hid, col, htext in burden:
    eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary(col)
    add(it, hid, htext, "novel",
        f"stats.ttest_ind by {col}",
        f"Mean PFS = {m1:.2f} ({col}=1, n={n1}) vs {m0:.2f} (=0, n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
        p, eff)
# ECOG (continuous-ish)
m = ols_summary("pfs_months ~ ecog_ps")
b = m.params["ecog_ps"]; p = m.pvalues["ecog_ps"]
add(it, "h3_7", "Higher ECOG performance status (ecog_ps) is associated with lower mean pfs_months.", "novel",
    "ols('pfs_months ~ ecog_ps', df)",
    f"OLS slope on ecog_ps = {b:+.3f} mo per unit (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 4: Treatment x biomarker interactions (key clinical pairings)
# ============================================================================
it = 4
# Trastuzumab x HER2-positive
m = ols_summary("pfs_months ~ treatment_trastuzumab * her2_positive")
b = m.params["treatment_trastuzumab:her2_positive"]; p = m.pvalues["treatment_trastuzumab:her2_positive"]
add(it, "h4_1",
    "The PFS benefit of treatment_trastuzumab is larger in HER2-positive patients than in HER2-negative patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_trastuzumab * her2_positive')",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}); positive => larger trastuzumab benefit when HER2+.",
    float(p), float(b))

# Tamoxifen x ER-positive
m = ols_summary("pfs_months ~ treatment_tamoxifen * er_positive")
b = m.params["treatment_tamoxifen:er_positive"]; p = m.pvalues["treatment_tamoxifen:er_positive"]
add(it, "h4_2",
    "The PFS benefit of treatment_tamoxifen is larger in ER-positive patients than in ER-negative patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_tamoxifen * er_positive')",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Palbociclib x ER-positive
m = ols_summary("pfs_months ~ treatment_palbociclib * er_positive")
b = m.params["treatment_palbociclib:er_positive"]; p = m.pvalues["treatment_palbociclib:er_positive"]
add(it, "h4_3",
    "The PFS benefit of treatment_palbociclib is larger in ER-positive patients than in ER-negative patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_palbociclib * er_positive')",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Olaparib x BRCA1 OR BRCA2
df["brca_any"] = ((df["brca1_mutation"] == 1) | (df["brca2_mutation"] == 1)).astype(int)
m = ols_summary("pfs_months ~ treatment_olaparib * brca_any")
b = m.params["treatment_olaparib:brca_any"]; p = m.pvalues["treatment_olaparib:brca_any"]
add(it, "h4_4",
    "The PFS benefit of treatment_olaparib is larger in BRCA1- or BRCA2-mutated patients than in BRCA-wildtype patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_olaparib * brca_any') where brca_any = brca1_mutation OR brca2_mutation",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Sacituzumab govitecan x her2_low
m = ols_summary("pfs_months ~ treatment_sacituzumab_govitecan * her2_low")
b = m.params["treatment_sacituzumab_govitecan:her2_low"]; p = m.pvalues["treatment_sacituzumab_govitecan:her2_low"]
add(it, "h4_5",
    "The PFS benefit of treatment_sacituzumab_govitecan is larger in HER2-low patients than in non-HER2-low patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_sacituzumab_govitecan * her2_low')",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Pembrolizumab x MSI-high
m = ols_summary("pfs_months ~ treatment_pembrolizumab * msi_high")
b = m.params["treatment_pembrolizumab:msi_high"]; p = m.pvalues["treatment_pembrolizumab:msi_high"]
add(it, "h4_6",
    "The PFS benefit of treatment_pembrolizumab is larger in MSI-high patients than in microsatellite-stable patients (positive interaction effect).",
    "novel",
    "ols('pfs_months ~ treatment_pembrolizumab * msi_high')",
    f"Interaction coefficient = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 5: Demographic effects
# ============================================================================
it = 5
# Age (continuous)
m = ols_summary("pfs_months ~ age_years")
b = m.params["age_years"]; p = m.pvalues["age_years"]
add(it, "h5_1", "Older age (age_years) is associated with shorter mean pfs_months.", "novel",
    "ols('pfs_months ~ age_years')",
    f"Slope = {b:+.4f} mo per year of age (p={p:.3g}).", float(p), float(b))

# Postmenopausal
eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary("postmenopausal")
add(it, "h5_2",
    "Postmenopausal patients have different mean pfs_months than premenopausal patients.", "novel",
    "stats.ttest_ind by postmenopausal",
    f"Mean PFS = {m1:.2f} (postmeno, n={n1}) vs {m0:.2f} (premeno, n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
    p, eff)

# Sex_female (~all should be female in breast cohort, sanity)
female_rate = df["sex_female"].mean()
add(it, "h5_3",
    "The proportion of female patients (sex_female) differs from 100% in this breast cancer cohort (sanity check).",
    "novel",
    "df['sex_female'].mean()",
    f"sex_female mean = {female_rate:.4f} ({df['sex_female'].sum()} of {N}).",
    None, float(female_rate - 1.0), False if abs(female_rate - 1.0) < 0.01 else True)

# Race/ethnicity ANOVA
groups = [df.loc[df["race_ethnicity"] == g, "pfs_months"].values for g in df["race_ethnicity"].unique()]
f, p = stats.f_oneway(*groups)
mean_by_race = df.groupby("race_ethnicity")["pfs_months"].mean().to_dict()
diff = max(mean_by_race.values()) - min(mean_by_race.values())
add(it, "h5_4", "Mean pfs_months differs across race_ethnicity groups (white, black, hispanic, asian, other).", "novel",
    "stats.f_oneway across race_ethnicity groups",
    f"ANOVA F p={p:.3g}; group means = {mean_by_race}.", float(p), float(diff))

# Insurance
groups = [df.loc[df["insurance_type"] == g, "pfs_months"].values for g in df["insurance_type"].unique()]
f, p = stats.f_oneway(*groups)
mean_by_ins = df.groupby("insurance_type")["pfs_months"].mean().to_dict()
diff = max(mean_by_ins.values()) - min(mean_by_ins.values())
add(it, "h5_5", "Mean pfs_months differs across insurance_type groups.", "novel",
    "stats.f_oneway across insurance_type",
    f"ANOVA F p={p:.3g}; group means = {mean_by_ins}.", float(p), float(diff))

# ============================================================================
# ITERATION 6: Lab-value / inflammation main effects on PFS
# ============================================================================
it = 6
labs = [
    ("h6_1", "albumin_g_dl", "Higher serum albumin (albumin_g_dl) is associated with longer mean pfs_months."),
    ("h6_2", "ldh_u_l", "Higher LDH (ldh_u_l) is associated with shorter mean pfs_months."),
    ("h6_3", "crp_mg_l", "Higher CRP (crp_mg_l) is associated with shorter mean pfs_months."),
    ("h6_4", "nlr", "Higher neutrophil-to-lymphocyte ratio (nlr) is associated with shorter mean pfs_months."),
    ("h6_5", "hemoglobin_g_dl", "Higher hemoglobin (hemoglobin_g_dl) is associated with longer mean pfs_months."),
    ("h6_6", "alkaline_phosphatase_u_l", "Higher alkaline phosphatase is associated with shorter mean pfs_months."),
]
for hid, col, htext in labs:
    m = ols_summary(f"pfs_months ~ {col}")
    b = m.params[col]; p = m.pvalues[col]
    add(it, hid, htext, "novel", f"ols('pfs_months ~ {col}')",
        f"Slope = {b:+.5f} mo per unit {col} (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 7: Tumor / proliferation / nutrition continuous features
# ============================================================================
it = 7
cont = [
    ("h7_1", "ki67_pct", "Higher Ki-67 percentage (ki67_pct) is associated with shorter mean pfs_months."),
    ("h7_2", "tumor_size_cm", "Larger tumor size (tumor_size_cm) is associated with shorter mean pfs_months."),
    ("h7_3", "weight_loss_pct_6mo", "Greater 6-month weight loss percentage is associated with shorter mean pfs_months."),
    ("h7_4", "prior_lines_of_therapy", "More prior lines of therapy are associated with shorter mean pfs_months."),
    ("h7_5", "bmi", "Higher BMI is associated with longer mean pfs_months."),
]
for hid, col, htext in cont:
    m = ols_summary(f"pfs_months ~ {col}")
    b = m.params[col]; p = m.pvalues[col]
    add(it, hid, htext, "novel", f"ols('pfs_months ~ {col}')",
        f"Slope = {b:+.5f} mo per unit {col} (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 8: Symptom grades vs PFS
# ============================================================================
it = 8
sx = [
    ("h8_1", "fatigue_grade", "Higher fatigue_grade is associated with shorter mean pfs_months."),
    ("h8_2", "pain_nrs", "Higher pain numeric rating (pain_nrs) is associated with shorter mean pfs_months."),
    ("h8_3", "dyspnea_grade", "Higher dyspnea_grade is associated with shorter mean pfs_months."),
    ("h8_4", "appetite_loss_grade", "Higher appetite_loss_grade is associated with shorter mean pfs_months."),
    ("h8_5", "cough_grade", "Higher cough_grade is associated with shorter mean pfs_months."),
]
for hid, col, htext in sx:
    m = ols_summary(f"pfs_months ~ {col}")
    b = m.params[col]; p = m.pvalues[col]
    add(it, hid, htext, "novel", f"ols('pfs_months ~ {col}')",
        f"Slope = {b:+.4f} mo per unit {col} (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 9: Comorbidity binary effects
# ============================================================================
it = 9
comorbid = [
    ("h9_1", "diabetes_mellitus", "Diabetes mellitus is associated with shorter mean pfs_months."),
    ("h9_2", "chronic_kidney_disease", "Chronic kidney disease is associated with shorter mean pfs_months."),
    ("h9_3", "heart_failure", "Heart failure is associated with shorter mean pfs_months."),
    ("h9_4", "copd", "COPD is associated with shorter mean pfs_months."),
    ("h9_5", "autoimmune_disease", "Autoimmune disease is associated with different mean pfs_months."),
    ("h9_6", "depression_anxiety_diagnosis", "Depression/anxiety diagnosis is associated with shorter mean pfs_months."),
    ("h9_7", "venous_thromboembolism_history", "Prior VTE is associated with shorter mean pfs_months."),
]
for hid, col, htext in comorbid:
    eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary(col)
    add(it, hid, htext, "novel", f"ttest by {col}",
        f"Mean PFS {m1:.2f} ({col}=1, n={n1}) vs {m0:.2f} (=0, n={n0}); diff = {eff:+.2f}, p={p:.3g}.",
        p, eff)

# ============================================================================
# ITERATION 10: Prior therapy / treatment history
# ============================================================================
it = 10
prior = [
    ("h10_1", "prior_chemotherapy", "Prior chemotherapy is associated with shorter mean pfs_months."),
    ("h10_2", "prior_radiation", "Prior radiation is associated with shorter mean pfs_months."),
    ("h10_3", "prior_surgery", "Prior surgery is associated with longer mean pfs_months (selection for less advanced disease)."),
    ("h10_4", "prior_immunotherapy", "Prior immunotherapy is associated with shorter mean pfs_months."),
    ("h10_5", "prior_targeted_therapy", "Prior targeted therapy is associated with shorter mean pfs_months."),
    ("h10_6", "prior_malignancy", "Prior malignancy is associated with shorter mean pfs_months."),
]
for hid, col, htext in prior:
    eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary(col)
    add(it, hid, htext, "novel", f"ttest by {col}",
        f"Mean PFS {m1:.2f} ({col}=1, n={n1}) vs {m0:.2f} (=0, n={n0}); diff = {eff:+.2f}, p={p:.3g}.",
        p, eff)

# ============================================================================
# ITERATION 11: Adjusted treatment effects (multivariable)
# ============================================================================
it = 11
covariates = "age_years + ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + albumin_g_dl + ldh_u_l + er_positive + her2_positive + prior_lines_of_therapy"
adj_treat = [
    ("h11_1", "treatment_trastuzumab"),
    ("h11_2", "treatment_tamoxifen"),
    ("h11_3", "treatment_palbociclib"),
    ("h11_4", "treatment_olaparib"),
    ("h11_5", "treatment_sacituzumab_govitecan"),
    ("h11_6", "treatment_pembrolizumab"),
]
for hid, tx in adj_treat:
    m = ols_summary(f"pfs_months ~ {tx} + {covariates}")
    b = m.params[tx]; p = m.pvalues[tx]
    add(it, hid,
        f"After adjusting for age, ECOG, stage, sites of metastasis, labs, ER/HER2 status, and prior lines, {tx} is independently associated with longer pfs_months.",
        "refined",
        f"ols('pfs_months ~ {tx} + age_years + ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + albumin_g_dl + ldh_u_l + er_positive + her2_positive + prior_lines_of_therapy')",
        f"Adjusted coefficient on {tx} = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 12: SNP / pharmacogenomic main effects
# ============================================================================
it = 12
snps = ["snp_rs1045642", "snp_rs1065852", "snp_rs1799853", "snp_rs1800566", "snp_rs2228001",
        "snp_rs4244285", "snp_rs1801133", "snp_rs429358", "snp_rs7412", "snp_rs1800629",
        "snp_rs4880", "snp_rs1050828"]
for i, s in enumerate(snps, start=1):
    m = ols_summary(f"pfs_months ~ {s}")
    b = m.params[s]; p = m.pvalues[s]
    add(it, f"h12_{i}", f"The genotype dosage at {s} is associated with mean pfs_months.", "novel",
        f"ols('pfs_months ~ {s}')",
        f"Slope = {b:+.4f} mo per allele (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 13: Olaparib by BRCA1 vs BRCA2 separately
# ============================================================================
it = 13
m = ols_summary("pfs_months ~ treatment_olaparib * brca1_mutation")
b = m.params["treatment_olaparib:brca1_mutation"]; p = m.pvalues["treatment_olaparib:brca1_mutation"]
add(it, "h13_1",
    "The PFS benefit of treatment_olaparib is larger in BRCA1-mutated patients than in BRCA1-wildtype patients (positive interaction).",
    "refined", "ols('pfs_months ~ treatment_olaparib * brca1_mutation')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))
m = ols_summary("pfs_months ~ treatment_olaparib * brca2_mutation")
b = m.params["treatment_olaparib:brca2_mutation"]; p = m.pvalues["treatment_olaparib:brca2_mutation"]
add(it, "h13_2",
    "The PFS benefit of treatment_olaparib is larger in BRCA2-mutated patients than in BRCA2-wildtype patients (positive interaction).",
    "refined", "ols('pfs_months ~ treatment_olaparib * brca2_mutation')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 14: Within-subgroup treatment effects (the "matched" subgroup)
# ============================================================================
it = 14
def subgroup_treat(sub_col, tx_col, hid, htext):
    sub = df[df[sub_col] == 1]
    g1 = sub.loc[sub[tx_col] == 1, "pfs_months"]
    g0 = sub.loc[sub[tx_col] == 0, "pfs_months"]
    res = stats.ttest_ind(g1, g0, equal_var=False)
    eff = float(g1.mean() - g0.mean())
    add(it, hid, htext, "refined",
        f"Within {sub_col}==1, ttest pfs_months by {tx_col}",
        f"In {sub_col}==1 (n={len(sub)}): PFS {g1.mean():.2f} on {tx_col} (n={len(g1)}) vs {g0.mean():.2f} off (n={len(g0)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
        float(res.pvalue), eff)

subgroup_treat("her2_positive", "treatment_trastuzumab", "h14_1",
    "Within HER2-positive patients, those receiving treatment_trastuzumab have higher mean pfs_months than those not.")
subgroup_treat("er_positive", "treatment_tamoxifen", "h14_2",
    "Within ER-positive patients, those receiving treatment_tamoxifen have higher mean pfs_months than those not.")
subgroup_treat("er_positive", "treatment_palbociclib", "h14_3",
    "Within ER-positive patients, those receiving treatment_palbociclib have higher mean pfs_months than those not.")
df["brca_any"] = ((df["brca1_mutation"] == 1) | (df["brca2_mutation"] == 1)).astype(int)
subgroup_treat("brca_any", "treatment_olaparib", "h14_4",
    "Within BRCA1- or BRCA2-mutated patients, those receiving treatment_olaparib have higher mean pfs_months than those not.")
subgroup_treat("her2_low", "treatment_sacituzumab_govitecan", "h14_5",
    "Within HER2-low patients, those receiving treatment_sacituzumab_govitecan have higher mean pfs_months than those not.")
subgroup_treat("msi_high", "treatment_pembrolizumab", "h14_6",
    "Within MSI-high patients, those receiving treatment_pembrolizumab have higher mean pfs_months than those not.")

# ============================================================================
# ITERATION 15: Reverse subgroups (off-target subgroups)
# ============================================================================
it = 15
def offtarget(sub_col, sub_val, tx_col, hid, htext):
    sub = df[df[sub_col] == sub_val]
    g1 = sub.loc[sub[tx_col] == 1, "pfs_months"]
    g0 = sub.loc[sub[tx_col] == 0, "pfs_months"]
    res = stats.ttest_ind(g1, g0, equal_var=False)
    eff = float(g1.mean() - g0.mean())
    add(it, hid, htext, "refined",
        f"Within {sub_col}=={sub_val}, ttest pfs_months by {tx_col}",
        f"In {sub_col}=={sub_val} (n={len(sub)}): PFS {g1.mean():.2f} on {tx_col} (n={len(g1)}) vs {g0.mean():.2f} off (n={len(g0)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
        float(res.pvalue), eff)

offtarget("her2_positive", 0, "treatment_trastuzumab", "h15_1",
    "In HER2-NEGATIVE patients, treatment_trastuzumab is NOT associated with longer pfs_months (off-target null).")
offtarget("er_positive", 0, "treatment_tamoxifen", "h15_2",
    "In ER-NEGATIVE patients, treatment_tamoxifen is NOT associated with longer pfs_months (off-target null).")
offtarget("brca_any", 0, "treatment_olaparib", "h15_3",
    "In BRCA-WILDTYPE patients, treatment_olaparib is NOT associated with longer pfs_months (off-target null).")
offtarget("msi_high", 0, "treatment_pembrolizumab", "h15_4",
    "In microsatellite-stable patients, treatment_pembrolizumab is NOT associated with longer pfs_months (off-target null).")

# ============================================================================
# ITERATION 16: Three-way interactions / refined biology
# ============================================================================
it = 16
# Trastuzumab benefit may scale with HER2 amplification
m = ols_summary("pfs_months ~ treatment_trastuzumab * her2_amplification")
b = m.params["treatment_trastuzumab:her2_amplification"]; p = m.pvalues["treatment_trastuzumab:her2_amplification"]
add(it, "h16_1",
    "The PFS benefit of treatment_trastuzumab is larger in HER2-amplified patients than in HER2-non-amplified patients (positive interaction).",
    "novel", "ols('pfs_months ~ treatment_trastuzumab * her2_amplification')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Palbociclib x CDKN2A loss
m = ols_summary("pfs_months ~ treatment_palbociclib * cdkn2a_loss")
b = m.params["treatment_palbociclib:cdkn2a_loss"]; p = m.pvalues["treatment_palbociclib:cdkn2a_loss"]
add(it, "h16_2",
    "The PFS effect of treatment_palbociclib is modified by CDKN2A loss (interaction non-zero).",
    "novel", "ols('pfs_months ~ treatment_palbociclib * cdkn2a_loss')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Tamoxifen x postmenopausal (premenopausal sometimes get tamoxifen preferentially)
m = ols_summary("pfs_months ~ treatment_tamoxifen * postmenopausal")
b = m.params["treatment_tamoxifen:postmenopausal"]; p = m.pvalues["treatment_tamoxifen:postmenopausal"]
add(it, "h16_3",
    "The PFS effect of treatment_tamoxifen differs between postmenopausal and premenopausal patients (interaction non-zero).",
    "novel", "ols('pfs_months ~ treatment_tamoxifen * postmenopausal')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 17: Insurance / socioeconomic disparities in outcome
# ============================================================================
it = 17
# Insurance: medicaid/uninsured vs private
df["insurance_disadvantaged"] = df["insurance_type"].isin(["medicaid", "uninsured"]).astype(int)
eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary("insurance_disadvantaged")
add(it, "h17_1",
    "Patients with disadvantaged insurance (medicaid or uninsured) have shorter mean pfs_months than those with private/medicare insurance.",
    "novel", "ttest by insurance_disadvantaged (medicaid+uninsured vs others)",
    f"Mean PFS = {m1:.2f} disadvantaged (n={n1}) vs {m0:.2f} other (n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
    p, eff)

# Rural residence
eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary("rural_residence")
add(it, "h17_2",
    "Patients with rural residence have shorter mean pfs_months than those without.",
    "novel", "ttest by rural_residence",
    f"Mean PFS = {m1:.2f} rural (n={n1}) vs {m0:.2f} (n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
    p, eff)

# Education years
m = ols_summary("pfs_months ~ education_years")
b = m.params["education_years"]; p = m.pvalues["education_years"]
add(it, "h17_3", "More education_years is associated with longer mean pfs_months.", "novel",
    "ols('pfs_months ~ education_years')",
    f"Slope = {b:+.4f} mo per year of education (p={p:.3g}).", float(p), float(b))

# Black vs white pfs
gB = df.loc[df["race_ethnicity"] == "black", "pfs_months"]
gW = df.loc[df["race_ethnicity"] == "white", "pfs_months"]
res = stats.ttest_ind(gB, gW, equal_var=False)
eff = float(gB.mean() - gW.mean())
add(it, "h17_4",
    "Black patients have shorter mean pfs_months than White patients.", "novel",
    "ttest pfs_months between race_ethnicity=='black' and 'white'",
    f"Mean PFS = {gB.mean():.2f} black (n={len(gB)}) vs {gW.mean():.2f} white (n={len(gW)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
    float(res.pvalue), eff)

# Hispanic vs white
gH = df.loc[df["race_ethnicity"] == "hispanic", "pfs_months"]
res = stats.ttest_ind(gH, gW, equal_var=False)
eff = float(gH.mean() - gW.mean())
add(it, "h17_5",
    "Hispanic patients have different mean pfs_months than White patients.", "novel",
    "ttest pfs_months between race_ethnicity=='hispanic' and 'white'",
    f"Mean PFS = {gH.mean():.2f} hispanic (n={len(gH)}) vs {gW.mean():.2f} white (n={len(gW)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
    float(res.pvalue), eff)

# ============================================================================
# ITERATION 18: Vital signs / labs (additional)
# ============================================================================
it = 18
labs2 = [
    ("h18_1", "platelets_k_ul", "Higher platelet count is associated with mean pfs_months (direction unspecified)."),
    ("h18_2", "wbc_k_ul", "Higher WBC count is associated with shorter mean pfs_months."),
    ("h18_3", "alc_k_ul", "Higher absolute lymphocyte count (alc_k_ul) is associated with longer mean pfs_months."),
    ("h18_4", "anc_k_ul", "Higher absolute neutrophil count (anc_k_ul) is associated with shorter mean pfs_months."),
    ("h18_5", "calcium_mg_dl", "Higher serum calcium is associated with shorter mean pfs_months."),
    ("h18_6", "ca_125_u_ml", "Higher CA-125 is associated with shorter mean pfs_months."),
    ("h18_7", "cea_ng_ml", "Higher CEA is associated with shorter mean pfs_months."),
    ("h18_8", "creatinine_mg_dl", "Higher creatinine is associated with shorter mean pfs_months."),
    ("h18_9", "total_bilirubin_mg_dl", "Higher total bilirubin is associated with shorter mean pfs_months."),
]
for hid, col, htext in labs2:
    m = ols_summary(f"pfs_months ~ {col}")
    b = m.params[col]; p = m.pvalues[col]
    add(it, hid, htext, "novel", f"ols('pfs_months ~ {col}')",
        f"Slope = {b:+.5f} mo per unit {col} (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 19: Continuous biomarker - Ki67 high vs low
# ============================================================================
it = 19
ki67_med = df["ki67_pct"].median()
df["ki67_high"] = (df["ki67_pct"] > 20).astype(int)  # standard >20% threshold
eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary("ki67_high")
add(it, "h19_1",
    "Patients with high Ki-67 (ki67_pct > 20) have shorter mean pfs_months than patients with low Ki-67.",
    "refined", "ttest by ki67_high (ki67_pct>20)",
    f"Mean PFS = {m1:.2f} high (n={n1}) vs {m0:.2f} low (n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
    p, eff)

# Stage IV x ECOG (interaction)
m = ols_summary("pfs_months ~ stage_iv * ecog_ps")
b = m.params["stage_iv:ecog_ps"]; p = m.pvalues["stage_iv:ecog_ps"]
add(it, "h19_2",
    "The negative effect of higher ECOG performance status on pfs_months is amplified in stage-IV patients (interaction).",
    "novel", "ols('pfs_months ~ stage_iv * ecog_ps')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Brain mets x systemic therapy access
df["any_targeted_or_io"] = (df[["treatment_trastuzumab", "treatment_olaparib",
                                  "treatment_sacituzumab_govitecan", "treatment_pembrolizumab"]].sum(axis=1) > 0).astype(int)
eff, p, m1, m0, n1, n0 = t_test_pfs_by_binary("any_targeted_or_io")
add(it, "h19_3",
    "Patients receiving any targeted/IO agent (trastuzumab, olaparib, sacituzumab govitecan, or pembrolizumab) have different mean pfs_months than those receiving none.",
    "novel", "ttest by any_targeted_or_io",
    f"Mean PFS = {m1:.2f} any (n={n1}) vs {m0:.2f} none (n={n0}); diff = {eff:+.2f}, p = {p:.3g}.",
    p, eff)

# ============================================================================
# ITERATION 20: Drug-class interactions (combinations)
# ============================================================================
it = 20
# Tamoxifen + Palbociclib combination effect
df["tam_and_palbo"] = ((df["treatment_tamoxifen"] == 1) & (df["treatment_palbociclib"] == 1)).astype(int)
m = ols_summary("pfs_months ~ treatment_tamoxifen * treatment_palbociclib")
b_int = m.params["treatment_tamoxifen:treatment_palbociclib"]; p_int = m.pvalues["treatment_tamoxifen:treatment_palbociclib"]
add(it, "h20_1",
    "The combination of treatment_tamoxifen and treatment_palbociclib has a synergistic (positive) interaction effect on pfs_months.",
    "novel", "ols('pfs_months ~ treatment_tamoxifen * treatment_palbociclib')",
    f"Interaction = {b_int:+.3f} mo (p={p_int:.3g}).", float(p_int), float(b_int))

# Trastuzumab + Pembrolizumab? (used in HER2+ sometimes)
m = ols_summary("pfs_months ~ treatment_trastuzumab * treatment_pembrolizumab")
b_int = m.params["treatment_trastuzumab:treatment_pembrolizumab"]; p_int = m.pvalues["treatment_trastuzumab:treatment_pembrolizumab"]
add(it, "h20_2",
    "The combination of treatment_trastuzumab and treatment_pembrolizumab has a non-zero interaction effect on pfs_months.",
    "novel", "ols('pfs_months ~ treatment_trastuzumab * treatment_pembrolizumab')",
    f"Interaction = {b_int:+.3f} mo (p={p_int:.3g}).", float(p_int), float(b_int))

# Palbociclib x PIK3CA (since both pathways relevant in ER+ disease)
m = ols_summary("pfs_months ~ treatment_palbociclib * pik3ca_mutation")
b_int = m.params["treatment_palbociclib:pik3ca_mutation"]; p_int = m.pvalues["treatment_palbociclib:pik3ca_mutation"]
add(it, "h20_3",
    "The PFS effect of treatment_palbociclib is modified by PIK3CA mutation status (interaction non-zero).",
    "novel", "ols('pfs_months ~ treatment_palbociclib * pik3ca_mutation')",
    f"Interaction = {b_int:+.3f} mo (p={p_int:.3g}).", float(p_int), float(b_int))

# ============================================================================
# ITERATION 21: Big multivariable model - net adjusted treatment / biomarker effects
# ============================================================================
it = 21
big_model = (
    "pfs_months ~ "
    "treatment_tamoxifen + treatment_palbociclib + treatment_trastuzumab + "
    "treatment_olaparib + treatment_sacituzumab_govitecan + treatment_pembrolizumab + "
    "age_years + ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + "
    "node_positive + er_positive + pr_positive + her2_positive + her2_low + "
    "brca1_mutation + brca2_mutation + pik3ca_mutation + tp53_mutation + msi_high + "
    "albumin_g_dl + ldh_u_l + crp_mg_l + nlr + hemoglobin_g_dl + ki67_pct + "
    "tumor_size_cm + prior_lines_of_therapy + weight_loss_pct_6mo"
)
m = ols_summary(big_model)
key = ["treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
       "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
       "er_positive", "her2_positive", "stage_iv", "ecog_ps", "albumin_g_dl",
       "ldh_u_l", "ki67_pct", "tumor_size_cm", "weight_loss_pct_6mo"]
for i, k in enumerate(key, start=1):
    b = float(m.params[k]); p = float(m.pvalues[k])
    sig = p < 0.05
    add(it, f"h21_{i}",
        f"In a fully adjusted multivariable OLS model of pfs_months, {k} has a non-zero independent association with pfs_months.",
        "refined",
        "ols(big multivariable model)",
        f"Adjusted coefficient on {k} = {b:+.4f} (p={p:.3g}); R^2={m.rsquared:.3f}.",
        p, b, sig)

# ============================================================================
# ITERATION 22: Three-way interaction: HER2 amplification x trastuzumab benefit refined
# ============================================================================
it = 22
# Within HER2-amplified, trastuzumab effect
sub = df[df["her2_amplification"] == 1]
g1 = sub.loc[sub["treatment_trastuzumab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_trastuzumab"] == 0, "pfs_months"]
res = stats.ttest_ind(g1, g0, equal_var=False)
eff = float(g1.mean() - g0.mean())
add(it, "h22_1",
    "Within HER2-amplified patients, those receiving treatment_trastuzumab have higher mean pfs_months than those not.",
    "refined", "ttest within her2_amplification==1",
    f"In her2_amplification==1 (n={len(sub)}): PFS {g1.mean():.2f} on trastuzumab (n={len(g1)}) vs {g0.mean():.2f} off (n={len(g0)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
    float(res.pvalue), eff)

# Within HER2 non-amplified, trastuzumab effect (off-target)
sub = df[df["her2_amplification"] == 0]
g1 = sub.loc[sub["treatment_trastuzumab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_trastuzumab"] == 0, "pfs_months"]
res = stats.ttest_ind(g1, g0, equal_var=False)
eff = float(g1.mean() - g0.mean())
add(it, "h22_2",
    "Within HER2-non-amplified patients, treatment_trastuzumab is not associated with longer pfs_months.",
    "refined", "ttest within her2_amplification==0",
    f"In her2_amplification==0 (n={len(sub)}): PFS {g1.mean():.2f} on trastuzumab (n={len(g1)}) vs {g0.mean():.2f} off (n={len(g0)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
    float(res.pvalue), eff)

# her2_positive AND her2_amplification are likely correlated; check
xt = pd.crosstab(df["her2_positive"], df["her2_amplification"])
chi2, p_chi, dof, exp = stats.chi2_contingency(xt)
add(it, "h22_3",
    "her2_positive status and her2_amplification status are highly correlated.",
    "novel", "chi2_contingency(her2_positive, her2_amplification)",
    f"Chi2 p={p_chi:.3g}; cross-tab counts = {xt.to_dict()}.", float(p_chi),
    float(xt.iloc[1, 1] / max(1, xt.iloc[1, :].sum()) - xt.iloc[0, 1] / max(1, xt.iloc[0, :].sum())))

# ============================================================================
# ITERATION 23: Treatment x ECOG (does benefit attenuate in poor PS?)
# ============================================================================
it = 23
for i, tx in enumerate(["treatment_trastuzumab", "treatment_palbociclib",
                         "treatment_pembrolizumab", "treatment_olaparib"], start=1):
    m = ols_summary(f"pfs_months ~ {tx} * ecog_ps")
    b = m.params[f"{tx}:ecog_ps"]; p = m.pvalues[f"{tx}:ecog_ps"]
    add(it, f"h23_{i}",
        f"The PFS benefit of {tx} is attenuated (or amplified) in patients with higher ECOG performance status (interaction non-zero).",
        "novel", f"ols('pfs_months ~ {tx} * ecog_ps')",
        f"Interaction = {b:+.4f} mo per unit ECOG (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 24: HRD/BRCA biology cross-checks
# ============================================================================
it = 24
# BRCA1 vs BRCA2 mutation rate by er_positive (BRCA1 typically TNBC; ER- enriched)
ct = pd.crosstab(df["er_positive"], df["brca1_mutation"])
chi2, p_chi, _, _ = stats.chi2_contingency(ct)
prop_brca1_er_pos = df.loc[df["er_positive"] == 1, "brca1_mutation"].mean()
prop_brca1_er_neg = df.loc[df["er_positive"] == 0, "brca1_mutation"].mean()
add(it, "h24_1",
    "BRCA1 mutation rate is higher in ER-negative patients than in ER-positive patients.",
    "novel", "chi2_contingency(er_positive, brca1_mutation)",
    f"Prop BRCA1+ in ER+ = {prop_brca1_er_pos:.4f}; in ER- = {prop_brca1_er_neg:.4f}; chi2 p={p_chi:.3g}.",
    float(p_chi), float(prop_brca1_er_neg - prop_brca1_er_pos))

# triple-negative rate vs other - and check if pembrolizumab benefit is concentrated there
df["tnbc"] = ((df["er_positive"] == 0) & (df["pr_positive"] == 0) & (df["her2_positive"] == 0)).astype(int)
m = ols_summary("pfs_months ~ treatment_pembrolizumab * tnbc")
b = m.params["treatment_pembrolizumab:tnbc"]; p = m.pvalues["treatment_pembrolizumab:tnbc"]
add(it, "h24_2",
    "The PFS benefit of treatment_pembrolizumab is larger in triple-negative breast cancer (TNBC: ER-/PR-/HER2-) than in non-TNBC patients (positive interaction).",
    "novel", "ols('pfs_months ~ treatment_pembrolizumab * tnbc')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}); TNBC rate = {df['tnbc'].mean():.3f}.", float(p), float(b))

# Sacituzumab govitecan x TNBC
m = ols_summary("pfs_months ~ treatment_sacituzumab_govitecan * tnbc")
b = m.params["treatment_sacituzumab_govitecan:tnbc"]; p = m.pvalues["treatment_sacituzumab_govitecan:tnbc"]
add(it, "h24_3",
    "The PFS benefit of treatment_sacituzumab_govitecan is larger in TNBC patients than in non-TNBC patients (positive interaction).",
    "novel", "ols('pfs_months ~ treatment_sacituzumab_govitecan * tnbc')",
    f"Interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# ============================================================================
# ITERATION 25: Summary multivariable + selected refinements
# ============================================================================
it = 25
# Treatment x biomarker for trastuzumab, but only within stage_iv to test extreme phenotype
sub = df[df["stage_iv"] == 1]
m = ols_summary_subset = ols("pfs_months ~ treatment_trastuzumab * her2_positive", data=sub).fit()
b = m.params["treatment_trastuzumab:her2_positive"]; p = m.pvalues["treatment_trastuzumab:her2_positive"]
add(it, "h25_1",
    "Within stage-IV patients, the PFS benefit of treatment_trastuzumab is larger in HER2-positive than in HER2-negative patients (positive interaction).",
    "refined", "ols('pfs_months ~ treatment_trastuzumab * her2_positive', data=stage_iv subset)",
    f"In stage_iv subset (n={len(sub)}), interaction = {b:+.3f} mo (p={p:.3g}).", float(p), float(b))

# Albumin x Stage IV interaction
m = ols_summary("pfs_months ~ albumin_g_dl * stage_iv")
b = m.params["albumin_g_dl:stage_iv"]; p = m.pvalues["albumin_g_dl:stage_iv"]
add(it, "h25_2",
    "The protective effect of higher albumin on pfs_months is modified by stage IV (interaction non-zero).",
    "refined", "ols('pfs_months ~ albumin_g_dl * stage_iv')",
    f"Interaction = {b:+.4f} mo per unit albumin*stage_iv (p={p:.3g}).", float(p), float(b))

# Treatment x age tertile for palbociclib
df["age_tertile"] = pd.qcut(df["age_years"], 3, labels=[0, 1, 2]).astype(int)
m = ols_summary("pfs_months ~ treatment_palbociclib * age_tertile")
b = m.params["treatment_palbociclib:age_tertile"]; p = m.pvalues["treatment_palbociclib:age_tertile"]
add(it, "h25_3",
    "The PFS benefit of treatment_palbociclib differs by age tertile (interaction non-zero).",
    "refined", "ols('pfs_months ~ treatment_palbociclib * age_tertile')",
    f"Interaction = {b:+.4f} mo per age tertile (p={p:.3g}).", float(p), float(b))

# Olaparib net effect within BRCA-any subset
sub = df[df["brca_any"] == 1]
g1 = sub.loc[sub["treatment_olaparib"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_olaparib"] == 0, "pfs_months"]
res = stats.ttest_ind(g1, g0, equal_var=False)
eff = float(g1.mean() - g0.mean())
add(it, "h25_4",
    "Within BRCA1- or BRCA2-mutated patients, those receiving treatment_olaparib have substantially higher mean pfs_months than those not (clinical confirmation).",
    "refined", "ttest pfs_months by treatment_olaparib within brca_any==1",
    f"In brca_any==1 (n={len(sub)}): PFS {g1.mean():.2f} on olaparib (n={len(g1)}) vs {g0.mean():.2f} off (n={len(g0)}); diff = {eff:+.2f}, p = {res.pvalue:.3g}.",
    float(res.pvalue), eff)

# Final synthesis: prognostic score = albumin - LDH/200 + (-ecog_ps) impact on PFS
df["prog_score"] = df["albumin_g_dl"] - df["ldh_u_l"] / 200.0 - df["ecog_ps"]
m = ols_summary("pfs_months ~ prog_score")
b = m.params["prog_score"]; p = m.pvalues["prog_score"]
add(it, "h25_5",
    "A composite prognostic score (albumin_g_dl - ldh_u_l/200 - ecog_ps) is positively associated with pfs_months.",
    "refined", "ols('pfs_months ~ prog_score') with prog_score = albumin - ldh/200 - ecog_ps",
    f"Slope = {b:+.4f} mo per unit prog_score (p={p:.3g}).", float(p), float(b))

# ============================================================================
# Persist results
# ============================================================================
with open("results.json", "w") as fh:
    json.dump(records, fh, indent=2)
print(f"Wrote {len(records)} analysis records across iterations 1-25.")
