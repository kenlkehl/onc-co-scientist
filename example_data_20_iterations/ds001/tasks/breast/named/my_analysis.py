"""
Iterative analysis of ds001_breast.

Each iteration block prints results and stores them so we can build the transcript at the end.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
print("loaded", df.shape)

results = []  # one dict per analysis


def add(it, hid, htext, kind, summary, p, effect, sig, code):
    results.append({
        "iteration": it,
        "hid": hid,
        "htext": htext,
        "kind": kind,
        "summary": summary,
        "p": None if p is None else float(p),
        "effect": None if effect is None else float(effect),
        "sig": bool(sig) if sig is not None else None,
        "code": code,
    })


def ttest_grp(col, group_col, sub=None):
    d = df if sub is None else sub
    a = d.loc[d[group_col] == 1, col].dropna()
    b = d.loc[d[group_col] == 0, col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean()), float(b.mean()), float(p), float(a.mean() - b.mean())


# =============================================================================
# ITERATION 1 — Cohort-wide treatment main effects on PFS
# =============================================================================
print("\n=== ITER 1: Treatment main effects on PFS ===")
TREATMENTS = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]
for tx in TREATMENTS:
    m1, m0, p, eff = ttest_grp("pfs_months", tx)
    summary = f"Mean PFS months: {tx}=1 -> {m1:.3f} vs {tx}=0 -> {m0:.3f}; diff={eff:.3f}, t-test p={p:.4g}"
    print(summary)
    add(1, f"h1_{tx}",
        f"Mean pfs_months differs between patients with {tx}=1 and {tx}=0 in the full cohort.",
        "novel", summary, p, eff, p < 0.05,
        f"stats.ttest_ind(df.loc[df['{tx}']==1,'pfs_months'], df.loc[df['{tx}']==0,'pfs_months'], equal_var=False)")

# =============================================================================
# ITERATION 2 — Biomarker-targeted treatment subgroup analyses
# =============================================================================
print("\n=== ITER 2: Biomarker-targeted subgroup PFS comparisons ===")

# tamoxifen in ER+
sub = df[df.er_positive == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_tamoxifen", sub)
add(2, "h2_tam_er",
    "Among er_positive=1 patients, mean pfs_months is higher with treatment_tamoxifen=1 than =0.",
    "novel", f"ER+: tam={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "df_er=df[df.er_positive==1]; ttest pfs by treatment_tamoxifen")
print("tam in ER+:", m1, m0, p)

# tamoxifen in ER- (should not benefit)
sub = df[df.er_positive == 0]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_tamoxifen", sub)
add(2, "h2_tam_erneg",
    "Among er_positive=0 patients, mean pfs_months does not differ between treatment_tamoxifen=1 and =0.",
    "novel", f"ER-: tam={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "df_er0=df[df.er_positive==0]; ttest pfs by treatment_tamoxifen")
print("tam in ER-:", m1, m0, p)

# trastuzumab in HER2+
sub = df[df.her2_positive == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_trastuzumab", sub)
add(2, "h2_tras_her2",
    "Among her2_positive=1 patients, mean pfs_months is higher with treatment_trastuzumab=1 than =0.",
    "novel", f"HER2+: tras={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "df_h=df[df.her2_positive==1]; ttest pfs by treatment_trastuzumab")
print("tras in HER2+:", m1, m0, p)

# trastuzumab in HER2-
sub = df[df.her2_positive == 0]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_trastuzumab", sub)
add(2, "h2_tras_her2neg",
    "Among her2_positive=0 patients, mean pfs_months does not differ between treatment_trastuzumab=1 and =0.",
    "novel", f"HER2-: tras={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "ttest within HER2-")
print("tras in HER2-:", m1, m0, p)

# olaparib in BRCA1/2 mutated
sub = df[(df.brca1_mutation == 1) | (df.brca2_mutation == 1)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(2, "h2_ola_brca",
    "Among patients with brca1_mutation=1 or brca2_mutation=1, mean pfs_months is higher with treatment_olaparib=1 than =0.",
    "novel", f"BRCA1/2-mut: olap={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "subset BRCA mutated; ttest pfs by treatment_olaparib")
print("ola in BRCA+:", m1, m0, p)

# olaparib in BRCA wt
sub = df[(df.brca1_mutation == 0) & (df.brca2_mutation == 0)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(2, "h2_ola_brcawt",
    "Among patients with brca1_mutation=0 and brca2_mutation=0, mean pfs_months does not differ between treatment_olaparib=1 and =0.",
    "novel", f"BRCA-wt: olap={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "subset BRCA wt; ttest pfs by treatment_olaparib")
print("ola in BRCA-:", m1, m0, p)

# palbociclib in ER+/HER2-
sub = df[(df.er_positive == 1) & (df.her2_positive == 0)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_palbociclib", sub)
add(2, "h2_palbo_erp_her2n",
    "Among er_positive=1 and her2_positive=0 patients, mean pfs_months is higher with treatment_palbociclib=1 than =0.",
    "novel", f"ER+/HER2-: palbo={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "subset ER+/HER2-; ttest pfs by treatment_palbociclib")
print("palbo in ER+/HER2-:", m1, m0, p)

# sacituzumab in TNBC
sub = df[(df.er_positive == 0) & (df.pr_positive == 0) & (df.her2_positive == 0)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_sacituzumab_govitecan", sub)
add(2, "h2_sg_tnbc",
    "Among triple-negative (er_positive=0, pr_positive=0, her2_positive=0) patients, mean pfs_months is higher with treatment_sacituzumab_govitecan=1 than =0.",
    "novel", f"TNBC: SG={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "TNBC subset; ttest pfs by treatment_sacituzumab_govitecan")
print("SG in TNBC:", m1, m0, p)

# pembrolizumab in MSI-high
sub = df[df.msi_high == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_pembrolizumab", sub)
add(2, "h2_pembro_msi",
    "Among msi_high=1 patients, mean pfs_months is higher with treatment_pembrolizumab=1 than =0.",
    "novel", f"MSI-H: pembro={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "subset msi_high; ttest pfs by treatment_pembrolizumab")
print("pembro in MSI-H:", m1, m0, p)

# =============================================================================
# ITERATION 3 — Continuous & binary prognostic main effects on PFS
# =============================================================================
print("\n=== ITER 3: Prognostic main effects on PFS ===")
for col in ["age_years", "ecog_ps", "albumin_g_dl", "ldh_u_l", "weight_loss_pct_6mo",
            "crp_mg_l", "nlr", "tumor_size_cm", "ki67_pct", "hemoglobin_g_dl",
            "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l"]:
    rho, p = stats.pearsonr(df[col], df["pfs_months"])
    summary = f"Pearson r({col}, pfs_months) = {rho:.4f}, p={p:.4g}"
    print(summary)
    add(3, f"h3_corr_{col}",
        f"Higher {col} is associated with shorter pfs_months (negative linear correlation).",
        "novel", summary, p, rho, p < 0.05,
        f"stats.pearsonr(df['{col}'], df['pfs_months'])")

for col in ["stage_iv", "has_brain_mets", "liver_mets", "bone_mets", "node_positive",
            "pleural_effusion", "pericardial_effusion", "adrenal_mets"]:
    m1, m0, p, eff = ttest_grp("pfs_months", col)
    summary = f"PFS: {col}=1 mean {m1:.3f} vs =0 mean {m0:.3f}; diff={eff:.3f}, p={p:.4g}"
    print(summary)
    add(3, f"h3_grp_{col}",
        f"Mean pfs_months is lower in {col}=1 than {col}=0.",
        "novel", summary, p, eff, p < 0.05,
        f"ttest pfs by {col}")

# =============================================================================
# ITERATION 4 — Formal interaction tests (treatment × biomarker)
# =============================================================================
print("\n=== ITER 4: Formal treatment x biomarker interactions on PFS ===")
interactions = [
    ("treatment_tamoxifen", "er_positive"),
    ("treatment_tamoxifen", "pr_positive"),
    ("treatment_trastuzumab", "her2_positive"),
    ("treatment_trastuzumab", "her2_amplification"),
    ("treatment_olaparib", "brca1_mutation"),
    ("treatment_olaparib", "brca2_mutation"),
    ("treatment_palbociclib", "er_positive"),
    ("treatment_pembrolizumab", "msi_high"),
    ("treatment_sacituzumab_govitecan", "her2_low"),
]
for tx, bm in interactions:
    f = f"pfs_months ~ {tx} * {bm}"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:{bm}"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    summary = f"OLS pfs_months ~ {tx}*{bm}: interaction coef={coef:.4f}, p={p:.4g}"
    print(summary)
    add(4, f"h4_int_{tx}_{bm}",
        f"There is a positive {tx}-by-{bm} interaction on pfs_months: the benefit of {tx} is greater when {bm}=1 than when {bm}=0.",
        "novel", summary, p, coef, p < 0.05,
        f"smf.ols('{f}', data=df).fit() -> coefficient on '{term}'")

# =============================================================================
# ITERATION 5 — Sociodemographic, comorbidity, lifestyle effects on PFS
# =============================================================================
print("\n=== ITER 5: Demographic, social, comorbidity effects ===")

m1, m0, p, eff = ttest_grp("pfs_months", "sex_female")
add(5, "h5_sex",
    "Mean pfs_months differs between sex_female=1 and sex_female=0 patients (positive coef = female longer PFS).",
    "novel", f"PFS female={m1:.3f} vs male={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "ttest by sex_female")
print("sex:", m1, m0, p)

m1, m0, p, eff = ttest_grp("pfs_months", "rural_residence")
add(5, "h5_rural",
    "Mean pfs_months is lower in rural_residence=1 than rural_residence=0.",
    "novel", f"PFS rural={m1:.3f} vs urban={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "ttest by rural_residence")
print("rural:", m1, m0, p)

groups = [df.loc[df.insurance_type == c, "pfs_months"] for c in df.insurance_type.unique()]
F, p = stats.f_oneway(*groups)
mns = df.groupby("insurance_type")["pfs_months"].mean().to_dict()
add(5, "h5_insurance",
    "Mean pfs_months differs across insurance_type categories.",
    "novel", f"ANOVA F={F:.3f}, p={p:.4g}; means={mns}", p, F, p < 0.05,
    "stats.f_oneway across insurance_type")
print("insurance:", F, p, mns)

groups = [df.loc[df.race_ethnicity == c, "pfs_months"] for c in df.race_ethnicity.unique()]
F, p = stats.f_oneway(*groups)
mns = df.groupby("race_ethnicity")["pfs_months"].mean().to_dict()
add(5, "h5_race",
    "Mean pfs_months differs across race_ethnicity categories.",
    "novel", f"ANOVA F={F:.3f}, p={p:.4g}; means={mns}", p, F, p < 0.05,
    "stats.f_oneway across race_ethnicity")
print("race:", F, p, mns)

for col in ["diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
            "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
            "venous_thromboembolism_history", "depression_anxiety_diagnosis",
            "interstitial_lung_disease_history", "prior_malignancy"]:
    m1, m0, p, eff = ttest_grp("pfs_months", col)
    add(5, f"h5_co_{col}",
        f"Mean pfs_months is lower in {col}=1 vs {col}=0.",
        "novel", f"PFS {col}=1 {m1:.3f} vs =0 {m0:.3f}; diff={eff:.3f}, p={p:.4g}",
        p, eff, p < 0.05,
        f"ttest by {col}")
    print(col, ":", m1, m0, p)

rho, p = stats.pearsonr(df["smoking_pack_years"], df["pfs_months"])
add(5, "h5_smoking",
    "Higher smoking_pack_years is associated with shorter pfs_months.",
    "novel", f"r({rho:.4f}) p={p:.4g}", p, rho, p < 0.05,
    "stats.pearsonr(smoking_pack_years, pfs_months)")
print("smoke:", rho, p)

rho, p = stats.pearsonr(df["education_years"], df["pfs_months"])
add(5, "h5_education",
    "Higher education_years is associated with longer pfs_months.",
    "novel", f"r={rho:.4f} p={p:.4g}", p, rho, p < 0.05,
    "stats.pearsonr(education_years, pfs_months)")
print("educ:", rho, p)

# =============================================================================
# ITERATION 6 — Symptom grades and laboratory derangements
# =============================================================================
print("\n=== ITER 6: Symptoms & lab derangements ===")
for col in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]:
    rho, p = stats.pearsonr(df[col], df["pfs_months"])
    add(6, f"h6_sym_{col}",
        f"Higher {col} is associated with shorter pfs_months.",
        "novel", f"r({col}, pfs_months)={rho:.4f}, p={p:.4g}", p, rho, p < 0.05,
        f"pearsonr({col}, pfs_months)")
    print(col, rho, p)

for col in ["total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl", "glucose_mg_dl",
            "platelets_k_ul", "wbc_k_ul", "anc_k_ul", "alc_k_ul", "ca_125_u_ml",
            "cea_ng_ml", "psa_ng_ml", "tsh_uiu_ml", "inr", "bmi",
            "systolic_bp_mmhg", "diastolic_bp_mmhg", "heart_rate_bpm", "spo2_pct",
            "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]:
    rho, p = stats.pearsonr(df[col], df["pfs_months"])
    add(6, f"h6_lab_{col}",
        f"Higher {col} is associated with shorter pfs_months.",
        "novel", f"r({col}, pfs_months)={rho:.4f}, p={p:.4g}", p, rho, p < 0.05,
        f"pearsonr({col}, pfs_months)")
    if abs(rho) > 0.05:
        print(col, rho, p)

# =============================================================================
# ITERATION 7 — Prior therapy main effects
# =============================================================================
print("\n=== ITER 7: Prior therapies ===")
for col in ["prior_chemotherapy", "prior_radiation", "prior_surgery",
            "prior_immunotherapy", "prior_targeted_therapy"]:
    m1, m0, p, eff = ttest_grp("pfs_months", col)
    add(7, f"h7_prior_{col}",
        f"Mean pfs_months is lower in {col}=1 vs {col}=0.",
        "novel", f"{col}=1 {m1:.3f} vs =0 {m0:.3f}; diff={eff:.3f}, p={p:.4g}",
        p, eff, p < 0.05,
        f"ttest by {col}")
    print(col, m1, m0, p)

rho, p = stats.pearsonr(df["prior_lines_of_therapy"], df["pfs_months"])
add(7, "h7_prior_lines",
    "Higher prior_lines_of_therapy is associated with shorter pfs_months.",
    "novel", f"r={rho:.4f} p={p:.4g}", p, rho, p < 0.05,
    "pearsonr")
print("prior_lines:", rho, p)

rho, p = stats.pearsonr(df["years_since_diagnosis"], df["pfs_months"])
add(7, "h7_years_dx",
    "Higher years_since_diagnosis is associated with shorter pfs_months.",
    "novel", f"r={rho:.4f} p={p:.4g}", p, rho, p < 0.05,
    "pearsonr")
print("years_since_dx:", rho, p)

# =============================================================================
# ITERATION 8 — SNP main effects on PFS (single-variant scan)
# =============================================================================
print("\n=== ITER 8: SNP scan ===")
SNPS = [c for c in df.columns if c.startswith("snp_")]
for snp in SNPS:
    m1, m0, p, eff = ttest_grp("pfs_months", snp)
    add(8, f"h8_{snp}",
        f"Mean pfs_months differs between {snp}=1 and {snp}=0 carriers.",
        "novel", f"{snp}=1 {m1:.3f} vs =0 {m0:.3f}; diff={eff:.3f}, p={p:.4g}",
        p, eff, p < 0.05,
        f"ttest by {snp}")
    if p < 0.01:
        print(snp, m1, m0, p)

# =============================================================================
# ITERATION 9 — Refine: multivariable adjusted treatment effects
# =============================================================================
print("\n=== ITER 9: Multivariable-adjusted treatment effects ===")
adjusters = ("age_years + ecog_ps + stage_iv + has_brain_mets + albumin_g_dl + "
             "ldh_u_l + tumor_size_cm + nlr + crp_mg_l + prior_lines_of_therapy")
for tx in TREATMENTS:
    f = f"pfs_months ~ {tx} + {adjusters}"
    m = smf.ols(f, data=df).fit()
    coef = float(m.params[tx]); p = float(m.pvalues[tx])
    summary = f"Adjusted PFS coef on {tx}: {coef:.4f}, p={p:.4g}"
    print(summary)
    add(9, f"h9_adj_{tx}",
        f"After adjusting for age_years, ecog_ps, stage_iv, has_brain_mets, albumin_g_dl, ldh_u_l, tumor_size_cm, nlr, crp_mg_l and prior_lines_of_therapy, {tx}=1 is associated with longer pfs_months.",
        "refined", summary, p, coef, p < 0.05,
        f"smf.ols('pfs_months ~ {tx} + {adjusters}', data=df).fit()")

# =============================================================================
# ITERATION 10 — Refine: adjusted biomarker-targeted treatment subgroups
# =============================================================================
print("\n=== ITER 10: Adjusted treatment effects within biomarker subgroups ===")
def adj_in_sub(tx, sub_expr, hid, htext):
    sub = df.query(sub_expr)
    f = f"pfs_months ~ {tx} + {adjusters}"
    m = smf.ols(f, data=sub).fit()
    coef = float(m.params[tx]); p = float(m.pvalues[tx])
    summary = f"Within {sub_expr}, adjusted coef({tx})={coef:.4f}, p={p:.4g}, n={len(sub)}"
    print(summary)
    add(10, hid, htext, "refined", summary, p, coef, p < 0.05,
        f"smf.ols on subset {sub_expr}")

adj_in_sub("treatment_tamoxifen", "er_positive == 1", "h10_tam_er",
           "After multivariable adjustment within er_positive=1, treatment_tamoxifen=1 is associated with longer pfs_months.")
adj_in_sub("treatment_trastuzumab", "her2_positive == 1", "h10_tras_her2",
           "After multivariable adjustment within her2_positive=1, treatment_trastuzumab=1 is associated with longer pfs_months.")
adj_in_sub("treatment_olaparib", "(brca1_mutation == 1) | (brca2_mutation == 1)", "h10_ola_brca",
           "After multivariable adjustment within BRCA1/2-mutated patients, treatment_olaparib=1 is associated with longer pfs_months.")
adj_in_sub("treatment_palbociclib", "(er_positive == 1) & (her2_positive == 0)", "h10_palbo_erp_her2n",
           "After multivariable adjustment within er_positive=1 and her2_positive=0, treatment_palbociclib=1 is associated with longer pfs_months.")
adj_in_sub("treatment_sacituzumab_govitecan", "(er_positive == 0) & (pr_positive == 0) & (her2_positive == 0)", "h10_sg_tnbc",
           "After multivariable adjustment within triple-negative patients, treatment_sacituzumab_govitecan=1 is associated with longer pfs_months.")
adj_in_sub("treatment_pembrolizumab", "msi_high == 1", "h10_pembro_msi",
           "After multivariable adjustment within msi_high=1, treatment_pembrolizumab=1 is associated with longer pfs_months.")

# =============================================================================
# ITERATION 11 — More refined: biomarker × treatment interactions, adjusted
# =============================================================================
print("\n=== ITER 11: Adjusted treatment × biomarker interactions ===")
for tx, bm in interactions:
    f = f"pfs_months ~ {tx} * {bm} + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + tumor_size_cm + prior_lines_of_therapy"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:{bm}"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    summary = f"Adjusted interaction {tx}*{bm}: coef={coef:.4f}, p={p:.4g}"
    print(summary)
    add(11, f"h11_intadj_{tx}_{bm}",
        f"After adjusting for clinical covariates (age_years, ecog_ps, stage_iv, albumin_g_dl, ldh_u_l, tumor_size_cm, prior_lines_of_therapy), the {tx}-by-{bm} interaction effect on pfs_months is positive.",
        "refined", summary, p, coef, p < 0.05,
        f"smf.ols pfs_months ~ {tx}*{bm}+covariates")

# =============================================================================
# ITERATION 12 — HER2-low and her2_amplification subgroup checks
# =============================================================================
print("\n=== ITER 12: HER2-low / her2_amplification effects ===")
sub = df[df.her2_low == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_sacituzumab_govitecan", sub)
add(12, "h12_sg_her2low",
    "Among her2_low=1 patients, mean pfs_months is higher with treatment_sacituzumab_govitecan=1 than =0.",
    "novel", f"HER2-low: SG={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05,
    "subset her2_low; ttest pfs by treatment_sacituzumab_govitecan")
print("SG in HER2-low:", m1, m0, p)

sub = df[df.her2_amplification == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_trastuzumab", sub)
add(12, "h12_tras_her2amp",
    "Among her2_amplification=1 patients, mean pfs_months is higher with treatment_trastuzumab=1 than =0.",
    "novel", f"HER2 amp: tras={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05,
    "subset her2_amplification; ttest pfs by treatment_trastuzumab")
print("tras in HER2-amp:", m1, m0, p)

# pik3ca interaction with palbociclib
f = "pfs_months ~ treatment_palbociclib * pik3ca_mutation"
m = smf.ols(f, data=df).fit()
coef = float(m.params["treatment_palbociclib:pik3ca_mutation"])
p = float(m.pvalues["treatment_palbociclib:pik3ca_mutation"])
add(12, "h12_palbo_pik3ca",
    "There is a positive treatment_palbociclib by pik3ca_mutation interaction on pfs_months.",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
print("palbo*pik3ca:", coef, p)

# =============================================================================
# ITERATION 13 — Postmenopausal x tamoxifen
# =============================================================================
print("\n=== ITER 13: Postmenopausal status, age, and ECOG modifiers ===")
sub = df[df.postmenopausal == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_tamoxifen", sub)
add(13, "h13_tam_postmeno",
    "Among postmenopausal=1 patients, mean pfs_months is higher with treatment_tamoxifen=1 than =0.",
    "novel", f"Postmeno: tam={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}", p, eff, p < 0.05,
    "subset postmenopausal; ttest pfs by treatment_tamoxifen")
print("tam in postmeno:", m1, m0, p)

f = "pfs_months ~ treatment_tamoxifen * postmenopausal"
m = smf.ols(f, data=df).fit()
coef = float(m.params["treatment_tamoxifen:postmenopausal"])
p = float(m.pvalues["treatment_tamoxifen:postmenopausal"])
add(13, "h13_int_tam_postmeno",
    "There is a positive treatment_tamoxifen by postmenopausal interaction on pfs_months: tamoxifen benefit is larger in postmenopausal=1 vs =0.",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
print("tam*postmeno int:", coef, p)

# Tamoxifen × age (continuous)
f = "pfs_months ~ treatment_tamoxifen * age_years"
m = smf.ols(f, data=df).fit()
coef = float(m.params["treatment_tamoxifen:age_years"])
p = float(m.pvalues["treatment_tamoxifen:age_years"])
add(13, "h13_int_tam_age",
    "There is a positive treatment_tamoxifen by age_years interaction on pfs_months (greater tamoxifen benefit in older patients).",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
print("tam*age int:", coef, p)

# ECOG×treatment_pembrolizumab (often poor PS does worse on IO)
f = "pfs_months ~ treatment_pembrolizumab * ecog_ps"
m = smf.ols(f, data=df).fit()
coef = float(m.params["treatment_pembrolizumab:ecog_ps"])
p = float(m.pvalues["treatment_pembrolizumab:ecog_ps"])
add(13, "h13_int_pembro_ecog",
    "There is a negative treatment_pembrolizumab by ecog_ps interaction on pfs_months (smaller pembrolizumab benefit at higher ECOG).",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
print("pembro*ecog int:", coef, p)

# =============================================================================
# ITERATION 14 — Brain mets x chemoradiation/treatment interactions
# =============================================================================
print("\n=== ITER 14: Brain mets, stage, line interactions ===")
for tx in TREATMENTS:
    f = f"pfs_months ~ {tx} * has_brain_mets"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:has_brain_mets"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    add(14, f"h14_int_{tx}_brain",
        f"There is a negative {tx} by has_brain_mets interaction on pfs_months (smaller {tx} benefit when brain mets are present).",
        "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)

for tx in TREATMENTS:
    f = f"pfs_months ~ {tx} * stage_iv"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:stage_iv"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    add(14, f"h14_int_{tx}_stage4",
        f"There is a negative {tx} by stage_iv interaction on pfs_months (smaller {tx} benefit in stage IV vs earlier).",
        "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)

# =============================================================================
# ITERATION 15 — Compound biomarker phenotypes
# =============================================================================
print("\n=== ITER 15: Compound phenotypes ===")
# triple-negative (TN) vs hormone-positive
df["tnbc"] = ((df.er_positive == 0) & (df.pr_positive == 0) & (df.her2_positive == 0)).astype(int)
m1, m0, p, eff = ttest_grp("pfs_months", "tnbc")
add(15, "h15_tnbc_pfs",
    "Mean pfs_months is lower in tnbc=1 (er_positive=0 & pr_positive=0 & her2_positive=0) vs tnbc=0 patients.",
    "novel", f"TNBC PFS={m1:.3f} vs non-TNBC={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "ttest pfs by derived tnbc")
print("TNBC:", m1, m0, p)

df["luminal_a_like"] = ((df.er_positive == 1) & (df.her2_positive == 0) &
                       (df.ki67_pct < 20)).astype(int)
m1, m0, p, eff = ttest_grp("pfs_months", "luminal_a_like")
add(15, "h15_lumA",
    "Mean pfs_months is higher in luminal-A-like patients (er_positive=1 & her2_positive=0 & ki67_pct<20) vs others.",
    "novel", f"PFS={m1:.3f} vs {m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "ttest pfs by luminal_a_like")
print("luminal A:", m1, m0, p)

# BRCA mutation regardless of treatment
df["brca_any"] = ((df.brca1_mutation == 1) | (df.brca2_mutation == 1)).astype(int)
m1, m0, p, eff = ttest_grp("pfs_months", "brca_any")
add(15, "h15_brca_any",
    "Mean pfs_months differs between BRCA1/2-mutated and non-mutated patients overall.",
    "novel", f"BRCA-any={m1:.3f} vs wt={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "ttest pfs by brca_any")
print("brca_any:", m1, m0, p)

# tp53 mutation main effect
m1, m0, p, eff = ttest_grp("pfs_months", "tp53_mutation")
add(15, "h15_tp53",
    "Mean pfs_months is lower in tp53_mutation=1 vs =0.",
    "novel", f"TP53 mut={m1:.3f} vs wt={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "ttest pfs by tp53_mutation")
print("TP53:", m1, m0, p)

# =============================================================================
# ITERATION 16 — Olaparib-specific interactions (BRCA1 vs BRCA2)
# =============================================================================
print("\n=== ITER 16: Olaparib in BRCA1 vs BRCA2 ===")
sub = df[df.brca1_mutation == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(16, "h16_ola_brca1",
    "Among brca1_mutation=1 patients, mean pfs_months is higher with treatment_olaparib=1 than =0.",
    "novel", f"BRCA1: ola={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset BRCA1; ttest")
print("ola in BRCA1:", m1, m0, p)

sub = df[df.brca2_mutation == 1]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(16, "h16_ola_brca2",
    "Among brca2_mutation=1 patients, mean pfs_months is higher with treatment_olaparib=1 than =0.",
    "novel", f"BRCA2: ola={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset BRCA2; ttest")
print("ola in BRCA2:", m1, m0, p)

# =============================================================================
# ITERATION 17 — Sex distribution in breast cancer & sex×treatment
# =============================================================================
print("\n=== ITER 17: Sex effects on treatment in breast cohort ===")
prop_female = df.sex_female.mean()
add(17, "h17_sex_dist",
    "Proportion of sex_female=1 patients in this breast cancer cohort exceeds 0.5.",
    "novel", f"Proportion female = {prop_female:.4f} (n=50000)", None,
    prop_female - 0.5, prop_female > 0.5,
    "df.sex_female.mean()")
print("prop female:", prop_female)

# trastuzumab × sex
f = "pfs_months ~ treatment_trastuzumab * sex_female"
m = smf.ols(f, data=df).fit()
coef = float(m.params["treatment_trastuzumab:sex_female"])
p = float(m.pvalues["treatment_trastuzumab:sex_female"])
add(17, "h17_int_tras_sex",
    "There is a positive treatment_trastuzumab by sex_female interaction on pfs_months (greater trastuzumab benefit in females).",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
print("tras*sex:", coef, p)

# =============================================================================
# ITERATION 18 — Three-way: tamoxifen x ER x postmenopausal
# =============================================================================
print("\n=== ITER 18: Three-way tamoxifen x ER x postmenopausal ===")
sub = df[(df.er_positive == 1) & (df.postmenopausal == 1)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_tamoxifen", sub)
add(18, "h18_tam_er_postmeno",
    "Among er_positive=1 and postmenopausal=1 patients, mean pfs_months is higher with treatment_tamoxifen=1 than =0.",
    "refined", f"ER+/postmeno: tam={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset ER+&postmeno; ttest")
print("tam in ER+&postmeno:", m1, m0, p)

sub = df[(df.er_positive == 1) & (df.postmenopausal == 0)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_tamoxifen", sub)
add(18, "h18_tam_er_premeno",
    "Among er_positive=1 and postmenopausal=0 patients, mean pfs_months is higher with treatment_tamoxifen=1 than =0.",
    "refined", f"ER+/premeno: tam={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset ER+&premeno; ttest")
print("tam in ER+&premeno:", m1, m0, p)

# =============================================================================
# ITERATION 19 — Synergy of palbociclib with tamoxifen in ER+/HER2-
# =============================================================================
print("\n=== ITER 19: Palbociclib×Tamoxifen in ER+/HER2- ===")
sub = df[(df.er_positive == 1) & (df.her2_positive == 0)]
f = "pfs_months ~ treatment_palbociclib * treatment_tamoxifen"
m = smf.ols(f, data=sub).fit()
term = "treatment_palbociclib:treatment_tamoxifen"
coef = float(m.params[term]); p = float(m.pvalues[term])
add(19, "h19_palbo_tam_int",
    "Within er_positive=1 & her2_positive=0 patients, there is a positive treatment_palbociclib by treatment_tamoxifen interaction on pfs_months.",
    "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05,
    f"smf.ols on subset; {f}")
print("palbo*tam:", coef, p)

# Both vs single agent
sub = df[(df.er_positive == 1) & (df.her2_positive == 0)]
both = sub[(sub.treatment_palbociclib == 1) & (sub.treatment_tamoxifen == 1)].pfs_months
palbo = sub[(sub.treatment_palbociclib == 1) & (sub.treatment_tamoxifen == 0)].pfs_months
tam = sub[(sub.treatment_palbociclib == 0) & (sub.treatment_tamoxifen == 1)].pfs_months
none = sub[(sub.treatment_palbociclib == 0) & (sub.treatment_tamoxifen == 0)].pfs_months
t, p = stats.ttest_ind(both, palbo, equal_var=False)
add(19, "h19_combo_vs_palbo",
    "Within er_positive=1 & her2_positive=0, mean pfs_months is higher with palbociclib+tamoxifen vs palbociclib alone.",
    "refined",
    f"both n={len(both)} mean={both.mean():.3f}; palbo-only n={len(palbo)} mean={palbo.mean():.3f}; t-test p={p:.4g}",
    float(p), float(both.mean() - palbo.mean()), p < 0.05,
    "subset; compare combo vs palbo-only")
print("combo vs palbo:", both.mean(), palbo.mean(), p)

# =============================================================================
# ITERATION 20 — Inflammation and treatment efficacy modifiers
# =============================================================================
print("\n=== ITER 20: Inflammation modifiers of pembrolizumab ===")
for mod in ["nlr", "albumin_g_dl", "crp_mg_l"]:
    f = f"pfs_months ~ treatment_pembrolizumab * {mod}"
    m = smf.ols(f, data=df).fit()
    term = f"treatment_pembrolizumab:{mod}"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    direction_word = "negative" if "nlr" in mod or "crp" in mod else "positive"
    add(20, f"h20_pembro_{mod}",
        f"There is a {direction_word} treatment_pembrolizumab by {mod} interaction on pfs_months.",
        "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
    print(f"pembro*{mod}:", coef, p)

# =============================================================================
# ITERATION 21 — SNP scan: pharmacogenomic interactions
# =============================================================================
print("\n=== ITER 21: SNP × treatment interactions ===")
SNPS_TARGETED = [
    ("snp_rs1045642", "treatment_palbociclib"),  # ABCB1 -> palbo
    ("snp_rs1065852", "treatment_tamoxifen"),    # CYP2D6 -> tamoxifen
    ("snp_rs1799853", "treatment_tamoxifen"),
    ("snp_rs4244285", "treatment_tamoxifen"),    # CYP2C19
    ("snp_rs1801133", "treatment_pembrolizumab"),
    ("snp_rs1800629", "treatment_pembrolizumab"),
    ("snp_rs429358", "treatment_olaparib"),
]
for snp, tx in SNPS_TARGETED:
    f = f"pfs_months ~ {tx} * {snp}"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:{snp}"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    add(21, f"h21_int_{tx}_{snp}",
        f"There is a non-zero {tx} by {snp} interaction on pfs_months (pharmacogenomic effect).",
        "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
    print(f"{tx}*{snp}:", coef, p)

# =============================================================================
# ITERATION 22 — Refine: olaparib in HER2-/BRCA-mut subgroup
# =============================================================================
print("\n=== ITER 22: Refined olaparib subgroups ===")
sub = df[((df.brca1_mutation == 1) | (df.brca2_mutation == 1)) & (df.her2_positive == 0)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(22, "h22_ola_brca_her2neg",
    "Within BRCA1/2-mutated and her2_positive=0 patients, mean pfs_months is higher with treatment_olaparib=1 than =0.",
    "refined", f"BRCA-mut & HER2-: ola={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset BRCA & HER2-; ttest")
print("ola in BRCA & HER2-:", m1, m0, p)

sub = df[((df.brca1_mutation == 1) | (df.brca2_mutation == 1)) & (df.er_positive == 1)]
m1, m0, p, eff = ttest_grp("pfs_months", "treatment_olaparib", sub)
add(22, "h22_ola_brca_erpos",
    "Within BRCA1/2-mutated and er_positive=1 patients, mean pfs_months is higher with treatment_olaparib=1 than =0.",
    "refined", f"BRCA-mut & ER+: ola={m1:.3f} vs none={m0:.3f}; diff={eff:.3f}, p={p:.4g}",
    p, eff, p < 0.05, "subset BRCA & ER+; ttest")
print("ola in BRCA & ER+:", m1, m0, p)

# =============================================================================
# ITERATION 23 — Composite multivariate model
# =============================================================================
print("\n=== ITER 23: Big multivariate OLS ===")
big_form = ("pfs_months ~ age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + "
            "albumin_g_dl + ldh_u_l + nlr + tumor_size_cm + ki67_pct + prior_lines_of_therapy + "
            "er_positive + pr_positive + her2_positive + brca1_mutation + brca2_mutation + "
            "msi_high + tp53_mutation + "
            "treatment_tamoxifen + treatment_palbociclib + treatment_trastuzumab + "
            "treatment_olaparib + treatment_sacituzumab_govitecan + treatment_pembrolizumab")
m = smf.ols(big_form, data=df).fit()
print(m.summary().tables[1])
for k in [
    "age_years", "ecog_ps", "stage_iv", "has_brain_mets", "albumin_g_dl",
    "ldh_u_l", "nlr", "tumor_size_cm", "ki67_pct", "prior_lines_of_therapy",
    "er_positive", "her2_positive", "brca1_mutation", "msi_high",
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]:
    coef = float(m.params[k]); p = float(m.pvalues[k])
    add(23, f"h23_big_{k}",
        f"In a multivariable OLS for pfs_months adjusting for clinical and tumor covariates plus all treatments, the coefficient on {k} is non-zero (sign indicates direction).",
        "refined", f"coef({k})={coef:.4f}, p={p:.4g}", p, coef, p < 0.05,
        f"smf.ols('{big_form}', data=df).fit()")

# =============================================================================
# ITERATION 24 — Pembrolizumab × tumor mutational burden surrogates
# =============================================================================
print("\n=== ITER 24: Pembrolizumab response modifiers ===")
for mod in ["msi_high", "tp53_mutation", "smoking_pack_years"]:
    f = f"pfs_months ~ treatment_pembrolizumab * {mod}"
    m = smf.ols(f, data=df).fit()
    term = f"treatment_pembrolizumab:{mod}"
    coef = float(m.params[term]); p = float(m.pvalues[term])
    add(24, f"h24_pembro_{mod}",
        f"There is a positive treatment_pembrolizumab by {mod} interaction on pfs_months (greater benefit at higher {mod}).",
        "novel", f"interaction coef={coef:.4f}, p={p:.4g}", p, coef, p < 0.05, f)
    print(f"pembro*{mod}:", coef, p)

# =============================================================================
# ITERATION 25 — Final synthesis: rank treatments by adjusted PFS in target subgroup
# =============================================================================
print("\n=== ITER 25: Adjusted treatment effects within indicated subgroups + sanity ===")

def fit_in(tx, sub_expr, hid, htext):
    sub = df.query(sub_expr)
    f = f"pfs_months ~ {tx} + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + tumor_size_cm + nlr + crp_mg_l + prior_lines_of_therapy"
    m = smf.ols(f, data=sub).fit()
    coef = float(m.params[tx]); p = float(m.pvalues[tx])
    summary = f"{tx} in {sub_expr}: adj coef={coef:.4f}, p={p:.4g}, n={len(sub)}"
    print(summary)
    add(25, hid, htext, "refined", summary, p, coef, p < 0.05,
        f"smf.ols on subset {sub_expr}: {f}")

fit_in("treatment_tamoxifen", "er_positive == 1 and postmenopausal == 1", "h25_tam_erpost",
       "Within er_positive=1 and postmenopausal=1 patients, after multivariable adjustment, treatment_tamoxifen=1 is associated with longer pfs_months.")
fit_in("treatment_trastuzumab", "her2_positive == 1 or her2_amplification == 1", "h25_tras_her2any",
       "Within HER2+ or HER2-amplified patients, after adjustment, treatment_trastuzumab=1 is associated with longer pfs_months.")
fit_in("treatment_olaparib", "brca1_mutation == 1 or brca2_mutation == 1", "h25_ola_brca",
       "Within BRCA1/2-mutated patients, after adjustment, treatment_olaparib=1 is associated with longer pfs_months.")

# Save raw results
with open("results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n=== Saved {len(results)} analyses ===")
