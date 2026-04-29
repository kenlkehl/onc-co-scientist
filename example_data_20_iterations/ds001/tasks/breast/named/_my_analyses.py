"""
Iterative analysis of ds001_breast cohort.
Tests prognostic factors, treatment main effects, treatment-biomarker interactions,
lab/symptom correlates, demographic disparities, and pharmacogenomic SNP effects on PFS.
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
results = {}


def linreg(formula):
    model = smf.ols(formula, data=df).fit()
    return model


def t_test(group_col, outcome="pfs_months"):
    a = df.loc[df[group_col] == 1, outcome]
    b = df.loc[df[group_col] == 0, outcome]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(p), int(a.size), int(b.size)


def fmt(d):
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}


# =========================================================
# Iteration 1: ECOG performance status
# =========================================================
m = linreg("pfs_months ~ C(ecog_ps)")
mean_by_ecog = df.groupby("ecog_ps")["pfs_months"].mean().to_dict()
m_lin = linreg("pfs_months ~ ecog_ps")
results["it1_ecog"] = {
    "means": mean_by_ecog,
    "p_anova": float(m.f_pvalue),
    "linear_beta": float(m_lin.params["ecog_ps"]),
    "linear_p": float(m_lin.pvalues["ecog_ps"]),
}

# Iteration 1: stage IV
beta, p, n1, n0 = t_test("stage_iv")
results["it1_stage_iv"] = {"diff_pfs_mo": beta, "p": p, "n_pos": n1, "n_neg": n0}

# Iteration 1: brain mets
beta, p, n1, n0 = t_test("has_brain_mets")
results["it1_brain_mets"] = {"diff_pfs_mo": beta, "p": p, "n_pos": n1, "n_neg": n0}

# =========================================================
# Iteration 2: Other metastatic sites + tumor burden
# =========================================================
for c in ["liver_mets", "bone_mets", "adrenal_mets",
          "pleural_effusion", "pericardial_effusion", "contralateral_lung_mets",
          "node_positive"]:
    beta, p, n1, n0 = t_test(c)
    results[f"it2_{c}"] = {"diff_pfs_mo": beta, "p": p, "n_pos": n1, "n_neg": n0}

# Tumor size as continuous
m = linreg("pfs_months ~ tumor_size_cm")
results["it2_tumor_size"] = {
    "beta_per_cm": float(m.params["tumor_size_cm"]),
    "p": float(m.pvalues["tumor_size_cm"]),
}

# Ki67 as continuous
m = linreg("pfs_months ~ ki67_pct")
results["it2_ki67"] = {
    "beta_per_pct": float(m.params["ki67_pct"]),
    "p": float(m.pvalues["ki67_pct"]),
}

# =========================================================
# Iteration 3: Lab markers (albumin, LDH, CRP, hemoglobin, NLR)
# =========================================================
for c in ["albumin_g_dl", "ldh_u_l", "crp_mg_l", "hemoglobin_g_dl",
          "nlr", "weight_loss_pct_6mo", "alkaline_phosphatase_u_l",
          "calcium_mg_dl", "platelets_k_ul"]:
    m = linreg(f"pfs_months ~ {c}")
    results[f"it3_{c}"] = {"beta": float(m.params[c]), "p": float(m.pvalues[c])}

# =========================================================
# Iteration 4: Treatment main effects (each treatment vs not)
# =========================================================
treatments = [
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]
for t in treatments:
    beta, p, n1, n0 = t_test(t)
    results[f"it4_{t}"] = {"diff_pfs_mo": beta, "p": p, "n_pos": n1, "n_neg": n0}

# =========================================================
# Iteration 5: Trastuzumab x HER2-positive interaction
# =========================================================
m = linreg("pfs_months ~ treatment_trastuzumab * her2_positive")
results["it5_trastuzumab_her2"] = {
    "main_trastuzumab": float(m.params["treatment_trastuzumab"]),
    "main_her2": float(m.params["her2_positive"]),
    "interaction": float(m.params["treatment_trastuzumab:her2_positive"]),
    "p_interaction": float(m.pvalues["treatment_trastuzumab:her2_positive"]),
}
# Sub-group means
for her2 in [0, 1]:
    sub = df[df.her2_positive == her2]
    a = sub.loc[sub.treatment_trastuzumab == 1, "pfs_months"]
    b = sub.loc[sub.treatment_trastuzumab == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it5_trastuzumab_her2{her2}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

# =========================================================
# Iteration 6: Tamoxifen x ER-positive
# =========================================================
m = linreg("pfs_months ~ treatment_tamoxifen * er_positive")
results["it6_tamox_er"] = {
    "interaction": float(m.params["treatment_tamoxifen:er_positive"]),
    "p_interaction": float(m.pvalues["treatment_tamoxifen:er_positive"]),
}
for er in [0, 1]:
    sub = df[df.er_positive == er]
    a = sub.loc[sub.treatment_tamoxifen == 1, "pfs_months"]
    b = sub.loc[sub.treatment_tamoxifen == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it6_tamox_er{er}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

# =========================================================
# Iteration 7: Olaparib x BRCA1/2
# =========================================================
df["brca_any"] = ((df.brca1_mutation == 1) | (df.brca2_mutation == 1)).astype(int)
m = linreg("pfs_months ~ treatment_olaparib * brca_any")
results["it7_olap_brca"] = {
    "interaction": float(m.params["treatment_olaparib:brca_any"]),
    "p_interaction": float(m.pvalues["treatment_olaparib:brca_any"]),
}
for brca in [0, 1]:
    sub = df[df.brca_any == brca]
    a = sub.loc[sub.treatment_olaparib == 1, "pfs_months"]
    b = sub.loc[sub.treatment_olaparib == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it7_olap_brca{brca}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

# =========================================================
# Iteration 8: Palbociclib x ER+ postmenopausal
# =========================================================
df["er_postmen"] = ((df.er_positive == 1) & (df.postmenopausal == 1)).astype(int)
m = linreg("pfs_months ~ treatment_palbociclib * er_postmen")
results["it8_palbo_erpm"] = {
    "interaction": float(m.params["treatment_palbociclib:er_postmen"]),
    "p_interaction": float(m.pvalues["treatment_palbociclib:er_postmen"]),
}
for grp in [0, 1]:
    sub = df[df.er_postmen == grp]
    a = sub.loc[sub.treatment_palbociclib == 1, "pfs_months"]
    b = sub.loc[sub.treatment_palbociclib == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it8_palbo_erpm{grp}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

# Also test palbociclib in ER+ regardless of menopausal status
for er in [0, 1]:
    sub = df[df.er_positive == er]
    a = sub.loc[sub.treatment_palbociclib == 1, "pfs_months"]
    b = sub.loc[sub.treatment_palbociclib == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it8_palbo_er{er}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p),
        }

# =========================================================
# Iteration 9: Pembrolizumab x MSI-high; pembro x triple-negative
# =========================================================
df["tnbc"] = ((df.er_positive == 0) & (df.pr_positive == 0) & (df.her2_positive == 0)).astype(int)
m = linreg("pfs_months ~ treatment_pembrolizumab * msi_high")
results["it9_pembro_msi"] = {
    "interaction": float(m.params["treatment_pembrolizumab:msi_high"]),
    "p_interaction": float(m.pvalues["treatment_pembrolizumab:msi_high"]),
}
for msi in [0, 1]:
    sub = df[df.msi_high == msi]
    a = sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"]
    b = sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it9_pembro_msi{msi}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

m = linreg("pfs_months ~ treatment_pembrolizumab * tnbc")
results["it9_pembro_tnbc"] = {
    "interaction": float(m.params["treatment_pembrolizumab:tnbc"]),
    "p_interaction": float(m.pvalues["treatment_pembrolizumab:tnbc"]),
}
for grp in [0, 1]:
    sub = df[df.tnbc == grp]
    a = sub.loc[sub.treatment_pembrolizumab == 1, "pfs_months"]
    b = sub.loc[sub.treatment_pembrolizumab == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it9_pembro_tnbc{grp}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

# =========================================================
# Iteration 10: Sacituzumab govitecan in TNBC vs other / HER2-low
# =========================================================
m = linreg("pfs_months ~ treatment_sacituzumab_govitecan * tnbc")
results["it10_sg_tnbc"] = {
    "interaction": float(m.params["treatment_sacituzumab_govitecan:tnbc"]),
    "p_interaction": float(m.pvalues["treatment_sacituzumab_govitecan:tnbc"]),
}
for grp in [0, 1]:
    sub = df[df.tnbc == grp]
    a = sub.loc[sub.treatment_sacituzumab_govitecan == 1, "pfs_months"]
    b = sub.loc[sub.treatment_sacituzumab_govitecan == 0, "pfs_months"]
    if len(a) > 5 and len(b) > 5:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        results[f"it10_sg_tnbc{grp}"] = {
            "mean_on": float(a.mean()), "mean_off": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "p": float(p), "n_on": int(a.size), "n_off": int(b.size),
        }

m = linreg("pfs_months ~ treatment_sacituzumab_govitecan * her2_low")
results["it10_sg_her2low"] = {
    "interaction": float(m.params["treatment_sacituzumab_govitecan:her2_low"]),
    "p_interaction": float(m.pvalues["treatment_sacituzumab_govitecan:her2_low"]),
}

# =========================================================
# Iteration 11: PIK3CA mutation effect
# =========================================================
beta, p, n1, n0 = t_test("pik3ca_mutation")
results["it11_pik3ca"] = {"diff": beta, "p": p, "n_pos": n1, "n_neg": n0}

# Iteration 11: TP53 mutation effect
beta, p, n1, n0 = t_test("tp53_mutation")
results["it11_tp53"] = {"diff": beta, "p": p, "n_pos": n1, "n_neg": n0}

# =========================================================
# Iteration 12: Symptom burden (fatigue, pain, dyspnea, appetite loss, cough)
# =========================================================
for c in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "appetite_loss_grade", "cough_grade"]:
    m = linreg(f"pfs_months ~ {c}")
    results[f"it12_{c}"] = {"beta": float(m.params[c]), "p": float(m.pvalues[c])}

# =========================================================
# Iteration 13: Comorbidities
# =========================================================
for c in ["diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
          "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
          "venous_thromboembolism_history", "autoimmune_disease",
          "depression_anxiety_diagnosis", "prior_malignancy"]:
    beta, p, n1, n0 = t_test(c)
    results[f"it13_{c}"] = {"diff": beta, "p": p, "n_pos": n1, "n_neg": n0}

# =========================================================
# Iteration 14: Age & prior treatment lines
# =========================================================
m = linreg("pfs_months ~ age_years")
results["it14_age"] = {"beta_per_year": float(m.params["age_years"]), "p": float(m.pvalues["age_years"])}

m = linreg("pfs_months ~ prior_lines_of_therapy")
results["it14_prior_lines"] = {
    "beta_per_line": float(m.params["prior_lines_of_therapy"]),
    "p": float(m.pvalues["prior_lines_of_therapy"]),
}

m = linreg("pfs_months ~ years_since_diagnosis")
results["it14_years_since_dx"] = {
    "beta_per_year": float(m.params["years_since_diagnosis"]),
    "p": float(m.pvalues["years_since_diagnosis"]),
}

# =========================================================
# Iteration 15: Race/ethnicity, insurance, rurality, education
# =========================================================
m = linreg("pfs_months ~ C(race_ethnicity)")
results["it15_race"] = {"p_anova": float(m.f_pvalue), "n_levels": df.race_ethnicity.nunique()}

m = linreg("pfs_months ~ C(insurance_type)")
results["it15_insurance"] = {"p_anova": float(m.f_pvalue), "n_levels": df.insurance_type.nunique()}

beta, p, n1, n0 = t_test("rural_residence")
results["it15_rural"] = {"diff": beta, "p": p}

m = linreg("pfs_months ~ education_years")
results["it15_education"] = {"beta_per_year": float(m.params["education_years"]), "p": float(m.pvalues["education_years"])}

# =========================================================
# Iteration 16: Multivariable model with major prognostic factors
# =========================================================
m = linreg(
    "pfs_months ~ ecog_ps + stage_iv + has_brain_mets + liver_mets + bone_mets + "
    "albumin_g_dl + ldh_u_l + crp_mg_l + age_years + prior_lines_of_therapy"
)
results["it16_multivar"] = {
    "params": {k: float(v) for k, v in m.params.items()},
    "pvalues": {k: float(v) for k, v in m.pvalues.items()},
    "rsquared": float(m.rsquared),
}

# =========================================================
# Iteration 17: Pharmacogenomics — CYP2D6 SNPs (rs1065852, rs3813867) x tamoxifen
# These are CYP2D6 loss-of-function variants; expected to reduce tamoxifen efficacy
# =========================================================
for snp in ["snp_rs1065852", "snp_rs3813867", "snp_rs1800566"]:
    m = linreg(f"pfs_months ~ treatment_tamoxifen * {snp}")
    coef = f"treatment_tamoxifen:{snp}"
    if coef in m.params:
        results[f"it17_tamox_x_{snp}"] = {
            "interaction_beta": float(m.params[coef]),
            "p_interaction": float(m.pvalues[coef]),
        }

# =========================================================
# Iteration 18: SNP main effects on PFS (multiple-testing aware)
# =========================================================
snps = [c for c in df.columns if c.startswith("snp_")]
snp_main = []
for s in snps:
    m = linreg(f"pfs_months ~ {s}")
    snp_main.append((s, float(m.params[s]), float(m.pvalues[s])))
snp_main.sort(key=lambda x: x[2])
results["it18_snp_main_top"] = [
    {"snp": s, "beta": b, "p": p, "bonferroni_sig": p * len(snps) < 0.05}
    for s, b, p in snp_main[:10]
]
results["it18_snp_main_bottom_p"] = float(snp_main[-1][2])
results["it18_snp_main_minp"] = float(snp_main[0][2])
results["it18_n_snps"] = len(snps)

# =========================================================
# Iteration 19: BRCA1 and BRCA2 separately x olaparib
# =========================================================
for col in ["brca1_mutation", "brca2_mutation"]:
    m = linreg(f"pfs_months ~ treatment_olaparib * {col}")
    coef = f"treatment_olaparib:{col}"
    results[f"it19_olap_x_{col}"] = {
        "interaction_beta": float(m.params[coef]),
        "p_interaction": float(m.pvalues[coef]),
    }
    for grp in [0, 1]:
        sub = df[df[col] == grp]
        a = sub.loc[sub.treatment_olaparib == 1, "pfs_months"]
        b = sub.loc[sub.treatment_olaparib == 0, "pfs_months"]
        if len(a) > 5 and len(b) > 5:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            results[f"it19_olap_{col}{grp}"] = {
                "mean_on": float(a.mean()), "mean_off": float(b.mean()),
                "diff": float(a.mean() - b.mean()), "p": float(p),
                "n_on": int(a.size), "n_off": int(b.size),
            }

# =========================================================
# Iteration 20: Postmenopausal main effect, tumor burden composite
# =========================================================
beta, p, n1, n0 = t_test("postmenopausal")
results["it20_postmen"] = {"diff": beta, "p": p}

beta, p, n1, n0 = t_test("er_positive")
results["it20_er"] = {"diff": beta, "p": p}

beta, p, n1, n0 = t_test("pr_positive")
results["it20_pr"] = {"diff": beta, "p": p}

beta, p, n1, n0 = t_test("her2_positive")
results["it20_her2"] = {"diff": beta, "p": p}

beta, p, n1, n0 = t_test("her2_low")
results["it20_her2_low"] = {"diff": beta, "p": p}

beta, p, n1, n0 = t_test("tnbc")
results["it20_tnbc"] = {"diff": beta, "p": p}

# =========================================================
# Iteration 21: Vital signs (heart rate, SpO2, BP)
# =========================================================
for c in ["heart_rate_bpm", "spo2_pct", "systolic_bp_mmhg", "diastolic_bp_mmhg", "bmi"]:
    m = linreg(f"pfs_months ~ {c}")
    results[f"it21_{c}"] = {"beta": float(m.params[c]), "p": float(m.pvalues[c])}

# =========================================================
# Iteration 22: Liver/kidney function effects
# =========================================================
for c in ["ast_u_l", "alt_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl",
          "bun_mg_dl", "sodium_meq_l", "potassium_meq_l", "glucose_mg_dl"]:
    m = linreg(f"pfs_months ~ {c}")
    results[f"it22_{c}"] = {"beta": float(m.params[c]), "p": float(m.pvalues[c])}

# =========================================================
# Iteration 23: Tumor markers (CEA, CA-125), hematologic indices
# =========================================================
for c in ["cea_ng_ml", "ca_125_u_ml", "wbc_k_ul", "anc_k_ul", "alc_k_ul", "inr"]:
    m = linreg(f"pfs_months ~ {c}")
    results[f"it23_{c}"] = {"beta": float(m.params[c]), "p": float(m.pvalues[c])}

# =========================================================
# Iteration 24: Trastuzumab x HER2-low (off-label?), Palbociclib x HER2 (negative selection)
# =========================================================
m = linreg("pfs_months ~ treatment_trastuzumab * her2_low")
results["it24_tras_her2low"] = {
    "interaction": float(m.params["treatment_trastuzumab:her2_low"]),
    "p_interaction": float(m.pvalues["treatment_trastuzumab:her2_low"]),
}

# Iteration 24: HER2-amplified (separate from her2_positive)
beta, p, n1, n0 = t_test("her2_amplification")
results["it24_her2_ampl"] = {"diff": beta, "p": p, "n_pos": n1, "n_neg": n0}

m = linreg("pfs_months ~ treatment_trastuzumab * her2_amplification")
coef = "treatment_trastuzumab:her2_amplification"
results["it24_tras_her2ampl"] = {
    "interaction": float(m.params[coef]),
    "p_interaction": float(m.pvalues[coef]),
}

# =========================================================
# Iteration 25: Multivariable PFS model adjusting for treatments and biomarker matches
# =========================================================
m = linreg(
    "pfs_months ~ ecog_ps + stage_iv + has_brain_mets + liver_mets + albumin_g_dl + ldh_u_l + "
    "treatment_trastuzumab * her2_positive + treatment_tamoxifen * er_positive + "
    "treatment_olaparib * brca_any + treatment_palbociclib * er_postmen + "
    "treatment_pembrolizumab * msi_high"
)
results["it25_full"] = {
    "interactions": {
        "tras_her2": float(m.params.get("treatment_trastuzumab:her2_positive", np.nan)),
        "p_tras_her2": float(m.pvalues.get("treatment_trastuzumab:her2_positive", np.nan)),
        "tamox_er": float(m.params.get("treatment_tamoxifen:er_positive", np.nan)),
        "p_tamox_er": float(m.pvalues.get("treatment_tamoxifen:er_positive", np.nan)),
        "olap_brca": float(m.params.get("treatment_olaparib:brca_any", np.nan)),
        "p_olap_brca": float(m.pvalues.get("treatment_olaparib:brca_any", np.nan)),
        "palbo_erpm": float(m.params.get("treatment_palbociclib:er_postmen", np.nan)),
        "p_palbo_erpm": float(m.pvalues.get("treatment_palbociclib:er_postmen", np.nan)),
        "pembro_msi": float(m.params.get("treatment_pembrolizumab:msi_high", np.nan)),
        "p_pembro_msi": float(m.pvalues.get("treatment_pembrolizumab:msi_high", np.nan)),
    },
    "rsquared": float(m.rsquared),
    "n": int(m.nobs),
}

# Save raw results
with open("_my_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("Done. Results saved to _my_results.json")
print(f"Total result keys: {len(results)}")
