"""Iterative hypothesis-testing analysis for ds001_prostate.

Outputs detailed result records that we use to populate transcript.json
and analysis_summary.txt.
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
N = len(df)

OUT = []  # list of dicts: {iter, hyp_id, text, code, summary, p, eff, sig, kind}

def record(it, hid, htext, code, summary, p, eff, kind="novel"):
    sig = (p is not None and not (isinstance(p, float) and np.isnan(p)) and p < 0.05)
    OUT.append({
        "iter": it,
        "hid": hid,
        "text": htext,
        "code": code,
        "summary": summary,
        "p": p,
        "eff": eff,
        "sig": sig,
        "kind": kind,
    })

def ttest_mean_diff(group1_mask, group0_mask, y_col):
    y1 = df.loc[group1_mask, y_col]
    y0 = df.loc[group0_mask, y_col]
    if len(y1) < 2 or len(y0) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(y1), len(y0)
    t, p = stats.ttest_ind(y1, y0, equal_var=False)
    return float(y1.mean() - y0.mean()), float(p), float(y1.mean()), float(y0.mean()), len(y1), len(y0)

def report_coef(m, term):
    return float(m.params[term]), float(m.pvalues[term])

# =====================================================================
# Iteration 1: Main effects of cancer treatments on pfs_months
# =====================================================================
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    diff, p, m1, m0, n1, n0 = ttest_mean_diff(df[tx]==1, df[tx]==0, "pfs_months")
    hid = f"h1_{tx[10:]}"
    record(1, hid,
           f"Mean pfs_months differs between patients receiving {tx} (=1) and those not (=0); we hypothesize a positive treatment effect (longer PFS on treatment) in this unselected aggregate analysis.",
           f"stats.ttest_ind(df.loc[df['{tx}']==1,'pfs_months'], df.loc[df['{tx}']==0,'pfs_months'], equal_var=False)",
           f"On {tx}: mean PFS={m1:.3f} (n={n1}); off: {m0:.3f} (n={n0}); diff={diff:+.3f} months; Welch t-test p={p:.4g}.",
           p, diff)

# =====================================================================
# Iteration 2: Pre-specified predictive biomarker × treatment interactions
# =====================================================================
def interaction_test(tx, marker):
    """Returns (interaction coef, p-value, eff in marker+, p in marker+, eff in marker-, p in marker-)."""
    f = f"pfs_months ~ {tx} * {marker}"
    m = smf.ols(f, data=df).fit()
    int_term = f"{tx}:{marker}"
    coef = float(m.params[int_term])
    p = float(m.pvalues[int_term])
    eff_pos, p_pos, _, _, _, _ = ttest_mean_diff((df[tx]==1)&(df[marker]==1),
                                                   (df[tx]==0)&(df[marker]==1),
                                                   "pfs_months")
    eff_neg, p_neg, _, _, _, _ = ttest_mean_diff((df[tx]==1)&(df[marker]==0),
                                                   (df[tx]==0)&(df[marker]==0),
                                                   "pfs_months")
    return coef, p, eff_pos, p_pos, eff_neg, p_neg

# olaparib × BRCA2
coef, p, ep, pp, en, pn = interaction_test("treatment_olaparib", "brca2_mutation")
record(2, "h2_olaparib_brca2",
       "The PFS benefit of treatment_olaparib is greater (more positive) in brca2_mutation=1 patients than in brca2_mutation=0 patients (positive interaction; PARP inhibitors target HR-deficient disease).",
       "smf.ols('pfs_months ~ treatment_olaparib * brca2_mutation', data=df).fit()",
       f"BRCA2+ subgroup olaparib effect={ep:+.3f} mo (p={pp:.3g}); BRCA2- subgroup olaparib effect={en:+.3f} mo (p={pn:.3g}); interaction coef={coef:+.3f} (p={p:.3g}).",
       p, coef)

# pembrolizumab × MSI
coef, p, ep, pp, en, pn = interaction_test("treatment_pembrolizumab", "msi_high")
record(2, "h2_pembro_msi",
       "The PFS benefit of treatment_pembrolizumab is greater (more positive) in msi_high=1 patients than in msi_high=0 patients (positive interaction; PD-1 blockade benefits MSI-high tumors).",
       "smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df).fit()",
       f"MSI-high subgroup pembrolizumab effect={ep:+.3f} mo (p={pp:.3g}); MSS subgroup effect={en:+.3f} mo (p={pn:.3g}); interaction coef={coef:+.3f} (p={p:.3g}).",
       p, coef)

# lu177-PSMA × psma_high
coef, p, ep, pp, en, pn = interaction_test("treatment_lu177_psma", "psma_high")
record(2, "h2_lu177_psma",
       "The PFS benefit of treatment_lu177_psma is greater (more positive) in psma_high=1 patients than in psma_high=0 patients (positive interaction; the radioligand depends on PSMA expression).",
       "smf.ols('pfs_months ~ treatment_lu177_psma * psma_high', data=df).fit()",
       f"PSMA-high subgroup lu177 effect={ep:+.3f} mo (p={pp:.3g}); PSMA-low subgroup effect={en:+.3f} mo (p={pn:.3g}); interaction coef={coef:+.3f} (p={p:.3g}).",
       p, coef)

# enzalutamide × AR-V7 — expected NEGATIVE interaction
coef, p, ep, pp, en, pn = interaction_test("treatment_enzalutamide", "ar_v7_positive")
record(2, "h2_enza_arv7",
       "The PFS effect of treatment_enzalutamide is more negative in ar_v7_positive=1 patients than in ar_v7_positive=0 patients (negative interaction; AR-V7 splice variant confers resistance to AR-targeted therapy).",
       "smf.ols('pfs_months ~ treatment_enzalutamide * ar_v7_positive', data=df).fit()",
       f"AR-V7+ subgroup enzalutamide effect={ep:+.3f} mo (p={pp:.3g}); AR-V7- subgroup effect={en:+.3f} mo (p={pn:.3g}); interaction coef={coef:+.3f} (p={p:.3g}).",
       p, coef)

# abiraterone × AR-V7
coef, p, ep, pp, en, pn = interaction_test("treatment_abiraterone", "ar_v7_positive")
record(2, "h2_abi_arv7",
       "The PFS effect of treatment_abiraterone is more negative in ar_v7_positive=1 patients than in ar_v7_positive=0 patients (negative interaction).",
       "smf.ols('pfs_months ~ treatment_abiraterone * ar_v7_positive', data=df).fit()",
       f"AR-V7+ abi effect={ep:+.3f} mo (p={pp:.3g}); AR-V7- abi effect={en:+.3f} mo (p={pn:.3g}); interaction coef={coef:+.3f} (p={p:.3g}).",
       p, coef)

# =====================================================================
# Iteration 3: Marker / disease-burden main effects
# =====================================================================
for marker, hid_suf, txt in [
    ("brca2_mutation", "brca2",
     "brca2_mutation=1 patients have different mean pfs_months than brca2_mutation=0 patients (direction unclear unadjusted: aggressive HR-deficient biology vs olaparib responders)."),
    ("ar_v7_positive", "arv7",
     "ar_v7_positive=1 patients have shorter pfs_months than ar_v7_positive=0 because the splice variant marks treatment-refractory disease."),
    ("msi_high", "msi",
     "msi_high=1 patients have different pfs_months than msi_high=0 patients (direction uncertain in unadjusted comparison)."),
    ("psma_high", "psma",
     "psma_high=1 patients have different pfs_months than psma_high=0 patients."),
    ("mcrpc", "mcrpc",
     "Patients with mcrpc=1 have shorter pfs_months than mcrpc=0 patients."),
    ("visceral_mets", "vismets",
     "Patients with visceral_mets=1 have shorter pfs_months than visceral_mets=0."),
    ("liver_mets", "livmets",
     "Patients with liver_mets=1 have shorter pfs_months than liver_mets=0."),
    ("bone_mets", "bonemets",
     "Patients with bone_mets=1 have shorter pfs_months than bone_mets=0."),
    ("adrenal_mets", "admets",
     "Patients with adrenal_mets=1 have shorter pfs_months than adrenal_mets=0."),
    ("pleural_effusion", "pleff",
     "Patients with pleural_effusion=1 have shorter pfs_months than =0."),
    ("pericardial_effusion", "pereff",
     "Patients with pericardial_effusion=1 have shorter pfs_months than =0."),
]:
    diff, p, m1, m0, n1, n0 = ttest_mean_diff(df[marker]==1, df[marker]==0, "pfs_months")
    record(3, f"h3_{hid_suf}", txt,
           f"stats.ttest_ind by {marker}",
           f"{marker}+ mean={m1:.3f} (n={n1}); {marker}- mean={m0:.3f} (n={n0}); diff={diff:+.3f}; Welch p={p:.3g}.",
           p, diff)

# =====================================================================
# Iteration 4: Continuous prognostic main effects (univariate OLS)
# =====================================================================
cont_hyps = [
    ("age_years", "age", "Older age_years is associated with shorter pfs_months (negative slope)."),
    ("ecog_ps", "ecog", "Higher ecog_ps is associated with shorter pfs_months (negative slope)."),
    ("psa_ng_ml", "psa", "Higher psa_ng_ml is associated with shorter pfs_months (negative slope; high PSA marks burden)."),
    ("gleason_score", "gleason", "Higher gleason_score is associated with shorter pfs_months (negative slope)."),
    ("albumin_g_dl", "alb", "Higher albumin_g_dl is associated with longer pfs_months (positive slope)."),
    ("ldh_u_l", "ldh", "Higher ldh_u_l is associated with shorter pfs_months (negative slope; tumor burden)."),
    ("hemoglobin_g_dl", "hgb", "Higher hemoglobin_g_dl is associated with longer pfs_months (positive slope)."),
    ("alkaline_phosphatase_u_l", "alp", "Higher alkaline_phosphatase_u_l is associated with shorter pfs_months (bone disease)."),
    ("crp_mg_l", "crp", "Higher crp_mg_l is associated with shorter pfs_months (inflammation)."),
    ("nlr", "nlr", "Higher nlr is associated with shorter pfs_months (inflammation/immune surveillance)."),
    ("weight_loss_pct_6mo", "wtloss", "Higher weight_loss_pct_6mo is associated with shorter pfs_months."),
    ("pain_nrs", "pain", "Higher pain_nrs is associated with shorter pfs_months."),
    ("fatigue_grade", "fatigue", "Higher fatigue_grade is associated with shorter pfs_months."),
    ("appetite_loss_grade", "appetite", "Higher appetite_loss_grade is associated with shorter pfs_months."),
    ("dyspnea_grade", "dyspnea", "Higher dyspnea_grade is associated with shorter pfs_months."),
    ("cea_ng_ml", "cea", "Higher cea_ng_ml is associated with shorter pfs_months."),
    ("calcium_mg_dl", "ca", "Higher calcium_mg_dl is associated with shorter pfs_months (hypercalcemia)."),
    ("creatinine_mg_dl", "creat", "Higher creatinine_mg_dl is associated with shorter pfs_months."),
]
for col, hid_suf, txt in cont_hyps:
    m = smf.ols(f"pfs_months ~ {col}", data=df).fit()
    c, p = report_coef(m, col)
    record(4, f"h4_{hid_suf}", txt,
           f"smf.ols('pfs_months ~ {col}', data=df).fit()",
           f"OLS slope on {col}: beta={c:+.4g} months/unit; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 5: Comorbidity main effects
# =====================================================================
for col, hid_suf, txt in [
    ("diabetes_mellitus", "dm", "diabetes_mellitus=1 is associated with shorter pfs_months."),
    ("hypertension", "htn", "hypertension=1 is associated with different pfs_months than =0."),
    ("chronic_kidney_disease", "ckd", "chronic_kidney_disease=1 is associated with shorter pfs_months."),
    ("heart_failure", "hf", "heart_failure=1 is associated with shorter pfs_months."),
    ("coronary_artery_disease", "cad", "coronary_artery_disease=1 is associated with shorter pfs_months."),
    ("copd", "copd", "copd=1 is associated with shorter pfs_months."),
    ("atrial_fibrillation", "afib", "atrial_fibrillation=1 is associated with shorter pfs_months."),
    ("venous_thromboembolism_history", "vte", "venous_thromboembolism_history=1 is associated with shorter pfs_months."),
    ("autoimmune_disease", "autoimmune", "autoimmune_disease=1 is associated with different pfs_months."),
    ("hepatitis_c_history", "hcv", "hepatitis_c_history=1 is associated with shorter pfs_months."),
    ("hiv_positive", "hiv", "hiv_positive=1 is associated with shorter pfs_months."),
    ("prior_malignancy", "pm", "prior_malignancy=1 is associated with shorter pfs_months."),
    ("depression_anxiety_diagnosis", "depanx", "depression_anxiety_diagnosis=1 is associated with shorter pfs_months."),
    ("interstitial_lung_disease_history", "ild", "interstitial_lung_disease_history=1 is associated with shorter pfs_months."),
]:
    diff, p, m1, m0, n1, n0 = ttest_mean_diff(df[col]==1, df[col]==0, "pfs_months")
    record(5, f"h5_{hid_suf}", txt,
           f"ttest by {col}",
           f"{col}=1 mean={m1:.3f} (n={n1}); =0 mean={m0:.3f} (n={n0}); diff={diff:+.3f}; p={p:.3g}.",
           p, diff)

# =====================================================================
# Iteration 6: SDOH and demographics
# =====================================================================
m = smf.ols("pfs_months ~ C(race_ethnicity, Treatment(reference='white'))", data=df).fit()
for level in ['black', 'hispanic', 'asian', 'other']:
    term = f"C(race_ethnicity, Treatment(reference='white'))[T.{level}]"
    if term in m.params:
        c, p = float(m.params[term]), float(m.pvalues[term])
        record(6, f"h6_race_{level}",
               f"Mean pfs_months in race_ethnicity={level} differs from race_ethnicity=white (reference).",
               "OLS pfs_months ~ race_ethnicity (white reference)",
               f"race_ethnicity={level} vs white: diff={c:+.3f} months; p={p:.3g}.",
               p, c)

m = smf.ols("pfs_months ~ C(insurance_type, Treatment(reference='private'))", data=df).fit()
for level in ['medicare', 'medicaid', 'uninsured']:
    term = f"C(insurance_type, Treatment(reference='private'))[T.{level}]"
    if term in m.params:
        c, p = float(m.params[term]), float(m.pvalues[term])
        record(6, f"h6_ins_{level}",
               f"Mean pfs_months in insurance_type={level} differs from insurance_type=private (reference). Hypothesis: medicaid/uninsured shorter PFS due to access disparities.",
               "OLS pfs_months ~ insurance_type (private reference)",
               f"insurance_type={level} vs private: diff={c:+.3f} months; p={p:.3g}.",
               p, c)

diff, p, m1, m0, n1, n0 = ttest_mean_diff(df["rural_residence"]==1, df["rural_residence"]==0, "pfs_months")
record(6, "h6_rural",
       "Rural residents (rural_residence=1) have shorter pfs_months than urban (rural_residence=0).",
       "ttest by rural_residence",
       f"rural=1 mean={m1:.3f} (n={n1}); =0 mean={m0:.3f} (n={n0}); diff={diff:+.3f}; p={p:.3g}.",
       p, diff)

m = smf.ols("pfs_months ~ smoking_pack_years", data=df).fit()
c, p = report_coef(m, "smoking_pack_years")
record(6, "h6_smoking",
       "Higher smoking_pack_years is associated with shorter pfs_months.",
       "smf.ols('pfs_months ~ smoking_pack_years', data=df).fit()",
       f"OLS slope: beta={c:+.4g}; p={p:.3g}.",
       p, c)

m = smf.ols("pfs_months ~ education_years", data=df).fit()
c, p = report_coef(m, "education_years")
record(6, "h6_education",
       "Higher education_years is associated with longer pfs_months (positive slope).",
       "smf.ols('pfs_months ~ education_years', data=df).fit()",
       f"OLS slope: beta={c:+.4g}; p={p:.3g}.",
       p, c)

# =====================================================================
# Iteration 7: Prior-treatment exposure
# =====================================================================
for col, hid_suf, txt in [
    ("prior_lines_of_therapy", "lines", "More prior_lines_of_therapy is associated with shorter pfs_months (negative slope)."),
    ("years_since_diagnosis", "ysd", "Longer years_since_diagnosis is associated with different pfs_months."),
    ("prior_chemotherapy", "prior_chemo", "prior_chemotherapy=1 is associated with shorter pfs_months."),
    ("prior_radiation", "prior_rad", "prior_radiation=1 is associated with different pfs_months."),
    ("prior_surgery", "prior_surg", "prior_surgery=1 is associated with different pfs_months."),
    ("prior_immunotherapy", "prior_io", "prior_immunotherapy=1 is associated with shorter pfs_months."),
    ("prior_targeted_therapy", "prior_tt", "prior_targeted_therapy=1 is associated with shorter pfs_months."),
]:
    if df[col].nunique() == 2:
        diff, p, m1, m0, n1, n0 = ttest_mean_diff(df[col]==1, df[col]==0, "pfs_months")
        record(7, f"h7_{hid_suf}", txt,
               f"ttest by {col}",
               f"{col}=1 mean={m1:.3f} (n={n1}); =0 mean={m0:.3f} (n={n0}); diff={diff:+.3f}; p={p:.3g}.",
               p, diff)
    else:
        m = smf.ols(f"pfs_months ~ {col}", data=df).fit()
        c, p = report_coef(m, col)
        record(7, f"h7_{hid_suf}", txt,
               f"smf.ols('pfs_months ~ {col}')",
               f"OLS slope: beta={c:+.4g}; p={p:.3g}.",
               p, c)

# =====================================================================
# Iteration 8: SNP main-effect screen
# =====================================================================
snp_cols = [c for c in df.columns if c.startswith("snp_")]
for snp in snp_cols:
    m = smf.ols(f"pfs_months ~ {snp}", data=df).fit()
    c, p = report_coef(m, snp)
    record(8, f"h8_{snp}",
           f"The dosage of {snp} (0/1/2 alleles) is associated with pfs_months under an additive model.",
           f"smf.ols('pfs_months ~ {snp}', data=df).fit()",
           f"OLS slope: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 9: Tumor co-mutation main effects
# =====================================================================
for col, hid_suf, txt in [
    ("tp53_mutation", "tp53", "tp53_mutation=1 is associated with shorter pfs_months."),
    ("pten_loss", "pten", "pten_loss=1 is associated with shorter pfs_months."),
    ("cdkn2a_loss", "cdkn2a", "cdkn2a_loss=1 is associated with shorter pfs_months."),
    ("pik3ca_mutation", "pik3ca", "pik3ca_mutation=1 is associated with different pfs_months."),
    ("her2_amplification", "her2", "her2_amplification=1 is associated with shorter pfs_months."),
    ("braf_v600e", "braf", "braf_v600e=1 is associated with different pfs_months."),
    ("fgfr_alteration", "fgfr", "fgfr_alteration=1 is associated with different pfs_months."),
    ("met_exon14_skipping", "met", "met_exon14_skipping=1 is associated with different pfs_months."),
    ("ret_fusion", "ret", "ret_fusion=1 is associated with different pfs_months."),
    ("ros1_fusion", "ros1", "ros1_fusion=1 is associated with different pfs_months."),
    ("ntrk_fusion", "ntrk", "ntrk_fusion=1 is associated with different pfs_months."),
    ("nrg1_fusion", "nrg1", "nrg1_fusion=1 is associated with different pfs_months."),
    ("keap1_mutation", "keap1", "keap1_mutation=1 is associated with different pfs_months."),
]:
    diff, p, m1, m0, n1, n0 = ttest_mean_diff(df[col]==1, df[col]==0, "pfs_months")
    record(9, f"h9_{hid_suf}", txt,
           f"ttest by {col}",
           f"{col}=1 mean={m1:.3f} (n={n1}); =0 mean={m0:.3f} (n={n0}); diff={diff:+.3f}; p={p:.3g}.",
           p, diff)

# =====================================================================
# Iteration 10: Secondary labs and vitals
# =====================================================================
for col, hid_suf, txt in [
    ("ast_u_l", "ast", "Higher ast_u_l is associated with shorter pfs_months."),
    ("alt_u_l", "alt", "Higher alt_u_l is associated with shorter pfs_months."),
    ("total_bilirubin_mg_dl", "tbili", "Higher total_bilirubin_mg_dl is associated with shorter pfs_months."),
    ("bun_mg_dl", "bun", "Higher bun_mg_dl is associated with shorter pfs_months."),
    ("sodium_meq_l", "na", "Higher sodium_meq_l (less hyponatremia) is associated with longer pfs_months."),
    ("potassium_meq_l", "k", "potassium_meq_l is associated with pfs_months."),
    ("glucose_mg_dl", "glu", "Higher glucose_mg_dl is associated with shorter pfs_months."),
    ("platelets_k_ul", "plt", "platelets_k_ul is associated with pfs_months."),
    ("wbc_k_ul", "wbc", "Higher wbc_k_ul is associated with shorter pfs_months."),
    ("anc_k_ul", "anc", "Higher anc_k_ul is associated with shorter pfs_months."),
    ("alc_k_ul", "alc", "Higher alc_k_ul is associated with longer pfs_months."),
    ("ca_125_u_ml", "ca125", "Higher ca_125_u_ml is associated with shorter pfs_months."),
    ("tsh_uiu_ml", "tsh", "tsh_uiu_ml is associated with pfs_months."),
    ("inr", "inr", "Higher inr is associated with shorter pfs_months."),
    ("bmi", "bmi", "Higher bmi is associated with longer pfs_months."),
    ("systolic_bp_mmhg", "sbp", "systolic_bp_mmhg is associated with pfs_months."),
    ("diastolic_bp_mmhg", "dbp", "diastolic_bp_mmhg is associated with pfs_months."),
    ("heart_rate_bpm", "hr", "Higher heart_rate_bpm is associated with shorter pfs_months."),
    ("spo2_pct", "spo2", "Higher spo2_pct is associated with longer pfs_months."),
]:
    m = smf.ols(f"pfs_months ~ {col}", data=df).fit()
    c, p = report_coef(m, col)
    record(10, f"h10_{hid_suf}", txt,
           f"smf.ols('pfs_months ~ {col}')",
           f"OLS slope: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 11: ECOG × treatment interactions
# =====================================================================
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * ecog_ps", data=df).fit()
    term = f"{tx}:ecog_ps"
    c, p = report_coef(m, term)
    record(11, f"h11_{tx[10:]}_ecog",
           f"The PFS effect of {tx} depends on ecog_ps (interaction). Hypothesized: more negative interaction for docetaxel (chemo poorly tolerated at higher ECOG).",
           f"smf.ols('pfs_months ~ {tx} * ecog_ps')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 12: visceral_mets × treatment interactions
# =====================================================================
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * visceral_mets", data=df).fit()
    term = f"{tx}:visceral_mets"
    c, p = report_coef(m, term)
    record(12, f"h12_{tx[10:]}_visceral",
           f"The PFS effect of {tx} differs by visceral_mets status (interaction).",
           f"smf.ols('pfs_months ~ {tx} * visceral_mets')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 13: mcrpc × treatment interactions
# =====================================================================
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * mcrpc", data=df).fit()
    term = f"{tx}:mcrpc"
    c, p = report_coef(m, term)
    record(13, f"h13_{tx[10:]}_mcrpc",
           f"The PFS effect of {tx} differs by mcrpc status (interaction).",
           f"smf.ols('pfs_months ~ {tx} * mcrpc')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 14: Adjusted multivariable prognostic + treatment model
# =====================================================================
formula14 = ("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + liver_mets + bone_mets "
             "+ albumin_g_dl + ldh_u_l + hemoglobin_g_dl + alkaline_phosphatase_u_l "
             "+ psa_ng_ml + gleason_score + crp_mg_l + nlr + weight_loss_pct_6mo "
             "+ pain_nrs + fatigue_grade + appetite_loss_grade "
             "+ brca2_mutation + ar_v7_positive + msi_high + psma_high + tp53_mutation + pten_loss "
             "+ treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel "
             "+ treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab")
mv = smf.ols(formula14, data=df).fit()
print("Iter 14 R^2:", mv.rsquared)

for term, hid in [
    ("ecog_ps", "h14_ecog"),
    ("albumin_g_dl", "h14_alb"),
    ("ldh_u_l", "h14_ldh"),
    ("hemoglobin_g_dl", "h14_hgb"),
    ("psa_ng_ml", "h14_psa"),
    ("crp_mg_l", "h14_crp"),
    ("nlr", "h14_nlr"),
    ("weight_loss_pct_6mo", "h14_wtloss"),
    ("alkaline_phosphatase_u_l", "h14_alp"),
    ("ar_v7_positive", "h14_arv7"),
    ("brca2_mutation", "h14_brca2"),
    ("msi_high", "h14_msi"),
    ("treatment_olaparib", "h14_olaparib"),
    ("treatment_pembrolizumab", "h14_pembro"),
    ("treatment_lu177_psma", "h14_lu177"),
    ("treatment_enzalutamide", "h14_enza"),
    ("treatment_abiraterone", "h14_abi"),
    ("treatment_docetaxel", "h14_docetaxel"),
    ("liver_mets", "h14_livmets"),
    ("visceral_mets", "h14_vismets"),
    ("mcrpc", "h14_mcrpc"),
]:
    if term in mv.params:
        c, p = float(mv.params[term]), float(mv.pvalues[term])
        record(14, hid,
               f"After adjusting for clinical, pathologic, biomarker and treatment covariates, {term} is independently associated with pfs_months in the expected direction (negative for adverse markers, positive for favorable markers and biomarker-matched treatments).",
               "Multivariable OLS pfs_months ~ full prognostic + biomarker + treatment covariate set.",
               f"Adjusted beta for {term}={c:+.4g}; p={p:.3g}.",
               p, c)

# =====================================================================
# Iteration 15: Refined predictive interactions, multivariable-adjusted
# =====================================================================
base_adj = ("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + liver_mets + bone_mets "
            "+ albumin_g_dl + ldh_u_l + hemoglobin_g_dl + alkaline_phosphatase_u_l + psa_ng_ml "
            "+ tp53_mutation + pten_loss")

for tx, marker, hid_suf, txt in [
    ("treatment_olaparib", "brca2_mutation", "olaparib_brca2_adj",
     "Refined: After adjusting for prognostic covariates, the PFS benefit of treatment_olaparib remains greater in BRCA2-mutated patients (positive interaction)."),
    ("treatment_pembrolizumab", "msi_high", "pembro_msi_adj",
     "Refined: After adjusting for prognostic covariates, the PFS benefit of treatment_pembrolizumab remains greater in MSI-high patients (positive interaction)."),
    ("treatment_lu177_psma", "psma_high", "lu177_psma_adj",
     "Refined: After adjusting for prognostic covariates, the PFS benefit of treatment_lu177_psma remains greater in PSMA-high patients (positive interaction)."),
    ("treatment_enzalutamide", "ar_v7_positive", "enza_arv7_adj",
     "Refined: After adjusting for prognostic covariates, the PFS effect of treatment_enzalutamide is more negative in AR-V7-positive patients (negative interaction)."),
    ("treatment_abiraterone", "ar_v7_positive", "abi_arv7_adj",
     "Refined: After adjusting for prognostic covariates, the PFS effect of treatment_abiraterone is more negative in AR-V7-positive patients (negative interaction)."),
]:
    f = f"{base_adj} + {tx}*{marker}"
    m = smf.ols(f, data=df).fit()
    term = f"{tx}:{marker}"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(15, f"h15_{hid_suf}", txt,
           f"smf.ols('{f}')",
           f"Adjusted interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c, kind="refined")

# =====================================================================
# Iteration 16: Secondary predictive interactions
# =====================================================================
for tx, mod, hid_suf, txt in [
    ("treatment_lu177_psma", "psa_ng_ml", "lu177_psa",
     "psa_ng_ml modifies treatment_lu177_psma effect (interaction; uncertain direction — high PSA marks burden but also more PSMA target)."),
    ("treatment_docetaxel", "visceral_mets", "doce_visceral",
     "treatment_docetaxel benefit is greater (more positive) in visceral_mets=1 patients (positive interaction)."),
    ("treatment_olaparib", "tp53_mutation", "olap_tp53",
     "treatment_olaparib effect differs by tp53_mutation status (interaction)."),
    ("treatment_pembrolizumab", "tp53_mutation", "pembro_tp53",
     "treatment_pembrolizumab effect differs by tp53_mutation status (interaction)."),
    ("treatment_enzalutamide", "tp53_mutation", "enza_tp53",
     "treatment_enzalutamide effect is more negative in tp53_mutation=1 patients (interaction)."),
    ("treatment_pembrolizumab", "ar_v7_positive", "pembro_arv7",
     "treatment_pembrolizumab effect differs by ar_v7_positive (exploratory interaction)."),
]:
    m = smf.ols(f"pfs_months ~ {tx} * {mod}", data=df).fit()
    term = f"{tx}:{mod}"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(16, f"h16_{hid_suf}", txt,
           f"smf.ols('pfs_months ~ {tx} * {mod}')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 17: Race × treatment interactions
# =====================================================================
df["race_black"] = (df["race_ethnicity"] == "black").astype(int)
for tx in ["treatment_enzalutamide", "treatment_abiraterone", "treatment_docetaxel",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * race_black", data=df).fit()
    term = f"{tx}:race_black"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(17, f"h17_{tx[10:]}_black",
           f"The PFS effect of {tx} differs in race_ethnicity='black' patients vs others (interaction; tests for differential treatment efficacy by race).",
           f"smf.ols('pfs_months ~ {tx} * race_black')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 18: Insurance × treatment interactions
# =====================================================================
df["ins_medicaid"] = (df["insurance_type"] == "medicaid").astype(int)
for tx in ["treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * ins_medicaid", data=df).fit()
    term = f"{tx}:ins_medicaid"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(18, f"h18_{tx[10:]}_medicaid",
           f"The PFS effect of {tx} differs by ins_medicaid status (interaction; tests for differential efficacy by insurance/access).",
           f"smf.ols('pfs_months ~ {tx} * ins_medicaid')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 19: Stratified subgroup confirmation of predictive markers
# =====================================================================
def strat(tx, marker, marker_val, label):
    sub = df.loc[df[marker] == marker_val]
    if (sub[tx]==1).sum() < 5 or (sub[tx]==0).sum() < 5:
        return
    t, p = stats.ttest_ind(sub.loc[sub[tx]==1,"pfs_months"],
                            sub.loc[sub[tx]==0,"pfs_months"], equal_var=False)
    eff = sub.loc[sub[tx]==1,"pfs_months"].mean() - sub.loc[sub[tx]==0,"pfs_months"].mean()
    return float(eff), float(p), int((sub[tx]==1).sum()), int((sub[tx]==0).sum())

# olaparib stratified by BRCA2
for v, lbl in [(1, "BRCA2-positive"), (0, "BRCA2-negative")]:
    res = strat("treatment_olaparib", "brca2_mutation", v, lbl)
    if res:
        eff, p, n1, n0 = res
        record(19, f"h19_olap_brca2_{v}",
               f"In the {lbl} subgroup, treatment_olaparib improves pfs_months relative to no olaparib.",
               f"stratified ttest in {lbl}",
               f"{lbl}: olaparib effect={eff:+.3f} mo (n_tx={n1}, n_ctrl={n0}); p={p:.3g}.",
               p, eff, kind="refined")

for v, lbl in [(1, "MSI-high"), (0, "MSS")]:
    res = strat("treatment_pembrolizumab", "msi_high", v, lbl)
    if res:
        eff, p, n1, n0 = res
        record(19, f"h19_pembro_msi_{v}",
               f"In the {lbl} subgroup, treatment_pembrolizumab improves pfs_months relative to no pembrolizumab.",
               f"stratified ttest in {lbl}",
               f"{lbl}: pembrolizumab effect={eff:+.3f} mo (n_tx={n1}, n_ctrl={n0}); p={p:.3g}.",
               p, eff, kind="refined")

for v, lbl in [(1, "PSMA-high"), (0, "PSMA-low")]:
    res = strat("treatment_lu177_psma", "psma_high", v, lbl)
    if res:
        eff, p, n1, n0 = res
        record(19, f"h19_lu177_psma_{v}",
               f"In the {lbl} subgroup, treatment_lu177_psma improves pfs_months relative to no lu177-PSMA.",
               f"stratified ttest in {lbl}",
               f"{lbl}: lu177 effect={eff:+.3f} mo (n_tx={n1}, n_ctrl={n0}); p={p:.3g}.",
               p, eff, kind="refined")

for v, lbl in [(1, "AR-V7-positive"), (0, "AR-V7-negative")]:
    res = strat("treatment_enzalutamide", "ar_v7_positive", v, lbl)
    if res:
        eff, p, n1, n0 = res
        # Hypothesized: positive in AR-V7-, negative or null in AR-V7+
        record(19, f"h19_enza_arv7_{v}",
               f"In the {lbl} subgroup, treatment_enzalutamide changes pfs_months: hypothesized {'no benefit / negative' if v==1 else 'positive benefit'}.",
               f"stratified ttest in {lbl}",
               f"{lbl}: enzalutamide effect={eff:+.3f} mo (n_tx={n1}, n_ctrl={n0}); p={p:.3g}.",
               p, eff, kind="refined")

# =====================================================================
# Iteration 20: Symptom burden composite
# =====================================================================
df["symptom_burden"] = (df["fatigue_grade"] + df["pain_nrs"]/2.0 + df["dyspnea_grade"]
                         + df["cough_grade"] + df["appetite_loss_grade"])
m = smf.ols("pfs_months ~ symptom_burden", data=df).fit()
c, p = report_coef(m, "symptom_burden")
record(20, "h20_sympburden",
       "A composite symptom-burden score (fatigue_grade + pain_nrs/2 + dyspnea_grade + cough_grade + appetite_loss_grade) is associated with shorter pfs_months (negative slope).",
       "smf.ols('pfs_months ~ symptom_burden')",
       f"OLS slope on symptom_burden: beta={c:+.4g}; p={p:.3g}.",
       p, c)

# =====================================================================
# Iteration 21: Inflammation × treatment
# =====================================================================
m = smf.ols("pfs_months ~ treatment_pembrolizumab * nlr", data=df).fit()
term = "treatment_pembrolizumab:nlr"
c, p = float(m.params[term]), float(m.pvalues[term])
record(21, "h21_pembro_nlr",
       "Higher nlr (neutrophil-lymphocyte ratio) reduces the PFS benefit of treatment_pembrolizumab (negative interaction; high NLR is a known biomarker of poor immunotherapy response).",
       "smf.ols('pfs_months ~ treatment_pembrolizumab * nlr')",
       f"Interaction beta={c:+.4g}; p={p:.3g}.",
       p, c)

m = smf.ols("pfs_months ~ treatment_docetaxel * crp_mg_l", data=df).fit()
term = "treatment_docetaxel:crp_mg_l"
c, p = float(m.params[term]), float(m.pvalues[term])
record(21, "h21_doce_crp",
       "Higher crp_mg_l reduces the PFS benefit of treatment_docetaxel (negative interaction; high inflammation marks poor chemo tolerance).",
       "smf.ols('pfs_months ~ treatment_docetaxel * crp_mg_l')",
       f"Interaction beta={c:+.4g}; p={p:.3g}.",
       p, c)

# =====================================================================
# Iteration 22: Age × treatment interactions
# =====================================================================
for tx in ["treatment_docetaxel", "treatment_enzalutamide", "treatment_abiraterone",
           "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * age_years", data=df).fit()
    term = f"{tx}:age_years"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(22, f"h22_{tx[10:]}_age",
           f"The PFS effect of {tx} depends on age_years (interaction). Hypothesized: more negative interaction for docetaxel (chemo less tolerated in older patients).",
           f"smf.ols('pfs_months ~ {tx} * age_years')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 23: Albumin × treatment interactions
# =====================================================================
for tx in ["treatment_docetaxel", "treatment_olaparib", "treatment_lu177_psma", "treatment_pembrolizumab"]:
    m = smf.ols(f"pfs_months ~ {tx} * albumin_g_dl", data=df).fit()
    term = f"{tx}:albumin_g_dl"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(23, f"h23_{tx[10:]}_alb",
           f"Higher albumin_g_dl is associated with greater PFS benefit of {tx} (positive interaction; better-nourished patients derive more benefit).",
           f"smf.ols('pfs_months ~ {tx} * albumin_g_dl')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 24: Co-mutation interactions on PFS
# =====================================================================
for a, b, hid_suf, txt in [
    ("brca2_mutation", "tp53_mutation", "brca2_tp53",
     "BRCA2 and TP53 co-mutation interact such that double-positive patients have especially short pfs_months (negative interaction)."),
    ("tp53_mutation", "pten_loss", "tp53_pten",
     "tp53_mutation and pten_loss interact such that double-positive patients have especially short pfs_months (negative interaction)."),
    ("ar_v7_positive", "tp53_mutation", "arv7_tp53",
     "ar_v7_positive and tp53_mutation interact (negative interaction)."),
]:
    m = smf.ols(f"pfs_months ~ {a} * {b}", data=df).fit()
    term = f"{a}:{b}"
    c, p = float(m.params[term]), float(m.pvalues[term])
    record(24, f"h24_{hid_suf}", txt,
           f"smf.ols('pfs_months ~ {a} * {b}')",
           f"Interaction {term}: beta={c:+.4g}; p={p:.3g}.",
           p, c)

# =====================================================================
# Iteration 25: Final adjusted model with all four predictive interactions
# =====================================================================
final = ("pfs_months ~ age_years + ecog_ps + mcrpc + visceral_mets + liver_mets + bone_mets "
         "+ albumin_g_dl + ldh_u_l + hemoglobin_g_dl + alkaline_phosphatase_u_l + psa_ng_ml "
         "+ tp53_mutation + pten_loss + crp_mg_l + nlr + weight_loss_pct_6mo "
         "+ treatment_olaparib*brca2_mutation "
         "+ treatment_pembrolizumab*msi_high "
         "+ treatment_lu177_psma*psma_high "
         "+ treatment_enzalutamide*ar_v7_positive "
         "+ treatment_abiraterone*ar_v7_positive "
         "+ treatment_docetaxel")
mfinal = smf.ols(final, data=df).fit()
print("Iter 25 final R^2:", mfinal.rsquared)

for term, hid, txt in [
    ("treatment_olaparib:brca2_mutation", "h25_olap_brca2_final",
     "Final refined hypothesis: in the fully-adjusted multivariable model with all four predictive biomarker × treatment interactions included simultaneously, treatment_olaparib × brca2_mutation interaction is positive and significant."),
    ("treatment_pembrolizumab:msi_high", "h25_pembro_msi_final",
     "Final refined hypothesis: in the fully-adjusted model, treatment_pembrolizumab × msi_high interaction is positive and significant."),
    ("treatment_lu177_psma:psma_high", "h25_lu177_psma_final",
     "Final refined hypothesis: in the fully-adjusted model, treatment_lu177_psma × psma_high interaction is positive and significant."),
    ("treatment_enzalutamide:ar_v7_positive", "h25_enza_arv7_final",
     "Final refined hypothesis: in the fully-adjusted model, treatment_enzalutamide × ar_v7_positive interaction is negative and significant."),
    ("treatment_abiraterone:ar_v7_positive", "h25_abi_arv7_final",
     "Final refined hypothesis: in the fully-adjusted model, treatment_abiraterone × ar_v7_positive interaction is negative and significant."),
]:
    if term in mfinal.params:
        c, p = float(mfinal.params[term]), float(mfinal.pvalues[term])
        record(25, hid, txt,
               "Final adjusted multivariable OLS with all four biomarker×treatment interactions.",
               f"Adjusted interaction {term}: beta={c:+.4g}; p={p:.3g}.",
               p, c, kind="refined")

print("Total records:", len(OUT))

with open("_agent_records.json", "w") as f:
    json.dump(OUT, f, indent=2,
              default=lambda o: None if (isinstance(o, float) and (np.isnan(o) or np.isinf(o))) else o)

print("Saved _agent_records.json")
