"""Iterations 10-25: deeper subgroup, comorbidity, SNP, refinement analyses.

Loads the existing transcript.json (iterations 1-9) and appends.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
DF["log_psa"] = np.log1p(DF["psa_ng_ml"])
DF["log_ldh"] = np.log(DF["ldh_u_l"])
DF["log_alp"] = np.log(DF["alkaline_phosphatase_u_l"])
DF["log_crp"] = np.log1p(DF["crp_mg_l"])
OUT = "pfs_months"

# Load existing transcript and continue
TRANSCRIPT = json.load(open("transcript.json"))
ITERATIONS = TRANSCRIPT["iterations"]


def _round(x, n=4):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return round(float(x), n)
    except Exception:
        return None


def add_iteration(idx, hypotheses, analyses):
    ITERATIONS.append({"index": idx, "proposed_hypotheses": hypotheses, "analyses": analyses})


def beta_p(model, term):
    return float(model.params[term]), float(model.pvalues[term])


def ols_fit(formula, df=None):
    if df is None:
        df = DF
    return smf.ols(formula, data=df).fit()


# ------------------- Iteration 10 ------------------- #
# Age: univariable vs adjusted. Earlier iter showed positive adjusted coef. Investigate.
m10u = ols_fit(f"{OUT} ~ age_years")
b10u, p10u = beta_p(m10u, "age_years")
m10a = ols_fit(f"{OUT} ~ age_years + ecog_ps + mcrpc + albumin_g_dl + log_psa + log_ldh + hemoglobin_g_dl + visceral_mets")
b10a, p10a = beta_p(m10a, "age_years")
# Check correlation of age with ecog and albumin
corr_age_ecog = float(DF[["age_years","ecog_ps"]].corr().iloc[0,1])
corr_age_alb = float(DF[["age_years","albumin_g_dl"]].corr().iloc[0,1])

add_iteration(
    10,
    [
        {"id":"h10.1","text":"Older age_years is univariably associated with shorter pfs_months (negative coefficient).","kind":"refined"},
        {"id":"h10.2","text":"After adjusting for ECOG, mCRPC, albumin, log PSA, log LDH, hemoglobin, and visceral mets, age_years has a positive (suppressed) coefficient on pfs_months, indicating its raw negative effect is mediated by these prognostic factors.","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h10.1"],"code":"OLS pfs ~ age",
         "result_summary": f"Univariable age beta={b10u:.4f} months/year (p={p10u:.3g}). Pearson r(age, ecog)={corr_age_ecog:.3f}, r(age, albumin)={corr_age_alb:.3f}.",
         "p_value": _round(p10u),"effect_estimate":_round(b10u),"significant":p10u<0.05},
        {"hypothesis_ids":["h10.2"],"code":"OLS pfs ~ age + clinical covars",
         "result_summary": f"Adjusted age beta={b10a:.4f} months/year (p={p10a:.3g}).",
         "p_value": _round(p10a),"effect_estimate":_round(b10a),"significant":p10a<0.05},
    ],
)
print("Iter10 done")

# ------------------- Iteration 11 ------------------- #
# Symptoms: fatigue, pain, dyspnea, cough, appetite — predict PFS
m11 = ols_fit(f"{OUT} ~ fatigue_grade + pain_nrs + dyspnea_grade + cough_grade + appetite_loss_grade + ecog_ps")
res11 = {t: beta_p(m11, t) for t in ["fatigue_grade","pain_nrs","dyspnea_grade","cough_grade","appetite_loss_grade"]}

add_iteration(
    11,
    [
        {"id":"h11.1","text":"Higher fatigue_grade is associated with shorter pfs_months after adjusting for ECOG.","kind":"novel"},
        {"id":"h11.2","text":"Higher pain_nrs is associated with shorter pfs_months after adjusting for ECOG.","kind":"novel"},
        {"id":"h11.3","text":"Higher dyspnea_grade is associated with shorter pfs_months after adjusting for ECOG.","kind":"novel"},
        {"id":"h11.4","text":"Higher cough_grade is associated with shorter pfs_months after adjusting for ECOG.","kind":"novel"},
        {"id":"h11.5","text":"Higher appetite_loss_grade is associated with shorter pfs_months after adjusting for ECOG.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":[f"h11.{i+1}"],"code":"OLS pfs ~ symptoms + ecog",
         "result_summary": f"{t} beta={res11[t][0]:.4f} (p={res11[t][1]:.3g}).",
         "p_value": _round(res11[t][1]),"effect_estimate":_round(res11[t][0]),"significant":res11[t][1]<0.05}
        for i,t in enumerate(["fatigue_grade","pain_nrs","dyspnea_grade","cough_grade","appetite_loss_grade"])
    ],
)
print("Iter11 done")

# ------------------- Iteration 12 ------------------- #
# Inflammation/nutrition markers: NLR, CRP, weight loss, ALC
m12 = ols_fit(f"{OUT} ~ nlr + log_crp + weight_loss_pct_6mo + alc_k_ul + bmi")
res12 = {t: beta_p(m12, t) for t in ["nlr","log_crp","weight_loss_pct_6mo","alc_k_ul","bmi"]}

add_iteration(
    12,
    [
        {"id":"h12.1","text":"Higher neutrophil-to-lymphocyte ratio (nlr) is associated with shorter pfs_months.","kind":"novel"},
        {"id":"h12.2","text":"Higher log(crp_mg_l) is associated with shorter pfs_months.","kind":"novel"},
        {"id":"h12.3","text":"Greater weight_loss_pct_6mo is associated with shorter pfs_months.","kind":"novel"},
        {"id":"h12.4","text":"Higher absolute lymphocyte count (alc_k_ul) is associated with longer pfs_months.","kind":"novel"},
        {"id":"h12.5","text":"Higher BMI (bmi) is associated with longer pfs_months.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":[f"h12.{i+1}"],"code":"OLS pfs ~ inflammation/nutrition markers",
         "result_summary": f"{t} beta={res12[t][0]:.4f} (p={res12[t][1]:.3g}).",
         "p_value": _round(res12[t][1]),"effect_estimate":_round(res12[t][0]),"significant":res12[t][1]<0.05}
        for i,t in enumerate(["nlr","log_crp","weight_loss_pct_6mo","alc_k_ul","bmi"])
    ],
)
print("Iter12 done")

# ------------------- Iteration 13 ------------------- #
# Comorbidities effects on PFS
comorbids = ["diabetes_mellitus","hypertension","copd","chronic_kidney_disease","heart_failure",
             "coronary_artery_disease","atrial_fibrillation","autoimmune_disease","prior_malignancy"]
m13 = ols_fit(f"{OUT} ~ " + " + ".join(comorbids) + " + age_years + ecog_ps")
res13 = {t: beta_p(m13, t) for t in comorbids}

add_iteration(
    13,
    [
        {"id":f"h13.{i+1}","text":f"Patients with {c}=1 have shorter pfs_months than those without (negative coefficient), after adjusting for age and ECOG.","kind":"novel"}
        for i,c in enumerate(comorbids)
    ],
    [
        {"hypothesis_ids":[f"h13.{i+1}"],"code":"OLS pfs ~ 9 comorbidities + age + ecog",
         "result_summary": f"{t} beta={res13[t][0]:.4f} (p={res13[t][1]:.3g}).",
         "p_value": _round(res13[t][1]),"effect_estimate":_round(res13[t][0]),"significant":res13[t][1]<0.05}
        for i,t in enumerate(comorbids)
    ],
)
print("Iter13 done")

# ------------------- Iteration 14 ------------------- #
# Prior therapy & lines of therapy
m14 = ols_fit(f"{OUT} ~ prior_chemotherapy + prior_radiation + prior_surgery + prior_targeted_therapy + prior_lines_of_therapy + years_since_diagnosis")
terms14 = ["prior_chemotherapy","prior_radiation","prior_surgery","prior_targeted_therapy","prior_lines_of_therapy","years_since_diagnosis"]
res14 = {t: beta_p(m14, t) for t in terms14}

add_iteration(
    14,
    [
        {"id":"h14.1","text":"prior_chemotherapy=1 is associated with shorter pfs_months than prior_chemotherapy=0 (negative coefficient).","kind":"novel"},
        {"id":"h14.2","text":"prior_radiation=1 is associated with shorter pfs_months than prior_radiation=0.","kind":"novel"},
        {"id":"h14.3","text":"prior_surgery=1 is associated with shorter pfs_months than prior_surgery=0.","kind":"novel"},
        {"id":"h14.4","text":"prior_targeted_therapy=1 is associated with shorter pfs_months than prior_targeted_therapy=0.","kind":"novel"},
        {"id":"h14.5","text":"More prior_lines_of_therapy is associated with shorter pfs_months (negative coefficient).","kind":"novel"},
        {"id":"h14.6","text":"Longer years_since_diagnosis is associated with shorter pfs_months (negative coefficient).","kind":"novel"},
    ],
    [
        {"hypothesis_ids":[f"h14.{i+1}"],"code":"OLS pfs ~ prior therapy variables",
         "result_summary": f"{t} beta={res14[t][0]:.4f} (p={res14[t][1]:.3g}).",
         "p_value": _round(res14[t][1]),"effect_estimate":_round(res14[t][0]),"significant":res14[t][1]<0.05}
        for i,t in enumerate(terms14)
    ],
)
print("Iter14 done")

# ------------------- Iteration 15 ------------------- #
# mCRPC subgroup: do treatment effects differ?
sub_mcrpc = DF[DF.mcrpc==1]
sub_hsens = DF[DF.mcrpc==0]
trt = ["treatment_enzalutamide","treatment_abiraterone","treatment_docetaxel",
       "treatment_olaparib","treatment_lu177_psma","treatment_pembrolizumab"]
covars = " + ecog_ps + log_psa + log_ldh + albumin_g_dl"

m15a = ols_fit(f"{OUT} ~ " + " + ".join(trt) + covars, df=sub_mcrpc)
m15b = ols_fit(f"{OUT} ~ " + " + ".join(trt) + covars, df=sub_hsens)
res_mcrpc = {t: beta_p(m15a, t) for t in trt}
res_hsens = {t: beta_p(m15b, t) for t in trt}

add_iteration(
    15,
    [
        {"id":"h15.1","text":"Among mCRPC patients (mcrpc=1), the PFS effect of treatment_olaparib is positive (beta>0) when controlling for clinical prognostic factors.","kind":"novel"},
        {"id":"h15.2","text":"Among non-mCRPC patients (mcrpc=0), the PFS effect of treatment_olaparib is also positive.","kind":"novel"},
        {"id":"h15.3","text":"Among mCRPC patients, the PFS effect of treatment_lu177_psma is positive when controlling for prognostic covariates.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":["h15.1"],"code":"subset mcrpc==1; OLS pfs ~ treatments + clin",
         "result_summary": f"Among mCRPC (n={len(sub_mcrpc)}), olaparib beta={res_mcrpc['treatment_olaparib'][0]:.4f} (p={res_mcrpc['treatment_olaparib'][1]:.3g}).",
         "p_value": _round(res_mcrpc['treatment_olaparib'][1]),"effect_estimate":_round(res_mcrpc['treatment_olaparib'][0]),"significant":res_mcrpc['treatment_olaparib'][1]<0.05},
        {"hypothesis_ids":["h15.2"],"code":"subset mcrpc==0",
         "result_summary": f"Among non-mCRPC (n={len(sub_hsens)}), olaparib beta={res_hsens['treatment_olaparib'][0]:.4f} (p={res_hsens['treatment_olaparib'][1]:.3g}).",
         "p_value": _round(res_hsens['treatment_olaparib'][1]),"effect_estimate":_round(res_hsens['treatment_olaparib'][0]),"significant":res_hsens['treatment_olaparib'][1]<0.05},
        {"hypothesis_ids":["h15.3"],"code":"subset mcrpc==1; lu177 term",
         "result_summary": f"Among mCRPC, lu177_psma beta={res_mcrpc['treatment_lu177_psma'][0]:.4f} (p={res_mcrpc['treatment_lu177_psma'][1]:.3g}).",
         "p_value": _round(res_mcrpc['treatment_lu177_psma'][1]),"effect_estimate":_round(res_mcrpc['treatment_lu177_psma'][0]),"significant":res_mcrpc['treatment_lu177_psma'][1]<0.05},
    ],
)
print("Iter15 done")

# ------------------- Iteration 16 ------------------- #
# Three-way refinement: olaparib x BRCA2 x mCRPC
sub_brca = DF[DF.brca2_mutation==1]
m16 = ols_fit(f"{OUT} ~ treatment_olaparib * mcrpc + ecog_ps + log_psa + log_ldh + albumin_g_dl", df=sub_brca)
b_ola_b, p_ola_b = beta_p(m16, "treatment_olaparib")
b_int_b, p_int_b = beta_p(m16, "treatment_olaparib:mcrpc")

# Stratified means
strat16 = {}
for mcr in [0,1]:
    for ola in [0,1]:
        sub = sub_brca[(sub_brca.mcrpc==mcr)&(sub_brca.treatment_olaparib==ola)][OUT]
        strat16[(mcr,ola)] = (float(sub.mean()), int(len(sub)))

add_iteration(
    16,
    [
        {"id":"h16.1","text":"Within BRCA2-mutated patients, the PFS benefit of treatment_olaparib (vs no olaparib) is positive in both mCRPC and non-mCRPC subsets, indicating the olaparib×BRCA2 interaction is not restricted to mCRPC.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":["h16.1"],"code":"subset brca2==1; OLS with olap*mcrpc",
         "result_summary": f"In BRCA2+ subset (n={len(sub_brca)}): olaparib main beta (mcrpc=0)={b_ola_b:.3f} (p={p_ola_b:.3g}); olaparib*mcrpc interaction = {b_int_b:.3f} (p={p_int_b:.3g}). Stratified means: BRCA2+/non-mCRPC ola+={strat16[(0,1)][0]:.2f} (n={strat16[(0,1)][1]}) vs ola-={strat16[(0,0)][0]:.2f} (n={strat16[(0,0)][1]}); BRCA2+/mCRPC ola+={strat16[(1,1)][0]:.2f} (n={strat16[(1,1)][1]}) vs ola-={strat16[(1,0)][0]:.2f} (n={strat16[(1,0)][1]}).",
         "p_value": _round(p_ola_b),"effect_estimate":_round(b_ola_b),"significant":p_ola_b<0.05},
    ],
)
print("Iter16 done")

# ------------------- Iteration 17 ------------------- #
# Race/ethnicity, insurance, rural residence, education effect on PFS
DF["race_white"] = (DF["race_ethnicity"]=="white").astype(int)
DF["race_black"] = (DF["race_ethnicity"]=="black").astype(int)
DF["race_hispanic"] = (DF["race_ethnicity"]=="hispanic").astype(int)
DF["race_asian"] = (DF["race_ethnicity"]=="asian").astype(int)
DF["ins_medicare"] = (DF["insurance_type"]=="medicare").astype(int)
DF["ins_private"] = (DF["insurance_type"]=="private").astype(int)
DF["ins_uninsured"] = (DF["insurance_type"]=="uninsured").astype(int)
DF["ins_medicaid"] = (DF["insurance_type"]=="medicaid").astype(int)

m17 = ols_fit(f"{OUT} ~ C(race_ethnicity) + C(insurance_type) + rural_residence + education_years + age_years + ecog_ps")
race_terms = [t for t in m17.pvalues.index if t.startswith("C(race_ethnicity)")]
ins_terms = [t for t in m17.pvalues.index if t.startswith("C(insurance_type)")]
b_rural, p_rural = beta_p(m17, "rural_residence")
b_edu, p_edu = beta_p(m17, "education_years")
race_p_overall = float(m17.f_test(" = ".join(race_terms) + " = 0").pvalue) if len(race_terms)>=2 else None

add_iteration(
    17,
    [
        {"id":"h17.1","text":"After adjusting for age and ECOG, mean pfs_months differs across race_ethnicity categories (joint F-test p<0.05).","kind":"novel"},
        {"id":"h17.2","text":"After adjusting for age and ECOG, rural_residence=1 patients have shorter pfs_months than rural_residence=0 patients (negative coefficient).","kind":"novel"},
        {"id":"h17.3","text":"After adjusting for age and ECOG, more education_years is associated with longer pfs_months.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":["h17.1"],"code":"OLS pfs ~ race + insurance + rural + edu + age + ecog (joint F-test on race terms)",
         "result_summary": "Joint F-test on race_ethnicity: p=" + str(race_p_overall) + ". Individual coefficients (vs reference): " + ", ".join([t.replace("C(race_ethnicity)[T.","").replace("]","") + ": " + format(m17.params[t],".3f") + " (p=" + format(m17.pvalues[t],".3g") + ")" for t in race_terms]),
         "p_value": _round(race_p_overall),"effect_estimate": _round(max([abs(m17.params[t]) for t in race_terms])) if race_terms else None,"significant": (race_p_overall is not None and race_p_overall<0.05)},
        {"hypothesis_ids":["h17.2"],"code":"OLS, rural_residence term",
         "result_summary": f"rural_residence beta={b_rural:.4f} months (p={p_rural:.3g}).",
         "p_value": _round(p_rural),"effect_estimate":_round(b_rural),"significant":p_rural<0.05},
        {"hypothesis_ids":["h17.3"],"code":"OLS, education_years term",
         "result_summary": f"education_years beta={b_edu:.4f} months/year (p={p_edu:.3g}).",
         "p_value": _round(p_edu),"effect_estimate":_round(b_edu),"significant":p_edu<0.05},
    ],
)
print("Iter17 done")

# ------------------- Iteration 18 ------------------- #
# SNPs — survey 10 SNPs for PFS effect (univariable)
snps = ["snp_rs1045642","snp_rs1065852","snp_rs1799853","snp_rs1800566","snp_rs2228001",
        "snp_rs3813867","snp_rs4244285","snp_rs4986893","snp_rs1801133","snp_rs1800896"]
res18 = {}
for s in snps:
    m = ols_fit(f"{OUT} ~ {s}")
    res18[s] = beta_p(m, s)
sig_snps = [s for s,(b,p) in res18.items() if p<0.05]

add_iteration(
    18,
    [
        {"id":f"h18.{i+1}","text":f"Variant {s} is univariably associated with pfs_months (any direction; tested two-sided).","kind":"novel"}
        for i,s in enumerate(snps)
    ],
    [
        {"hypothesis_ids":[f"h18.{i+1}"],"code":f"OLS pfs ~ {s}",
         "result_summary": f"{s} beta={res18[s][0]:.4f} (p={res18[s][1]:.3g}).",
         "p_value": _round(res18[s][1]),"effect_estimate":_round(res18[s][0]),"significant":res18[s][1]<0.05}
        for i,s in enumerate(snps)
    ],
)
print(f"Iter18 done ({len(sig_snps)} sig of {len(snps)})")

# ------------------- Iteration 19 ------------------- #
# Cross-treatment interactions: combo of enzalutamide + abiraterone, etc., test additive vs interaction
m19 = ols_fit(f"{OUT} ~ treatment_enzalutamide * treatment_abiraterone + ecog_ps + mcrpc + log_psa + albumin_g_dl")
b_int19, p_int19 = beta_p(m19, "treatment_enzalutamide:treatment_abiraterone")

m19b = ols_fit(f"{OUT} ~ treatment_docetaxel * treatment_abiraterone + ecog_ps + mcrpc + log_psa + albumin_g_dl")
b_int19b, p_int19b = beta_p(m19b, "treatment_docetaxel:treatment_abiraterone")

m19c = ols_fit(f"{OUT} ~ treatment_olaparib * treatment_lu177_psma + ecog_ps + mcrpc + log_psa + albumin_g_dl")
b_int19c, p_int19c = beta_p(m19c, "treatment_olaparib:treatment_lu177_psma")

add_iteration(
    19,
    [
        {"id":"h19.1","text":"There is an interaction between treatment_enzalutamide and treatment_abiraterone on pfs_months (the combination's effect differs from the sum of single-agent effects); two-sided test.","kind":"novel"},
        {"id":"h19.2","text":"There is an interaction between treatment_docetaxel and treatment_abiraterone on pfs_months.","kind":"novel"},
        {"id":"h19.3","text":"There is an interaction between treatment_olaparib and treatment_lu177_psma on pfs_months.","kind":"novel"},
    ],
    [
        {"hypothesis_ids":["h19.1"],"code":"OLS pfs ~ enza*abi + covariates",
         "result_summary": f"enza:abi interaction beta={b_int19:.4f} (p={p_int19:.3g}).",
         "p_value": _round(p_int19),"effect_estimate":_round(b_int19),"significant":p_int19<0.05},
        {"hypothesis_ids":["h19.2"],"code":"OLS pfs ~ doc*abi + covariates",
         "result_summary": f"doc:abi interaction beta={b_int19b:.4f} (p={p_int19b:.3g}).",
         "p_value": _round(p_int19b),"effect_estimate":_round(b_int19b),"significant":p_int19b<0.05},
        {"hypothesis_ids":["h19.3"],"code":"OLS pfs ~ olap*lu177 + covariates",
         "result_summary": f"olap:lu177 interaction beta={b_int19c:.4f} (p={p_int19c:.3g}).",
         "p_value": _round(p_int19c),"effect_estimate":_round(b_int19c),"significant":p_int19c<0.05},
    ],
)
print("Iter19 done")

# ------------------- Iteration 20 ------------------- #
# Lab values: bilirubin, creatinine, sodium, potassium, calcium, glucose, platelets, wbc
m20 = ols_fit(f"{OUT} ~ total_bilirubin_mg_dl + creatinine_mg_dl + sodium_meq_l + potassium_meq_l + calcium_mg_dl + glucose_mg_dl + platelets_k_ul + wbc_k_ul")
terms20 = ["total_bilirubin_mg_dl","creatinine_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl","glucose_mg_dl","platelets_k_ul","wbc_k_ul"]
res20 = {t: beta_p(m20, t) for t in terms20}

add_iteration(
    20,
    [
        {"id":f"h20.{i+1}","text":f"After adjusting for the other listed labs, {t} is associated with pfs_months (two-sided).","kind":"novel"}
        for i,t in enumerate(terms20)
    ],
    [
        {"hypothesis_ids":[f"h20.{i+1}"],"code":"OLS pfs ~ 8 chemistry labs",
         "result_summary": f"{t} beta={res20[t][0]:.5f} (p={res20[t][1]:.3g}).",
         "p_value": _round(res20[t][1]),"effect_estimate":_round(res20[t][0],5),"significant":res20[t][1]<0.05}
        for i,t in enumerate(terms20)
    ],
)
print("Iter20 done")

# ------------------- Iteration 21 ------------------- #
# Refine: BRCA2*olaparib confirmation in fully adjusted model
m21 = ols_fit(
    f"{OUT} ~ treatment_olaparib * brca2_mutation + ecog_ps + mcrpc + visceral_mets + "
    f"log_psa + log_ldh + albumin_g_dl + hemoglobin_g_dl + log_alp + age_years + "
    f"nlr + weight_loss_pct_6mo + log_crp + "
    f"treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + "
    f"treatment_lu177_psma + treatment_pembrolizumab"
)
b_int21, p_int21 = beta_p(m21, "treatment_olaparib:brca2_mutation")
b_ola21, p_ola21 = beta_p(m21, "treatment_olaparib")
b_brca21, p_brca21 = beta_p(m21, "brca2_mutation")

add_iteration(
    21,
    [
        {"id":"h21.1","text":"The positive treatment_olaparib × brca2_mutation interaction on pfs_months remains significant and large after adjusting for all other treatments, ECOG, mCRPC, visceral mets, age, and core labs (PSA, LDH, albumin, hemoglobin, ALP, NLR, CRP, weight loss).","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h21.1"],"code":"fully-adjusted OLS",
         "result_summary": f"Adjusted olaparib:brca2 interaction beta={b_int21:.4f} (p={p_int21:.3g}). Olaparib main (BRCA2=0) beta={b_ola21:.4f} (p={p_ola21:.3g}). BRCA2 main (no olaparib) beta={b_brca21:.4f} (p={p_brca21:.3g}).",
         "p_value": _round(p_int21),"effect_estimate":_round(b_int21),"significant":p_int21<0.05},
    ],
)
print("Iter21 done")

# ------------------- Iteration 22 ------------------- #
# Test additional treatment-biomarker interactions that might exist:
# docetaxel x visceral_mets; abiraterone x mcrpc; pembrolizumab x tp53; olaparib x tp53
inters = [
    ("treatment_docetaxel","visceral_mets"),
    ("treatment_abiraterone","mcrpc"),
    ("treatment_pembrolizumab","tp53_mutation"),
    ("treatment_olaparib","tp53_mutation"),
    ("treatment_olaparib","pten_loss"),
    ("treatment_enzalutamide","tp53_mutation"),
]
res22 = {}
for tx, bm in inters:
    m = ols_fit(f"{OUT} ~ {tx} * {bm} + ecog_ps + log_psa + log_ldh + albumin_g_dl")
    b, p = beta_p(m, f"{tx}:{bm}")
    res22[(tx,bm)] = (b, p)

hyps22 = []
ans22 = []
for i,(tx,bm) in enumerate(inters):
    hid=f"h22.{i+1}"
    hyps22.append({"id":hid,"text":f"There is an interaction between {tx} and {bm} on pfs_months (two-sided test).","kind":"novel"})
    b,p = res22[(tx,bm)]
    ans22.append({"hypothesis_ids":[hid],"code":f"OLS pfs ~ {tx}*{bm} + clin",
                  "result_summary": f"{tx}:{bm} interaction beta={b:.4f} (p={p:.3g}).",
                  "p_value": _round(p),"effect_estimate":_round(b),"significant":p<0.05})
add_iteration(22, hyps22, ans22)
print("Iter22 done")

# ------------------- Iteration 23 ------------------- #
# More biomarker × treatment interactions: HER2, fgfr, braf — each with relevant treatments
# Also pleural_effusion, bone_mets x treatments
inters23 = [
    ("treatment_pembrolizumab","brca2_mutation"),
    ("treatment_pembrolizumab","msi_high"),  # repeat with full adjustment
    ("treatment_lu177_psma","bone_mets"),
    ("treatment_docetaxel","bone_mets"),
    ("treatment_abiraterone","ar_v7_positive"),  # check refined
    ("treatment_olaparib","ar_v7_positive"),
]
res23 = {}
for tx, bm in inters23:
    m = ols_fit(f"{OUT} ~ {tx} * {bm} + ecog_ps + log_psa + log_ldh + albumin_g_dl + age_years + mcrpc")
    b, p = beta_p(m, f"{tx}:{bm}")
    res23[(tx,bm)] = (b, p)

hyps23 = []
ans23 = []
for i,(tx,bm) in enumerate(inters23):
    hid=f"h23.{i+1}"
    hyps23.append({"id":hid,"text":f"There is an interaction between {tx} and {bm} on pfs_months after adjusting for ECOG, log PSA, log LDH, albumin, age, and mCRPC (two-sided).","kind":"novel"})
    b,p = res23[(tx,bm)]
    ans23.append({"hypothesis_ids":[hid],"code":f"adjusted OLS pfs ~ {tx}*{bm}",
                  "result_summary": f"{tx}:{bm} adjusted interaction beta={b:.4f} (p={p:.3g}).",
                  "p_value": _round(p),"effect_estimate":_round(b),"significant":p<0.05})
add_iteration(23, hyps23, ans23)
print("Iter23 done")

# ------------------- Iteration 24 ------------------- #
# Composite prognostic index built from significant predictors,
# and check whether olaparib × BRCA2 effect is preserved across PI tertiles.
# Build PI from pre-treatment variables.
pi_features = ["ecog_ps","log_psa","log_ldh","albumin_g_dl","mcrpc","age_years","nlr","weight_loss_pct_6mo","log_crp"]
DF["_PI_input_check"] = DF[pi_features].isnull().any(axis=1)
m24 = smf.ols(f"{OUT} ~ " + " + ".join(pi_features), data=DF).fit()
DF["risk_score"] = -m24.predict(DF)  # higher = worse prognosis (negative predicted PFS)
DF["risk_tertile"] = pd.qcut(DF["risk_score"], 3, labels=["low","mid","high"]).astype(str)

m24a = smf.ols(f"{OUT} ~ treatment_olaparib * brca2_mutation + " + " + ".join(pi_features), data=DF[DF.risk_tertile=="low"]).fit()
m24b = smf.ols(f"{OUT} ~ treatment_olaparib * brca2_mutation + " + " + ".join(pi_features), data=DF[DF.risk_tertile=="high"]).fit()
b_lo, p_lo = beta_p(m24a, "treatment_olaparib:brca2_mutation")
b_hi, p_hi = beta_p(m24b, "treatment_olaparib:brca2_mutation")

add_iteration(
    24,
    [
        {"id":"h24.1","text":"The positive olaparib × BRCA2 interaction on pfs_months is present in the low-risk tertile (low baseline prognostic-index risk).","kind":"refined"},
        {"id":"h24.2","text":"The positive olaparib × BRCA2 interaction on pfs_months is present in the high-risk tertile (high baseline prognostic-index risk).","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h24.1"],"code":"PI tertile=low; OLS pfs ~ olap*brca2 + PI features",
         "result_summary": f"Low-risk tertile (n={int((DF.risk_tertile=='low').sum())}): olaparib:brca2 beta={b_lo:.4f} (p={p_lo:.3g}).",
         "p_value": _round(p_lo),"effect_estimate":_round(b_lo),"significant":p_lo<0.05},
        {"hypothesis_ids":["h24.2"],"code":"PI tertile=high",
         "result_summary": f"High-risk tertile (n={int((DF.risk_tertile=='high').sum())}): olaparib:brca2 beta={b_hi:.4f} (p={p_hi:.3g}).",
         "p_value": _round(p_hi),"effect_estimate":_round(b_hi),"significant":p_hi<0.05},
    ],
)
print("Iter24 done")

# ------------------- Iteration 25 ------------------- #
# Final omnibus model: comprehensive adjusted model, report top predictors and confirm the headline finding.
formula25 = (
    f"{OUT} ~ ecog_ps + mcrpc + visceral_mets + log_psa + log_ldh + albumin_g_dl + "
    f"hemoglobin_g_dl + log_alp + age_years + nlr + weight_loss_pct_6mo + log_crp + "
    f"fatigue_grade + pain_nrs + appetite_loss_grade + "
    f"brca2_mutation + ar_v7_positive + msi_high + psma_high + tp53_mutation + pten_loss + "
    f"treatment_enzalutamide + treatment_abiraterone + treatment_docetaxel + "
    f"treatment_olaparib + treatment_lu177_psma + treatment_pembrolizumab + "
    f"treatment_olaparib:brca2_mutation + treatment_pembrolizumab:msi_high + "
    f"treatment_lu177_psma:psma_high + treatment_enzalutamide:ar_v7_positive"
)
m25 = ols_fit(formula25)
key_terms25 = [
    "ecog_ps","mcrpc","albumin_g_dl","log_psa","log_ldh","log_alp","nlr","weight_loss_pct_6mo",
    "log_crp","fatigue_grade","pain_nrs","appetite_loss_grade","brca2_mutation",
    "treatment_olaparib","treatment_olaparib:brca2_mutation","treatment_pembrolizumab:msi_high",
    "treatment_lu177_psma:psma_high","treatment_enzalutamide:ar_v7_positive",
]
res25 = {t: beta_p(m25, t) for t in key_terms25 if t in m25.params.index}

add_iteration(
    25,
    [
        {"id":"h25.1","text":"In a comprehensive multivariable model including all six treatments, six key biomarkers, four pre-specified treatment×biomarker interactions, and core clinical/lab/symptom prognostic factors, the only large and significant treatment×biomarker interaction on pfs_months is treatment_olaparib × brca2_mutation (positive sign).","kind":"refined"},
        {"id":"h25.2","text":"In the same comprehensive multivariable model, the strongest negative prognostic factors for pfs_months are higher ecog_ps and lower albumin_g_dl, while higher log_psa, log_ldh, NLR, log_crp, weight_loss_pct_6mo, and the patient-reported symptom grades all contribute negatively.","kind":"refined"},
    ],
    [
        {"hypothesis_ids":["h25.1"],"code":"comprehensive OLS w/ four treatment×biomarker interactions",
         "result_summary": "; ".join([f"{t} beta={v[0]:.4f} (p={v[1]:.3g})" for t,v in res25.items() if 'treatment' in t and ':' in t]),
         "p_value": _round(res25.get("treatment_olaparib:brca2_mutation",(None,1))[1]),
         "effect_estimate": _round(res25.get("treatment_olaparib:brca2_mutation",(0,1))[0]),
         "significant": res25.get("treatment_olaparib:brca2_mutation",(0,1))[1]<0.05},
        {"hypothesis_ids":["h25.2"],"code":"same comprehensive model, prognostic-factor terms",
         "result_summary": "; ".join([f"{t} beta={v[0]:.4f} (p={v[1]:.3g})" for t,v in res25.items() if 'treatment' not in t]),
         "p_value": _round(res25.get("ecog_ps",(None,1))[1]),
         "effect_estimate": _round(res25.get("ecog_ps",(0,1))[0]),
         "significant": res25.get("ecog_ps",(0,1))[1]<0.05},
    ],
)
print("Iter25 done")

TRANSCRIPT["iterations"] = ITERATIONS
with open("transcript.json","w") as f:
    json.dump(TRANSCRIPT, f, indent=2, default=str)
print("Final transcript saved.")
