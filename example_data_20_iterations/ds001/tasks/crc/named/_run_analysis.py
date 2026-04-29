"""Iterative analysis of ds001_crc.

Outputs JSON with a list of iteration result blocks; the transcript is
assembled separately.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet("dataset.parquet")
print("loaded", df.shape)

# Pre-derive useful flags
df["ras_wt"] = ((df["kras_mutation"] == 0) & (df["nras_mutation"] == 0)).astype(int)
df["left_sided"] = (df["right_sided_primary"] == 0).astype(int)

results = {}

def linreg(formula, name):
    m = smf.ols(formula, data=df).fit()
    return m, name

def report(m, term):
    coef = m.params[term]
    p = m.pvalues[term]
    return float(coef), float(p), bool(p < 0.05)

def ttest(g1, g0):
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return float(np.mean(g1) - np.mean(g0)), float(p), bool(p < 0.05)

# ----- Iteration 1: ECOG, stage IV main effects on PFS -----
out1 = {}
m = smf.ols("pfs_months ~ ecog_ps", data=df).fit()
out1["ecog_ps"] = report(m, "ecog_ps")
m = smf.ols("pfs_months ~ stage_iv", data=df).fit()
out1["stage_iv"] = report(m, "stage_iv")
results["iter1"] = out1

# ----- Iteration 2: albumin, LDH, CRP, NLR, weight loss -----
out2 = {}
for v in ["albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr", "weight_loss_pct_6mo"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out2[v] = report(m, v)
results["iter2"] = out2

# ----- Iteration 3: tumor side, age, sex -----
out3 = {}
m = smf.ols("pfs_months ~ right_sided_primary", data=df).fit()
out3["right_sided_primary"] = report(m, "right_sided_primary")
m = smf.ols("pfs_months ~ age_years", data=df).fit()
out3["age_years"] = report(m, "age_years")
m = smf.ols("pfs_months ~ sex_female", data=df).fit()
out3["sex_female"] = report(m, "sex_female")
results["iter3"] = out3

# ----- Iteration 4: liver mets, bone mets, pleural effusion -----
out4 = {}
for v in ["liver_mets", "bone_mets", "pleural_effusion", "adrenal_mets",
          "contralateral_lung_mets", "pericardial_effusion"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out4[v] = report(m, v)
results["iter4"] = out4

# ----- Iteration 5: CEA, alk phos, hemoglobin, calcium -----
out5 = {}
for v in ["cea_ng_ml", "alkaline_phosphatase_u_l", "hemoglobin_g_dl",
          "calcium_mg_dl", "platelets_k_ul"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out5[v] = report(m, v)
results["iter5"] = out5

# ----- Iteration 6: driver mutations main effects -----
out6 = {}
for v in ["braf_v600e", "kras_mutation", "nras_mutation", "msi_high",
          "her2_amplified", "ntrk_fusion", "tp53_mutation", "pik3ca_mutation"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out6[v] = report(m, v)
results["iter6"] = out6

# ----- Iteration 7: treatment main effects -----
out7 = {}
for v in ["treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
          "treatment_encorafenib", "treatment_trastuzumab_tucatinib",
          "treatment_regorafenib"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out7[v] = report(m, v)
results["iter7"] = out7

# ----- Iteration 8: cetuximab × KRAS/NRAS interaction (RAS WT vs MUT) -----
out8 = {}
m = smf.ols("pfs_months ~ treatment_cetuximab * kras_mutation", data=df).fit()
out8["cet_main"] = report(m, "treatment_cetuximab")
out8["kras_main"] = report(m, "kras_mutation")
out8["cet_x_kras"] = report(m, "treatment_cetuximab:kras_mutation")
m = smf.ols("pfs_months ~ treatment_cetuximab * ras_wt", data=df).fit()
out8["cet_main_v_raswt"] = report(m, "treatment_cetuximab")
out8["raswt_main"] = report(m, "ras_wt")
out8["cet_x_raswt"] = report(m, "treatment_cetuximab:ras_wt")
# subgroup deltas
for sub_col, sub_val, label in [("ras_wt", 1, "RAS_WT"), ("ras_wt", 0, "RAS_MUT"),
                                ("kras_mutation", 0, "KRAS_WT"), ("kras_mutation", 1, "KRAS_MUT")]:
    sub = df[df[sub_col] == sub_val]
    g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
    g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
    out8[f"cet_in_{label}"] = ttest(g1, g0)
results["iter8"] = out8

# ----- Iteration 9: cetuximab × tumor side -----
out9 = {}
m = smf.ols("pfs_months ~ treatment_cetuximab * right_sided_primary", data=df).fit()
out9["cet_x_right"] = report(m, "treatment_cetuximab:right_sided_primary")
out9["cet_main"] = report(m, "treatment_cetuximab")
out9["right_main"] = report(m, "right_sided_primary")
for sub_val, label in [(0, "left_sided"), (1, "right_sided")]:
    sub = df[df["right_sided_primary"] == sub_val]
    g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
    g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
    out9[f"cet_in_{label}"] = ttest(g1, g0)
# triple: cetuximab × side × ras_wt
m = smf.ols("pfs_months ~ treatment_cetuximab * right_sided_primary * ras_wt", data=df).fit()
out9["triple_term_p"] = float(m.pvalues["treatment_cetuximab:right_sided_primary:ras_wt"])
out9["triple_term_coef"] = float(m.params["treatment_cetuximab:right_sided_primary:ras_wt"])
# Best subgroup: left-sided RAS WT
sub = df[(df["ras_wt"] == 1) & (df["right_sided_primary"] == 0)]
g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
out9["cet_in_leftRASWT"] = ttest(g1, g0)
sub = df[(df["ras_wt"] == 0) | (df["right_sided_primary"] == 1)]
g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
out9["cet_in_unfavorable"] = ttest(g1, g0)
results["iter9"] = out9

# ----- Iteration 10: pembrolizumab × MSI-high -----
out10 = {}
m = smf.ols("pfs_months ~ treatment_pembrolizumab * msi_high", data=df).fit()
out10["pembro_main"] = report(m, "treatment_pembrolizumab")
out10["msi_main"] = report(m, "msi_high")
out10["pembro_x_msi"] = report(m, "treatment_pembrolizumab:msi_high")
for sub_val, label in [(1, "MSI_high"), (0, "MSS")]:
    sub = df[df["msi_high"] == sub_val]
    g1 = sub.loc[sub["treatment_pembrolizumab"] == 1, "pfs_months"]
    g0 = sub.loc[sub["treatment_pembrolizumab"] == 0, "pfs_months"]
    out10[f"pembro_in_{label}"] = ttest(g1, g0)
results["iter10"] = out10

# ----- Iteration 11: encorafenib × BRAF V600E -----
out11 = {}
m = smf.ols("pfs_months ~ treatment_encorafenib * braf_v600e", data=df).fit()
out11["enco_main"] = report(m, "treatment_encorafenib")
out11["braf_main"] = report(m, "braf_v600e")
out11["enco_x_braf"] = report(m, "treatment_encorafenib:braf_v600e")
for sub_val, label in [(1, "BRAFmut"), (0, "BRAFwt")]:
    sub = df[df["braf_v600e"] == sub_val]
    g1 = sub.loc[sub["treatment_encorafenib"] == 1, "pfs_months"]
    g0 = sub.loc[sub["treatment_encorafenib"] == 0, "pfs_months"]
    out11[f"enco_in_{label}"] = ttest(g1, g0)
results["iter11"] = out11

# ----- Iteration 12: trastuzumab+tucatinib × HER2 -----
out12 = {}
m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified", data=df).fit()
out12["t_main"] = report(m, "treatment_trastuzumab_tucatinib")
out12["her2_main"] = report(m, "her2_amplified")
out12["t_x_her2"] = report(m, "treatment_trastuzumab_tucatinib:her2_amplified")
for sub_val, label in [(1, "HER2amp"), (0, "HER2neg")]:
    sub = df[df["her2_amplified"] == sub_val]
    g1 = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 1, "pfs_months"]
    g0 = sub.loc[sub["treatment_trastuzumab_tucatinib"] == 0, "pfs_months"]
    out12[f"t_in_{label}"] = ttest(g1, g0)
results["iter12"] = out12

# ----- Iteration 13: bevacizumab × tumor side, & main multivariable adj -----
out13 = {}
m = smf.ols("pfs_months ~ treatment_bevacizumab * right_sided_primary", data=df).fit()
out13["bev_x_right"] = report(m, "treatment_bevacizumab:right_sided_primary")
out13["bev_main"] = report(m, "treatment_bevacizumab")
m = smf.ols("pfs_months ~ treatment_bevacizumab + ecog_ps + albumin_g_dl + ldh_u_l + stage_iv + age_years + crp_mg_l", data=df).fit()
out13["bev_adj"] = report(m, "treatment_bevacizumab")
results["iter13"] = out13

# ----- Iteration 14: regorafenib main, prior lines of therapy -----
out14 = {}
m = smf.ols("pfs_months ~ treatment_regorafenib", data=df).fit()
out14["rego_main"] = report(m, "treatment_regorafenib")
m = smf.ols("pfs_months ~ prior_lines_of_therapy", data=df).fit()
out14["prior_lines"] = report(m, "prior_lines_of_therapy")
m = smf.ols("pfs_months ~ treatment_regorafenib + prior_lines_of_therapy", data=df).fit()
out14["rego_adj_lines"] = report(m, "treatment_regorafenib")
out14["lines_adj_rego"] = report(m, "prior_lines_of_therapy")
results["iter14"] = out14

# ----- Iteration 15: symptom burden -----
out15 = {}
for v in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "cough_grade", "appetite_loss_grade"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out15[v] = report(m, v)
results["iter15"] = out15

# ----- Iteration 16: comorbidities -----
out16 = {}
for v in ["diabetes_mellitus", "hypertension", "copd", "chronic_kidney_disease",
          "heart_failure", "coronary_artery_disease", "atrial_fibrillation",
          "venous_thromboembolism_history", "autoimmune_disease",
          "interstitial_lung_disease_history", "depression_anxiety_diagnosis",
          "prior_malignancy"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out16[v] = report(m, v)
results["iter16"] = out16

# ----- Iteration 17: race / insurance / rural disparities -----
out17 = {}
m = smf.ols("pfs_months ~ C(race_ethnicity, Treatment(reference='white'))", data=df).fit()
for term in m.params.index:
    if "race_ethnicity" in term:
        out17[term] = (float(m.params[term]), float(m.pvalues[term]),
                       bool(m.pvalues[term] < 0.05))
m = smf.ols("pfs_months ~ C(insurance_type, Treatment(reference='private'))", data=df).fit()
for term in m.params.index:
    if "insurance_type" in term:
        out17[term] = (float(m.params[term]), float(m.pvalues[term]),
                       bool(m.pvalues[term] < 0.05))
m = smf.ols("pfs_months ~ rural_residence", data=df).fit()
out17["rural_residence"] = report(m, "rural_residence")
results["iter17"] = out17

# ----- Iteration 18: SNPs - main effects screen, adjust for ECOG/albumin/LDH -----
out18 = {}
snps = [c for c in df.columns if c.startswith("snp_")]
for snp in snps:
    m = smf.ols(f"pfs_months ~ {snp}", data=df).fit()
    out18[snp] = report(m, snp)
results["iter18"] = out18

# ----- Iteration 19: BMI, smoking, education -----
out19 = {}
for v in ["bmi", "smoking_pack_years", "education_years"]:
    m = smf.ols(f"pfs_months ~ {v}", data=df).fit()
    out19[v] = report(m, v)
results["iter19"] = out19

# ----- Iteration 20: combined favorable/unfavorable cetuximab subgroup with adjustment -----
out20 = {}
m = smf.ols("pfs_months ~ treatment_cetuximab * ras_wt * left_sided + ecog_ps + albumin_g_dl + ldh_u_l + stage_iv", data=df).fit()
for term in ["treatment_cetuximab", "treatment_cetuximab:ras_wt",
             "treatment_cetuximab:left_sided",
             "treatment_cetuximab:ras_wt:left_sided"]:
    out20[term] = (float(m.params[term]), float(m.pvalues[term]),
                   bool(m.pvalues[term] < 0.05))
results["iter20"] = out20

# ----- Iteration 21: pembrolizumab benefit adjusted -----
out21 = {}
m = smf.ols("pfs_months ~ treatment_pembrolizumab * msi_high + ecog_ps + albumin_g_dl + ldh_u_l + stage_iv + age_years", data=df).fit()
out21["pembro_adj"] = (float(m.params["treatment_pembrolizumab"]),
                      float(m.pvalues["treatment_pembrolizumab"]),
                      bool(m.pvalues["treatment_pembrolizumab"] < 0.05))
out21["pembro_x_msi_adj"] = (float(m.params["treatment_pembrolizumab:msi_high"]),
                             float(m.pvalues["treatment_pembrolizumab:msi_high"]),
                             bool(m.pvalues["treatment_pembrolizumab:msi_high"] < 0.05))
out21["msi_adj"] = (float(m.params["msi_high"]), float(m.pvalues["msi_high"]),
                   bool(m.pvalues["msi_high"] < 0.05))
results["iter21"] = out21

# ----- Iteration 22: encorafenib adjusted, plus BRAF prognostic adj -----
out22 = {}
m = smf.ols("pfs_months ~ treatment_encorafenib * braf_v600e + ecog_ps + albumin_g_dl + ldh_u_l + stage_iv", data=df).fit()
out22["enco_adj"] = (float(m.params["treatment_encorafenib"]),
                     float(m.pvalues["treatment_encorafenib"]),
                     bool(m.pvalues["treatment_encorafenib"] < 0.05))
out22["enco_x_braf_adj"] = (float(m.params["treatment_encorafenib:braf_v600e"]),
                            float(m.pvalues["treatment_encorafenib:braf_v600e"]),
                            bool(m.pvalues["treatment_encorafenib:braf_v600e"] < 0.05))
out22["braf_adj"] = (float(m.params["braf_v600e"]),
                    float(m.pvalues["braf_v600e"]),
                    bool(m.pvalues["braf_v600e"] < 0.05))
results["iter22"] = out22

# ----- Iteration 23: HER2 trastuzumab+tucatinib adjusted -----
out23 = {}
m = smf.ols("pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified + ecog_ps + albumin_g_dl + ldh_u_l + stage_iv", data=df).fit()
out23["t_adj"] = (float(m.params["treatment_trastuzumab_tucatinib"]),
                 float(m.pvalues["treatment_trastuzumab_tucatinib"]),
                 bool(m.pvalues["treatment_trastuzumab_tucatinib"] < 0.05))
out23["t_x_her2_adj"] = (float(m.params["treatment_trastuzumab_tucatinib:her2_amplified"]),
                         float(m.pvalues["treatment_trastuzumab_tucatinib:her2_amplified"]),
                         bool(m.pvalues["treatment_trastuzumab_tucatinib:her2_amplified"] < 0.05))
out23["her2_adj"] = (float(m.params["her2_amplified"]),
                    float(m.pvalues["her2_amplified"]),
                    bool(m.pvalues["her2_amplified"] < 0.05))
results["iter23"] = out23

# ----- Iteration 24: full adjusted prognostic model -----
out24 = {}
formula = ("pfs_months ~ ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + crp_mg_l + nlr "
           "+ weight_loss_pct_6mo + age_years + liver_mets + bone_mets + cea_ng_ml "
           "+ right_sided_primary + braf_v600e + msi_high + kras_mutation "
           "+ treatment_cetuximab + treatment_bevacizumab + treatment_pembrolizumab "
           "+ treatment_encorafenib + treatment_trastuzumab_tucatinib + treatment_regorafenib "
           "+ ras_wt:treatment_cetuximab + msi_high:treatment_pembrolizumab "
           "+ braf_v600e:treatment_encorafenib + her2_amplified:treatment_trastuzumab_tucatinib")
m = smf.ols(formula, data=df).fit()
for t in m.params.index:
    out24[t] = (float(m.params[t]), float(m.pvalues[t]),
               bool(m.pvalues[t] < 0.05))
results["iter24"] = out24
results["iter24_r2"] = float(m.rsquared)

# ----- Iteration 25: cetuximab×right+ras combined effect, regorafenib in late lines -----
out25 = {}
# Cetuximab benefit in left-sided RAS-WT vs harm in right-sided RAS-MUT
sub = df[(df["right_sided_primary"] == 0) & (df["ras_wt"] == 1)]
g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
out25["cet_left_RASWT"] = ttest(g1, g0)
sub = df[(df["right_sided_primary"] == 1) & (df["ras_wt"] == 0)]
g1 = sub.loc[sub["treatment_cetuximab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_cetuximab"] == 0, "pfs_months"]
out25["cet_right_RASmut"] = ttest(g1, g0)
# regorafenib by prior lines (>=2 vs <2)
df["heavily_pretreated"] = (df["prior_lines_of_therapy"] >= 2).astype(int)
m = smf.ols("pfs_months ~ treatment_regorafenib * heavily_pretreated", data=df).fit()
out25["rego_x_heavy"] = (float(m.params["treatment_regorafenib:heavily_pretreated"]),
                         float(m.pvalues["treatment_regorafenib:heavily_pretreated"]),
                         bool(m.pvalues["treatment_regorafenib:heavily_pretreated"] < 0.05))
# Pembrolizumab in MSI-high w/ adjustment final
sub = df[df["msi_high"] == 1]
g1 = sub.loc[sub["treatment_pembrolizumab"] == 1, "pfs_months"]
g0 = sub.loc[sub["treatment_pembrolizumab"] == 0, "pfs_months"]
out25["pembro_in_MSI"] = ttest(g1, g0)
# Combined biomarker score: NLR>3 & albumin<3.5
df["high_inflam"] = ((df["nlr"] > 3) & (df["albumin_g_dl"] < 3.5)).astype(int)
m = smf.ols("pfs_months ~ high_inflam", data=df).fit()
out25["high_inflam"] = report(m, "high_inflam")
results["iter25"] = out25

with open("_run_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("done")
