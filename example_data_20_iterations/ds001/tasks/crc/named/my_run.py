"""Run a structured set of analyses across iterations and dump results to JSON."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
print("Loaded:", df.shape)

results = {}


def lin(formula, key, focal):
    """Fit OLS, return coef, p, n for the focal term."""
    m = smf.ols(formula, data=df).fit()
    coef = m.params.get(focal, np.nan)
    pv = m.pvalues.get(focal, np.nan)
    return {"coef": float(coef), "p": float(pv), "n": int(m.nobs), "formula": formula, "focal": focal}


def ttest_pfs(mask_label, outcome="pfs_months"):
    """t-test PFS by binary mask label name."""
    g1 = df.loc[df[mask_label] == 1, outcome]
    g0 = df.loc[df[mask_label] == 0, outcome]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return {"mean1": float(g1.mean()), "mean0": float(g0.mean()), "diff": float(g1.mean()-g0.mean()),
            "p": float(p), "n1": int(g1.size), "n0": int(g0.size)}


# -------- ITER 1: main treatment effects on PFS --------
for trt in ["treatment_cetuximab", "treatment_bevacizumab", "treatment_pembrolizumab",
            "treatment_encorafenib", "treatment_trastuzumab_tucatinib", "treatment_regorafenib"]:
    results[f"i1_{trt}"] = ttest_pfs(trt)

# -------- ITER 2: biomarker main effects --------
for bm in ["kras_mutation", "nras_mutation", "braf_v600e", "msi_high",
           "her2_amplified", "ntrk_fusion", "tp53_mutation", "pik3ca_mutation"]:
    results[f"i2_{bm}"] = ttest_pfs(bm)

# -------- ITER 3: clinical baseline --------
for v in ["stage_iv", "right_sided_primary"]:
    results[f"i3_{v}"] = ttest_pfs(v)
# ECOG continuous
results["i3_ecog"] = lin("pfs_months ~ ecog_ps", "ecog", "ecog_ps")
results["i3_age"] = lin("pfs_months ~ age_years", "age", "age_years")
results["i3_albumin"] = lin("pfs_months ~ albumin_g_dl", "alb", "albumin_g_dl")
results["i3_ldh"] = lin("pfs_months ~ ldh_u_l", "ldh", "ldh_u_l")
results["i3_cea"] = lin("pfs_months ~ cea_ng_ml", "cea", "cea_ng_ml")

# -------- ITER 4: cetuximab × RAS (KRAS, NRAS) interaction --------
results["i4_cet_kras_main"] = lin(
    "pfs_months ~ treatment_cetuximab * kras_mutation",
    "cet_kras_main", "treatment_cetuximab"
)
results["i4_cet_kras_ix"] = lin(
    "pfs_months ~ treatment_cetuximab * kras_mutation",
    "cet_kras_ix", "treatment_cetuximab:kras_mutation"
)
results["i4_cet_nras_ix"] = lin(
    "pfs_months ~ treatment_cetuximab * nras_mutation",
    "cet_nras_ix", "treatment_cetuximab:nras_mutation"
)
# Combined RAS wild-type
df["ras_wt"] = ((df["kras_mutation"] == 0) & (df["nras_mutation"] == 0)).astype(int)
results["i4_cet_raswt_ix"] = lin(
    "pfs_months ~ treatment_cetuximab * ras_wt",
    "cet_raswt_ix", "treatment_cetuximab:ras_wt"
)
# Stratified
for sub in [0, 1]:
    sub_df = df[df["kras_mutation"] == sub]
    g1 = sub_df.loc[sub_df["treatment_cetuximab"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_cetuximab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i4_cet_in_kras{sub}"] = {"mean1": float(g1.mean()), "mean0": float(g0.mean()),
                                        "diff": float(g1.mean()-g0.mean()), "p": float(p),
                                        "n1": int(g1.size), "n0": int(g0.size)}

# -------- ITER 5: encorafenib × BRAF V600E interaction --------
results["i5_enc_brafix"] = lin(
    "pfs_months ~ treatment_encorafenib * braf_v600e",
    "enc_brafix", "treatment_encorafenib:braf_v600e"
)
for sub in [0, 1]:
    sub_df = df[df["braf_v600e"] == sub]
    g1 = sub_df.loc[sub_df["treatment_encorafenib"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_encorafenib"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i5_enc_in_braf{sub}"] = {"mean1": float(g1.mean()), "mean0": float(g0.mean()),
                                        "diff": float(g1.mean()-g0.mean()), "p": float(p),
                                        "n1": int(g1.size), "n0": int(g0.size)}

# -------- ITER 6: pembrolizumab × MSI-high interaction --------
results["i6_pem_msi_ix"] = lin(
    "pfs_months ~ treatment_pembrolizumab * msi_high",
    "pem_msi_ix", "treatment_pembrolizumab:msi_high"
)
for sub in [0, 1]:
    sub_df = df[df["msi_high"] == sub]
    g1 = sub_df.loc[sub_df["treatment_pembrolizumab"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_pembrolizumab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i6_pem_in_msi{sub}"] = {"mean1": float(g1.mean()), "mean0": float(g0.mean()),
                                       "diff": float(g1.mean()-g0.mean()), "p": float(p),
                                       "n1": int(g1.size), "n0": int(g0.size)}

# -------- ITER 7: trastuzumab+tucatinib × HER2 amplified --------
results["i7_her2_ix"] = lin(
    "pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified",
    "her2_ix", "treatment_trastuzumab_tucatinib:her2_amplified"
)
for sub in [0, 1]:
    sub_df = df[df["her2_amplified"] == sub]
    g1 = sub_df.loc[sub_df["treatment_trastuzumab_tucatinib"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_trastuzumab_tucatinib"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i7_her2trt_in_her2{sub}"] = {"mean1": float(g1.mean()), "mean0": float(g0.mean()),
                                            "diff": float(g1.mean()-g0.mean()), "p": float(p),
                                            "n1": int(g1.size), "n0": int(g0.size)}

# -------- ITER 8: cetuximab × right-sided interaction (CRC sidedness) --------
results["i8_cet_side_ix"] = lin(
    "pfs_months ~ treatment_cetuximab * right_sided_primary",
    "cet_side_ix", "treatment_cetuximab:right_sided_primary"
)
for sub in [0, 1]:
    sub_df = df[df["right_sided_primary"] == sub]
    g1 = sub_df.loc[sub_df["treatment_cetuximab"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_cetuximab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i8_cet_in_side{sub}"] = {"mean1": float(g1.mean()), "mean0": float(g0.mean()),
                                        "diff": float(g1.mean()-g0.mean()), "p": float(p),
                                        "n1": int(g1.size), "n0": int(g0.size)}

# -------- ITER 9: inflammation/nutrition labs as continuous predictors --------
for v in ["nlr", "crp_mg_l", "albumin_g_dl", "hemoglobin_g_dl", "alkaline_phosphatase_u_l",
         "weight_loss_pct_6mo", "platelets_k_ul"]:
    results[f"i9_{v}"] = lin(f"pfs_months ~ {v}", v, v)

# -------- ITER 10: metastatic site burden --------
for v in ["liver_mets", "bone_mets", "adrenal_mets"]:
    results[f"i10_{v}"] = ttest_pfs(v)

# Mets count
df["mets_count"] = df[["liver_mets", "bone_mets", "adrenal_mets"]].sum(axis=1)
results["i10_mets_count"] = lin("pfs_months ~ mets_count", "mc", "mets_count")

# -------- ITER 11: symptom burden / composite --------
for v in ["fatigue_grade", "pain_nrs", "dyspnea_grade", "appetite_loss_grade"]:
    results[f"i11_{v}"] = lin(f"pfs_months ~ {v}", v, v)

df["sx_burden"] = df[["fatigue_grade", "dyspnea_grade", "cough_grade",
                      "appetite_loss_grade"]].sum(axis=1) + df["pain_nrs"]
results["i11_sx_burden"] = lin("pfs_months ~ sx_burden", "sx", "sx_burden")

# -------- ITER 12: demographics & social --------
results["i12_sex"] = ttest_pfs("sex_female")
results["i12_rural"] = ttest_pfs("rural_residence")
results["i12_smoking"] = lin("pfs_months ~ smoking_pack_years", "smk", "smoking_pack_years")
results["i12_education"] = lin("pfs_months ~ education_years", "edu", "education_years")
# race/insurance
m = smf.ols("pfs_months ~ C(race_ethnicity)", data=df).fit()
results["i12_race_F"] = {"F": float(m.fvalue), "p": float(m.f_pvalue), "n": int(m.nobs)}
m = smf.ols("pfs_months ~ C(insurance_type)", data=df).fit()
results["i12_insurance_F"] = {"F": float(m.fvalue), "p": float(m.f_pvalue), "n": int(m.nobs)}

# -------- ITER 13: SNP main effects (PGx-flavored) --------
snp_cols = [c for c in df.columns if c.startswith("snp_")]
snp_main = {}
for s in snp_cols:
    g1 = df.loc[df[s] == 1, "pfs_months"]
    g0 = df.loc[df[s] == 0, "pfs_months"]
    if g1.size > 50 and g0.size > 50:
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        snp_main[s] = {"diff": float(g1.mean()-g0.mean()), "p": float(p)}
results["i13_snp_main"] = snp_main

# -------- ITER 14: multivariable model controlling key prognostic factors --------
mv_formula = ("pfs_months ~ ecog_ps + age_years + stage_iv + albumin_g_dl + ldh_u_l + "
              "nlr + crp_mg_l + treatment_cetuximab*kras_mutation + "
              "treatment_encorafenib*braf_v600e + treatment_pembrolizumab*msi_high + "
              "treatment_trastuzumab_tucatinib*her2_amplified + treatment_bevacizumab + "
              "treatment_regorafenib + right_sided_primary + liver_mets")
mv = smf.ols(mv_formula, data=df).fit()
results["i14_mv"] = {"params": mv.params.to_dict(), "p": mv.pvalues.to_dict(),
                     "rsq": float(mv.rsquared), "n": int(mv.nobs)}

# -------- ITER 15: ECOG x treatment_cetuximab interaction (does fitness modify benefit?) --------
results["i15_ecog_cet_ix"] = lin(
    "pfs_months ~ ecog_ps * treatment_cetuximab",
    "ecog_cet", "ecog_ps:treatment_cetuximab"
)

# -------- ITER 16: Combined RAS-WT + cetuximab + left-sided (canonical CRC story) --------
df["raswt_left_cet"] = ((df["ras_wt"] == 1) & (df["right_sided_primary"] == 0) &
                         (df["treatment_cetuximab"] == 1)).astype(int)
results["i16_raswt_left_cet"] = ttest_pfs("raswt_left_cet")
# Three-way: cetuximab benefit in RAS-WT stratified by side
ras_wt_df = df[df["ras_wt"] == 1]
for sub_label, sub_df in [("left", ras_wt_df[ras_wt_df["right_sided_primary"] == 0]),
                           ("right", ras_wt_df[ras_wt_df["right_sided_primary"] == 1])]:
    g1 = sub_df.loc[sub_df["treatment_cetuximab"] == 1, "pfs_months"]
    g0 = sub_df.loc[sub_df["treatment_cetuximab"] == 0, "pfs_months"]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    results[f"i16_cet_raswt_{sub_label}"] = {
        "mean1": float(g1.mean()), "mean0": float(g0.mean()),
        "diff": float(g1.mean()-g0.mean()), "p": float(p),
        "n1": int(g1.size), "n0": int(g0.size)
    }

# -------- ITER 17: bevacizumab effect overall and in subgroups (KRAS, side) --------
results["i17_bev_kras_ix"] = lin(
    "pfs_months ~ treatment_bevacizumab * kras_mutation",
    "bev_kras", "treatment_bevacizumab:kras_mutation"
)
results["i17_bev_side_ix"] = lin(
    "pfs_months ~ treatment_bevacizumab * right_sided_primary",
    "bev_side", "treatment_bevacizumab:right_sided_primary"
)
# bev × liver mets (anti-VEGF in visceral disease)
results["i17_bev_liver_ix"] = lin(
    "pfs_months ~ treatment_bevacizumab * liver_mets",
    "bev_liver", "treatment_bevacizumab:liver_mets"
)

# -------- ITER 18: Regorafenib in late lines --------
results["i18_reg_lines_ix"] = lin(
    "pfs_months ~ treatment_regorafenib * prior_lines_of_therapy",
    "reg_lines", "treatment_regorafenib:prior_lines_of_therapy"
)

# -------- ITER 19: NLR-stratified treatment effects (immune fitness) --------
df["nlr_high"] = (df["nlr"] >= df["nlr"].median()).astype(int)
results["i19_pem_nlr_ix"] = lin(
    "pfs_months ~ treatment_pembrolizumab * nlr_high",
    "pem_nlr", "treatment_pembrolizumab:nlr_high"
)

# -------- ITER 20: Final adjusted treatment effects controlling biomarkers --------
final_formula = ("pfs_months ~ treatment_cetuximab + treatment_bevacizumab + "
                 "treatment_pembrolizumab + treatment_encorafenib + "
                 "treatment_trastuzumab_tucatinib + treatment_regorafenib + "
                 "kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + "
                 "ecog_ps + age_years + stage_iv + albumin_g_dl + ldh_u_l + "
                 "right_sided_primary + liver_mets + nlr + crp_mg_l")
fm = smf.ols(final_formula, data=df).fit()
results["i20_final"] = {"params": fm.params.to_dict(), "p": fm.pvalues.to_dict(),
                        "rsq": float(fm.rsquared), "n": int(fm.nobs)}

# Save
with open("my_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("DONE. Keys:", len(results))
print("Sample treatment cetux:", results["i1_treatment_cetuximab"])
print("cet x kras IX:", results["i4_cet_kras_ix"])
print("enc x braf IX:", results["i5_enc_brafix"])
print("pem x msi IX:", results["i6_pem_msi_ix"])
print("her2 trt x her2 IX:", results["i7_her2_ix"])
print("Final R^2:", results["i20_final"]["rsq"])
