"""Comprehensive 25-iteration analysis for ds001_breast.

Outputs analysis results to ./my_results.json keyed by analysis tag.
The transcript is then assembled from these results by build_my_transcript.py.
"""
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
RESULTS = {}

OUT = "pfs_months"


def store(tag, *, p, eff, sig, summary):
    """Store one analysis result; sig auto p<0.05 if None."""
    if sig is None and p is not None:
        sig = bool(p < 0.05)
    RESULTS[tag] = {
        "p_value": float(p) if p is not None else None,
        "effect_estimate": float(eff) if eff is not None else None,
        "significant": bool(sig) if sig is not None else None,
        "result_summary": summary,
    }
    print(f"[{tag}] eff={eff} p={p} sig={sig}")


def ttest(group_col, label):
    g1 = DF.loc[DF[group_col] == 1, OUT]
    g0 = DF.loc[DF[group_col] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    eff = float(g1.mean() - g0.mean())
    return t.pvalue, eff, g1.mean(), g0.mean(), len(g1), len(g0)


def linreg(formula, focus_term):
    m = smf.ols(formula, data=DF).fit()
    p = m.pvalues.get(focus_term)
    eff = m.params.get(focus_term)
    return float(p) if p is not None else None, float(eff) if eff is not None else None, m


# =========================================================
# ITERATION 1 — Main effects of each treatment on PFS
# =========================================================
print("\n=== ITERATION 1: treatment main effects ===")
for trt in [
    "treatment_tamoxifen",
    "treatment_palbociclib",
    "treatment_trastuzumab",
    "treatment_olaparib",
    "treatment_sacituzumab_govitecan",
    "treatment_pembrolizumab",
]:
    p, eff, m1, m0, n1, n0 = ttest(trt, trt)
    store(
        f"i1_{trt}_main",
        p=p, eff=eff, sig=None,
        summary=f"Mean PFS {m1:.3f} on {trt} (n={n1}) vs {m0:.3f} off (n={n0}); diff={eff:+.3f} mo (Welch t-test).",
    )


# =========================================================
# ITERATION 2 — Tamoxifen × ER status interaction
# =========================================================
print("\n=== ITERATION 2: tamoxifen × ER interaction ===")
# ER+ subgroup
sub = DF[DF["er_positive"] == 1]
g1 = sub.loc[sub["treatment_tamoxifen"] == 1, OUT]
g0 = sub.loc[sub["treatment_tamoxifen"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i2_tam_in_erpos",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"In ER+ (n={len(sub)}): tamoxifen mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
)
# ER- subgroup
sub = DF[DF["er_positive"] == 0]
g1 = sub.loc[sub["treatment_tamoxifen"] == 1, OUT]
g0 = sub.loc[sub["treatment_tamoxifen"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i2_tam_in_erneg",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"In ER- (n={len(sub)}): tamoxifen mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
)
# Interaction term in OLS
p, eff, m = linreg(
    f"{OUT} ~ treatment_tamoxifen * er_positive",
    "treatment_tamoxifen:er_positive",
)
store(
    "i2_tam_x_er_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction tamoxifen:er_positive coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 3 — Trastuzumab × HER2 interaction
# =========================================================
print("\n=== ITERATION 3: trastuzumab × HER2 interaction ===")
for tag, mask, label in [
    ("i3_tras_in_her2pos", DF["her2_positive"] == 1, "HER2+"),
    ("i3_tras_in_her2neg", DF["her2_positive"] == 0, "HER2-"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_trastuzumab"] == 1, OUT]
    g0 = sub.loc[sub["treatment_trastuzumab"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): trastuzumab mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )

p, eff, m = linreg(
    f"{OUT} ~ treatment_trastuzumab * her2_positive",
    "treatment_trastuzumab:her2_positive",
)
store(
    "i3_tras_x_her2_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction trastuzumab:her2_positive coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 4 — Olaparib × BRCA mutation interaction
# =========================================================
print("\n=== ITERATION 4: olaparib × BRCA ===")
DF["brca_any"] = ((DF["brca1_mutation"] == 1) | (DF["brca2_mutation"] == 1)).astype(int)
for tag, mask, label in [
    ("i4_olap_in_brca_any", DF["brca_any"] == 1, "BRCA1/2 mutated"),
    ("i4_olap_in_brca_wt", DF["brca_any"] == 0, "BRCA wild-type"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_olaparib"] == 1, OUT]
    g0 = sub.loc[sub["treatment_olaparib"] == 0, OUT]
    if len(g1) > 5 and len(g0) > 5:
        t = stats.ttest_ind(g1, g0, equal_var=False)
        store(
            tag,
            p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
            summary=f"In {label} (n={len(sub)}): olaparib mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
        )

p, eff, m = linreg(
    f"{OUT} ~ treatment_olaparib * brca_any",
    "treatment_olaparib:brca_any",
)
store(
    "i4_olap_x_brca_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction olaparib:brca_any coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 5 — Sacituzumab × HER2-low interaction
# =========================================================
print("\n=== ITERATION 5: sacituzumab × HER2-low ===")
for tag, mask, label in [
    ("i5_sg_in_her2low", DF["her2_low"] == 1, "HER2-low"),
    ("i5_sg_in_her2_not_low", DF["her2_low"] == 0, "not HER2-low"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_sacituzumab_govitecan"] == 1, OUT]
    g0 = sub.loc[sub["treatment_sacituzumab_govitecan"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): sacituzumab mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )

p, eff, m = linreg(
    f"{OUT} ~ treatment_sacituzumab_govitecan * her2_low",
    "treatment_sacituzumab_govitecan:her2_low",
)
store(
    "i5_sg_x_her2low_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction sacituzumab:her2_low coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 6 — Pembrolizumab × MSI-high (and PD-L1 surrogates)
# =========================================================
print("\n=== ITERATION 6: pembrolizumab × MSI-high ===")
for tag, mask, label in [
    ("i6_pembro_in_msi_high", DF["msi_high"] == 1, "MSI-high"),
    ("i6_pembro_in_msi_low", DF["msi_high"] == 0, "MSI-low/MSS"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_pembrolizumab"] == 1, OUT]
    g0 = sub.loc[sub["treatment_pembrolizumab"] == 0, OUT]
    if len(g1) > 5 and len(g0) > 5:
        t = stats.ttest_ind(g1, g0, equal_var=False)
        store(
            tag,
            p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
            summary=f"In {label} (n={len(sub)}): pembrolizumab mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
        )

p, eff, m = linreg(
    f"{OUT} ~ treatment_pembrolizumab * msi_high",
    "treatment_pembrolizumab:msi_high",
)
store(
    "i6_pembro_x_msi_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction pembrolizumab:msi_high coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 7 — Disease burden: stage IV, brain mets, liver mets
# =========================================================
print("\n=== ITERATION 7: disease burden ===")
for var in ["stage_iv", "has_brain_mets", "liver_mets", "bone_mets",
            "node_positive", "pleural_effusion", "adrenal_mets"]:
    p, eff, m1, m0, n1, n0 = ttest(var, var)
    store(
        f"i7_{var}_main",
        p=p, eff=eff, sig=None,
        summary=f"{var}=1 mean PFS {m1:.3f} (n={n1}) vs =0 {m0:.3f} (n={n0}); diff={eff:+.3f} mo.",
    )


# =========================================================
# ITERATION 8 — ECOG, fatigue, pain, weight loss
# =========================================================
print("\n=== ITERATION 8: performance status / symptoms ===")
for var in ["ecog_ps", "fatigue_grade", "pain_nrs", "dyspnea_grade",
            "appetite_loss_grade", "weight_loss_pct_6mo"]:
    # Spearman correlation
    rho, p = stats.spearmanr(DF[var], DF[OUT])
    store(
        f"i8_{var}_corr",
        p=p, eff=rho, sig=None,
        summary=f"Spearman rho({var}, pfs_months) = {rho:+.4f}, p={p:.3g}.",
    )


# =========================================================
# ITERATION 9 — Continuous lab values
# =========================================================
print("\n=== ITERATION 9: lab values ===")
for var in ["albumin_g_dl", "ldh_u_l", "crp_mg_l", "nlr",
            "hemoglobin_g_dl", "alkaline_phosphatase_u_l", "calcium_mg_dl",
            "platelets_k_ul", "alc_k_ul", "anc_k_ul", "ki67_pct",
            "tumor_size_cm"]:
    rho, p = stats.spearmanr(DF[var], DF[OUT])
    store(
        f"i9_{var}_corr",
        p=p, eff=rho, sig=None,
        summary=f"Spearman rho({var}, pfs_months) = {rho:+.4f}, p={p:.3g}.",
    )


# =========================================================
# ITERATION 10 — Demographic / disparity main effects
# =========================================================
print("\n=== ITERATION 10: demographics / disparities ===")
# Age
rho, p = stats.spearmanr(DF["age_years"], DF[OUT])
store(
    "i10_age_corr",
    p=p, eff=rho, sig=None,
    summary=f"Spearman rho(age_years, pfs_months) = {rho:+.4f}, p={p:.3g}.",
)
# Sex
p, eff, m1, m0, n1, n0 = ttest("sex_female", "sex_female")
store(
    "i10_female_main",
    p=p, eff=eff, sig=None,
    summary=f"Female (n={n1}) PFS {m1:.3f} vs male (n={n0}) {m0:.3f}; diff={eff:+.3f} mo.",
)
# Race/ethnicity ANOVA & each vs white
groups = [DF.loc[DF["race_ethnicity"] == r, OUT] for r in DF["race_ethnicity"].unique()]
f, p = stats.f_oneway(*groups)
ref = DF.loc[DF["race_ethnicity"] == "white", OUT]
race_eff = []
for r in DF["race_ethnicity"].unique():
    if r == "white":
        continue
    g = DF.loc[DF["race_ethnicity"] == r, OUT]
    race_eff.append(f"{r} {g.mean():.3f}")
store(
    "i10_race_anova",
    p=p, eff=float(f), sig=None,
    summary=f"One-way ANOVA across race_ethnicity F={f:.2f} p={p:.3g}; means white={ref.mean():.3f}, " + ", ".join(race_eff) + ".",
)
# black vs white
g_black = DF.loc[DF["race_ethnicity"] == "black", OUT]
t = stats.ttest_ind(g_black, ref, equal_var=False)
store(
    "i10_black_vs_white",
    p=t.pvalue, eff=g_black.mean() - ref.mean(), sig=None,
    summary=f"Black PFS {g_black.mean():.3f} (n={len(g_black)}) vs white {ref.mean():.3f} (n={len(ref)}); diff={g_black.mean()-ref.mean():+.3f} mo.",
)
# Insurance
groups = [DF.loc[DF["insurance_type"] == i, OUT] for i in DF["insurance_type"].unique()]
f, p = stats.f_oneway(*groups)
ins_means = {i: DF.loc[DF["insurance_type"] == i, OUT].mean() for i in DF["insurance_type"].unique()}
store(
    "i10_insurance_anova",
    p=p, eff=float(f), sig=None,
    summary=f"One-way ANOVA across insurance_type F={f:.2f} p={p:.3g}; means " + ", ".join(f"{k}={v:.3f}" for k, v in ins_means.items()) + ".",
)
# Uninsured vs private
g_un = DF.loc[DF["insurance_type"] == "uninsured", OUT]
g_priv = DF.loc[DF["insurance_type"] == "private", OUT]
t = stats.ttest_ind(g_un, g_priv, equal_var=False)
store(
    "i10_uninsured_vs_private",
    p=t.pvalue, eff=g_un.mean() - g_priv.mean(), sig=None,
    summary=f"Uninsured PFS {g_un.mean():.3f} (n={len(g_un)}) vs private {g_priv.mean():.3f} (n={len(g_priv)}); diff={g_un.mean()-g_priv.mean():+.3f} mo.",
)
# Rural
p, eff, m1, m0, n1, n0 = ttest("rural_residence", "rural_residence")
store(
    "i10_rural_main",
    p=p, eff=eff, sig=None,
    summary=f"Rural (n={n1}) PFS {m1:.3f} vs urban (n={n0}) {m0:.3f}; diff={eff:+.3f} mo.",
)


# =========================================================
# ITERATION 11 — Tamoxifen × postmenopausal interaction
# =========================================================
print("\n=== ITERATION 11: tamoxifen × postmenopausal ===")
for tag, mask, label in [
    ("i11_tam_in_post", DF["postmenopausal"] == 1, "postmenopausal"),
    ("i11_tam_in_pre", DF["postmenopausal"] == 0, "premenopausal"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_tamoxifen"] == 1, OUT]
    g0 = sub.loc[sub["treatment_tamoxifen"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): tamoxifen mean PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )

p, eff, m = linreg(
    f"{OUT} ~ treatment_tamoxifen * postmenopausal",
    "treatment_tamoxifen:postmenopausal",
)
store(
    "i11_tam_x_post_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction tamoxifen:postmenopausal coef={eff:+.3f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 12 — Multivariable OLS adjusting for key covariates
# =========================================================
print("\n=== ITERATION 12: multivariable model ===")
formula = (
    f"{OUT} ~ age_years + sex_female + ecog_ps + stage_iv + has_brain_mets + "
    "liver_mets + bone_mets + er_positive + her2_positive + her2_low + "
    "brca1_mutation + brca2_mutation + albumin_g_dl + ldh_u_l + nlr + "
    "treatment_tamoxifen + treatment_palbociclib + treatment_trastuzumab + "
    "treatment_olaparib + treatment_sacituzumab_govitecan + treatment_pembrolizumab"
)
m = smf.ols(formula, data=DF).fit()
for term in [
    "ecog_ps", "stage_iv", "has_brain_mets", "liver_mets", "bone_mets",
    "albumin_g_dl", "ldh_u_l", "nlr",
    "treatment_tamoxifen", "treatment_palbociclib", "treatment_trastuzumab",
    "treatment_olaparib", "treatment_sacituzumab_govitecan", "treatment_pembrolizumab",
]:
    p = m.pvalues.get(term)
    eff = m.params.get(term)
    store(
        f"i12_mv_{term}",
        p=p, eff=eff, sig=None,
        summary=f"Multivariable OLS adjusted coef for {term}={eff:+.4f} mo, p={p:.3g}.",
    )


# =========================================================
# ITERATION 13 — Comorbidities
# =========================================================
print("\n=== ITERATION 13: comorbidities ===")
for var in ["diabetes_mellitus", "hypertension", "chronic_kidney_disease",
            "heart_failure", "copd", "depression_anxiety_diagnosis",
            "autoimmune_disease", "prior_malignancy", "venous_thromboembolism_history"]:
    p, eff, m1, m0, n1, n0 = ttest(var, var)
    store(
        f"i13_{var}_main",
        p=p, eff=eff, sig=None,
        summary=f"{var}=1 PFS {m1:.3f} (n={n1}) vs =0 {m0:.3f} (n={n0}); diff={eff:+.3f} mo.",
    )


# =========================================================
# ITERATION 14 — Palbociclib × ER-positive subgroup
# =========================================================
print("\n=== ITERATION 14: palbociclib × ER+ ===")
for tag, mask, label in [
    ("i14_palbo_in_erpos", DF["er_positive"] == 1, "ER+"),
    ("i14_palbo_in_erneg", DF["er_positive"] == 0, "ER-"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_palbociclib"] == 1, OUT]
    g0 = sub.loc[sub["treatment_palbociclib"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): palbociclib PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )

p, eff, m = linreg(
    f"{OUT} ~ treatment_palbociclib * er_positive",
    "treatment_palbociclib:er_positive",
)
store(
    "i14_palbo_x_er_interaction",
    p=p, eff=eff, sig=None,
    summary=f"OLS interaction palbociclib:er_positive coef={eff:+.3f} mo, p={p:.3g}.",
)
# Also palbociclib × HER2- (combined ER+/HER2-)
sub = DF[(DF["er_positive"] == 1) & (DF["her2_positive"] == 0)]
g1 = sub.loc[sub["treatment_palbociclib"] == 1, OUT]
g0 = sub.loc[sub["treatment_palbociclib"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i14_palbo_in_erpos_her2neg",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"In ER+/HER2- (n={len(sub)}): palbociclib PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
)


# =========================================================
# ITERATION 15 — SNP main effects (selected)
# =========================================================
print("\n=== ITERATION 15: SNPs ===")
for snp in ["snp_rs1045642", "snp_rs1065852", "snp_rs1799853",
            "snp_rs4244285", "snp_rs1801133", "snp_rs429358",
            "snp_rs7412", "snp_rs1050828", "snp_rs1800629"]:
    rho, p = stats.spearmanr(DF[snp], DF[OUT])
    store(
        f"i15_{snp}_corr",
        p=p, eff=rho, sig=None,
        summary=f"Spearman rho({snp}, pfs_months) = {rho:+.4f}, p={p:.3g}.",
    )


# =========================================================
# ITERATION 16 — BRCA-driven olaparib refinement: BRCA1 vs BRCA2 separately
# =========================================================
print("\n=== ITERATION 16: olaparib × BRCA1 vs BRCA2 ===")
for brca, label in [("brca1_mutation", "BRCA1+"), ("brca2_mutation", "BRCA2+")]:
    sub = DF[DF[brca] == 1]
    g1 = sub.loc[sub["treatment_olaparib"] == 1, OUT]
    g0 = sub.loc[sub["treatment_olaparib"] == 0, OUT]
    if len(g1) > 5 and len(g0) > 5:
        t = stats.ttest_ind(g1, g0, equal_var=False)
        store(
            f"i16_olap_in_{brca}",
            p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
            summary=f"In {label} (n={len(sub)}): olaparib PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
        )


# =========================================================
# ITERATION 17 — Trastuzumab × HER2 amplification (alternative HER2 marker)
# =========================================================
print("\n=== ITERATION 17: trastuzumab × HER2 amplification ===")
for tag, mask, label in [
    ("i17_tras_in_her2amp", DF["her2_amplification"] == 1, "HER2 amplified"),
    ("i17_tras_in_her2unamp", DF["her2_amplification"] == 0, "HER2 not amplified"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_trastuzumab"] == 1, OUT]
    g0 = sub.loc[sub["treatment_trastuzumab"] == 0, OUT]
    if len(g1) > 5 and len(g0) > 5:
        t = stats.ttest_ind(g1, g0, equal_var=False)
        store(
            tag,
            p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
            summary=f"In {label} (n={len(sub)}): trastuzumab PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
        )


# =========================================================
# ITERATION 18 — Sex × ER interaction (since this 'breast' cohort has 45% female)
# =========================================================
print("\n=== ITERATION 18: sex distribution & sex × treatment ===")
# Tamoxifen by sex
for tag, mask, label in [
    ("i18_tam_in_female", DF["sex_female"] == 1, "female"),
    ("i18_tam_in_male", DF["sex_female"] == 0, "male"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_tamoxifen"] == 1, OUT]
    g0 = sub.loc[sub["treatment_tamoxifen"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): tamoxifen PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )


# =========================================================
# ITERATION 19 — Race × treatment heterogeneity (palbociclib, pembrolizumab)
# =========================================================
print("\n=== ITERATION 19: race × treatment ===")
for trt in ["treatment_palbociclib", "treatment_pembrolizumab", "treatment_olaparib"]:
    for r in ["white", "black", "hispanic", "asian"]:
        sub = DF[DF["race_ethnicity"] == r]
        g1 = sub.loc[sub[trt] == 1, OUT]
        g0 = sub.loc[sub[trt] == 0, OUT]
        if len(g1) > 5 and len(g0) > 5:
            t = stats.ttest_ind(g1, g0, equal_var=False)
            store(
                f"i19_{trt}_in_{r}",
                p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
                summary=f"In {r} (n={len(sub)}): {trt} PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
            )


# =========================================================
# ITERATION 20 — Prior therapy lines (cumulative effect)
# =========================================================
print("\n=== ITERATION 20: prior therapy ===")
rho, p = stats.spearmanr(DF["prior_lines_of_therapy"], DF[OUT])
store(
    "i20_prior_lines_corr",
    p=p, eff=rho, sig=None,
    summary=f"Spearman rho(prior_lines_of_therapy, pfs_months) = {rho:+.4f}, p={p:.3g}.",
)
for var in ["prior_chemotherapy", "prior_radiation", "prior_surgery",
            "prior_immunotherapy", "prior_targeted_therapy"]:
    p, eff, m1, m0, n1, n0 = ttest(var, var)
    store(
        f"i20_{var}_main",
        p=p, eff=eff, sig=None,
        summary=f"{var}=1 PFS {m1:.3f} (n={n1}) vs =0 {m0:.3f} (n={n0}); diff={eff:+.3f} mo.",
    )


# =========================================================
# ITERATION 21 — PIK3CA mutation: prognostic and predictive (palbociclib?)
# =========================================================
print("\n=== ITERATION 21: PIK3CA ===")
p, eff, m1, m0, n1, n0 = ttest("pik3ca_mutation", "pik3ca_mutation")
store(
    "i21_pik3ca_main",
    p=p, eff=eff, sig=None,
    summary=f"PIK3CA mut (n={n1}) PFS {m1:.3f} vs WT (n={n0}) {m0:.3f}; diff={eff:+.3f} mo.",
)
# Olaparib × PIK3CA?
sub = DF[DF["pik3ca_mutation"] == 1]
g1 = sub.loc[sub["treatment_olaparib"] == 1, OUT]
g0 = sub.loc[sub["treatment_olaparib"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i21_olap_in_pik3ca",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"In PIK3CA-mut (n={len(sub)}): olaparib PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
)


# =========================================================
# ITERATION 22 — Interactions: trastuzumab × her2_low (off-target?), pembro × stage
# =========================================================
print("\n=== ITERATION 22: cross-treatment biomarker checks ===")
# trastuzumab in her2_low only (excluding her2_positive)
sub = DF[(DF["her2_positive"] == 0) & (DF["her2_low"] == 1)]
g1 = sub.loc[sub["treatment_trastuzumab"] == 1, OUT]
g0 = sub.loc[sub["treatment_trastuzumab"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i22_tras_in_her2low_only",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"In HER2-low (HER2-) (n={len(sub)}): trastuzumab PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
)
# Pembro in stage IV vs not
for tag, mask, label in [
    ("i22_pembro_in_stage4", DF["stage_iv"] == 1, "stage IV"),
    ("i22_pembro_in_not_stage4", DF["stage_iv"] == 0, "not stage IV"),
]:
    sub = DF[mask]
    g1 = sub.loc[sub["treatment_pembrolizumab"] == 1, OUT]
    g0 = sub.loc[sub["treatment_pembrolizumab"] == 0, OUT]
    t = stats.ttest_ind(g1, g0, equal_var=False)
    store(
        tag,
        p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
        summary=f"In {label} (n={len(sub)}): pembrolizumab PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
    )


# =========================================================
# ITERATION 23 — Albumin × treatment (frail vs fit)
# =========================================================
print("\n=== ITERATION 23: frailty × treatment effect ===")
DF["low_albumin"] = (DF["albumin_g_dl"] < DF["albumin_g_dl"].median()).astype(int)
for trt in ["treatment_palbociclib", "treatment_pembrolizumab"]:
    for tag_suf, mask, label in [
        ("low_alb", DF["low_albumin"] == 1, "below-median albumin"),
        ("high_alb", DF["low_albumin"] == 0, "at-or-above-median albumin"),
    ]:
        sub = DF[mask]
        g1 = sub.loc[sub[trt] == 1, OUT]
        g0 = sub.loc[sub[trt] == 0, OUT]
        t = stats.ttest_ind(g1, g0, equal_var=False)
        store(
            f"i23_{trt}_{tag_suf}",
            p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
            summary=f"In {label} (n={len(sub)}): {trt} PFS {g1.mean():.3f} vs {g0.mean():.3f}, diff={g1.mean()-g0.mean():+.3f} mo.",
        )


# =========================================================
# ITERATION 24 — Adjusted treatment effects within ER+/HER2- (palbo) and HER2+ (tras)
#                using OLS with key covariates
# =========================================================
print("\n=== ITERATION 24: adjusted within-subgroup treatment effects ===")
sub = DF[(DF["er_positive"] == 1) & (DF["her2_positive"] == 0)].copy()
m = smf.ols(
    f"{OUT} ~ treatment_palbociclib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr",
    data=sub,
).fit()
p = m.pvalues.get("treatment_palbociclib")
eff = m.params.get("treatment_palbociclib")
store(
    "i24_palbo_in_erpos_her2neg_adjusted",
    p=p, eff=eff, sig=None,
    summary=f"Adjusted palbociclib effect in ER+/HER2- subgroup (n={len(sub)}): coef={eff:+.4f} mo, p={p:.3g}.",
)
sub = DF[DF["her2_positive"] == 1].copy()
m = smf.ols(
    f"{OUT} ~ treatment_trastuzumab + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr",
    data=sub,
).fit()
p = m.pvalues.get("treatment_trastuzumab")
eff = m.params.get("treatment_trastuzumab")
store(
    "i24_tras_in_her2pos_adjusted",
    p=p, eff=eff, sig=None,
    summary=f"Adjusted trastuzumab effect in HER2+ subgroup (n={len(sub)}): coef={eff:+.4f} mo, p={p:.3g}.",
)
sub = DF[DF["brca_any"] == 1].copy()
m = smf.ols(
    f"{OUT} ~ treatment_olaparib + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr",
    data=sub,
).fit()
p = m.pvalues.get("treatment_olaparib")
eff = m.params.get("treatment_olaparib")
store(
    "i24_olap_in_brcaany_adjusted",
    p=p, eff=eff, sig=None,
    summary=f"Adjusted olaparib effect in BRCA1/2-mutated subgroup (n={len(sub)}): coef={eff:+.4f} mo, p={p:.3g}.",
)


# =========================================================
# ITERATION 25 — Wrap-up: composite biomarker-matched-treatment indicator
# =========================================================
print("\n=== ITERATION 25: composite biomarker-matched treatment ===")
# Define "biomarker-matched" treatment indicator:
#  - tamoxifen for ER+
#  - palbociclib for ER+/HER2-
#  - trastuzumab for HER2+
#  - olaparib for BRCA1/2 mutated
#  - sacituzumab for HER2-low (HER2-negative)
matched = (
    ((DF["treatment_tamoxifen"] == 1) & (DF["er_positive"] == 1)) |
    ((DF["treatment_palbociclib"] == 1) & (DF["er_positive"] == 1) & (DF["her2_positive"] == 0)) |
    ((DF["treatment_trastuzumab"] == 1) & (DF["her2_positive"] == 1)) |
    ((DF["treatment_olaparib"] == 1) & (DF["brca_any"] == 1)) |
    ((DF["treatment_sacituzumab_govitecan"] == 1) & (DF["her2_low"] == 1) & (DF["her2_positive"] == 0))
).astype(int)
DF["biomarker_matched_tx"] = matched
g1 = DF.loc[DF["biomarker_matched_tx"] == 1, OUT]
g0 = DF.loc[DF["biomarker_matched_tx"] == 0, OUT]
t = stats.ttest_ind(g1, g0, equal_var=False)
store(
    "i25_biomarker_matched_tx_main",
    p=t.pvalue, eff=g1.mean() - g0.mean(), sig=None,
    summary=f"Patients receiving any biomarker-matched targeted therapy (n={len(g1)}) PFS {g1.mean():.3f} vs others (n={len(g0)}) {g0.mean():.3f}; diff={g1.mean()-g0.mean():+.3f} mo.",
)
# Adjusted version
m = smf.ols(
    f"{OUT} ~ biomarker_matched_tx + age_years + ecog_ps + stage_iv + albumin_g_dl + ldh_u_l + nlr",
    data=DF,
).fit()
p = m.pvalues.get("biomarker_matched_tx")
eff = m.params.get("biomarker_matched_tx")
store(
    "i25_biomarker_matched_tx_adjusted",
    p=p, eff=eff, sig=None,
    summary=f"Adjusted biomarker-matched-tx effect (covariates: age, ECOG, stage IV, albumin, LDH, NLR): coef={eff:+.4f} mo, p={p:.3g}.",
)


# Save results
with open("my_results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2)
print(f"\n\nWrote {len(RESULTS)} analyses to my_results.json")
