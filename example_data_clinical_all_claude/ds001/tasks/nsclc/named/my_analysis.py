"""
Comprehensive analysis script for ds001_nsclc.
Runs main-effect, treatment, and heterogeneity analyses.
"""
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

DF = pd.read_parquet("dataset.parquet")
RESULTS = {}


def add_result(key, **kw):
    RESULTS[key] = kw
    print(f"\n[{key}]")
    for k, v in kw.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6g}")
        else:
            print(f"  {k}: {v}")


def linreg(formula, data=None):
    if data is None:
        data = DF
    m = smf.ols(formula, data=data).fit()
    return m


def coef_p(m, name):
    if name not in m.params.index:
        return None, None
    return float(m.params[name]), float(m.pvalues[name])


# ---------- Iteration 1: Univariate prognostic main effects on PFS ----------
print("=" * 60)
print("ITER 1: prognostic main effects")
print("=" * 60)

# ECOG
m = linreg("pfs_months ~ C(ecog_ps)")
e1, p1 = coef_p(m, "C(ecog_ps)[T.1]")
e2, p2 = coef_p(m, "C(ecog_ps)[T.2]")
add_result("ecog_main", coef_ecog1=e1, p_ecog1=p1, coef_ecog2=e2, p_ecog2=p2,
           mean_ecog0=float(DF.loc[DF.ecog_ps == 0, "pfs_months"].mean()),
           mean_ecog1=float(DF.loc[DF.ecog_ps == 1, "pfs_months"].mean()),
           mean_ecog2=float(DF.loc[DF.ecog_ps == 2, "pfs_months"].mean()))

# Stage IV
m = linreg("pfs_months ~ stage_iv")
e, p = coef_p(m, "stage_iv")
add_result("stage_iv_main", coef=e, p=p,
           mean1=float(DF.loc[DF.stage_iv == 1, "pfs_months"].mean()),
           mean0=float(DF.loc[DF.stage_iv == 0, "pfs_months"].mean()))

# Brain mets
m = linreg("pfs_months ~ has_brain_mets")
e, p = coef_p(m, "has_brain_mets")
add_result("brain_mets_main", coef=e, p=p,
           mean1=float(DF.loc[DF.has_brain_mets == 1, "pfs_months"].mean()),
           mean0=float(DF.loc[DF.has_brain_mets == 0, "pfs_months"].mean()))

# Albumin
m = linreg("pfs_months ~ albumin_g_dl")
e, p = coef_p(m, "albumin_g_dl")
add_result("albumin_main", coef=e, p=p)

# LDH
m = linreg("pfs_months ~ ldh_u_l")
e, p = coef_p(m, "ldh_u_l")
add_result("ldh_main", coef=e, p=p)

# Weight loss
m = linreg("pfs_months ~ weight_loss_pct_6mo")
e, p = coef_p(m, "weight_loss_pct_6mo")
add_result("weight_loss_main", coef=e, p=p)

# CRP
m = linreg("pfs_months ~ crp_mg_l")
e, p = coef_p(m, "crp_mg_l")
add_result("crp_main", coef=e, p=p)

# NLR
m = linreg("pfs_months ~ nlr")
e, p = coef_p(m, "nlr")
add_result("nlr_main", coef=e, p=p)

# Age
m = linreg("pfs_months ~ age_years")
e, p = coef_p(m, "age_years")
add_result("age_main", coef=e, p=p)

# Sex
m = linreg("pfs_months ~ sex_female")
e, p = coef_p(m, "sex_female")
add_result("sex_main", coef=e, p=p)

# Smoking
m = linreg("pfs_months ~ C(smoking_status, Treatment(reference='never'))")
add_result("smoking_main",
           coef_current=float(m.params.get("C(smoking_status, Treatment(reference='never'))[T.current]", np.nan)),
           p_current=float(m.pvalues.get("C(smoking_status, Treatment(reference='never'))[T.current]", np.nan)),
           coef_former=float(m.params.get("C(smoking_status, Treatment(reference='never'))[T.former]", np.nan)),
           p_former=float(m.pvalues.get("C(smoking_status, Treatment(reference='never'))[T.former]", np.nan)),
           mean_never=float(DF.loc[DF.smoking_status == "never", "pfs_months"].mean()),
           mean_former=float(DF.loc[DF.smoking_status == "former", "pfs_months"].mean()),
           mean_current=float(DF.loc[DF.smoking_status == "current", "pfs_months"].mean()))

# Histology
m = linreg("pfs_months ~ C(histology)")
add_result("histology_main",
           coef_squamous=float(m.params.get("C(histology)[T.squamous]", np.nan)),
           p_squamous=float(m.pvalues.get("C(histology)[T.squamous]", np.nan)),
           mean_adeno=float(DF.loc[DF.histology == "adenocarcinoma", "pfs_months"].mean()),
           mean_squamous=float(DF.loc[DF.histology == "squamous", "pfs_months"].mean()))

# Mutations (univariate)
for mut in ["egfr_mutation", "kras_g12c", "alk_fusion", "stk11_mutation", "brca2_mutation", "tmb_high"]:
    m = linreg(f"pfs_months ~ {mut}")
    e, p = coef_p(m, mut)
    add_result(f"{mut}_main_pfs", coef=e, p=p,
               mean1=float(DF.loc[DF[mut] == 1, "pfs_months"].mean()),
               mean0=float(DF.loc[DF[mut] == 0, "pfs_months"].mean()))

# pdl1 univariate
m = linreg("pfs_months ~ pdl1_tps")
e, p = coef_p(m, "pdl1_tps")
add_result("pdl1_main_pfs", coef=e, p=p)

# Lab univariate (others)
for lab in ["hemoglobin_g_dl", "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
            "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
            "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]:
    m = linreg(f"pfs_months ~ {lab}")
    e, p = coef_p(m, lab)
    add_result(f"{lab}_main_pfs", coef=e, p=p)

# ---------- Iteration 2: Treatment main effects (unadjusted) ----------
print("\n" + "=" * 60)
print("ITER 2: treatment main effects")
print("=" * 60)
for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    m = linreg(f"pfs_months ~ {tx}")
    e, p = coef_p(m, tx)
    add_result(f"{tx}_unadjusted_pfs", coef=e, p=p,
               mean_on=float(DF.loc[DF[tx] == 1, "pfs_months"].mean()),
               mean_off=float(DF.loc[DF[tx] == 0, "pfs_months"].mean()))

# ---------- Iteration 3: Treatment effects adjusted for prognostics ----------
print("\n" + "=" * 60)
print("ITER 3: adjusted treatment main effects")
print("=" * 60)
adj = ("C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + "
       "weight_loss_pct_6mo + crp_mg_l + nlr + age_years + sex_female + "
       "C(smoking_status) + C(histology) + egfr_mutation + kras_g12c + "
       "alk_fusion + stk11_mutation + brca2_mutation + pdl1_tps + tmb_high + "
       "hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + alt_u_l + "
       "total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + sodium_meq_l + "
       "potassium_meq_l + calcium_mg_dl")
for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    m = linreg(f"pfs_months ~ {tx} + {adj}")
    e, p = coef_p(m, tx)
    add_result(f"{tx}_adjusted_pfs", coef=e, p=p)

# ---------- Iteration 4: Targeted treatment-by-biomarker interactions ----------
print("\n" + "=" * 60)
print("ITER 4: targeted tx-by-biomarker")
print("=" * 60)

# pembro x pdl1
m = linreg("pfs_months ~ treatment_pembrolizumab * pdl1_tps")
e, p = coef_p(m, "treatment_pembrolizumab:pdl1_tps")
add_result("pembro_x_pdl1", coef=e, p=p,
           tx_main=float(m.params["treatment_pembrolizumab"]),
           pdl1_main=float(m.params["pdl1_tps"]))

# pembro x tmb
m = linreg("pfs_months ~ treatment_pembrolizumab * tmb_high")
e, p = coef_p(m, "treatment_pembrolizumab:tmb_high")
add_result("pembro_x_tmb", coef=e, p=p,
           tx_main=float(m.params["treatment_pembrolizumab"]),
           tmb_main=float(m.params["tmb_high"]))

# pembro x stk11 (known negative modifier)
m = linreg("pfs_months ~ treatment_pembrolizumab * stk11_mutation")
e, p = coef_p(m, "treatment_pembrolizumab:stk11_mutation")
add_result("pembro_x_stk11", coef=e, p=p,
           tx_main=float(m.params["treatment_pembrolizumab"]),
           stk11_main=float(m.params["stk11_mutation"]))

# pembro x egfr (known: less benefit)
m = linreg("pfs_months ~ treatment_pembrolizumab * egfr_mutation")
e, p = coef_p(m, "treatment_pembrolizumab:egfr_mutation")
add_result("pembro_x_egfr", coef=e, p=p,
           tx_main=float(m.params["treatment_pembrolizumab"]),
           egfr_main=float(m.params["egfr_mutation"]))

# sotorasib x KRAS G12C (expected positive)
m = linreg("pfs_months ~ treatment_sotorasib * kras_g12c")
e, p = coef_p(m, "treatment_sotorasib:kras_g12c")
add_result("soto_x_krasg12c", coef=e, p=p,
           tx_main=float(m.params["treatment_sotorasib"]),
           kras_main=float(m.params["kras_g12c"]))

# osimertinib x EGFR (expected positive)
m = linreg("pfs_months ~ treatment_osimertinib * egfr_mutation")
e, p = coef_p(m, "treatment_osimertinib:egfr_mutation")
add_result("osi_x_egfr", coef=e, p=p,
           tx_main=float(m.params["treatment_osimertinib"]),
           egfr_main=float(m.params["egfr_mutation"]))

# olaparib x BRCA2 (expected positive)
m = linreg("pfs_months ~ treatment_olaparib * brca2_mutation")
e, p = coef_p(m, "treatment_olaparib:brca2_mutation")
add_result("ola_x_brca2", coef=e, p=p,
           tx_main=float(m.params["treatment_olaparib"]),
           brca2_main=float(m.params["brca2_mutation"]))

# olaparib x ALK fusion (HRD-related? probably not)
m = linreg("pfs_months ~ treatment_olaparib * alk_fusion")
e, p = coef_p(m, "treatment_olaparib:alk_fusion")
add_result("ola_x_alk", coef=e, p=p)

# ---------- Iteration 5: stratified subgroup means ----------
print("\n" + "=" * 60)
print("ITER 5: stratified subgroup tx-effect estimates")
print("=" * 60)


def subgroup_tx_effect(label, mask, tx):
    sub = DF.loc[mask]
    if sub[tx].nunique() < 2 or len(sub) < 50:
        return None
    on = sub.loc[sub[tx] == 1, "pfs_months"]
    off = sub.loc[sub[tx] == 0, "pfs_months"]
    t = stats.ttest_ind(on, off, equal_var=False)
    return dict(label=label, n_on=int(len(on)), n_off=int(len(off)),
                mean_on=float(on.mean()), mean_off=float(off.mean()),
                effect=float(on.mean() - off.mean()),
                p=float(t.pvalue))


# Pembro stratified
add_result("pembro_in_pdl1_high", **subgroup_tx_effect(
    "pdl1_tps>=0.5", DF.pdl1_tps >= 0.5, "treatment_pembrolizumab"))
add_result("pembro_in_pdl1_low", **subgroup_tx_effect(
    "pdl1_tps<0.5", DF.pdl1_tps < 0.5, "treatment_pembrolizumab"))
add_result("pembro_in_tmb_high", **subgroup_tx_effect(
    "tmb_high=1", DF.tmb_high == 1, "treatment_pembrolizumab"))
add_result("pembro_in_tmb_low", **subgroup_tx_effect(
    "tmb_high=0", DF.tmb_high == 0, "treatment_pembrolizumab"))
add_result("pembro_in_stk11", **subgroup_tx_effect(
    "stk11=1", DF.stk11_mutation == 1, "treatment_pembrolizumab"))
add_result("pembro_in_no_stk11", **subgroup_tx_effect(
    "stk11=0", DF.stk11_mutation == 0, "treatment_pembrolizumab"))
add_result("pembro_in_egfr", **subgroup_tx_effect(
    "egfr=1", DF.egfr_mutation == 1, "treatment_pembrolizumab"))
add_result("pembro_in_no_egfr", **subgroup_tx_effect(
    "egfr=0", DF.egfr_mutation == 0, "treatment_pembrolizumab"))

# Sotorasib stratified by KRAS G12C
add_result("soto_in_krasg12c", **subgroup_tx_effect(
    "kras_g12c=1", DF.kras_g12c == 1, "treatment_sotorasib"))
add_result("soto_in_no_krasg12c", **subgroup_tx_effect(
    "kras_g12c=0", DF.kras_g12c == 0, "treatment_sotorasib"))

# Osimertinib stratified by EGFR
add_result("osi_in_egfr", **subgroup_tx_effect(
    "egfr=1", DF.egfr_mutation == 1, "treatment_osimertinib"))
add_result("osi_in_no_egfr", **subgroup_tx_effect(
    "egfr=0", DF.egfr_mutation == 0, "treatment_osimertinib"))

# Olaparib stratified by BRCA2
add_result("ola_in_brca2", **subgroup_tx_effect(
    "brca2=1", DF.brca2_mutation == 1, "treatment_olaparib"))
add_result("ola_in_no_brca2", **subgroup_tx_effect(
    "brca2=0", DF.brca2_mutation == 0, "treatment_olaparib"))

# ---------- Iteration 6: Treatment heterogeneity by every binary feature ----------
print("\n" + "=" * 60)
print("ITER 6: full tx-x-feature interaction screen")
print("=" * 60)

binary_feats = ["sex_female", "stage_iv", "has_brain_mets", "egfr_mutation",
                "kras_g12c", "alk_fusion", "stk11_mutation", "brca2_mutation",
                "tmb_high"]
cont_feats = ["age_years", "pdl1_tps", "albumin_g_dl", "ldh_u_l",
              "weight_loss_pct_6mo", "crp_mg_l", "nlr", "hemoglobin_g_dl",
              "alkaline_phosphatase_u_l", "ast_u_l", "alt_u_l",
              "total_bilirubin_mg_dl", "creatinine_mg_dl", "bun_mg_dl",
              "sodium_meq_l", "potassium_meq_l", "calcium_mg_dl"]

interaction_screen = {}
for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    for f in binary_feats + cont_feats:
        try:
            m = linreg(f"pfs_months ~ {tx} * {f}")
            ix_name = f"{tx}:{f}"
            e, p = coef_p(m, ix_name)
            interaction_screen[(tx, f)] = (e, p)
        except Exception as ex:
            interaction_screen[(tx, f)] = (None, None)

# Print top interactions per tx (most significant by p-value)
top_interactions = {}
for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    rows = [(f, interaction_screen[(tx, f)][0], interaction_screen[(tx, f)][1])
            for f in binary_feats + cont_feats]
    rows = [r for r in rows if r[2] is not None]
    rows.sort(key=lambda r: r[2])
    top_interactions[tx] = rows[:8]
    print(f"\nTop interactions for {tx}:")
    for r in rows[:8]:
        print(f"  {r[0]}: coef={r[1]:.4f}  p={r[2]:.3g}")

RESULTS["top_interactions"] = {tx: [(f, e, p) for f, e, p in rows]
                              for tx, rows in top_interactions.items()}

# ---------- Iteration 7: smoking strata for pembro ----------
print("\n" + "=" * 60)
print("ITER 7: smoking and histology modifiers")
print("=" * 60)
for s in ["never", "former", "current"]:
    add_result(f"pembro_in_smoking_{s}", **subgroup_tx_effect(
        f"smoking={s}", DF.smoking_status == s, "treatment_pembrolizumab"))

for h in ["adenocarcinoma", "squamous"]:
    add_result(f"pembro_in_histology_{h}", **subgroup_tx_effect(
        f"hist={h}", DF.histology == h, "treatment_pembrolizumab"))

# ECOG strata
for e_lvl in [0, 1, 2]:
    add_result(f"pembro_in_ecog_{e_lvl}", **subgroup_tx_effect(
        f"ecog={e_lvl}", DF.ecog_ps == e_lvl, "treatment_pembrolizumab"))

# ---------- Iteration 8: combined biomarker + STK11 stratification for pembro ----------
print("\n" + "=" * 60)
print("ITER 8: deeper pembro subgroups (TPS x STK11 etc)")
print("=" * 60)

# pdl1>=0.5 and stk11=0
m_high_no_stk11 = (DF.pdl1_tps >= 0.5) & (DF.stk11_mutation == 0)
m_high_stk11 = (DF.pdl1_tps >= 0.5) & (DF.stk11_mutation == 1)
m_low_no_stk11 = (DF.pdl1_tps < 0.5) & (DF.stk11_mutation == 0)
m_low_stk11 = (DF.pdl1_tps < 0.5) & (DF.stk11_mutation == 1)

add_result("pembro_pdl1high_stk11neg", **subgroup_tx_effect(
    "pdl1>=0.5 & stk11=0", m_high_no_stk11, "treatment_pembrolizumab"))
add_result("pembro_pdl1high_stk11pos", **subgroup_tx_effect(
    "pdl1>=0.5 & stk11=1", m_high_stk11, "treatment_pembrolizumab"))
add_result("pembro_pdl1low_stk11neg", **subgroup_tx_effect(
    "pdl1<0.5 & stk11=0", m_low_no_stk11, "treatment_pembrolizumab"))
add_result("pembro_pdl1low_stk11pos", **subgroup_tx_effect(
    "pdl1<0.5 & stk11=1", m_low_stk11, "treatment_pembrolizumab"))

# Triple interaction model: pembro * pdl1 * stk11
m = linreg("pfs_months ~ treatment_pembrolizumab * pdl1_tps * stk11_mutation")
print(m.summary().tables[1])
RESULTS["pembro_pdl1_stk11_triple"] = {
    name: (float(m.params[name]), float(m.pvalues[name]))
    for name in m.params.index
}

# pembro * pdl1 * tmb_high
m = linreg("pfs_months ~ treatment_pembrolizumab * pdl1_tps * tmb_high")
print(m.summary().tables[1])
RESULTS["pembro_pdl1_tmb_triple"] = {
    name: (float(m.params[name]), float(m.pvalues[name]))
    for name in m.params.index
}

# ---------- Iteration 9: KRAS G12C subgroup combos for sotorasib ----------
print("\n" + "=" * 60)
print("ITER 9: sotorasib subgroup combos")
print("=" * 60)
# soto in kras_g12c with stk11
add_result("soto_kras_stk11pos", **subgroup_tx_effect(
    "kras_g12c=1 & stk11=1", (DF.kras_g12c == 1) & (DF.stk11_mutation == 1),
    "treatment_sotorasib"))
add_result("soto_kras_stk11neg", **subgroup_tx_effect(
    "kras_g12c=1 & stk11=0", (DF.kras_g12c == 1) & (DF.stk11_mutation == 0),
    "treatment_sotorasib"))

# triple interaction
m = linreg("pfs_months ~ treatment_sotorasib * kras_g12c * stk11_mutation")
print(m.summary().tables[1])
RESULTS["soto_kras_stk11_triple"] = {
    name: (float(m.params[name]), float(m.pvalues[name]))
    for name in m.params.index
}

# soto x kras x ECOG
m = linreg("pfs_months ~ treatment_sotorasib * kras_g12c + C(ecog_ps)")
print(m.summary().tables[1])

# ---------- Iteration 10: osimertinib refinements ----------
print("\n" + "=" * 60)
print("ITER 10: osimertinib refinements")
print("=" * 60)
# osi in egfr by smoking, brain mets
add_result("osi_egfr_brainmets", **subgroup_tx_effect(
    "egfr=1 & brain_mets=1", (DF.egfr_mutation == 1) & (DF.has_brain_mets == 1),
    "treatment_osimertinib"))
add_result("osi_egfr_no_brainmets", **subgroup_tx_effect(
    "egfr=1 & brain_mets=0", (DF.egfr_mutation == 1) & (DF.has_brain_mets == 0),
    "treatment_osimertinib"))
add_result("osi_egfr_never_smoker", **subgroup_tx_effect(
    "egfr=1 & smoking=never",
    (DF.egfr_mutation == 1) & (DF.smoking_status == "never"),
    "treatment_osimertinib"))
add_result("osi_egfr_ever_smoker", **subgroup_tx_effect(
    "egfr=1 & smoking!=never",
    (DF.egfr_mutation == 1) & (DF.smoking_status != "never"),
    "treatment_osimertinib"))

# triple interaction osi x egfr x ecog
m = linreg("pfs_months ~ treatment_osimertinib * egfr_mutation * C(ecog_ps)")
print(m.summary().tables[1])
RESULTS["osi_egfr_ecog_triple"] = {
    name: (float(m.params[name]), float(m.pvalues[name]))
    for name in m.params.index
}

# ---------- Iteration 11: olaparib refinements ----------
print("\n" + "=" * 60)
print("ITER 11: olaparib refinements")
print("=" * 60)
add_result("ola_brca2_pdl1high",
           **(subgroup_tx_effect("brca2=1 & pdl1>=0.5",
                                 (DF.brca2_mutation == 1) & (DF.pdl1_tps >= 0.5),
                                 "treatment_olaparib") or {}))
add_result("ola_brca2_pdl1low",
           **(subgroup_tx_effect("brca2=1 & pdl1<0.5",
                                 (DF.brca2_mutation == 1) & (DF.pdl1_tps < 0.5),
                                 "treatment_olaparib") or {}))
add_result("ola_brca2_ecog0",
           **(subgroup_tx_effect("brca2=1 & ecog=0",
                                 (DF.brca2_mutation == 1) & (DF.ecog_ps == 0),
                                 "treatment_olaparib") or {}))
add_result("ola_brca2_ecog2",
           **(subgroup_tx_effect("brca2=1 & ecog=2",
                                 (DF.brca2_mutation == 1) & (DF.ecog_ps == 2),
                                 "treatment_olaparib") or {}))

# ola x brca2 x ecog triple
m = linreg("pfs_months ~ treatment_olaparib * brca2_mutation * C(ecog_ps)")
print(m.summary().tables[1])

# ola in non-brca? Should be null
m = linreg("pfs_months ~ treatment_olaparib", data=DF[DF.brca2_mutation == 0])
e, p = coef_p(m, "treatment_olaparib")
add_result("ola_no_brca2_clean", coef=e, p=p)

# ---------- Iteration 12: combined model w/ targeted interactions ----------
print("\n" + "=" * 60)
print("ITER 12: joint adjusted model w/ key interactions")
print("=" * 60)
formula = (
    "pfs_months ~ "
    "treatment_pembrolizumab * pdl1_tps + "
    "treatment_pembrolizumab * stk11_mutation + "
    "treatment_pembrolizumab * tmb_high + "
    "treatment_pembrolizumab * egfr_mutation + "
    "treatment_sotorasib * kras_g12c + "
    "treatment_osimertinib * egfr_mutation + "
    "treatment_olaparib * brca2_mutation + "
    "C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + ldh_u_l + "
    "weight_loss_pct_6mo + crp_mg_l + nlr + age_years + sex_female + "
    "C(smoking_status) + C(histology) + alk_fusion"
)
m_joint = linreg(formula)
print(m_joint.summary().tables[1])
RESULTS["joint_model"] = {
    name: (float(m_joint.params[name]), float(m_joint.pvalues[name]))
    for name in m_joint.params.index
}

# ---------- Iteration 13: unbiased subgroup heterogeneity for each tx (full screen w/ adj) ----------
print("\n" + "=" * 60)
print("ITER 13: adjusted interaction screen")
print("=" * 60)

# Adjusted interaction tests
adj_short = ("C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + "
             "ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + age_years")
adj_screen = {}
for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    for f in binary_feats + cont_feats:
        try:
            m = linreg(f"pfs_months ~ {tx} * {f} + {adj_short}")
            ix_name = f"{tx}:{f}"
            e, p = coef_p(m, ix_name)
            adj_screen[(tx, f)] = (e, p)
        except Exception as ex:
            adj_screen[(tx, f)] = (None, None)

for tx in ["treatment_pembrolizumab", "treatment_sotorasib",
           "treatment_olaparib", "treatment_osimertinib"]:
    rows = [(f, adj_screen[(tx, f)][0], adj_screen[(tx, f)][1])
            for f in binary_feats + cont_feats]
    rows = [r for r in rows if r[2] is not None]
    rows.sort(key=lambda r: r[2])
    print(f"\nTop adjusted interactions for {tx}:")
    for r in rows[:6]:
        print(f"  {r[0]}: coef={r[1]:.4f}  p={r[2]:.3g}")
    RESULTS[f"adj_top_interactions_{tx}"] = [(f, e, p) for f, e, p in rows[:10]]

# ---------- Iteration 14: Final concentrated subgroup definitions ----------
print("\n" + "=" * 60)
print("ITER 14: final subgroup definitions")
print("=" * 60)

# Pembro: PD-L1 high, STK11 negative, EGFR negative
m_pembro_final = ((DF.pdl1_tps >= 0.5) & (DF.stk11_mutation == 0) &
                  (DF.egfr_mutation == 0))
add_result("pembro_final_subgroup", **subgroup_tx_effect(
    "pdl1>=0.5 & stk11=0 & egfr=0", m_pembro_final, "treatment_pembrolizumab"))

# Pembro complement: anywhere else
add_result("pembro_complement", **subgroup_tx_effect(
    "complement", ~m_pembro_final, "treatment_pembrolizumab"))

# Sotorasib: KRAS G12C only - already strong
# Osimertinib: EGFR only - already strong
# Olaparib: BRCA2 only - already strong

# Save raw results
with open("my_raw_results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2, default=str)

print("\nDONE — wrote my_raw_results.json")
