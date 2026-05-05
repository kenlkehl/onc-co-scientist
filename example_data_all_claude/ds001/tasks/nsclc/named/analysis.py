"""Comprehensive analysis of ds001_nsclc dataset.

Runs hypothesis tests across iterations and stores results to results.json
for later transcript construction.
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


def add(name, **kw):
    results[name] = kw
    return kw


def fmt(x):
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def ttest_pfs(mask):
    a = df.loc[mask, "pfs_months"]
    b = df.loc[~mask, "pfs_months"]
    t = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_in": float(a.mean()),
        "mean_out": float(b.mean()),
        "diff": float(a.mean() - b.mean()),
        "p_value": float(t.pvalue),
        "n_in": int(mask.sum()),
        "n_out": int((~mask).sum()),
    }


def ols_with_treatment(formula):
    m = smf.ols(formula, data=df).fit()
    return m


def interaction_effect(treatment, modifier, modifier_pos=1, modifier_neg=0):
    """Compute treatment effect within modifier subgroups + interaction p-value."""
    a = df[df[modifier] == modifier_pos]
    b = df[df[modifier] == modifier_neg]
    eff_pos = (
        a.loc[a[treatment] == 1, "pfs_months"].mean()
        - a.loc[a[treatment] == 0, "pfs_months"].mean()
    )
    eff_neg = (
        b.loc[b[treatment] == 1, "pfs_months"].mean()
        - b.loc[b[treatment] == 0, "pfs_months"].mean()
    )
    formula = f"pfs_months ~ {treatment} * C({modifier})"
    m = smf.ols(formula, data=df).fit()
    interaction_term = [c for c in m.params.index if ":" in c][0]
    return {
        "effect_in_modifier_pos": float(eff_pos),
        "effect_in_modifier_neg": float(eff_neg),
        "interaction_coef": float(m.params[interaction_term]),
        "interaction_p": float(m.pvalues[interaction_term]),
        "n_pos": int(len(a)),
        "n_neg": int(len(b)),
    }


# ------------ ITERATION 1: simple feature-outcome main effects ------------
print("== Iter 1: clinical feature main effects on PFS ==")
# ECOG vs PFS
m = smf.ols("pfs_months ~ C(ecog_ps)", data=df).fit()
add("ecog_main",
    coef_ecog1=float(m.params.get("C(ecog_ps)[T.1]", np.nan)),
    coef_ecog2=float(m.params.get("C(ecog_ps)[T.2]", np.nan)),
    p_ecog1=float(m.pvalues.get("C(ecog_ps)[T.1]", np.nan)),
    p_ecog2=float(m.pvalues.get("C(ecog_ps)[T.2]", np.nan)),
    mean_ecog0=float(df.loc[df.ecog_ps==0,"pfs_months"].mean()),
    mean_ecog1=float(df.loc[df.ecog_ps==1,"pfs_months"].mean()),
    mean_ecog2=float(df.loc[df.ecog_ps==2,"pfs_months"].mean()),
)

# Stage IV
add("stage4_main", **ttest_pfs(df.stage_iv == 1))

# Brain mets
add("brainmets_main", **ttest_pfs(df.has_brain_mets == 1))

# Sex female
add("sex_main", **ttest_pfs(df.sex_female == 1))

# Histology
add("histology_main", **ttest_pfs(df.histology == "squamous"))

# Smoking
m = smf.ols("pfs_months ~ C(smoking_status)", data=df).fit()
means = df.groupby("smoking_status")["pfs_months"].mean().to_dict()
add("smoking_main",
    mean_never=float(means.get("never", np.nan)),
    mean_former=float(means.get("former", np.nan)),
    mean_current=float(means.get("current", np.nan)),
    p_former=float(m.pvalues.get("C(smoking_status)[T.former]", np.nan)),
    p_never=float(m.pvalues.get("C(smoking_status)[T.never]", np.nan)),
)

# Age (linear)
m = smf.ols("pfs_months ~ age_years", data=df).fit()
add("age_main",
    coef=float(m.params["age_years"]),
    p=float(m.pvalues["age_years"]),
)

# Albumin
m = smf.ols("pfs_months ~ albumin_g_dl", data=df).fit()
add("albumin_main",
    coef=float(m.params["albumin_g_dl"]),
    p=float(m.pvalues["albumin_g_dl"]),
)

# LDH
m = smf.ols("pfs_months ~ ldh_u_l", data=df).fit()
add("ldh_main",
    coef=float(m.params["ldh_u_l"]),
    p=float(m.pvalues["ldh_u_l"]),
)

# NLR
m = smf.ols("pfs_months ~ nlr", data=df).fit()
add("nlr_main",
    coef=float(m.params["nlr"]),
    p=float(m.pvalues["nlr"]),
)

# CRP
m = smf.ols("pfs_months ~ crp_mg_l", data=df).fit()
add("crp_main",
    coef=float(m.params["crp_mg_l"]),
    p=float(m.pvalues["crp_mg_l"]),
)

# Weight loss
m = smf.ols("pfs_months ~ weight_loss_pct_6mo", data=df).fit()
add("wtloss_main",
    coef=float(m.params["weight_loss_pct_6mo"]),
    p=float(m.pvalues["weight_loss_pct_6mo"]),
)

# PDL1 main
m = smf.ols("pfs_months ~ pdl1_tps", data=df).fit()
add("pdl1_main",
    coef=float(m.params["pdl1_tps"]),
    p=float(m.pvalues["pdl1_tps"]),
)

# TMB main
add("tmb_main", **ttest_pfs(df.tmb_high == 1))

# Hemoglobin
m = smf.ols("pfs_months ~ hemoglobin_g_dl", data=df).fit()
add("hgb_main",
    coef=float(m.params["hemoglobin_g_dl"]),
    p=float(m.pvalues["hemoglobin_g_dl"]),
)

# ------------ ITERATION 2: treatment main effects ------------
print("== Iter 2: treatment main effects ==")
for t in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    add(f"{t}_main", **ttest_pfs(df[t]==1))

# ------------ ITERATION 3: mutation main effects ------------
print("== Iter 3: mutation main effects ==")
for v in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation"]:
    add(f"{v}_main", **ttest_pfs(df[v]==1))

# ------------ ITERATION 4-7: Treatment x biomarker interactions ------------
print("== Iter 4: osimertinib x EGFR ==")
add("osi_x_egfr", **interaction_effect("treatment_osimertinib","egfr_mutation"))

print("== Iter 5: sotorasib x KRAS G12C ==")
add("soto_x_kras", **interaction_effect("treatment_sotorasib","kras_g12c"))

print("== Iter 6: olaparib x BRCA2 ==")
add("olap_x_brca2", **interaction_effect("treatment_olaparib","brca2_mutation"))

print("== Iter 7a: pembrolizumab x PDL1 (continuous) ==")
m = smf.ols("pfs_months ~ treatment_pembrolizumab * pdl1_tps", data=df).fit()
add("pembro_x_pdl1",
    pembro_main=float(m.params["treatment_pembrolizumab"]),
    pdl1_main=float(m.params["pdl1_tps"]),
    interaction_coef=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
    interaction_p=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
)
# in pdl1>=0.5
mask_high = df.pdl1_tps >= 0.5
eff_high = df.loc[mask_high & (df.treatment_pembrolizumab==1),"pfs_months"].mean() - df.loc[mask_high & (df.treatment_pembrolizumab==0),"pfs_months"].mean()
mask_low = df.pdl1_tps < 0.5
eff_low = df.loc[mask_low & (df.treatment_pembrolizumab==1),"pfs_months"].mean() - df.loc[mask_low & (df.treatment_pembrolizumab==0),"pfs_months"].mean()
add("pembro_pdl1_subgroup",
    n_high=int(mask_high.sum()), n_low=int(mask_low.sum()),
    pembro_effect_pdl1_high=float(eff_high),
    pembro_effect_pdl1_low=float(eff_low),
)

print("== Iter 7b: pembrolizumab x TMB ==")
add("pembro_x_tmb", **interaction_effect("treatment_pembrolizumab","tmb_high"))

# ------------ ITERATION 8: STK11 modifies pembrolizumab? ------------
print("== Iter 8: pembrolizumab x STK11 ==")
add("pembro_x_stk11", **interaction_effect("treatment_pembrolizumab","stk11_mutation"))

# ------------ ITERATION 9: histology x pembrolizumab ------------
print("== Iter 9: pembrolizumab x histology(squamous) ==")
df["sq"] = (df.histology == "squamous").astype(int)
add("pembro_x_squamous", **interaction_effect("treatment_pembrolizumab","sq"))

# ------------ ITERATION 10: pembrolizumab x ECOG ==
print("== Iter 10: pembrolizumab x ECOG (poor 2 vs good 0/1) ==")
df["ecog_poor"] = (df.ecog_ps == 2).astype(int)
add("pembro_x_ecogpoor", **interaction_effect("treatment_pembrolizumab","ecog_poor"))

# ------------ ITERATION 11: Multivariable model w/ all main effects ------------
print("== Iter 11: full main-effects model ==")
m = smf.ols(
    "pfs_months ~ age_years + sex_female + C(smoking_status) + C(ecog_ps) + "
    "C(histology) + stage_iv + has_brain_mets + egfr_mutation + kras_g12c + "
    "alk_fusion + stk11_mutation + brca2_mutation + pdl1_tps + tmb_high + "
    "albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + "
    "treatment_pembrolizumab + treatment_sotorasib + treatment_olaparib + "
    "treatment_osimertinib + hemoglobin_g_dl + alkaline_phosphatase_u_l + "
    "ast_u_l + alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
    "sodium_meq_l + potassium_meq_l + calcium_mg_dl",
    data=df,
).fit()
mv_coefs = {k: float(v) for k, v in m.params.items()}
mv_p = {k: float(v) for k, v in m.pvalues.items()}
add("multivariable_main",
    rsquared=float(m.rsquared),
    coefs=mv_coefs,
    pvalues=mv_p,
)

# ------------ ITERATION 12: Joint targeted-therapy interaction model ------------
print("== Iter 12: joint targeted-therapy x biomarker interaction model ==")
m = smf.ols(
    "pfs_months ~ treatment_osimertinib*egfr_mutation + "
    "treatment_sotorasib*kras_g12c + treatment_olaparib*brca2_mutation + "
    "treatment_pembrolizumab*pdl1_tps + treatment_pembrolizumab*stk11_mutation",
    data=df,
).fit()
add("joint_targeted_interactions",
    coefs={k:float(v) for k,v in m.params.items()},
    pvalues={k:float(v) for k,v in m.pvalues.items()},
    rsquared=float(m.rsquared),
)

# ------------ ITERATION 13: pembrolizumab effect among PDL1>=0.5 by STK11 ------------
print("== Iter 13: pembrolizumab x STK11 within PDL1-high ==")
high = df[df.pdl1_tps >= 0.5].copy()
m = smf.ols("pfs_months ~ treatment_pembrolizumab*stk11_mutation", data=high).fit()
add("pembro_stk11_in_pdl1_high",
    n=int(len(high)),
    coef_pembro=float(m.params["treatment_pembrolizumab"]),
    p_pembro=float(m.pvalues["treatment_pembrolizumab"]),
    coef_int=float(m.params["treatment_pembrolizumab:stk11_mutation"]),
    p_int=float(m.pvalues["treatment_pembrolizumab:stk11_mutation"]),
)
# subgroup means
for stk in [0,1]:
    sub = high[high.stk11_mutation==stk]
    eff = sub.loc[sub.treatment_pembrolizumab==1,"pfs_months"].mean() - sub.loc[sub.treatment_pembrolizumab==0,"pfs_months"].mean()
    results.setdefault("pembro_stk11_in_pdl1_high", {})[f"effect_stk11_{stk}"] = float(eff)
    results["pembro_stk11_in_pdl1_high"][f"n_stk11_{stk}"] = int(len(sub))

# ------------ ITERATION 14: pembrolizumab effect by PDL1 among STK11=0 ------------
print("== Iter 14: pembrolizumab x PDL1 among STK11=0 ==")
nonstk = df[df.stk11_mutation==0].copy()
m = smf.ols("pfs_months ~ treatment_pembrolizumab*pdl1_tps", data=nonstk).fit()
add("pembro_pdl1_in_stk11neg",
    n=int(len(nonstk)),
    coef_int=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
    p_int=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
    coef_pembro=float(m.params["treatment_pembrolizumab"]),
    p_pembro=float(m.pvalues["treatment_pembrolizumab"]),
)

# ------------ ITERATION 15: Triple-subgroup effect: pembrolizumab in PDL1>=0.5 & STK11=0 ------------
print("== Iter 15: pembrolizumab in PDL1>=0.5 & STK11=0 ==")
mask = (df.pdl1_tps >= 0.5) & (df.stk11_mutation == 0)
sub = df[mask]
m = smf.ols("pfs_months ~ treatment_pembrolizumab", data=sub).fit()
add("pembro_in_pdl1high_stk11neg",
    n=int(len(sub)),
    coef=float(m.params["treatment_pembrolizumab"]),
    p=float(m.pvalues["treatment_pembrolizumab"]),
    pfs_treated=float(sub.loc[sub.treatment_pembrolizumab==1,"pfs_months"].mean()),
    pfs_untreated=float(sub.loc[sub.treatment_pembrolizumab==0,"pfs_months"].mean()),
)
# also in PDL1>=0.5 & STK11=1 ("suppressed")
mask2 = (df.pdl1_tps >= 0.5) & (df.stk11_mutation == 1)
sub2 = df[mask2]
m2 = smf.ols("pfs_months ~ treatment_pembrolizumab", data=sub2).fit()
add("pembro_in_pdl1high_stk11pos",
    n=int(len(sub2)),
    coef=float(m2.params["treatment_pembrolizumab"]),
    p=float(m2.pvalues["treatment_pembrolizumab"]),
    pfs_treated=float(sub2.loc[sub2.treatment_pembrolizumab==1,"pfs_months"].mean()),
    pfs_untreated=float(sub2.loc[sub2.treatment_pembrolizumab==0,"pfs_months"].mean()),
)

# ------------ ITERATION 16: Targeted-therapy refined definitions ------------
print("== Iter 16: osimertinib effect in EGFR+ subgroup (and EGFR-) ==")
egfrpos = df[df.egfr_mutation==1]
m = smf.ols("pfs_months ~ treatment_osimertinib", data=egfrpos).fit()
add("osi_in_egfrpos",
    n=int(len(egfrpos)),
    coef=float(m.params["treatment_osimertinib"]),
    p=float(m.pvalues["treatment_osimertinib"]),
    pfs_treated=float(egfrpos.loc[egfrpos.treatment_osimertinib==1,"pfs_months"].mean()),
    pfs_untreated=float(egfrpos.loc[egfrpos.treatment_osimertinib==0,"pfs_months"].mean()),
)
egfrneg = df[df.egfr_mutation==0]
m = smf.ols("pfs_months ~ treatment_osimertinib", data=egfrneg).fit()
add("osi_in_egfrneg",
    n=int(len(egfrneg)),
    coef=float(m.params["treatment_osimertinib"]),
    p=float(m.pvalues["treatment_osimertinib"]),
)

print("== Iter 17: sotorasib effect in KRAS G12C+ vs - ==")
for v in [1,0]:
    sub = df[df.kras_g12c==v]
    m = smf.ols("pfs_months ~ treatment_sotorasib", data=sub).fit()
    add(f"soto_in_krasg12c_{v}",
        n=int(len(sub)),
        coef=float(m.params["treatment_sotorasib"]),
        p=float(m.pvalues["treatment_sotorasib"]),
    )

print("== Iter 18: olaparib effect in BRCA2+ vs - ==")
for v in [1,0]:
    sub = df[df.brca2_mutation==v]
    m = smf.ols("pfs_months ~ treatment_olaparib", data=sub).fit()
    add(f"olap_in_brca2_{v}",
        n=int(len(sub)),
        coef=float(m.params["treatment_olaparib"]),
        p=float(m.pvalues["treatment_olaparib"]),
    )

# ------------ ITERATION 19: targeted therapy in non-mutated population - is there harm? ------------
print("== Iter 19: targeted therapies in 'wrong' biomarker populations ==")
# explore cross-effects
add("osi_in_kras_pos",
    n=int((df.kras_g12c==1).sum()),
    diff=float(df.loc[(df.kras_g12c==1)&(df.treatment_osimertinib==1),"pfs_months"].mean()
              - df.loc[(df.kras_g12c==1)&(df.treatment_osimertinib==0),"pfs_months"].mean())
)

# ------------ ITERATION 20: heterogeneity search: pembro x feature interactions screen ------------
print("== Iter 20: pembrolizumab interaction screen across all features ==")
features = ["age_years","sex_female","stage_iv","has_brain_mets","egfr_mutation",
            "kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","pdl1_tps",
            "tmb_high","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
            "hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l",
            "total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl","sodium_meq_l",
            "potassium_meq_l","calcium_mg_dl"]
screen = {}
for f in features:
    formula = f"pfs_months ~ treatment_pembrolizumab*{f}"
    try:
        m = smf.ols(formula, data=df).fit()
        ikey = f"treatment_pembrolizumab:{f}"
        screen[f] = {"coef": float(m.params[ikey]), "p": float(m.pvalues[ikey])}
    except Exception as e:
        screen[f] = {"err": str(e)}
add("pembro_feature_screen", **screen)

print("== Iter 20b: osimertinib interaction screen (focused on EGFR conditional) ==")
screen2 = {}
for f in features:
    if f == "egfr_mutation":
        continue
    try:
        m = smf.ols(f"pfs_months ~ treatment_osimertinib*{f}", data=df).fit()
        ikey = f"treatment_osimertinib:{f}"
        screen2[f] = {"coef": float(m.params[ikey]), "p": float(m.pvalues[ikey])}
    except Exception as e:
        screen2[f] = {"err": str(e)}
add("osimertinib_feature_screen", **screen2)

print("== Iter 20c: sotorasib screen ==")
screen3 = {}
for f in features:
    if f == "kras_g12c":
        continue
    try:
        m = smf.ols(f"pfs_months ~ treatment_sotorasib*{f}", data=df).fit()
        ikey = f"treatment_sotorasib:{f}"
        screen3[f] = {"coef": float(m.params[ikey]), "p": float(m.pvalues[ikey])}
    except Exception as e:
        screen3[f] = {"err": str(e)}
add("sotorasib_feature_screen", **screen3)

print("== Iter 20d: olaparib screen ==")
screen4 = {}
for f in features:
    if f == "brca2_mutation":
        continue
    try:
        m = smf.ols(f"pfs_months ~ treatment_olaparib*{f}", data=df).fit()
        ikey = f"treatment_olaparib:{f}"
        screen4[f] = {"coef": float(m.params[ikey]), "p": float(m.pvalues[ikey])}
    except Exception as e:
        screen4[f] = {"err": str(e)}
add("olaparib_feature_screen", **screen4)

# ------------ ITERATION 21: refined osimertinib subgroup (EGFR+ and any modifiers) ------------
print("== Iter 21: osimertinib in EGFR+ refined by features ==")
egfrpos = df[df.egfr_mutation==1].copy()
# By brain mets
for v in [0,1]:
    sub = egfrpos[egfrpos.has_brain_mets==v]
    if len(sub)>20:
        m = smf.ols("pfs_months ~ treatment_osimertinib", data=sub).fit()
        add(f"osi_in_egfrpos_brainmets_{v}",
            n=int(len(sub)),
            coef=float(m.params["treatment_osimertinib"]),
            p=float(m.pvalues["treatment_osimertinib"]),
        )

# ------------ ITERATION 22: Concurrent STK11 in KRAS G12C subgroup for sotorasib ------------
print("== Iter 22: sotorasib in KRAS+ stratified by STK11 ==")
krasp = df[df.kras_g12c==1]
for stk in [0,1]:
    sub = krasp[krasp.stk11_mutation==stk]
    m = smf.ols("pfs_months ~ treatment_sotorasib", data=sub).fit()
    add(f"soto_in_krasp_stk11_{stk}",
        n=int(len(sub)),
        coef=float(m.params["treatment_sotorasib"]),
        p=float(m.pvalues["treatment_sotorasib"]),
    )

# ------------ ITERATION 23: pembro x PDL1 interaction in TMB-high ------------
print("== Iter 23: pembrolizumab x PDL1 in TMB-high vs TMB-low ==")
for tmb in [0,1]:
    sub = df[df.tmb_high==tmb]
    m = smf.ols("pfs_months ~ treatment_pembrolizumab*pdl1_tps", data=sub).fit()
    add(f"pembro_pdl1_in_tmb_{tmb}",
        n=int(len(sub)),
        int_coef=float(m.params["treatment_pembrolizumab:pdl1_tps"]),
        int_p=float(m.pvalues["treatment_pembrolizumab:pdl1_tps"]),
        pembro_main=float(m.params["treatment_pembrolizumab"]),
        pembro_p=float(m.pvalues["treatment_pembrolizumab"]),
    )

# ------------ ITERATION 24: full pembro 3-way subgroup search (PDL1>=.5, TMB-high, STK11-) ------------
print("== Iter 24: pembrolizumab 3-way subgroup ==")
mask = (df.pdl1_tps>=0.5) & (df.tmb_high==1) & (df.stk11_mutation==0)
sub = df[mask]
m = smf.ols("pfs_months ~ treatment_pembrolizumab", data=sub).fit()
add("pembro_3way_subgroup",
    n=int(len(sub)),
    coef=float(m.params["treatment_pembrolizumab"]),
    p=float(m.pvalues["treatment_pembrolizumab"]),
    pfs_treated=float(sub.loc[sub.treatment_pembrolizumab==1,"pfs_months"].mean()),
    pfs_untreated=float(sub.loc[sub.treatment_pembrolizumab==0,"pfs_months"].mean()),
)
# complementary
for label, mask in [
    ("pdl1_low_or_tmb_low_or_stk11", ~mask),
    ("pdl1_high_tmb_low_stk11_neg", (df.pdl1_tps>=0.5) & (df.tmb_high==0) & (df.stk11_mutation==0)),
    ("pdl1_high_tmb_high_stk11_pos", (df.pdl1_tps>=0.5) & (df.tmb_high==1) & (df.stk11_mutation==1)),
    ("pdl1_low_tmb_high_stk11_neg", (df.pdl1_tps<0.5) & (df.tmb_high==1) & (df.stk11_mutation==0)),
]:
    sub = df[mask]
    if len(sub) < 50: continue
    if sub.treatment_pembrolizumab.nunique()<2: continue
    m = smf.ols("pfs_months ~ treatment_pembrolizumab", data=sub).fit()
    add(f"pembro_subgroup_{label}",
        n=int(len(sub)),
        coef=float(m.params["treatment_pembrolizumab"]),
        p=float(m.pvalues["treatment_pembrolizumab"]),
    )

# ------------ ITERATION 25: comprehensive multivariable interaction model w/ STK11 ------------
print("== Iter 25: comprehensive joint model ==")
m = smf.ols(
    "pfs_months ~ age_years + C(ecog_ps) + stage_iv + has_brain_mets + albumin_g_dl + "
    "ldh_u_l + nlr + crp_mg_l + weight_loss_pct_6mo + "
    "treatment_osimertinib*egfr_mutation + "
    "treatment_sotorasib*kras_g12c + "
    "treatment_olaparib*brca2_mutation + "
    "treatment_pembrolizumab*pdl1_tps + "
    "treatment_pembrolizumab*stk11_mutation + "
    "treatment_pembrolizumab*tmb_high",
    data=df,
).fit()
add("comprehensive_model",
    rsquared=float(m.rsquared),
    coefs={k:float(v) for k,v in m.params.items()},
    pvalues={k:float(v) for k,v in m.pvalues.items()},
)

# Save
with open("results.json","w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved results.json")
print("Total result keys:", len(results))
