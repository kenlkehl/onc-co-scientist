"""
Comprehensive iterative analysis of the NSCLC dataset.
Outputs raw analysis dictionary to my_run_results.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DF = pd.read_parquet("dataset.parquet")
RESULTS = {}

def add(key, **kw):
    RESULTS[key] = kw

def lr_pfs(formula_predictors, df=None, label=None):
    """Linear regression of pfs_months on predictors. Returns coef and p of first predictor."""
    if df is None:
        df = DF
    y = df["pfs_months"].values
    X = df[formula_predictors].astype(float).values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def t_two_groups(mask, label_on="on", label_off="off"):
    on = DF.loc[mask, "pfs_months"]
    off = DF.loc[~mask, "pfs_months"]
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return {
        "n_on": int(mask.sum()),
        "n_off": int((~mask).sum()),
        "mean_on": float(on.mean()),
        "mean_off": float(off.mean()),
        "delta": float(on.mean() - off.mean()),
        "t": float(t),
        "p": float(p),
    }

# ====================================================================
# ITER 1 — Marginal main effects of each treatment on pfs_months
# ====================================================================
for t in ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]:
    r = t_two_groups(DF[t]==1)
    add(f"main_{t}", **r)

# ====================================================================
# ITER 2 — Marginal main effects of biomarkers on pfs_months
# ====================================================================
for m in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    r = t_two_groups(DF[m]==1)
    add(f"biomarker_{m}", **r)

# pdl1 continuous
m = lr_pfs(["pdl1_tps"])
add("biomarker_pdl1_tps", coef=float(m.params[1]), p=float(m.pvalues[1]))

# ====================================================================
# ITER 3 — Clinical/demographic main effects
# ====================================================================
# age
m = lr_pfs(["age_years"])
add("age_main", coef=float(m.params[1]), p=float(m.pvalues[1]))
# sex
add("sex_female_main", **t_two_groups(DF["sex_female"]==1))
# stage_iv
add("stage_iv_main", **t_two_groups(DF["stage_iv"]==1))
# brain mets
add("has_brain_mets_main", **t_two_groups(DF["has_brain_mets"]==1))
# ECOG: 0 vs 1+2 and ordinal
m = lr_pfs(["ecog_ps"])
add("ecog_main", coef=float(m.params[1]), p=float(m.pvalues[1]))
# histology squamous vs adeno
add("histology_squamous", **t_two_groups(DF["histology"]=="squamous"))
# smoking
for s in ["never","former","current"]:
    add(f"smoking_{s}", **t_two_groups(DF["smoking_status"]==s))

# ====================================================================
# ITER 4 — Lab main effects (continuous, linear regression)
# ====================================================================
labs = ["albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr",
        "hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l","alt_u_l",
        "total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl",
        "sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
for lab in labs:
    m = lr_pfs([lab])
    add(f"lab_{lab}", coef=float(m.params[1]), p=float(m.pvalues[1]))

# ====================================================================
# ITER 5 — Pembro x biomarker interactions
# ====================================================================
def interaction(treat, modifier, mod_kind="binary"):
    """Treatment by modifier interaction on pfs_months."""
    df = DF.copy()
    if mod_kind == "binary":
        df["mod"] = (df[modifier] == 1).astype(int)
    else:
        df["mod"] = df[modifier].astype(float)
    df["t"] = (df[treat] == 1).astype(int)
    df["interact"] = df["t"] * df["mod"]
    X = df[["t", "mod", "interact"]].values
    X = sm.add_constant(X)
    y = df["pfs_months"].values
    m = sm.OLS(y, X).fit()
    return {
        "main_t": float(m.params[1]),
        "main_mod": float(m.params[2]),
        "interact": float(m.params[3]),
        "p_interact": float(m.pvalues[3]),
        "p_main_t": float(m.pvalues[1]),
        "p_main_mod": float(m.pvalues[2]),
    }

for mod in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    add(f"int_pembro_{mod}", **interaction("treatment_pembrolizumab", mod))
add("int_pembro_pdl1", **interaction("treatment_pembrolizumab","pdl1_tps","cont"))

# squamous
df = DF.copy()
df["squamous"] = (df["histology"]=="squamous").astype(int)
df["t"] = df["treatment_pembrolizumab"]
df["interact"] = df["t"]*df["squamous"]
X = sm.add_constant(df[["t","squamous","interact"]].values)
m = sm.OLS(df["pfs_months"].values, X).fit()
add("int_pembro_squamous", main_t=float(m.params[1]), main_mod=float(m.params[2]),
    interact=float(m.params[3]), p_interact=float(m.pvalues[3]))

# smoking (current vs not)
df["current_smoker"] = (df["smoking_status"]=="current").astype(int)
df["interact"] = df["t"]*df["current_smoker"]
X = sm.add_constant(df[["t","current_smoker","interact"]].values)
m = sm.OLS(df["pfs_months"].values, X).fit()
add("int_pembro_currentsmoker", main_t=float(m.params[1]), main_mod=float(m.params[2]),
    interact=float(m.params[3]), p_interact=float(m.pvalues[3]))

# never-smoker
df["never_smoker"] = (df["smoking_status"]=="never").astype(int)
df["interact"] = df["t"]*df["never_smoker"]
X = sm.add_constant(df[["t","never_smoker","interact"]].values)
m = sm.OLS(df["pfs_months"].values, X).fit()
add("int_pembro_neversmoker", main_t=float(m.params[1]), main_mod=float(m.params[2]),
    interact=float(m.params[3]), p_interact=float(m.pvalues[3]))

# ====================================================================
# ITER 6 — Osimertinib x biomarker
# ====================================================================
for mod in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    add(f"int_osi_{mod}", **interaction("treatment_osimertinib", mod))
add("int_osi_pdl1", **interaction("treatment_osimertinib","pdl1_tps","cont"))

# ====================================================================
# ITER 7 — Sotorasib x biomarker
# ====================================================================
for mod in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    add(f"int_sot_{mod}", **interaction("treatment_sotorasib", mod))

# ====================================================================
# ITER 8 — Olaparib x biomarker
# ====================================================================
for mod in ["egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation","tmb_high"]:
    add(f"int_olap_{mod}", **interaction("treatment_olaparib", mod))

# ====================================================================
# ITER 9 — In-subgroup treatment effects (the matched biomarker)
# ====================================================================
def sub_effect(treat, sub_mask, label):
    on_mask = sub_mask & (DF[treat]==1)
    off_mask = sub_mask & (DF[treat]==0)
    on = DF.loc[on_mask, "pfs_months"]
    off = DF.loc[off_mask, "pfs_months"]
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return {
        "treat": treat, "sub": label,
        "n_on": int(on_mask.sum()), "n_off": int(off_mask.sum()),
        "mean_on": float(on.mean()) if len(on) else None,
        "mean_off": float(off.mean()) if len(off) else None,
        "delta": float(on.mean() - off.mean()) if len(on) and len(off) else None,
        "p": float(p)
    }

add("sub_osi_egfr", **sub_effect("treatment_osimertinib", DF["egfr_mutation"]==1, "EGFR+"))
add("sub_osi_no_egfr", **sub_effect("treatment_osimertinib", DF["egfr_mutation"]==0, "EGFR-"))
add("sub_sot_kras", **sub_effect("treatment_sotorasib", DF["kras_g12c"]==1, "KRAS G12C+"))
add("sub_sot_no_kras", **sub_effect("treatment_sotorasib", DF["kras_g12c"]==0, "KRAS G12C-"))
add("sub_olap_brca2", **sub_effect("treatment_olaparib", DF["brca2_mutation"]==1, "BRCA2+"))
add("sub_olap_no_brca2", **sub_effect("treatment_olaparib", DF["brca2_mutation"]==0, "BRCA2-"))
add("sub_pembro_pdl1high", **sub_effect("treatment_pembrolizumab", DF["pdl1_tps"]>=0.5, "PDL1>=50%"))
add("sub_pembro_pdl1low", **sub_effect("treatment_pembrolizumab", DF["pdl1_tps"]<0.5, "PDL1<50%"))
add("sub_pembro_tmbhigh", **sub_effect("treatment_pembrolizumab", DF["tmb_high"]==1, "TMB-H"))
add("sub_pembro_tmblow", **sub_effect("treatment_pembrolizumab", DF["tmb_high"]==0, "TMB-L"))

# ====================================================================
# ITER 10 — STK11 effect on pembrolizumab (suppressor candidate)
# ====================================================================
# Pembro effect by STK11 status
add("sub_pembro_stk11pos", **sub_effect("treatment_pembrolizumab", DF["stk11_mutation"]==1, "STK11+"))
add("sub_pembro_stk11neg", **sub_effect("treatment_pembrolizumab", DF["stk11_mutation"]==0, "STK11-"))

# Triple subgroup: PDL1>=50 AND STK11- vs PDL1>=50 AND STK11+
sub_a = (DF["pdl1_tps"]>=0.5) & (DF["stk11_mutation"]==0)
sub_b = (DF["pdl1_tps"]>=0.5) & (DF["stk11_mutation"]==1)
add("sub_pembro_pdl1hi_stk11neg", **sub_effect("treatment_pembrolizumab", sub_a, "PDL1>=50 & STK11-"))
add("sub_pembro_pdl1hi_stk11pos", **sub_effect("treatment_pembrolizumab", sub_b, "PDL1>=50 & STK11+"))

# ====================================================================
# ITER 11 — KRAS x STK11 in pembro/sotorasib (STK11 co-mut suppresses)
# ====================================================================
sub_kras_stk11neg = (DF["kras_g12c"]==1) & (DF["stk11_mutation"]==0)
sub_kras_stk11pos = (DF["kras_g12c"]==1) & (DF["stk11_mutation"]==1)
add("sub_sot_kras_stk11neg", **sub_effect("treatment_sotorasib", sub_kras_stk11neg, "KRAS+/STK11-"))
add("sub_sot_kras_stk11pos", **sub_effect("treatment_sotorasib", sub_kras_stk11pos, "KRAS+/STK11+"))

# ====================================================================
# ITER 12 — Treatment effects in clean monotherapy contexts (single-treatment patients)
# ====================================================================
treat_cols = ["treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]
n_treat = DF[treat_cols].sum(axis=1)
# pure monotherapy or none
pure_pembro = (DF["treatment_pembrolizumab"]==1) & (n_treat==1)
no_treat = (n_treat==0)
on = DF.loc[pure_pembro, "pfs_months"]; off = DF.loc[no_treat, "pfs_months"]
add("monothx_pembro_vs_none",
    n_on=int(pure_pembro.sum()), n_off=int(no_treat.sum()),
    mean_on=float(on.mean()), mean_off=float(off.mean()),
    delta=float(on.mean()-off.mean()),
    p=float(stats.ttest_ind(on,off,equal_var=False).pvalue))

for t,label in zip(["treatment_sotorasib","treatment_olaparib","treatment_osimertinib"],
                   ["sot","olap","osi"]):
    pure = (DF[t]==1) & (n_treat==1)
    on = DF.loc[pure, "pfs_months"]
    add(f"monothx_{label}_vs_none",
        n_on=int(pure.sum()), n_off=int(no_treat.sum()),
        mean_on=float(on.mean()), mean_off=float(off.mean()),
        delta=float(on.mean()-off.mean()),
        p=float(stats.ttest_ind(on,off,equal_var=False).pvalue))

# ====================================================================
# ITER 13 — Multivariable regression of PFS on all features
# ====================================================================
df = DF.copy()
df["squamous"] = (df["histology"]=="squamous").astype(int)
df["smoke_curr"] = (df["smoking_status"]=="current").astype(int)
df["smoke_form"] = (df["smoking_status"]=="former").astype(int)

features = ["age_years","sex_female","ecog_ps","squamous","smoke_curr","smoke_form",
            "stage_iv","has_brain_mets","egfr_mutation","kras_g12c","alk_fusion",
            "stk11_mutation","brca2_mutation","pdl1_tps","tmb_high","albumin_g_dl",
            "ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr","hemoglobin_g_dl",
            "alkaline_phosphatase_u_l","ast_u_l","alt_u_l","total_bilirubin_mg_dl",
            "creatinine_mg_dl","bun_mg_dl","sodium_meq_l","potassium_meq_l","calcium_mg_dl",
            "treatment_pembrolizumab","treatment_sotorasib","treatment_olaparib","treatment_osimertinib"]

X = sm.add_constant(df[features].astype(float).values)
m = sm.OLS(df["pfs_months"].values, X).fit()
mv = {}
for i, f in enumerate(["const"]+features):
    mv[f] = {"coef": float(m.params[i]), "p": float(m.pvalues[i])}
add("multivariable_all", coefs=mv, r2=float(m.rsquared))

# ====================================================================
# ITER 14 — Interaction: osi x EGFR with key co-mutations
# ====================================================================
sub_egfr_brain = (DF["egfr_mutation"]==1) & (DF["has_brain_mets"]==1)
sub_egfr_no_brain = (DF["egfr_mutation"]==1) & (DF["has_brain_mets"]==0)
add("sub_osi_egfr_brainmets", **sub_effect("treatment_osimertinib", sub_egfr_brain, "EGFR+/brain"))
add("sub_osi_egfr_no_brainmets", **sub_effect("treatment_osimertinib", sub_egfr_no_brain, "EGFR+/no_brain"))

sub_egfr_tp = (DF["egfr_mutation"]==1) & (DF["stk11_mutation"]==1)
sub_egfr_tn = (DF["egfr_mutation"]==1) & (DF["stk11_mutation"]==0)
add("sub_osi_egfr_stk11pos", **sub_effect("treatment_osimertinib", sub_egfr_tp, "EGFR+/STK11+"))
add("sub_osi_egfr_stk11neg", **sub_effect("treatment_osimertinib", sub_egfr_tn, "EGFR+/STK11-"))

# ====================================================================
# ITER 15 — Olaparib x BRCA2 stratified
# ====================================================================
sub_brca2_alb = (DF["brca2_mutation"]==1) & (DF["albumin_g_dl"]>=df["albumin_g_dl"].median())
add("sub_olap_brca2_alb_high", **sub_effect("treatment_olaparib", sub_brca2_alb, "BRCA2+/alb>=med"))

# ====================================================================
# ITER 16 — All pairwise treatment x feature interactions for pembrolizumab
# ====================================================================
pembro_int_results = {}
candidate_mods = ["age_years","sex_female","ecog_ps","stage_iv","has_brain_mets",
                  "egfr_mutation","kras_g12c","alk_fusion","stk11_mutation","brca2_mutation",
                  "pdl1_tps","tmb_high","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo",
                  "crp_mg_l","nlr","hemoglobin_g_dl","alkaline_phosphatase_u_l","ast_u_l",
                  "alt_u_l","total_bilirubin_mg_dl","creatinine_mg_dl","bun_mg_dl",
                  "sodium_meq_l","potassium_meq_l","calcium_mg_dl"]
for mod in candidate_mods:
    df = DF.copy()
    df["t"] = df["treatment_pembrolizumab"]
    df["m"] = df[mod].astype(float)
    df["i"] = df["t"]*df["m"]
    X = sm.add_constant(df[["t","m","i"]].values)
    res = sm.OLS(df["pfs_months"].values, X).fit()
    pembro_int_results[mod] = {
        "coef_int": float(res.params[3]),
        "p_int": float(res.pvalues[3]),
    }
add("pembro_interactions_screen", results=pembro_int_results)

# Same for osi, sot, olap
for t, lab in [("treatment_osimertinib","osi"),
               ("treatment_sotorasib","sot"),
               ("treatment_olaparib","olap")]:
    res_dict = {}
    for mod in candidate_mods:
        df = DF.copy()
        df["t"] = df[t]
        df["m"] = df[mod].astype(float)
        df["i"] = df["t"]*df["m"]
        X = sm.add_constant(df[["t","m","i"]].values)
        res = sm.OLS(df["pfs_months"].values, X).fit()
        res_dict[mod] = {
            "coef_int": float(res.params[3]),
            "p_int": float(res.pvalues[3]),
        }
    add(f"{lab}_interactions_screen", results=res_dict)

# ====================================================================
# Save
# ====================================================================
with open("my_run_results.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("Wrote my_run_results.json with", len(RESULTS), "keys")
