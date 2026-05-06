"""
Iter 23-25: deep refinement and sanity checks of treatment-effect heterogeneity.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

DF = pd.read_parquet("dataset.parquet")
RESULTS = json.load(open("my_run_results.json"))

def add(key, **kw):
    RESULTS[key] = kw

def sub_effect(treat, sub_mask, label):
    on_mask = sub_mask & (DF[treat]==1)
    off_mask = sub_mask & (DF[treat]==0)
    on = DF.loc[on_mask, "pfs_months"]
    off = DF.loc[off_mask, "pfs_months"]
    if len(on)<5 or len(off)<5:
        return {"treat": treat, "sub": label,
                "n_on": int(on_mask.sum()), "n_off": int(off_mask.sum()),
                "delta": None, "p": None}
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return {
        "treat": treat, "sub": label,
        "n_on": int(on_mask.sum()), "n_off": int(off_mask.sum()),
        "mean_on": float(on.mean()), "mean_off": float(off.mean()),
        "delta": float(on.mean() - off.mean()),
        "p": float(p)
    }

# ===========================================================
# Confirm KRAS+ x male is the right subgroup for sotorasib
# Try other variants
# ===========================================================
sub = (DF["kras_g12c"]==1) & (DF["sex_female"]==0) & (DF["alk_fusion"]==0)
add("sub_sot_kras_male_alkneg", **sub_effect("treatment_sotorasib", sub, "KRAS+/male/ALK-"))
sub = (DF["kras_g12c"]==1) & (DF["sex_female"]==0) & (DF["brca2_mutation"]==0)
add("sub_sot_kras_male_brca2neg", **sub_effect("treatment_sotorasib", sub, "KRAS+/male/BRCA2-"))
sub = (DF["kras_g12c"]==1) & (DF["sex_female"]==0) & (DF["alk_fusion"]==0) & (DF["brca2_mutation"]==0)
add("sub_sot_kras_male_clean", **sub_effect("treatment_sotorasib", sub, "KRAS+/male/ALK-/BRCA2-"))

# Even cleaner: any cofactor weakening signal?
sub = (DF["kras_g12c"]==1) & (DF["sex_female"]==0) & (DF["egfr_mutation"]==0)
add("sub_sot_kras_male_egfrneg", **sub_effect("treatment_sotorasib", sub, "KRAS+/male/EGFR-"))

# Verify all "off" effect is null among non-KRAS or non-male
sub = (DF["kras_g12c"]==0)  # all KRAS-
add("sub_sot_no_kras", **sub_effect("treatment_sotorasib", sub, "KRAS-"))
sub = (DF["sex_female"]==1) & (DF["kras_g12c"]==1)
add("sub_sot_kras_female", **sub_effect("treatment_sotorasib", sub, "KRAS+/female"))

# Also look at any mini effect in KRAS+/male (positive control)
# Test the joint subgroup hypothesis: sotorasib improves PFS only in KRAS G12C+ male
# Run regression: KRAS+ only
df = DF[DF["kras_g12c"]==1].copy()
# triple interaction: sotorasib x sex x stk11
df["t"] = df["treatment_sotorasib"].astype(float)
df["sx"] = df["sex_female"].astype(float)
df["sk"] = df["stk11_mutation"].astype(float)
df["t_sx"] = df["t"]*df["sx"]
df["t_sk"] = df["t"]*df["sk"]
df["sx_sk"] = df["sx"]*df["sk"]
df["t_sx_sk"] = df["t"]*df["sx"]*df["sk"]
X = df[["t","sx","sk","t_sx","t_sk","sx_sk","t_sx_sk"]].astype(float).values
X = sm.add_constant(X)
res = sm.OLS(df["pfs_months"].values, X).fit()
add("sot_kras_3way", coefs={n: {"coef": float(res.params[i]), "p": float(res.pvalues[i])}
    for i,n in enumerate(["const","t","sx","sk","t_sx","t_sk","sx_sk","t_sx_sk"])})

# ===========================================================
# Look for any pembro signal in subgroups beyond what we tried
# ===========================================================
# Try pembro effect in KRAS G12C-/EGFR-/ALK- (i.e. "no driver" + PDL1>=50%)
sub = (DF["egfr_mutation"]==0) & (DF["alk_fusion"]==0) & (DF["kras_g12c"]==0) & (DF["pdl1_tps"]>=0.5)
add("sub_pembro_no_driver_pdl1hi_v2", **sub_effect("treatment_pembrolizumab", sub, "no_driver_PDL1hi"))

# Pembro in females, in males
add("sub_pembro_male", **sub_effect("treatment_pembrolizumab", DF["sex_female"]==0, "male"))
add("sub_pembro_female", **sub_effect("treatment_pembrolizumab", DF["sex_female"]==1, "female"))

# Try pembro x age interaction — already in screen, was p~0.10
# Pembro x ecog
for e in [0,1,2]:
    add(f"sub_pembro_ecog{e}", **sub_effect("treatment_pembrolizumab", DF["ecog_ps"]==e, f"ECOG{e}"))

# ===========================================================
# Look for any osi signal
# ===========================================================
# Osi effect in EGFR+ AND EGFR-? Already done. Also check male/female interaction
df = DF[DF["egfr_mutation"]==1].copy()
df["t"] = df["treatment_osimertinib"].astype(float)
df["sx"] = df["sex_female"].astype(float)
df["i"] = df["t"]*df["sx"]
X = sm.add_constant(df[["t","sx","i"]].values)
res = sm.OLS(df["pfs_months"].values, X).fit()
add("osi_egfr_sex_interact", main_t=float(res.params[1]), main_mod=float(res.params[2]),
    interact=float(res.params[3]), p_int=float(res.pvalues[3]), p_main_t=float(res.pvalues[1]))

# Try osi in all-EGFR with brain mets and male
sub = (DF["egfr_mutation"]==1) & (DF["sex_female"]==0) & (DF["has_brain_mets"]==1)
add("sub_osi_egfr_male_brain", **sub_effect("treatment_osimertinib", sub, "EGFR+/male/brain"))
sub = (DF["egfr_mutation"]==1) & (DF["sex_female"]==1) & (DF["has_brain_mets"]==0)
add("sub_osi_egfr_female_nobrain", **sub_effect("treatment_osimertinib", sub, "EGFR+/female/no_brain"))

# Check for olaparib in BRCA2+/female/specific labs
sub = (DF["brca2_mutation"]==1) & (DF["sex_female"]==1) & (DF["albumin_g_dl"]>=4.0)
add("sub_olap_brca2_female_alb_hi", **sub_effect("treatment_olaparib", sub, "BRCA2+/female/alb>=4"))

# ===========================================================
# Final: confirm no main treatment effect for pembro/olap/osi in their "unselected" use
# ===========================================================
# pembro effect in patients without the sotorasib-relevant mutation
sub = (DF["kras_g12c"]==0)
add("sub_pembro_no_kras", **sub_effect("treatment_pembrolizumab", sub, "KRAS-"))
sub = (DF["egfr_mutation"]==0) & (DF["alk_fusion"]==0) & (DF["kras_g12c"]==0) & (DF["brca2_mutation"]==0)
add("sub_pembro_clean_unselected", **sub_effect("treatment_pembrolizumab", sub, "no_actionable_driver"))

# ===========================================================
# Save
# ===========================================================
with open("my_run_results.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("Wrote", len(RESULTS), "keys")
