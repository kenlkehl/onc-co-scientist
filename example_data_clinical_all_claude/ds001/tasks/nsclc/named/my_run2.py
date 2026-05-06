"""
Iter 17-25: Refined subgroup discovery and modifier-of-modifier searches.
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
# Focus on sotorasib effect within KRAS G12C+ — explore modifiers
# ===========================================================
kras = DF["kras_g12c"]==1

# By sex
add("sub_sot_kras_male", **sub_effect("treatment_sotorasib", kras & (DF["sex_female"]==0), "KRAS+/male"))
add("sub_sot_kras_female", **sub_effect("treatment_sotorasib", kras & (DF["sex_female"]==1), "KRAS+/female"))

# By histology
add("sub_sot_kras_adeno", **sub_effect("treatment_sotorasib", kras & (DF["histology"]=="adenocarcinoma"), "KRAS+/adeno"))
add("sub_sot_kras_squam", **sub_effect("treatment_sotorasib", kras & (DF["histology"]=="squamous"), "KRAS+/squamous"))

# By smoking
for s in ["current","former","never"]:
    add(f"sub_sot_kras_smoke_{s}", **sub_effect("treatment_sotorasib", kras & (DF["smoking_status"]==s), f"KRAS+/{s}"))

# By ECOG
for e in [0,1,2]:
    add(f"sub_sot_kras_ecog{e}", **sub_effect("treatment_sotorasib", kras & (DF["ecog_ps"]==e), f"KRAS+/ECOG{e}"))

# By stage_iv
add("sub_sot_kras_stage4", **sub_effect("treatment_sotorasib", kras & (DF["stage_iv"]==1), "KRAS+/stage4"))
add("sub_sot_kras_nonstage4", **sub_effect("treatment_sotorasib", kras & (DF["stage_iv"]==0), "KRAS+/nonstage4"))

# By brain mets
add("sub_sot_kras_brain", **sub_effect("treatment_sotorasib", kras & (DF["has_brain_mets"]==1), "KRAS+/brain"))
add("sub_sot_kras_nobrain", **sub_effect("treatment_sotorasib", kras & (DF["has_brain_mets"]==0), "KRAS+/no_brain"))

# By co-mutations
for m in ["egfr_mutation","alk_fusion","brca2_mutation","stk11_mutation"]:
    add(f"sub_sot_kras_{m}_pos", **sub_effect("treatment_sotorasib", kras & (DF[m]==1), f"KRAS+/{m}+"))
    add(f"sub_sot_kras_{m}_neg", **sub_effect("treatment_sotorasib", kras & (DF[m]==0), f"KRAS+/{m}-"))

# By PDL1
add("sub_sot_kras_pdl1high", **sub_effect("treatment_sotorasib", kras & (DF["pdl1_tps"]>=0.5), "KRAS+/PDL1>=50%"))
add("sub_sot_kras_pdl1low", **sub_effect("treatment_sotorasib", kras & (DF["pdl1_tps"]<0.5), "KRAS+/PDL1<50%"))
add("sub_sot_kras_tmb_high", **sub_effect("treatment_sotorasib", kras & (DF["tmb_high"]==1), "KRAS+/TMB-H"))
add("sub_sot_kras_tmb_low", **sub_effect("treatment_sotorasib", kras & (DF["tmb_high"]==0), "KRAS+/TMB-L"))

# By labs (binary high/low at median)
med_alb = DF["albumin_g_dl"].median()
add("sub_sot_kras_alb_high", **sub_effect("treatment_sotorasib", kras & (DF["albumin_g_dl"]>=med_alb), "KRAS+/alb>=med"))
add("sub_sot_kras_alb_low", **sub_effect("treatment_sotorasib", kras & (DF["albumin_g_dl"]<med_alb), "KRAS+/alb<med"))

# ===========================================================
# Sotorasib x sex interaction within KRAS G12C+ — does the benefit only appear in males?
# ===========================================================
df = DF[DF["kras_g12c"]==1].copy()
df["t"] = df["treatment_sotorasib"].astype(float)
df["m"] = df["sex_female"].astype(float)
df["i"] = df["t"]*df["m"]
X = sm.add_constant(df[["t","m","i"]].values)
res = sm.OLS(df["pfs_months"].values, X).fit()
add("sot_kras_sex_interact", main_t=float(res.params[1]),
    main_mod=float(res.params[2]), interact=float(res.params[3]),
    p_int=float(res.pvalues[3]), p_main_t=float(res.pvalues[1]))

# Triple subgroup KRAS+ x sex x ALK to see if interaction is real
sub_kras_male = (DF["kras_g12c"]==1) & (DF["sex_female"]==0)
sub_kras_female = (DF["kras_g12c"]==1) & (DF["sex_female"]==1)

# ===========================================================
# Pembro x weight_loss — significant interaction earlier
# ===========================================================
# Stratify weight_loss into tertiles
wl = DF["weight_loss_pct_6mo"]
wl_q = pd.qcut(wl, 3, labels=["low","mid","high"])
for tertile in ["low","mid","high"]:
    add(f"sub_pembro_wl_{tertile}", **sub_effect("treatment_pembrolizumab", wl_q==tertile, f"WL_{tertile}"))

# ===========================================================
# Pembro x stage IV
# ===========================================================
add("sub_pembro_stage4", **sub_effect("treatment_pembrolizumab", DF["stage_iv"]==1, "stage_iv=1"))
add("sub_pembro_nonstage4", **sub_effect("treatment_pembrolizumab", DF["stage_iv"]==0, "stage_iv=0"))

# ===========================================================
# Same for osi: subgroup discovery in EGFR+ for any modifier
# ===========================================================
egfr = DF["egfr_mutation"]==1
add("sub_osi_egfr_male", **sub_effect("treatment_osimertinib", egfr & (DF["sex_female"]==0), "EGFR+/male"))
add("sub_osi_egfr_female", **sub_effect("treatment_osimertinib", egfr & (DF["sex_female"]==1), "EGFR+/female"))
for s in ["current","former","never"]:
    add(f"sub_osi_egfr_{s}", **sub_effect("treatment_osimertinib", egfr & (DF["smoking_status"]==s), f"EGFR+/{s}"))
add("sub_osi_egfr_adeno", **sub_effect("treatment_osimertinib", egfr & (DF["histology"]=="adenocarcinoma"), "EGFR+/adeno"))
for e in [0,1,2]:
    add(f"sub_osi_egfr_ecog{e}", **sub_effect("treatment_osimertinib", egfr & (DF["ecog_ps"]==e), f"EGFR+/ECOG{e}"))

# Look at osi effect within EGFR+ AND various co-features
add("sub_osi_egfr_brain", **sub_effect("treatment_osimertinib", egfr & (DF["has_brain_mets"]==1), "EGFR+/brain"))
add("sub_osi_egfr_nobrain", **sub_effect("treatment_osimertinib", egfr & (DF["has_brain_mets"]==0), "EGFR+/no_brain"))

# ===========================================================
# Olap x BRCA2: do subgroups show effect?
# ===========================================================
brca = DF["brca2_mutation"]==1
add("sub_olap_brca2_male", **sub_effect("treatment_olaparib", brca & (DF["sex_female"]==0), "BRCA2+/male"))
add("sub_olap_brca2_female", **sub_effect("treatment_olaparib", brca & (DF["sex_female"]==1), "BRCA2+/female"))
for e in [0,1,2]:
    add(f"sub_olap_brca2_ecog{e}", **sub_effect("treatment_olaparib", brca & (DF["ecog_ps"]==e), f"BRCA2+/ECOG{e}"))
add("sub_olap_brca2_adeno", **sub_effect("treatment_olaparib", brca & (DF["histology"]=="adenocarcinoma"), "BRCA2+/adeno"))

# ===========================================================
# Pembro x PDL1 high in TMB-H AND non-squamous AND non-current smoker
# ===========================================================
sub = (DF["pdl1_tps"]>=0.5) & (DF["tmb_high"]==1)
add("sub_pembro_pdl1hi_tmbhi", **sub_effect("treatment_pembrolizumab", sub, "PDL1>=50/TMB-H"))
sub = (DF["pdl1_tps"]>=0.5) & (DF["tmb_high"]==1) & (DF["histology"]=="adenocarcinoma")
add("sub_pembro_pdl1hi_tmbhi_adeno", **sub_effect("treatment_pembrolizumab", sub, "PDL1>=50/TMB-H/adeno"))

# Pembro effect in patients without the targetable drivers (EGFR-, ALK-, KRAS-)
no_driver = (DF["egfr_mutation"]==0)&(DF["alk_fusion"]==0)&(DF["kras_g12c"]==0)
add("sub_pembro_no_driver", **sub_effect("treatment_pembrolizumab", no_driver, "no_driver"))
add("sub_pembro_no_driver_pdl1hi", **sub_effect("treatment_pembrolizumab", no_driver & (DF["pdl1_tps"]>=0.5), "no_driver/PDL1>=50%"))

# ===========================================================
# Treatment-effect heterogeneity: pembro within multivariable model
# ===========================================================
# Add interaction of pembro with multiple modifiers in one model
df = DF.copy()
df["squamous"] = (df["histology"]=="squamous").astype(int)
mods = ["pdl1_tps","tmb_high","stk11_mutation","egfr_mutation","alk_fusion","squamous","ecog_ps","stage_iv"]
df["t"] = df["treatment_pembrolizumab"]
ints = []
for m in mods:
    df[f"i_{m}"] = df["t"]*df[m].astype(float)
    ints.append(f"i_{m}")
X = df[["t"]+mods+ints].astype(float).values
X = sm.add_constant(X)
res = sm.OLS(df["pfs_months"].values, X).fit()
res_dict = {}
names = ["const","t"]+mods+ints
for i,n in enumerate(names):
    res_dict[n] = {"coef": float(res.params[i]), "p": float(res.pvalues[i])}
add("pembro_joint_interactions", coefs=res_dict)

# ===========================================================
# Save
# ===========================================================
with open("my_run_results.json","w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("Wrote", len(RESULTS), "keys")
