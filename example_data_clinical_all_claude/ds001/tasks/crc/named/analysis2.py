"""Deeper subgroup discovery, particularly for regorafenib."""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
results = {}

features = ("age_years + sex_female + C(ecog_ps) + stage_iv + right_sided_primary + "
            "kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + "
            "ntrk_fusion + cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + "
            "crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + "
            "alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
            "sodium_meq_l + potassium_meq_l + calcium_mg_dl")

# =========================================================
# 1. Regorafenib stratified by KRAS
# =========================================================
print("=== Regorafenib stratified by KRAS ===")
for kras_val in [0,1]:
    sub = df[df.kras_mutation==kras_val]
    m = smf.ols(f"pfs_months ~ {features.replace('+ kras_mutation ','')} + treatment_regorafenib", data=sub).fit()
    coef = m.params["treatment_regorafenib"]
    p = m.pvalues["treatment_regorafenib"]
    print(f"  KRAS={kras_val} (n={len(sub)}, treated n={sub.treatment_regorafenib.sum()}): "
          f"regorafenib coef={coef:+.4f}, p={p:.3g}")
    results[f"rego_in_kras_{kras_val}"] = {"coef": float(coef), "p": float(p),
                                            "n": int(len(sub)),
                                            "n_treated": int(sub.treatment_regorafenib.sum())}

# Unadjusted means
print("\n  Unadjusted regorafenib effect by KRAS:")
for kras_val in [0,1]:
    sub = df[df.kras_mutation==kras_val]
    a = sub.loc[sub.treatment_regorafenib==1, "pfs_months"]
    b = sub.loc[sub.treatment_regorafenib==0, "pfs_months"]
    t,p = stats.ttest_ind(a,b,equal_var=False)
    print(f"  KRAS={kras_val}: regorafenib mean={a.mean():.3f} vs no-rego mean={b.mean():.3f}, "
          f"diff={a.mean()-b.mean():+.3f}, p={p:.3g}, n_t={len(a)}")

# =========================================================
# 2. Regorafenib × every other variable
# =========================================================
print("\n=== Regorafenib interaction screen ===")
candidate_modifiers = ["sex_female","stage_iv","right_sided_primary","kras_mutation",
                       "nras_mutation","braf_v600e","msi_high","her2_amplified",
                       "ntrk_fusion"]
inter_results = {}
for bm in candidate_modifiers:
    formula = f"pfs_months ~ {features} + treatment_regorafenib * {bm}"
    m = smf.ols(formula, data=df).fit()
    inter = f"treatment_regorafenib:{bm}"
    inter_results[bm] = {
        "main_rego": float(m.params["treatment_regorafenib"]),
        "main_rego_p": float(m.pvalues["treatment_regorafenib"]),
        "interaction": float(m.params[inter]),
        "interaction_p": float(m.pvalues[inter]),
    }
    print(f"  rego × {bm}: interaction={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
results["rego_interactions"] = inter_results

# Continuous modifiers — split at median
print("\n=== Regorafenib × continuous (median split) ===")
cont_modifiers = ["age_years","cea_ng_ml","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo",
                  "crp_mg_l","nlr","hemoglobin_g_dl","calcium_mg_dl",
                  "alkaline_phosphatase_u_l","ast_u_l","alt_u_l","total_bilirubin_mg_dl",
                  "creatinine_mg_dl","bun_mg_dl","sodium_meq_l","potassium_meq_l"]
for c in cont_modifiers:
    df["_hi"] = (df[c] > df[c].median()).astype(int)
    formula = f"pfs_months ~ {features} + treatment_regorafenib * _hi"
    m = smf.ols(formula, data=df).fit()
    inter = "treatment_regorafenib:_hi"
    print(f"  rego × ({c} > median): interaction={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
    inter_results[f"_hi_{c}"] = {"interaction": float(m.params[inter]),
                                  "interaction_p": float(m.pvalues[inter])}
df.drop(columns=["_hi"], inplace=True, errors="ignore")

# ECOG categorical
print("\n=== Regorafenib × ECOG ===")
formula = f"pfs_months ~ {features} + treatment_regorafenib * C(ecog_ps)"
m = smf.ols(formula, data=df).fit()
for k in ["treatment_regorafenib:C(ecog_ps)[T.1]","treatment_regorafenib:C(ecog_ps)[T.2]"]:
    print(f"  {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")
    results[k] = {"coef": float(m.params[k]), "p": float(m.pvalues[k])}

# =========================================================
# 3. KRAS-WT regorafenib subgroup search:
#    Among KRAS WT (where rego works), what other variables modify it?
# =========================================================
print("\n=== Within KRAS-WT, regorafenib × other modifiers ===")
df_wt = df[df.kras_mutation==0].copy()
print(f"  KRAS-WT n={len(df_wt)}, n treated={df_wt.treatment_regorafenib.sum()}")

# Mean PFS by treatment status in KRAS WT
a = df_wt.loc[df_wt.treatment_regorafenib==1,"pfs_months"]
b = df_wt.loc[df_wt.treatment_regorafenib==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  KRAS WT: rego={a.mean():.3f} vs no-rego={b.mean():.3f}, diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["rego_in_kras_wt_unadj"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                                    "n_treated": int(len(a)), "n_untreated": int(len(b))}

feats_no_kras = features.replace(" + kras_mutation","")
for bm in ["sex_female","stage_iv","right_sided_primary","nras_mutation","braf_v600e",
           "msi_high","her2_amplified","ntrk_fusion"]:
    formula = f"pfs_months ~ {feats_no_kras} + treatment_regorafenib * {bm}"
    try:
        m = smf.ols(formula, data=df_wt).fit()
        inter = f"treatment_regorafenib:{bm}"
        print(f"  KRAS WT, rego × {bm}: interaction={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
    except Exception as e:
        print(f"  KRAS WT, rego × {bm}: failed ({e})")

# Try age, ECOG splits
for c in ["age_years","albumin_g_dl","ldh_u_l","cea_ng_ml","weight_loss_pct_6mo","nlr"]:
    df_wt["_hi"] = (df_wt[c] > df_wt[c].median()).astype(int)
    formula = f"pfs_months ~ {feats_no_kras} + treatment_regorafenib * _hi"
    m = smf.ols(formula, data=df_wt).fit()
    inter = "treatment_regorafenib:_hi"
    print(f"  KRAS WT, rego × ({c} > median): interaction={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
df_wt.drop(columns=["_hi"], inplace=True, errors="ignore")

# ECOG within KRAS WT
formula = f"pfs_months ~ {feats_no_kras} + treatment_regorafenib * C(ecog_ps)"
m = smf.ols(formula, data=df_wt).fit()
for k in ["treatment_regorafenib:C(ecog_ps)[T.1]","treatment_regorafenib:C(ecog_ps)[T.2]"]:
    print(f"  KRAS WT, {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")

# =========================================================
# 4. Other treatment × biomarker interactions screened broadly
# =========================================================
print("\n=== Other treatments × every biomarker (interaction screen) ===")
all_interaction_results = {}
for tx in ["treatment_cetuximab","treatment_bevacizumab","treatment_pembrolizumab",
          "treatment_encorafenib","treatment_trastuzumab_tucatinib"]:
    print(f"\n  -- {tx} --")
    for bm in candidate_modifiers + ["age_years","albumin_g_dl","ldh_u_l","cea_ng_ml",
                                      "weight_loss_pct_6mo","nlr"]:
        if bm in cont_modifiers:
            df["_hi"] = (df[bm] > df[bm].median()).astype(int)
            formula = f"pfs_months ~ {features} + {tx} * _hi"
            inter = f"{tx}:_hi"
        else:
            formula = f"pfs_months ~ {features} + {tx} * {bm}"
            inter = f"{tx}:{bm}"
        try:
            m = smf.ols(formula, data=df).fit()
            res = {"coef": float(m.params[inter]), "p": float(m.pvalues[inter])}
            all_interaction_results[f"{tx}_x_{bm}"] = res
            if res["p"] < 0.05:
                print(f"    {tx} × {bm}: coef={res['coef']:+.4f}, p={res['p']:.3g}  *")
            df.drop(columns=["_hi"], inplace=True, errors="ignore")
        except Exception as e:
            print(f"    {tx} × {bm}: failed ({e})")

results["all_interactions"] = all_interaction_results

# ECOG-by-treatment
print("\n=== Treatment × ECOG ===")
for tx in ["treatment_cetuximab","treatment_bevacizumab","treatment_pembrolizumab",
          "treatment_encorafenib","treatment_trastuzumab_tucatinib"]:
    formula = f"pfs_months ~ {features} + {tx} * C(ecog_ps)"
    m = smf.ols(formula, data=df).fit()
    for k in [f"{tx}:C(ecog_ps)[T.1]", f"{tx}:C(ecog_ps)[T.2]"]:
        if abs(m.pvalues[k]) < 0.05:
            print(f"  {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}  *")

# =========================================================
# 5. Three-way: regorafenib × KRAS × ECOG (look for full subgroup)
# =========================================================
print("\n=== Regorafenib × KRAS by ECOG strata ===")
for ecog_val in [0,1,2]:
    sub = df[df.ecog_ps==ecog_val]
    formula = f"pfs_months ~ {features.replace('+ C(ecog_ps) ','').replace('+ kras_mutation ','')} + treatment_regorafenib * kras_mutation"
    try:
        m = smf.ols(formula, data=sub).fit()
        inter = "treatment_regorafenib:kras_mutation"
        print(f"  ECOG={ecog_val} (n={len(sub)}): rego×kras inter={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
        # rego coef in KRAS WT within this ECOG
        sub_wt = sub[sub.kras_mutation==0]
        if len(sub_wt) > 50:
            mw = smf.ols(f"pfs_months ~ {features.replace('+ C(ecog_ps) ','').replace('+ kras_mutation ','')} + treatment_regorafenib", data=sub_wt).fit()
            print(f"    KRAS WT within ECOG={ecog_val}: rego coef={mw.params['treatment_regorafenib']:+.4f}, "
                  f"p={mw.pvalues['treatment_regorafenib']:.3g}, n_treated={int(sub_wt.treatment_regorafenib.sum())}")
    except Exception as e:
        print(f"  ECOG={ecog_val}: failed ({e})")

# =========================================================
# Save
# =========================================================
with open("analysis2_results.json","w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nDone (analysis2)")
