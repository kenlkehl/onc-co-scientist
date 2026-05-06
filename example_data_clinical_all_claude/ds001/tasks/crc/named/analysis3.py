"""Final subgroup verification + joint multivariable models."""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
results = {}

features = ("age_years + sex_female + C(ecog_ps) + stage_iv + "
            "nras_mutation + msi_high + her2_amplified + "
            "ntrk_fusion + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + "
            "crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + "
            "alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
            "sodium_meq_l + potassium_meq_l + calcium_mg_dl")

# 1) Joint subgroup: KRAS WT AND BRAF WT AND left-sided
print("=== Step 1: KRAS WT × BRAF WT × left-sided (3-way) ===")
df["_egfr_like"] = ((df.kras_mutation==0) & (df.braf_v600e==0) & (df.right_sided_primary==0)).astype(int)
print(f"  EGFR-like profile (KRAS WT & BRAF WT & left-sided): n={df._egfr_like.sum()} ({df._egfr_like.mean()*100:.1f}%)")
a = df.loc[(df._egfr_like==1)&(df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[(df._egfr_like==1)&(df.treatment_regorafenib==0),"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  In EGFR-like profile: rego mean={a.mean():.3f} (n={len(a)}) vs no-rego mean={b.mean():.3f} (n={len(b)}), "
      f"diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["egfr_like_rego"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                              "n_treated": int(len(a)), "n_untreated": int(len(b))}

a = df.loc[(df._egfr_like==0)&(df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[(df._egfr_like==0)&(df.treatment_regorafenib==0),"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Outside EGFR-like profile: rego mean={a.mean():.3f} (n={len(a)}) vs no-rego mean={b.mean():.3f} (n={len(b)}), "
      f"diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["non_egfr_like_rego"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                                  "n_treated": int(len(a)), "n_untreated": int(len(b))}

# Adjusted three-way interaction model
formula = (f"pfs_months ~ {features} + treatment_regorafenib + kras_mutation + braf_v600e + right_sided_primary + "
           "treatment_regorafenib:kras_mutation + treatment_regorafenib:braf_v600e + "
           "treatment_regorafenib:right_sided_primary")
m = smf.ols(formula, data=df).fit()
print("\n  Adjusted 3-way interaction:")
for k in ["treatment_regorafenib","treatment_regorafenib:kras_mutation",
          "treatment_regorafenib:braf_v600e","treatment_regorafenib:right_sided_primary"]:
    print(f"    {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")
    results[f"adj_{k}"] = {"coef": float(m.params[k]), "p": float(m.pvalues[k])}

# 2) Add CEA as a continuous interaction
print("\n=== Step 2: also adding CEA interaction ===")
df["_cea_low"] = (df.cea_ng_ml < df.cea_ng_ml.median()).astype(int)
df["_cea_hi"] = (df.cea_ng_ml >= df.cea_ng_ml.median()).astype(int)
df["_full_egfr"] = ((df.kras_mutation==0) & (df.braf_v600e==0) &
                    (df.right_sided_primary==0) & (df._cea_low==1)).astype(int)
print(f"  Full EGFR-like + low CEA: n={df._full_egfr.sum()} ({df._full_egfr.mean()*100:.1f}%)")

a = df.loc[(df._full_egfr==1)&(df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[(df._full_egfr==1)&(df.treatment_regorafenib==0),"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  In full profile: rego mean={a.mean():.3f} (n={len(a)}) vs no-rego mean={b.mean():.3f} (n={len(b)}), "
      f"diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["full_profile_rego"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                                  "n_treated": int(len(a)), "n_untreated": int(len(b))}

a = df.loc[(df._full_egfr==0)&(df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[(df._full_egfr==0)&(df.treatment_regorafenib==0),"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Outside full profile: rego mean={a.mean():.3f} (n={len(a)}) vs no-rego mean={b.mean():.3f} (n={len(b)}), "
      f"diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["non_full_profile_rego"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                                     "n_treated": int(len(a)), "n_untreated": int(len(b))}

# 3) CEA interaction — is it actually low CEA or specifically below median?
# Try splitting CEA into quartiles
print("\n=== Step 3: regorafenib effect by CEA quartile (within KRAS WT, BRAF WT, left-sided) ===")
sub = df[(df.kras_mutation==0)&(df.braf_v600e==0)&(df.right_sided_primary==0)].copy()
sub["_cea_q"] = pd.qcut(sub.cea_ng_ml, q=4, labels=False)
for q in range(4):
    s = sub[sub._cea_q == q]
    a = s.loc[s.treatment_regorafenib==1,"pfs_months"]
    b = s.loc[s.treatment_regorafenib==0,"pfs_months"]
    if len(a)>10 and len(b)>10:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        print(f"  CEA Q{q+1} (range {sub.cea_ng_ml.quantile(q*0.25):.2f}-{sub.cea_ng_ml.quantile((q+1)*0.25):.2f}): "
              f"rego={a.mean():.3f} (n={len(a)}), no-rego={b.mean():.3f} (n={len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

# 4) NRAS-mut subgroup — verify our finding (interaction was +0.88)
print("\n=== Step 4: NRAS interaction with rego (in KRAS WT) ===")
print("  Note: KRAS and NRAS mutations are typically mutually exclusive — verify.")
xt = pd.crosstab(df.kras_mutation, df.nras_mutation)
print(xt)
# Among KRAS WT NRAS WT
sub = df[(df.kras_mutation==0)&(df.nras_mutation==0)]
a = sub.loc[sub.treatment_regorafenib==1,"pfs_months"]
b = sub.loc[sub.treatment_regorafenib==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  KRAS WT & NRAS WT: rego={a.mean():.3f} (n={len(a)}), no-rego={b.mean():.3f} (n={len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
sub = df[(df.kras_mutation==0)&(df.nras_mutation==1)]
if len(sub)>20:
    a = sub.loc[sub.treatment_regorafenib==1,"pfs_months"]
    b = sub.loc[sub.treatment_regorafenib==0,"pfs_months"]
    if len(a)>5 and len(b)>5:
        t,p = stats.ttest_ind(a,b,equal_var=False)
        print(f"  KRAS WT & NRAS MUT (n={len(sub)}): rego={a.mean():.3f} (n={len(a)}), no-rego={b.mean():.3f} (n={len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

# 5) Final clean joint model — does subgroup definition fully capture the effect?
print("\n=== Step 5: full joint interaction model (KRAS, BRAF, sidedness, CEA hi) ===")
df["_cea_hi"] = (df.cea_ng_ml > df.cea_ng_ml.median()).astype(int)
formula = (f"pfs_months ~ {features} + kras_mutation + braf_v600e + right_sided_primary + cea_ng_ml + "
           "treatment_regorafenib + "
           "treatment_regorafenib:kras_mutation + "
           "treatment_regorafenib:braf_v600e + "
           "treatment_regorafenib:right_sided_primary + "
           "treatment_regorafenib:_cea_hi")
m = smf.ols(formula, data=df).fit()
for k in ["treatment_regorafenib","treatment_regorafenib:kras_mutation",
          "treatment_regorafenib:braf_v600e","treatment_regorafenib:right_sided_primary",
          "treatment_regorafenib:_cea_hi"]:
    print(f"  {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")
    results[f"final_{k}"] = {"coef": float(m.params[k]), "p": float(m.pvalues[k])}

# 6) Sanity: does cetuximab actually do nothing? Check in EGFR-like subgroup
print("\n=== Step 6: Cetuximab / pembrolizumab / encorafenib / trast-tuc in 'right' subgroup ===")
# Cetuximab in EGFR-like (KRAS WT, NRAS WT, BRAF WT, left-sided)
sub = df[(df.kras_mutation==0)&(df.nras_mutation==0)&(df.braf_v600e==0)&(df.right_sided_primary==0)]
a = sub.loc[sub.treatment_cetuximab==1,"pfs_months"]
b = sub.loc[sub.treatment_cetuximab==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Cetuximab in EGFR-like (n={len(sub)}): cet={a.mean():.3f} ({len(a)}) vs no-cet={b.mean():.3f} ({len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["cet_in_egfr_like"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                                "n_treated": int(len(a)), "n_untreated": int(len(b))}

# Pembrolizumab in MSI-high
sub = df[df.msi_high==1]
a = sub.loc[sub.treatment_pembrolizumab==1,"pfs_months"]
b = sub.loc[sub.treatment_pembrolizumab==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Pembrolizumab in MSI-high (n={len(sub)}): pembro={a.mean():.3f} ({len(a)}) vs no-pembro={b.mean():.3f} ({len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["pembro_in_msi"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                             "n_treated": int(len(a)), "n_untreated": int(len(b))}

# Encorafenib in BRAF V600E
sub = df[df.braf_v600e==1]
a = sub.loc[sub.treatment_encorafenib==1,"pfs_months"]
b = sub.loc[sub.treatment_encorafenib==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Encorafenib in BRAF V600E (n={len(sub)}): enco={a.mean():.3f} ({len(a)}) vs no-enco={b.mean():.3f} ({len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["enco_in_braf"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                            "n_treated": int(len(a)), "n_untreated": int(len(b))}

# Trastuzumab/tucatinib in HER2 amp
sub = df[df.her2_amplified==1]
a = sub.loc[sub.treatment_trastuzumab_tucatinib==1,"pfs_months"]
b = sub.loc[sub.treatment_trastuzumab_tucatinib==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Trast-tuc in HER2 amp (n={len(sub)}): tt={a.mean():.3f} ({len(a)}) vs no-tt={b.mean():.3f} ({len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")
results["tt_in_her2"] = {"diff": float(a.mean()-b.mean()), "p": float(p),
                          "n_treated": int(len(a)), "n_untreated": int(len(b))}

# Bevacizumab — overall sanity
sub = df
a = sub.loc[sub.treatment_bevacizumab==1,"pfs_months"]
b = sub.loc[sub.treatment_bevacizumab==0,"pfs_months"]
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  Bevacizumab overall: bev={a.mean():.3f} ({len(a)}) vs no-bev={b.mean():.3f} ({len(b)}), diff={a.mean()-b.mean():+.3f}, p={p:.3g}")

# 7) Decision tree on rego treatment effect
print("\n=== Step 7: Decision tree heterogeneity (T-learner-ish) ===")
from sklearn.tree import DecisionTreeRegressor
predictors = ["age_years","sex_female","ecog_ps","stage_iv","right_sided_primary",
              "kras_mutation","nras_mutation","braf_v600e","msi_high","her2_amplified",
              "ntrk_fusion","cea_ng_ml","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo",
              "crp_mg_l","nlr","hemoglobin_g_dl"]
X = df[predictors].values
y_treated = df[df.treatment_regorafenib==1].copy()
y_untreated = df[df.treatment_regorafenib==0].copy()

mt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(y_treated[predictors], y_treated["pfs_months"])
mc = DecisionTreeRegressor(max_depth=6, min_samples_leaf=200).fit(y_untreated[predictors], y_untreated["pfs_months"])
df["_pred_treat"] = mt.predict(df[predictors])
df["_pred_ctrl"] = mc.predict(df[predictors])
df["_cate"] = df["_pred_treat"] - df["_pred_ctrl"]
print(f"  Estimated CATE distribution: mean={df._cate.mean():.3f}, sd={df._cate.std():.3f}")
print(f"  CATE quartiles: {df._cate.quantile([0,0.25,0.5,0.75,1.0]).tolist()}")
df["_cate_q"] = pd.qcut(df._cate, q=4, labels=False)

# Within each CATE quartile, examine subgroup composition
print("\n  Top CATE quartile (q3) subgroup composition:")
top = df[df._cate_q==3]
bot = df[df._cate_q==0]
for col in ["kras_mutation","braf_v600e","right_sided_primary","msi_high","ecog_ps"]:
    print(f"    {col}: top mean={top[col].mean():.3f}, bottom mean={bot[col].mean():.3f}")
print(f"    cea_ng_ml: top median={top.cea_ng_ml.median():.2f}, bottom median={bot.cea_ng_ml.median():.2f}")

# Save
with open("analysis3_results.json","w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nDone (analysis3)")
