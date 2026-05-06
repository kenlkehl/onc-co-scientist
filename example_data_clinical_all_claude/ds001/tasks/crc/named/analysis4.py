"""Final verification & sensitivity analyses."""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

df = pd.read_parquet("dataset.parquet")
results = {}

# Define the candidate subgroup
df["_full_profile"] = ((df.kras_mutation==0) & (df.braf_v600e==0) &
                        (df.right_sided_primary==0) &
                        (df.cea_ng_ml < df.cea_ng_ml.median())).astype(int)

# 1) Within the subgroup, is the regorafenib effect modified by anything else?
print("=== Within EGFR-like + low CEA subgroup, does anything else modify the rego effect? ===")
sub = df[df._full_profile==1].copy()
print(f"Subgroup n={len(sub)}, treated n={sub.treatment_regorafenib.sum()}")

feats = ("age_years + sex_female + C(ecog_ps) + stage_iv + nras_mutation + msi_high + "
         "her2_amplified + ntrk_fusion + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + "
         "crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + "
         "alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
         "sodium_meq_l + potassium_meq_l + calcium_mg_dl + cea_ng_ml")

# Test interactions within subgroup
candidates = ["sex_female","stage_iv","nras_mutation","msi_high","her2_amplified",
              "ntrk_fusion"]
for bm in candidates:
    formula = f"pfs_months ~ {feats} + treatment_regorafenib * {bm}"
    try:
        m = smf.ols(formula, data=sub).fit()
        inter = f"treatment_regorafenib:{bm}"
        print(f"  rego × {bm}: inter={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
    except Exception as e:
        print(f"  rego × {bm}: failed ({e})")

# Continuous
for c in ["age_years","albumin_g_dl","ldh_u_l","weight_loss_pct_6mo","crp_mg_l","nlr"]:
    sub["_hi"] = (sub[c] > sub[c].median()).astype(int)
    formula = f"pfs_months ~ {feats} + treatment_regorafenib * _hi"
    m = smf.ols(formula, data=sub).fit()
    inter = "treatment_regorafenib:_hi"
    print(f"  rego × ({c} > median): inter={m.params[inter]:+.4f}, p={m.pvalues[inter]:.3g}")
sub.drop(columns=["_hi"], inplace=True, errors="ignore")

# ECOG within subgroup
formula = f"pfs_months ~ {feats} + treatment_regorafenib * C(ecog_ps)"
m = smf.ols(formula, data=sub).fit()
for k in ["treatment_regorafenib:C(ecog_ps)[T.1]","treatment_regorafenib:C(ecog_ps)[T.2]"]:
    print(f"  {k}: coef={m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")

# 2) Sensitivity: is the CEA cutoff at median right?
print("\n=== Search for optimal CEA cutoff in EGFR-like subset ===")
egfr_like = df[(df.kras_mutation==0)&(df.braf_v600e==0)&(df.right_sided_primary==0)].copy()
print(f"EGFR-like n={len(egfr_like)}, rego treated={egfr_like.treatment_regorafenib.sum()}")
for cut in [1, 2, 3, 4, 4.5, 5, 7, 10, 15, 20]:
    egfr_like["_below"] = (egfr_like.cea_ng_ml < cut).astype(int)
    sub_low = egfr_like[egfr_like._below==1]
    sub_hi = egfr_like[egfr_like._below==0]
    if len(sub_low)>50 and sub_low.treatment_regorafenib.sum()>10:
        a = sub_low.loc[sub_low.treatment_regorafenib==1,"pfs_months"]
        b = sub_low.loc[sub_low.treatment_regorafenib==0,"pfs_months"]
        diff_low = a.mean()-b.mean()
    else:
        diff_low = None
    if len(sub_hi)>50 and sub_hi.treatment_regorafenib.sum()>10:
        a = sub_hi.loc[sub_hi.treatment_regorafenib==1,"pfs_months"]
        b = sub_hi.loc[sub_hi.treatment_regorafenib==0,"pfs_months"]
        diff_hi = a.mean()-b.mean()
    else:
        diff_hi = None
    dl = f"{diff_low:+.3f}" if diff_low is not None else "NA"
    dh = f"{diff_hi:+.3f}" if diff_hi is not None else "NA"
    print(f"  CEA<{cut}: rego diff={dl} (n_below={len(sub_low)}); CEA>={cut}: rego diff={dh} (n_above={len(sub_hi)})")

# 3) Verify direction of all main effects in clean adjusted model
print("\n=== Final adjusted main-effects model (no treatment) ===")
all_features = ("age_years + sex_female + C(ecog_ps) + stage_iv + right_sided_primary + "
                "kras_mutation + nras_mutation + braf_v600e + msi_high + her2_amplified + "
                "ntrk_fusion + cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + "
                "crp_mg_l + nlr + hemoglobin_g_dl + alkaline_phosphatase_u_l + ast_u_l + "
                "alt_u_l + total_bilirubin_mg_dl + creatinine_mg_dl + bun_mg_dl + "
                "sodium_meq_l + potassium_meq_l + calcium_mg_dl")
m = smf.ols(f"pfs_months ~ {all_features}", data=df).fit()
print(f"R^2 = {m.rsquared:.4f}")
significant = []
for k in m.params.index:
    if k=="Intercept": continue
    if m.pvalues[k] < 0.001:
        significant.append((k, float(m.params[k]), float(m.pvalues[k])))
significant.sort(key=lambda x: x[2])
for k, c, p in significant:
    print(f"  {k}: {c:+.4f} (p={p:.3g})")

# 4) Verify the NRAS-mut peculiar finding doesn't break the subgroup
print("\n=== Final subgroup check ===")
profile = ((df.kras_mutation==0) & (df.braf_v600e==0) & (df.right_sided_primary==0)
           & (df.cea_ng_ml < df.cea_ng_ml.median()))
print(f"In profile: n={profile.sum()}")
a = df.loc[profile & (df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[profile & (df.treatment_regorafenib==0),"pfs_months"]
print(f"  rego mean={a.mean():.3f}, no-rego mean={b.mean():.3f}, diff={a.mean()-b.mean():+.3f}")
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  t-test p={p:.3g}")

# Out of profile
a = df.loc[~profile & (df.treatment_regorafenib==1),"pfs_months"]
b = df.loc[~profile & (df.treatment_regorafenib==0),"pfs_months"]
print(f"  Out of profile: rego={a.mean():.3f}, no-rego={b.mean():.3f}, diff={a.mean()-b.mean():+.3f}")
t,p = stats.ttest_ind(a,b,equal_var=False)
print(f"  t-test p={p:.3g}")

# Adjusted in subgroup
m = smf.ols(f"pfs_months ~ {feats} + treatment_regorafenib", data=df[profile]).fit()
print(f"\n  Adjusted regorafenib effect within profile: {m.params['treatment_regorafenib']:+.4f}, p={m.pvalues['treatment_regorafenib']:.3g}")

# 5) Look at age effect direction (it was positive — odd)
print("\n=== Age × treatment ===")
m = smf.ols(f"pfs_months ~ {all_features} + treatment_regorafenib*age_years", data=df).fit()
for k in ["age_years","treatment_regorafenib","treatment_regorafenib:age_years"]:
    print(f"  {k}: {m.params[k]:+.4f}, p={m.pvalues[k]:.3g}")

print("\nDone (analysis4)")
