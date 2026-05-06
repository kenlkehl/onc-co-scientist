"""Examine the CEA modifier of regorafenib effect within favorable subgroup."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_parquet("dataset.parquet")
df["_favorable"] = ((df.kras_mutation == 0) & (df.right_sided_primary == 0) & (df.braf_v600e == 0)).astype(int)
fav = df[df._favorable == 1].copy()
print(f"Favorable n = {len(fav)}; rego rate = {fav.treatment_regorafenib.mean():.3f}")
print(f"CEA distribution (favorable): min={fav.cea_ng_ml.min():.2f} median={fav.cea_ng_ml.median():.2f} "
      f"95th={fav.cea_ng_ml.quantile(0.95):.2f} 99th={fav.cea_ng_ml.quantile(0.99):.2f} max={fav.cea_ng_ml.max():.2f}")

# rego effect by CEA quartile within favorable
fav["_cea_q"] = pd.qcut(fav.cea_ng_ml, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\nRego effect by CEA quartile (favorable):")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    s = fav[fav._cea_q == q]
    a = s.loc[s.treatment_regorafenib == 1, "pfs_months"]
    b = s.loc[s.treatment_regorafenib == 0, "pfs_months"]
    cea_lo, cea_hi = s.cea_ng_ml.min(), s.cea_ng_ml.max()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  CEA {q} ({cea_lo:.2f}-{cea_hi:.2f}): n+={len(a):4d} n-={len(b):4d} "
          f"diff={a.mean()-b.mean():+.3f}  p={p:.3g}")

# Try thresholds — at what CEA does the benefit disappear?
print("\nRego effect by CEA threshold (favorable, treated as binary):")
for thr in [5, 10, 20, 30, 50]:
    lo = fav[fav.cea_ng_ml <= thr]
    hi = fav[fav.cea_ng_ml > thr]
    a = lo.loc[lo.treatment_regorafenib == 1, "pfs_months"]
    b = lo.loc[lo.treatment_regorafenib == 0, "pfs_months"]
    if len(a) > 3 and len(b) > 3:
        d_lo = a.mean() - b.mean()
        _, p_lo = stats.ttest_ind(a, b, equal_var=False)
    else:
        d_lo, p_lo = float("nan"), float("nan")
    a = hi.loc[hi.treatment_regorafenib == 1, "pfs_months"]
    b = hi.loc[hi.treatment_regorafenib == 0, "pfs_months"]
    if len(a) > 3 and len(b) > 3:
        d_hi = a.mean() - b.mean()
        _, p_hi = stats.ttest_ind(a, b, equal_var=False)
    else:
        d_hi, p_hi = float("nan"), float("nan")
    print(f"  CEA<={thr}: n_lo={len(lo):5d} diff_lo={d_lo:+.3f} p_lo={p_lo:.3g}  | "
          f"CEA>{thr}: n_hi={len(hi):5d} diff_hi={d_hi:+.3f} p_hi={p_hi:.3g}")

# Confirm CEA does not modify rego effect OUTSIDE favorable
print("\nRego x CEA interaction OUTSIDE favorable (should be null/uniform):")
unfav = df[df._favorable == 0]
m = smf.ols("pfs_months ~ treatment_regorafenib * cea_ng_ml", data=unfav).fit()
print(f"  rego x cea coef = {m.params.get('treatment_regorafenib:cea_ng_ml'):+.5f}, "
      f"p = {m.pvalues.get('treatment_regorafenib:cea_ng_ml'):.3g}")

# Check the favorable & low-CEA combined effect
print("\nFinal subgroup: KRAS-wt & left-sided & BRAF-wt & low CEA")
for thr in [5, 10, 20]:
    s = df[(df._favorable == 1) & (df.cea_ng_ml <= thr)]
    a = s.loc[s.treatment_regorafenib == 1, "pfs_months"]
    b = s.loc[s.treatment_regorafenib == 0, "pfs_months"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  CEA<={thr}: n+={len(a):4d} n-={len(b):4d} diff={a.mean()-b.mean():+.3f} p={p:.3g}")

# Also check NRAS — separate analysis
print("\nNRAS interaction with rego separately:")
print("Within favorable subgroup, by NRAS status:")
for nras in [0, 1]:
    s = df[(df._favorable == 1) & (df.nras_mutation == nras)]
    a = s.loc[s.treatment_regorafenib == 1, "pfs_months"]
    b = s.loc[s.treatment_regorafenib == 0, "pfs_months"]
    if len(a) > 3 and len(b) > 3:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"  NRAS={nras}: n+={len(a):4d} n-={len(b):4d} diff={a.mean()-b.mean():+.3f} p={p:.3g}")

# Outside favorable, is NRAS-mut still a special case?
print("\nNRAS-mut outside favorable (rego effect):")
for kras in [0, 1]:
    for rs in [0, 1]:
        for braf in [0, 1]:
            s = df[(df.nras_mutation == 1) & (df.kras_mutation == kras) &
                   (df.right_sided_primary == rs) & (df.braf_v600e == braf)]
            a = s.loc[s.treatment_regorafenib == 1, "pfs_months"]
            b = s.loc[s.treatment_regorafenib == 0, "pfs_months"]
            if len(a) > 5 and len(b) > 5:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                print(f"  NRAS-mut KRAS={kras} right={rs} BRAF={braf}: n+={len(a):3d} n-={len(b):4d} "
                      f"diff={a.mean()-b.mean():+.3f} p={p:.3g}")
