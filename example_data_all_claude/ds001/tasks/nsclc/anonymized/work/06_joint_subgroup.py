"""Iteration 6: Verify joint subgroup feature_016=1 & feature_031=0 for feature_018."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet').copy()

# Joint stratification
print("=== Joint stratification: PFS by feature_016, feature_031, feature_018 ===")
res = df.groupby(['feature_016','feature_031','feature_018'])['pfs_months'].agg(['mean','std','count']).round(3)
print(res)

# For each (feature_016, feature_031) cell, compute treatment effect of feature_018
print("\n=== feature_018 effect within (feature_016, feature_031) cells ===")
for f16 in [0,1]:
    for f31 in [0,1]:
        sub = df[(df['feature_016']==f16) & (df['feature_031']==f31)]
        a = sub.loc[sub['feature_018']==1, 'pfs_months']
        b = sub.loc[sub['feature_018']==0, 'pfs_months']
        if len(a) > 5 and len(b) > 5:
            t, p = stats.ttest_ind(a, b)
            print(f"f016={f16}, f31={f31}: n_T1={len(a)}, n_T0={len(b)}, "
                  f"mean_T1={a.mean():.3f}, mean_T0={b.mean():.3f}, "
                  f"effect={a.mean()-b.mean():+.3f}, p={p:.3e}")

# Adjusted within feature_016=1, feature_031=0
print("\n=== Adjusted (covariate) feature_018 effect in feature_016=1, feature_031=0 ===")
sub = df[(df['feature_016']==1) & (df['feature_031']==0)].copy()
sub['hist_squam'] = (sub['feature_006']=='squamous').astype(int)
sub['smk_current'] = (sub['feature_001']=='current').astype(int)
sub['smk_former'] = (sub['feature_001']=='former').astype(int)
m = smf.ols("pfs_months ~ feature_018 + feature_015 + feature_014 + feature_023 + feature_011 + "
            "feature_017 + feature_025 + hist_squam + smk_current + smk_former", data=sub).fit()
print(m.summary().tables[1])

# Also verify with: feature_018 effect in {feature_016=1, feature_031=0, feature_028=0}
print("\n=== Effect within feature_016=1 & feature_031=0 & feature_028=0 (additional negative gating) ===")
sub2 = df[(df['feature_016']==1) & (df['feature_031']==0) & (df['feature_028']==0)]
a = sub2.loc[sub2['feature_018']==1, 'pfs_months']
b = sub2.loc[sub2['feature_018']==0, 'pfs_months']
t, p = stats.ttest_ind(a, b)
print(f"n_T1={len(a)}, n_T0={len(b)}, effect={a.mean()-b.mean():+.3f}, p={p:.3e}")

# What about feature_005 (negative interaction within feature_016=1)?
print("\n=== Effect within feature_016=1 & feature_031=0 & feature_005=0 ===")
sub3 = df[(df['feature_016']==1) & (df['feature_031']==0) & (df['feature_005']==0)]
a = sub3.loc[sub3['feature_018']==1, 'pfs_months']
b = sub3.loc[sub3['feature_018']==0, 'pfs_months']
t, p = stats.ttest_ind(a, b)
print(f"n_T1={len(a)}, n_T0={len(b)}, effect={a.mean()-b.mean():+.3f}, p={p:.3e}")

# Combined gating
print("\n=== Effect within feature_016=1 & feature_031=0 & feature_028=0 & feature_005=0 ===")
sub4 = df[(df['feature_016']==1) & (df['feature_031']==0) & (df['feature_028']==0) & (df['feature_005']==0)]
a = sub4.loc[sub4['feature_018']==1, 'pfs_months']
b = sub4.loc[sub4['feature_018']==0, 'pfs_months']
t, p = stats.ttest_ind(a, b)
print(f"n_T1={len(a)}, n_T0={len(b)}, effect={a.mean()-b.mean():+.3f}, p={p:.3e}")
