"""Iteration 9: 3-way and continuous-modifier screens for feature_018, feature_012,
feature_020, feature_027. Also test feature_015 as a continuous modifier.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)

# 1. Within feature_016=1, feature_031=0: does the effect of feature_018 depend on continuous features?
print("=== Within f016=1 & f031=0: continuous modifiers of feature_018 effect ===")
sub = df[(df['feature_016']==1) & (df['feature_031']==0)].copy()
print(f"n={len(sub)}, n_T1={(sub['feature_018']==1).sum()}, n_T0={(sub['feature_018']==0).sum()}")
cont_feats = ['feature_015','feature_019','feature_025','feature_032','feature_017',
              'feature_024','feature_010','feature_003','feature_030','feature_008',
              'feature_004','feature_013','feature_026','feature_029','feature_007',
              'feature_033','feature_009']
results = []
for X in cont_feats:
    xv = (sub[X] - sub[X].mean()) / sub[X].std()
    d = pd.DataFrame({'y': sub['pfs_months'].values, 'T': sub['feature_018'].astype(float).values, 'X': xv.values})
    m = smf.ols("y ~ T*X", data=d).fit()
    results.append({'X': X, 'beta_T': m.params.get('T'), 'beta_TX': m.params.get('T:X'),
                    'p_TX': m.pvalues.get('T:X')})
print(pd.DataFrame(results).sort_values('p_TX').head(10).to_string(index=False))

# 2. Same but binary modifiers within f016=1 & f031=0
print("\n=== Within f016=1 & f031=0: binary modifiers of feature_018 effect ===")
binary_feats = ['feature_023','feature_011','feature_022','feature_028','feature_002',
                'feature_005','feature_021','feature_020','feature_027','feature_012',
                'hist_squam','smk_current','smk_former']
results = []
for X in binary_feats:
    if sub[X].nunique() < 2:
        continue
    d = pd.DataFrame({'y': sub['pfs_months'].values, 'T': sub['feature_018'].astype(float).values, 'X': sub[X].astype(float).values})
    m = smf.ols("y ~ T*X", data=d).fit()
    results.append({'X': X, 'beta_T': m.params.get('T'), 'beta_TX': m.params.get('T:X'),
                    'p_TX': m.pvalues.get('T:X')})
print(pd.DataFrame(results).sort_values('p_TX').head(15).to_string(index=False))

# 3. Cleanest 3-way subgroup test: feature_018 effect by combination of negative gates
print("\n=== Cleanest subgroup: f016=1 & f031=0 & f028=0 & f005=0 ===")
sub2 = df[(df['feature_016']==1) & (df['feature_031']==0) & (df['feature_028']==0) & (df['feature_005']==0)]
a = sub2.loc[sub2['feature_018']==1, 'pfs_months']
b = sub2.loc[sub2['feature_018']==0, 'pfs_months']
print(f"n={len(sub2)}, n_T1={len(a)}, n_T0={len(b)}, mean_T1={a.mean():.3f}, mean_T0={b.mean():.3f}, "
      f"effect={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# Check effect outside this subgroup (everywhere else)
print("\n=== Outside subgroup: f016=0 OR f031=1 OR f028=1 OR f005=1 ===")
mask = ~((df['feature_016']==1) & (df['feature_031']==0) & (df['feature_028']==0) & (df['feature_005']==0))
sub3 = df[mask]
a = sub3.loc[sub3['feature_018']==1, 'pfs_months']
b = sub3.loc[sub3['feature_018']==0, 'pfs_months']
print(f"n={len(sub3)}, n_T1={len(a)}, n_T0={len(b)}, mean_T1={a.mean():.3f}, mean_T0={b.mean():.3f}, "
      f"effect={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# 4. Test main effect of feature_012 in joint subgroups too
print("\n=== feature_012 effect in feature_016=1 ===")
for f16 in [0,1]:
    sub = df[df['feature_016']==f16]
    a = sub.loc[sub['feature_012']==1, 'pfs_months']
    b = sub.loc[sub['feature_012']==0, 'pfs_months']
    print(f"  f016={f16}: n={len(sub)}, eff={a.mean()-b.mean():+.4f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# 5. feature_020 across feature_023 and feature_014 (most prognostic)
print("\n=== feature_020 effect by stage/ECOG ===")
for f23 in [0,1]:
    for f14 in [0,1,2]:
        sub = df[(df['feature_023']==f23) & (df['feature_014']==f14)]
        if len(sub) > 100:
            a = sub.loc[sub['feature_020']==1, 'pfs_months']
            b = sub.loc[sub['feature_020']==0, 'pfs_months']
            if len(a) > 50 and len(b) > 50:
                print(f"  f023={f23}, f14={f14}: n={len(sub)}, eff={a.mean()-b.mean():+.4f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# 6. feature_027 across various subgroups
print("\n=== feature_027 effect by feature_028, feature_005 ===")
for X in ['feature_028','feature_005','feature_011','feature_023']:
    for v in [0,1]:
        sub = df[df[X]==v]
        a = sub.loc[sub['feature_027']==1, 'pfs_months']
        b = sub.loc[sub['feature_027']==0, 'pfs_months']
        if len(a) > 30 and len(b) > 30:
            print(f"  {X}={v}: n={len(sub)}, eff={a.mean()-b.mean():+.4f}, p={stats.ttest_ind(a,b).pvalue:.3e}")
