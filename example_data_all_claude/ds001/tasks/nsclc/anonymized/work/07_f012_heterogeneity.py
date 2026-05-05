"""Iteration 7: feature_012 heterogeneity. Search for the modifier(s) of its effect.
Note: top single interaction was f012:f017 (continuous, p=3e-4). Test split + nonparametric.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)

# Tertile split of feature_017 to look for non-linear interaction
df['f017_t'] = pd.qcut(df['feature_017'], 3, labels=['low','mid','high'])
print("=== feature_012 effect by tertile of feature_017 ===")
for t in ['low','mid','high']:
    sub = df[df['f017_t']==t]
    a = sub.loc[sub['feature_012']==1, 'pfs_months']
    b = sub.loc[sub['feature_012']==0, 'pfs_months']
    print(f"  f017 {t}: n_T1={len(a)}, n_T0={len(b)}, eff={a.mean()-b.mean():+.4f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# Let me look at feature_012 effect in a CART-like way: try every binary feature as a split
print("\n=== feature_012 effect by every binary feature ===")
binary_feats = ['feature_031','feature_023','feature_011','feature_022','feature_016',
                'feature_028','feature_002','feature_005','feature_021','feature_018',
                'feature_020','feature_027','hist_squam','smk_current','smk_former']
for f in binary_feats:
    for v in [0,1]:
        sub = df[df[f]==v]
        a = sub.loc[sub['feature_012']==1, 'pfs_months']
        b = sub.loc[sub['feature_012']==0, 'pfs_months']
        if len(a) > 50 and len(b) > 50:
            eff = a.mean() - b.mean()
            p = stats.ttest_ind(a,b).pvalue
            if p < 0.01:
                print(f"  {f}={v}: n={len(sub)}, eff={eff:+.4f}, p={p:.3e}")

# Adjusted multivariable model with feature_012:feature_023 interaction
print("\n=== feature_012 × feature_023 (adjusted) ===")
m = smf.ols("pfs_months ~ feature_012 * feature_023 + feature_015 + feature_014 + feature_011 + "
            "feature_016 + feature_017 + feature_025 + feature_018 + hist_squam + smk_current + smk_former + feature_031",
            data=df).fit()
print(m.summary().tables[1])

# How about within feature_023=1 only, screen feature_012 modifiers
print("\n=== Within feature_023=1, feature_012 modifiers ===")
sub = df[df['feature_023']==1]
all_X = ['feature_031','feature_011','feature_022','feature_016','feature_028','feature_002',
         'feature_005','feature_021','feature_018','feature_020','feature_027','hist_squam',
         'smk_current','smk_former','feature_014','feature_015','feature_017','feature_025']
results = []
for X in all_X:
    if X in ['feature_015','feature_017','feature_025']:
        xv = (sub[X] - sub[X].mean()) / sub[X].std()
    else:
        xv = sub[X].astype(float)
    d = pd.DataFrame({'y': sub['pfs_months'].values, 'T': sub['feature_012'].astype(float).values, 'X': xv.values})
    m2 = smf.ols("y ~ T*X", data=d).fit()
    results.append({'X': X, 'beta_T': m2.params.get('T'), 'beta_TX': m2.params.get('T:X'),
                    'p_TX': m2.pvalues.get('T:X')})
R = pd.DataFrame(results).sort_values('p_TX')
print(R.head(15).to_string(index=False))
