"""Iteration 5: verify feature_018 × feature_016 interaction with stratified analyses,
and screen for additional modifiers within feature_016=1.
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

# Stratified means: T=feature_018 by feature_016
print("=== Stratified mean PFS: feature_018 × feature_016 ===")
ct = df.groupby(['feature_016', 'feature_018'])['pfs_months'].agg(['mean','std','count']).round(3)
print(ct)

print("\n=== Within feature_016=1 (n biomarker positive) ===")
sub = df[df['feature_016'] == 1]
print(f"n = {len(sub)}")
print(f"mean PFS by feature_018: {sub.groupby('feature_018')['pfs_months'].agg(['mean','std','count']).round(3)}")
t, p = stats.ttest_ind(sub.loc[sub['feature_018']==1, 'pfs_months'],
                       sub.loc[sub['feature_018']==0, 'pfs_months'])
print(f"t-test: t={t:.3f}, p={p:.3e}")
diff = sub.loc[sub['feature_018']==1, 'pfs_months'].mean() - sub.loc[sub['feature_018']==0, 'pfs_months'].mean()
print(f"effect (T1-T0) = {diff:.3f} months")

print("\n=== Within feature_016=0 (biomarker negative) ===")
sub0 = df[df['feature_016'] == 0]
print(f"n = {len(sub0)}")
print(f"mean PFS by feature_018: {sub0.groupby('feature_018')['pfs_months'].agg(['mean','std','count']).round(3)}")
t, p = stats.ttest_ind(sub0.loc[sub0['feature_018']==1, 'pfs_months'],
                       sub0.loc[sub0['feature_018']==0, 'pfs_months'])
print(f"t-test: t={t:.3f}, p={p:.3e}")
diff = sub0.loc[sub0['feature_018']==1, 'pfs_months'].mean() - sub0.loc[sub0['feature_018']==0, 'pfs_months'].mean()
print(f"effect (T1-T0) = {diff:.3f} months")

# Within feature_016=1 patients, screen for further modifiers of the feature_018 effect
print("\n=== Modifiers of feature_018 effect WITHIN feature_016=1 ===")
binary_feats = ['feature_031','feature_023','feature_011','feature_022',
                'feature_028','feature_002','feature_005','feature_021','feature_020',
                'feature_027','feature_012','hist_squam','smk_current','smk_former']
ord_feats = ['feature_014']
cont_feats = ['feature_015','feature_019','feature_025','feature_032','feature_017',
              'feature_024','feature_010','feature_003','feature_030','feature_008',
              'feature_004','feature_013','feature_026','feature_029','feature_007',
              'feature_033','feature_009']
all_X = binary_feats + ord_feats + cont_feats

results = []
for X in all_X:
    if X in cont_feats:
        xv = (sub[X] - sub[X].mean()) / sub[X].std()
    else:
        xv = sub[X].astype(float)
    d = pd.DataFrame({'y': sub['pfs_months'].values, 'T': sub['feature_018'].astype(float).values, 'X': xv.values})
    m = smf.ols("y ~ T*X", data=d).fit()
    results.append({'X': X, 'beta_T': m.params.get('T'), 'beta_X': m.params.get('X'),
                    'beta_TX': m.params.get('T:X'), 'p_TX': m.pvalues.get('T:X')})
R = pd.DataFrame(results).sort_values('p_TX')
print(R.head(20).to_string(index=False))

R.to_csv('work/05_within_f016pos.csv', index=False)
