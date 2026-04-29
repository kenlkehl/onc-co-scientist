import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))

# Confirm synergy: 2x2 group means
g00 = df[(df.feature_109==0)&(df.feature_039==0)]
g10 = df[(df.feature_109==1)&(df.feature_039==0)]
g01 = df[(df.feature_109==0)&(df.feature_039==1)]
g11 = df[(df.feature_109==1)&(df.feature_039==1)]

print('2x2 PFS table by f109/f039:')
print(f'  Neither (n={len(g00)}): mean PFS = {g00.pfs_months.mean():.3f}')
print(f'  Only f109 (n={len(g10)}): mean PFS = {g10.pfs_months.mean():.3f}')
print(f'  Only f039 (n={len(g01)}): mean PFS = {g01.pfs_months.mean():.3f}')
print(f'  Both (n={len(g11)}): mean PFS = {g11.pfs_months.mean():.3f}')

# Adjusted interaction model
print('\n=== Interaction model (f109 x f039) ===')
m = smf.ols('pfs_months ~ feature_109 * feature_039 + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in ['feature_109','feature_039','feature_109:feature_039']:
    print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.2e}')

# Does this synergy exist across all biomarker subgroups?
print('\n=== Synergy by feature_057 (PS) ===')
for ps in [0,1,2]:
    sub = df[df.feature_057==ps]
    g11s = sub[(sub.feature_109==1)&(sub.feature_039==1)]
    g00s = sub[(sub.feature_109==0)&(sub.feature_039==0)]
    g10s = sub[(sub.feature_109==1)&(sub.feature_039==0)]
    g01s = sub[(sub.feature_109==0)&(sub.feature_039==1)]
    if len(g11s) < 20: continue
    print(f'  PS={ps}: neither={g00s.pfs_months.mean():.2f} (n={len(g00s)}), only109={g10s.pfs_months.mean():.2f} (n={len(g10s)}), only039={g01s.pfs_months.mean():.2f} (n={len(g01s)}), both={g11s.pfs_months.mean():.2f} (n={len(g11s)})')

# Does dual receipt vary by race/insurance?
print('\n=== Dual treatment receipt by race ===')
df['dual'] = ((df.feature_109==1)&(df.feature_039==1)).astype(int)
print(df.groupby('feature_018')['dual'].agg(['mean','sum']))
print('\n=== Dual treatment receipt by insurance ===')
print(df.groupby('feature_045')['dual'].agg(['mean','sum']))

# Statistical test for disparity in dual treatment receipt
from scipy import stats
ct = pd.crosstab(df['feature_018'], df['dual'])
chi2 = stats.chi2_contingency(ct)
print(f'\nChi-square (dual receipt by race): chi2={chi2.statistic:.3f}, p={chi2.pvalue:.4f}')

ct = pd.crosstab(df['feature_045'], df['dual'])
chi2 = stats.chi2_contingency(ct)
print(f'Chi-square (dual receipt by insurance): chi2={chi2.statistic:.3f}, p={chi2.pvalue:.4f}')

# Synergy effect by race
print('\n=== Synergy effect (dual vs neither) by race ===')
for r in df.feature_018.unique():
    sub = df[df.feature_018==r]
    g11s = sub[(sub.feature_109==1)&(sub.feature_039==1)]
    g00s = sub[(sub.feature_109==0)&(sub.feature_039==0)]
    if len(g11s) < 10 or len(g00s) < 100: continue
    diff = g11s.pfs_months.mean() - g00s.pfs_months.mean()
    print(f'  {r}: dual_mean={g11s.pfs_months.mean():.2f} (n={len(g11s)}), no_treat={g00s.pfs_months.mean():.2f}, diff={diff:.3f}')

