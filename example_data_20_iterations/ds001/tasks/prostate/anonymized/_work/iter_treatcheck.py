import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')

# Check if f109 and f039 overlap
print('feature_109 vs feature_039 cross-tab:')
print(pd.crosstab(df['feature_109'], df['feature_039']))
print()

# Concordance
both = ((df['feature_109']==1) & (df['feature_039']==1)).sum()
either = ((df['feature_109']==1) | (df['feature_039']==1)).sum()
print(f'Both 1: {both}, Either 1: {either}')

# Check if feature_109 + feature_039 = something exclusive
df['t1'] = df['feature_109']
df['t2'] = df['feature_039']
print('Joint dist:')
print(pd.crosstab(df['t1'], df['t2'], normalize='all'))

# Race x treatment (f109) interactions
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))

# Effect of treatment (f109) by race (f018)
print('\n\n=== Race x Treatment interaction (f109) ===')
m = smf.ols('pfs_months ~ feature_109 * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
# Print main + interaction terms only
print('Coefficients of interest:')
for k in m.params.index:
    if 'feature_109' in k or 'feature_018' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Stratified treatment effect by race
print('\n=== Stratified f109 effect by race ===')
for r in df['feature_018'].unique():
    sub = df[df['feature_018']==r]
    if len(sub) < 100:
        continue
    n_treated = (sub['feature_109']==1).sum()
    if n_treated < 20:
        continue
    m = smf.ols('pfs_months ~ feature_109 + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=sub).fit()
    print(f"  {r} (n={len(sub)}, n_treated={n_treated}): beta={m.params['feature_109']:.4f}, p={m.pvalues['feature_109']:.2e}")

# Treatment receipt by race
print('\n=== Treatment receipt rates by race ===')
print('feature_109:')
print(df.groupby('feature_018')['feature_109'].mean())
print('\nfeature_039:')
print(df.groupby('feature_018')['feature_039'].mean())
print('\nfeature_109 by insurance:')
print(df.groupby('feature_045')['feature_109'].mean())
print('\nfeature_039 by insurance:')
print(df.groupby('feature_045')['feature_039'].mean())
