import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))

# Treatment x biomarker interactions for feature_109
print('=== feature_109 x performance status ===')
m = smf.ols('pfs_months ~ feature_109 * C(feature_057) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092', data=df).fit()
for k in m.params.index:
    if 'feature_109' in k or 'feature_057' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Stratified
print('\nStratified by feature_057:')
for v in [0,1,2]:
    sub = df[df['feature_057']==v]
    n_t = (sub['feature_109']==1).sum()
    m = smf.ols('pfs_months ~ feature_109 + feature_078 + feature_009 + feature_006 + log_f013 + log_f092', data=sub).fit()
    print(f"  PS={v} (n={len(sub)}, n_treated={n_t}): beta={m.params['feature_109']:.4f}, p={m.pvalues['feature_109']:.2e}")

# Treatment x log_PSA interaction
print('\n=== feature_109 x log_PSA ===')
df['log_f013_c'] = df['log_f013'] - df['log_f013'].mean()
m = smf.ols('pfs_months ~ feature_109 * log_f013_c + feature_078 + feature_009 + feature_006 + log_f092 + C(feature_057)', data=df).fit()
print(f"  feature_109: beta={m.params['feature_109']:.4f}, p={m.pvalues['feature_109']:.4f}")
print(f"  feature_109:log_f013_c interaction: beta={m.params['feature_109:log_f013_c']:.4f}, p={m.pvalues['feature_109:log_f013_c']:.4f}")

# Treatment x age interaction
print('\n=== feature_109 x age (f078) ===')
df['f078_c'] = df['feature_078'] - df['feature_078'].mean()
m = smf.ols('pfs_months ~ feature_109 * f078_c + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
print(f"  feature_109: beta={m.params['feature_109']:.4f}, p={m.pvalues['feature_109']:.4f}")
print(f"  feature_109:f078_c interaction: beta={m.params['feature_109:f078_c']:.4f}, p={m.pvalues['feature_109:f078_c']:.4f}")

# Treatment x insurance interaction
print('\n=== feature_109 x insurance (f045) ===')
m = smf.ols('pfs_months ~ feature_109 * C(feature_045, Treatment(reference="private")) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'feature_109' in k or 'feature_045' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Same for feature_039
print('\n\n=== feature_039 x performance status ===')
m = smf.ols('pfs_months ~ feature_039 * C(feature_057) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092', data=df).fit()
for k in m.params.index:
    if 'feature_039' in k or 'feature_057' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Stratified treatment effect of f039 by race
print('\n=== f039 effect by race ===')
for r in df['feature_018'].unique():
    sub = df[df['feature_018']==r]
    n_t = (sub['feature_039']==1).sum()
    if n_t < 20: continue
    m = smf.ols('pfs_months ~ feature_039 + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=sub).fit()
    print(f"  {r} (n={len(sub)}, n_treated={n_t}): beta={m.params['feature_039']:.4f}, p={m.pvalues['feature_039']:.2e}")

