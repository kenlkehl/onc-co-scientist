import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('dataset.parquet')
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))

# Check if there's a sex effect (this is prostate but maybe...)
# Let's investigate other interactions

# Race * f039
print('=== feature_039 x race interaction ===')
m = smf.ols('pfs_months ~ feature_039 * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'feature_039' in k or 'feature_018' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Dual treatment (f109+f039)
print('\n=== Dual treatment groups ===')
df['dual'] = ((df['feature_109']==1)&(df['feature_039']==1)).astype(int)
df['only109'] = ((df['feature_109']==1)&(df['feature_039']==0)).astype(int)
df['only039'] = ((df['feature_109']==0)&(df['feature_039']==1)).astype(int)
m = smf.ols('pfs_months ~ only109 + only039 + dual + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in ['only109','only039','dual']:
    print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.2e}, n={(df[k]==1).sum()}')

# Hemoglobin x race interaction (potential disparity)
print('\n=== Anemia x race interaction (does low hgb hurt black patients more?) ===')
df['hgb_low'] = (df['feature_038'] < 11).astype(int)
m = smf.ols('pfs_months ~ hgb_low * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'hgb_low' in k or 'feature_018' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Insurance x age interaction
print('\n=== Insurance x age ===')
df['f078_c'] = df['feature_078'] - df['feature_078'].mean()
m = smf.ols('pfs_months ~ f078_c * C(feature_045, Treatment(reference="private")) + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'f078_c' in k or 'feature_045' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# How about race x lab interactions
print('\n=== Race x log_PSA interaction ===')
df['log_f013_c'] = df['log_f013'] - df['log_f013'].mean()
m = smf.ols('pfs_months ~ log_f013_c * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + feature_006 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'log_f013_c' in k or 'feature_018' in k:
        if 'log_f013_c' in k:
            print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Race x feature_006
print('\n=== Race x f006 interaction ===')
df['f006_c'] = df['feature_006'] - df['feature_006'].mean()
m = smf.ols('pfs_months ~ f006_c * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'f006_c' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

# Examine the high-prevalence treatment-like features for race differences
# feature_027 (60% prev): adjusted +0.025
print('\n=== feature_027 x race ===')
m = smf.ols('pfs_months ~ feature_027 * C(feature_018, Treatment(reference="white")) + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df).fit()
for k in m.params.index:
    if 'feature_027' in k:
        print(f'  {k}: beta={m.params[k]:.4f}, p={m.pvalues[k]:.4f}')

print('\nf027 receipt by race:')
print(df.groupby('feature_018')['feature_027'].mean())
print('f027 receipt by insurance:')
print(df.groupby('feature_045')['feature_027'].mean())

