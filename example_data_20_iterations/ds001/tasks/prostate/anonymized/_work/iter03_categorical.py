"""Iteration 3: race (feature_018) and insurance (feature_045) effects on pfs_months."""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

print('=== feature_018 (race-like) groups ===')
gb = df.groupby('feature_018')['pfs_months'].agg(['count','mean','std','median'])
print(gb)
print()
groups = [df.loc[df['feature_018']==v,'pfs_months'] for v in df['feature_018'].unique()]
F, p = stats.f_oneway(*groups)
print(f'One-way ANOVA across feature_018 levels: F={F:.3f}, p={p:.3e}')
# pairwise vs reference 'white'
ref = df.loc[df['feature_018']=='white','pfs_months']
print('\nPairwise t-tests vs white:')
for v in df['feature_018'].unique():
    if v=='white': continue
    g = df.loc[df['feature_018']==v,'pfs_months']
    t, pp = stats.ttest_ind(g, ref, equal_var=False)
    print(f'  {v}: mean={g.mean():.3f}, white={ref.mean():.3f}, diff={g.mean()-ref.mean():+.3f}, t={t:.2f}, p={pp:.3e}')

print()
print('=== feature_045 (insurance-like) groups ===')
gb = df.groupby('feature_045')['pfs_months'].agg(['count','mean','std','median'])
print(gb)
groups = [df.loc[df['feature_045']==v,'pfs_months'] for v in df['feature_045'].unique()]
F, p = stats.f_oneway(*groups)
print(f'One-way ANOVA across feature_045 levels: F={F:.3f}, p={p:.3e}')
ref = df.loc[df['feature_045']=='private','pfs_months']
print('\nPairwise t-tests vs private:')
for v in df['feature_045'].unique():
    if v=='private': continue
    g = df.loc[df['feature_045']==v,'pfs_months']
    t, pp = stats.ttest_ind(g, ref, equal_var=False)
    print(f'  {v}: mean={g.mean():.3f}, private={ref.mean():.3f}, diff={g.mean()-ref.mean():+.3f}, t={t:.2f}, p={pp:.3e}')
