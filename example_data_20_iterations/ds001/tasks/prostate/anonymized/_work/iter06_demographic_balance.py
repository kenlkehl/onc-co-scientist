"""Iteration 6: Are key predictors balanced across race / insurance groups?"""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

key_cont = ['feature_078','log_feature_013','feature_006','feature_009','feature_092']
key_bin = ['feature_109','feature_039','feature_051']

print('=== Continuous predictors by race ===')
for c in key_cont:
    groups = [df.loc[df['feature_018']==v, c].values for v in df['feature_018'].unique()]
    F, p = stats.f_oneway(*groups)
    means = df.groupby('feature_018')[c].mean().to_dict()
    print(f'  {c}: ANOVA F={F:.3f} p={p:.3e}; means={ {k:round(v,3) for k,v in means.items()} }')

print()
print('=== Continuous predictors by insurance ===')
for c in key_cont:
    groups = [df.loc[df['feature_045']==v, c].values for v in df['feature_045'].unique()]
    F, p = stats.f_oneway(*groups)
    means = df.groupby('feature_045')[c].mean().to_dict()
    print(f'  {c}: ANOVA F={F:.3f} p={p:.3e}; means={ {k:round(v,3) for k,v in means.items()} }')

print()
print('=== Binary predictors by race (chi-square) ===')
for c in key_bin:
    ct = pd.crosstab(df['feature_018'], df[c])
    chi, p, dof, _ = stats.chi2_contingency(ct)
    rates = (df.groupby('feature_018')[c].mean()).to_dict()
    print(f'  {c}: chi2={chi:.3f} p={p:.3e}; rates={ {k:round(v,3) for k,v in rates.items()} }')

print()
print('=== Binary predictors by insurance (chi-square) ===')
for c in key_bin:
    ct = pd.crosstab(df['feature_045'], df[c])
    chi, p, dof, _ = stats.chi2_contingency(ct)
    rates = (df.groupby('feature_045')[c].mean()).to_dict()
    print(f'  {c}: chi2={chi:.3f} p={p:.3e}; rates={ {k:round(v,3) for k,v in rates.items()} }')
