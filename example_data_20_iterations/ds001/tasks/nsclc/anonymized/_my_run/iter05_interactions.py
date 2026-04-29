"""Iteration 5: pairwise interactions among top predictors via likelihood ratio tests."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values

# Top predictors
core = ['feature_013', 'feature_067', 'feature_006', 'feature_007', 'feature_039',
        'feature_011', 'feature_099', 'feature_063', 'feature_092', 'feature_051']

# Standardize continuous
cont = ['feature_011', 'feature_099', 'feature_063', 'feature_092', 'feature_051']
data = pd.DataFrame()
for c in core:
    if c in cont:
        data[c] = (df[c] - df[c].mean()) / df[c].std()
    else:
        data[c] = df[c].astype(float)

# Base model: all main effects
X_base = sm.add_constant(data).astype(float)
res_base = sm.Logit(y, X_base).fit(disp=False, maxiter=200)
ll_base = res_base.llf

results = []
for a, b in combinations(core, 2):
    inter = data[a] * data[b]
    X = X_base.copy()
    X[f'{a}*{b}'] = inter
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)
    lr = 2 * (res.llf - ll_base)
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(lr, df=1)
    coef = res.params[f'{a}*{b}']
    results.append({'a': a, 'b': b, 'coef': coef, 'p_value': p})

res_df = pd.DataFrame(results).sort_values('p_value')
res_df.to_csv('iter05_interactions.csv', index=False)
print(res_df.to_string(index=False))
print(f'\n# Interactions with p<0.05: {(res_df.p_value<0.05).sum()}')
print(f'# Interactions with p<0.01: {(res_df.p_value<0.01).sum()} (Bonferroni 0.05/45 = 0.0011)')
print(f'# Interactions with p<0.0011: {(res_df.p_value<0.0011).sum()}')
