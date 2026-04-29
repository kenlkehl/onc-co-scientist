"""Iteration 3: ordinal int features and categorical string features."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

print('=== ORDINAL INTS ===')
ordinal = ['feature_051', 'feature_026', 'feature_033', 'feature_018', 'feature_045', 'feature_042']
for c in ordinal:
    tab = df.groupby(c)['objective_response'].agg(['mean', 'count'])
    print(f'\n{c}:')
    print(tab.to_string())
    # Test as ordinal (linear trend)
    x = df[c].values.astype(float)
    z = (x - x.mean()) / x.std()
    X = sm.add_constant(z)
    res = sm.Logit(y.values, X).fit(disp=False)
    print(f'  Linear-trend logistic beta_per_sd={res.params[1]:.4f}, p={res.pvalues[1]:.3g}')

print('\n=== CATEGORICAL STRINGS ===')
cat_str = ['feature_057', 'feature_043', 'feature_123', 'feature_005']
for c in cat_str:
    tab = df.groupby(c)['objective_response'].agg(['mean', 'count'])
    print(f'\n{c}:')
    print(tab.to_string())
    # Chi-square
    ct = pd.crosstab(df[c], y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f'  Chi-square p={p:.3g}')
