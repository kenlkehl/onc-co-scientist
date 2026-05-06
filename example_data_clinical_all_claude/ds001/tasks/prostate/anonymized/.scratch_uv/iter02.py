"""Iteration 2: full multivariable logistic model."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
features = [c for c in df.columns if c not in ('patient_id', 'objective_response', 'feature_030')]

# standardize continuous to make coefficients comparable
X = df[features].astype(float).copy()
for c in features:
    if df[c].nunique() > 2:
        s = X[c]
        X[c] = (s - s.mean()) / s.std()
X = sm.add_constant(X)

mod = sm.Logit(y, X).fit(disp=0, maxiter=200)
print(mod.summary())
print()
print('--- Sorted by |z| ---')
zs = mod.tvalues.drop('const')
for name, z, p, beta in sorted(zip(zs.index, zs.values, mod.pvalues.drop('const').values, mod.params.drop('const').values),
                                key=lambda r: -abs(r[1])):
    print(f'{name:14s}  beta={beta:+.4f}  z={z:+.2f}  p={p:.2e}')
