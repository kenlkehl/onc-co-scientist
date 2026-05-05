import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
feature_cols = [c for c in df.columns if c.startswith('feature_') and c != 'feature_030']

X = df[feature_cols].astype(float).copy()
# Standardize continuous feats with many unique values for interpretability
for c in feature_cols:
    if X[c].nunique() > 5:
        X[c] = (X[c] - X[c].mean()) / X[c].std()
Xc = sm.add_constant(X)
m = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
print(m.summary())
print('\nFeature coefs sorted by p:')
res = []
for c in feature_cols:
    res.append((c, m.params[c], m.pvalues[c]))
res.sort(key=lambda r: r[2])
for c, b, p in res:
    print(f'{c:<14} coef={b:+.4f} p={p:.3e}')
