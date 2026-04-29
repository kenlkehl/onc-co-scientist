"""Iteration 4: multivariable logistic regression with top predictors from iter1-3."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values

# Top predictors from screening
top = ['feature_013', 'feature_067', 'feature_006', 'feature_007', 'feature_039',
       'feature_011', 'feature_099', 'feature_063', 'feature_092',
       'feature_051', 'feature_033']

# Standardize continuous
cont = ['feature_011', 'feature_099', 'feature_063', 'feature_092']
ord_ = ['feature_051', 'feature_033']

X = pd.DataFrame()
for c in top:
    if c in cont or c in ord_:
        X[c] = (df[c] - df[c].mean()) / df[c].std()
    else:
        X[c] = df[c]

# Add categorical strings as dummies
for c in ['feature_057', 'feature_043', 'feature_123', 'feature_005']:
    dummies = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(int)
    X = pd.concat([X, dummies], axis=1)

X = sm.add_constant(X).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(res.summary())

# Save coefficient table
coef_tab = pd.DataFrame({
    'coef': res.params,
    'se': res.bse,
    'z': res.tvalues,
    'p_value': res.pvalues,
    'or': np.exp(res.params),
})
coef_tab.to_csv('iter04_multivar.csv')
print('\n--- Sorted by |z| ---')
print(coef_tab.assign(abs_z=coef_tab['z'].abs()).sort_values('abs_z', ascending=False)[['coef','se','p_value','or']].to_string())
