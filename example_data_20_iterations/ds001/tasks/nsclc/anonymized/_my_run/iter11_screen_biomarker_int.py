"""Iteration 11: scan ALL binary features × feature_092_hi interaction; find any other treatments
that depend on biomarker."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)

binary_cols = [c for c in df.columns
               if c not in ('patient_id','objective_response','f092_hi')
               and df[c].dtype != 'object'
               and df[c].nunique() == 2]

results = []
for c in binary_cols:
    x_hi = df['f092_hi'].astype(float).values
    x_c = df[c].astype(float).values
    # Build base model with main effects only
    X_base = np.column_stack([np.ones(len(df)), x_hi, x_c])
    X_full = np.column_stack([X_base, x_hi*x_c])
    try:
        res_b = sm.Logit(y, X_base).fit(disp=False, maxiter=100)
        res_f = sm.Logit(y, X_full).fit(disp=False, maxiter=100)
        lr = 2*(res_f.llf - res_b.llf)
        p = 1 - chi2_dist.cdf(lr, df=1)
        coef = res_f.params[3]
        results.append({'feature': c, 'coef': coef, 'p_value': p})
    except Exception as e:
        results.append({'feature': c, 'coef': np.nan, 'p_value': np.nan})

res_df = pd.DataFrame(results).sort_values('p_value')
res_df.to_csv('iter11_biomarker_x_binary.csv', index=False)
print(res_df.head(20).to_string(index=False))
print(f'\n# Significant (p<0.001): {(res_df.p_value<0.001).sum()}')
print(f'# Significant (p<0.05/77 = {0.05/77:.4f}): {(res_df.p_value<0.05/77).sum()}')

# For top hits, print stratified ORR
print('\n=== Stratified ORR for top biomarker × binary hits ===')
for c in res_df.head(8)['feature']:
    print(f'\n{c}:')
    tab = df.groupby(['f092_hi', c])['objective_response'].agg(['mean','count']).unstack(c)
    print(tab.to_string())
