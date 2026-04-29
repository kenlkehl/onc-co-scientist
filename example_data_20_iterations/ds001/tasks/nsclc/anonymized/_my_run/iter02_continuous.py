"""Iteration 2: univariate logistic regression for each continuous feature."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')

binary_cols = [c for c in df.columns
               if c not in ('patient_id', 'objective_response')
               and df[c].dtype != 'object'
               and df[c].nunique() == 2]
multi_cat_int = [c for c in df.columns
                 if df[c].dtype == 'int64' and 2 < df[c].nunique() <= 10
                 and c != 'patient_id']
continuous_cols = [c for c in df.columns
                   if c not in ('patient_id', 'objective_response')
                   and df[c].dtype in ('float64', 'int64')
                   and c not in binary_cols and c not in multi_cat_int]

y = df['objective_response'].values
rows = []
for c in continuous_cols:
    x = df[c].values.astype(float)
    # Standardize so that beta is per-SD change
    sd = x.std()
    z = (x - x.mean()) / sd
    X = sm.add_constant(z)
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=100)
        beta = res.params[1]
        se = res.bse[1]
        p = res.pvalues[1]
        rows.append({'feature': c, 'mean': x.mean(), 'sd': sd,
                     'beta_per_sd': beta, 'se': se, 'p_value': p})
    except Exception as e:
        rows.append({'feature': c, 'mean': x.mean(), 'sd': sd,
                     'beta_per_sd': np.nan, 'se': np.nan, 'p_value': np.nan})

res_df = pd.DataFrame(rows).sort_values('p_value')
res_df.to_csv('iter02_continuous_screen.csv', index=False)
print(res_df.head(20).to_string(index=False))
print(f'\nTotal continuous tested: {len(res_df)}')
print(f'Significant at p<0.001: {(res_df.p_value<0.001).sum()}')
print(f'Significant at p<0.05: {(res_df.p_value<0.05).sum()}')
