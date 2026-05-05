"""Iteration 4: treatment-effect heterogeneity screen.

Treat each binary feature as a candidate 'treatment' and test
treatment * feature interactions across all other features.
Report the strongest interactions (lowest p-values).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
features = [c for c in df.columns if c not in ('patient_id', 'objective_response', 'feature_030')]
binaries = [c for c in features if df[c].nunique() == 2]
y = df['objective_response'].astype(int).values

# standardize continuous features
df_std = df.copy()
for c in features:
    if df[c].nunique() > 2 and pd.api.types.is_numeric_dtype(df[c]):
        s = df_std[c].astype(float)
        df_std[c] = (s - s.mean()) / s.std()

records = []
for tx in binaries:
    for cov in features:
        if cov == tx:
            continue
        x_tx = df_std[tx].astype(float).values
        x_cov = df_std[cov].astype(float).values
        x_int = x_tx * x_cov
        X = np.column_stack([np.ones(len(y)), x_tx, x_cov, x_int])
        try:
            mod = sm.Logit(y, X).fit(disp=0, maxiter=80)
            beta_int = mod.params[3]
            p_int = mod.pvalues[3]
            beta_tx = mod.params[1]
            p_tx = mod.pvalues[1]
        except Exception:
            beta_int, p_int, beta_tx, p_tx = np.nan, np.nan, np.nan, np.nan
        records.append({'tx': tx, 'cov': cov, 'beta_int': beta_int, 'p_int': p_int,
                        'beta_tx': beta_tx, 'p_tx': p_tx})

rec = pd.DataFrame(records).dropna(subset=['p_int']).sort_values('p_int')
print('--- top 40 interactions overall ---')
print(rec.head(40).to_string(index=False))
print()
print('--- top interactions per tx ---')
for tx in binaries:
    sub = rec[rec.tx == tx].head(5)
    print(f'\n=== {tx} (rate={df[tx].mean():.3f}) ===')
    print(sub.to_string(index=False))
