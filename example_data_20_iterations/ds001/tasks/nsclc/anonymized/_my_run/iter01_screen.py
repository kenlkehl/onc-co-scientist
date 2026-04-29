"""Iteration 1: univariate screen of all binary features for association with objective_response."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

binary_cols = [c for c in df.columns
               if c not in ('patient_id', 'objective_response')
               and df[c].dtype != 'object'
               and df[c].nunique() == 2]

rows = []
y = df['objective_response']
for c in binary_cols:
    x = df[c]
    p1 = y[x == 1].mean()
    p0 = y[x == 0].mean()
    n1 = int((x == 1).sum())
    n0 = int((x == 0).sum())
    if min(n0, n1) < 5:
        continue
    tbl = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(tbl)
    # log-odds ratio (signed)
    a = ((x == 1) & (y == 1)).sum()
    b = ((x == 1) & (y == 0)).sum()
    c0 = ((x == 0) & (y == 1)).sum()
    d = ((x == 0) & (y == 0)).sum()
    or_ = (a * d) / max(b * c0, 1)
    log_or = np.log(or_) if or_ > 0 else np.nan
    rows.append({
        'feature': c, 'n1': n1, 'p1': p1, 'p0': p0,
        'diff': p1 - p0, 'log_or': log_or, 'p_value': p,
    })

res = pd.DataFrame(rows).sort_values('p_value')
res.to_csv('iter01_binary_screen.csv', index=False)
print(res.head(25).to_string(index=False))
print('...')
print(f'Total binary features tested: {len(res)}')
print(f'Significant at p<0.001: {(res.p_value<0.001).sum()}')
print(f'Significant at p<0.05: {(res.p_value<0.05).sum()}')
