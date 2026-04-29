"""Iteration 1: univariate continuous scan vs pfs_months (Spearman + linear regression)."""
import pandas as pd, numpy as np, json
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
nunique = df.nunique()
cont = [c for c in df.columns
        if c not in ['patient_id','pfs_months']
        and nunique[c] > 10
        and df[c].dtype != object]

results = []
for c in cont:
    x = df[c].values
    y = df['pfs_months'].values
    rho, p = stats.spearmanr(x, y)
    # linear slope (effect_estimate per unit)
    slope, intercept, r, plin, se = stats.linregress(x, y)
    results.append({'feature': c, 'spearman_rho': rho, 'spearman_p': p,
                    'lin_slope': slope, 'lin_p': plin})

res = pd.DataFrame(results).sort_values('spearman_p')
res.to_csv('iter01_univariate_continuous.csv', index=False)
print('Top 25 by Spearman p:')
print(res.head(25).to_string(index=False))
print()
print('Bottom 10 (least significant):')
print(res.tail(10).to_string(index=False))
