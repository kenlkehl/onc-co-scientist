"""Iteration 2: univariate scan of binary features vs pfs_months (Welch t-test)."""
import pandas as pd, numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
nunique = df.nunique()
binary_cols = [c for c in df.columns if c not in ['patient_id','pfs_months'] and nunique[c]==2 and df[c].dtype != object]

results = []
for c in binary_cols:
    g1 = df.loc[df[c]==1,'pfs_months']
    g0 = df.loc[df[c]==0,'pfs_months']
    if len(g1) < 30 or len(g0) < 30:
        continue
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    diff = g1.mean() - g0.mean()
    results.append({'feature': c, 'n1': len(g1), 'n0': len(g0),
                    'mean_1': g1.mean(), 'mean_0': g0.mean(),
                    'diff_1_minus_0': diff, 't': t, 'p': p})

res = pd.DataFrame(results).sort_values('p')
res.to_csv('iter02_univariate_binary.csv', index=False)
print('Top 30 binary features by p (effect = mean(pfs|x=1) - mean(pfs|x=0)):')
print(res.head(30).to_string(index=False))
print()
print(f'Significant at p<0.001: {(res["p"]<0.001).sum()} / {len(res)}')
print(f'Significant at p<0.05: {(res["p"]<0.05).sum()} / {len(res)}')
