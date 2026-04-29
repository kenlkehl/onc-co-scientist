import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet('dataset.parquet')
pfs = df['pfs_months']

# Univariate analysis of all binary features
int_cols = df.select_dtypes(include='int64').columns.tolist()
binary_cols = [c for c in int_cols if df[c].nunique()==2]
print(f'Testing {len(binary_cols)} binary features:')

results = {}
for c in binary_cols:
    a = pfs[df[c]==1]
    b = pfs[df[c]==0]
    if len(a) < 10 or len(b) < 10:
        continue
    t = stats.ttest_ind(a, b, equal_var=False)
    results[c] = {'mean_1': float(a.mean()),
                  'mean_0': float(b.mean()),
                  'diff': float(a.mean()-b.mean()),
                  'p': float(t.pvalue),
                  't': float(t.statistic),
                  'n_1': int(len(a)),
                  'prevalence': float(a.shape[0]/df.shape[0])}

ranked = sorted(results.items(), key=lambda x: x[1]['p'])
print('Top 25 binary predictors of PFS by p-value:')
for f, r in ranked[:25]:
    print(f"{f}: diff={r['diff']:+.3f}, mean_1={r['mean_1']:.2f}, mean_0={r['mean_0']:.2f}, prev={r['prevalence']:.3f}, p={r['p']:.2e}")

with open('_work/binary_results.json','w') as fp:
    json.dump(results, fp, indent=2)
