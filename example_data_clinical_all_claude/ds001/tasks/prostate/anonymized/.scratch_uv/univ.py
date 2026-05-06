import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']
print(f"Overall response rate: {y.mean():.4f}\n")

feats = [c for c in df.columns if c.startswith('feature_') and c != 'feature_030']

# Univariate associations
rows = []
for c in feats:
    x = df[c]
    n_uniq = x.nunique()
    if n_uniq <= 5:
        # categorical / binary
        ct = pd.crosstab(x, y)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        # response rate per level
        rr = df.groupby(c)['objective_response'].mean().to_dict()
        rows.append({'feat': c, 'kind': f'cat({n_uniq})', 'p': p, 'rates': rr})
    else:
        # continuous → split by median; also Mann-Whitney
        med = x.median()
        hi = x > med
        rr_hi = y[hi].mean()
        rr_lo = y[~hi].mean()
        # use point-biserial
        r, p = stats.pointbiserialr(y, x)
        rows.append({'feat': c, 'kind': 'cont', 'p': p, 'r': r, 'rr_hi(>med)': rr_hi, 'rr_lo': rr_lo, 'med': med})

rows.sort(key=lambda r: r['p'])
for r in rows:
    print(r)
