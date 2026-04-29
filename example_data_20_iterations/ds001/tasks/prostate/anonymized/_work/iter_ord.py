import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet('dataset.parquet')
pfs = df['pfs_months']

# Ordinal/multi-level features
ord_cols = ['feature_057','feature_067','feature_016','feature_071','feature_026',
            'feature_096','feature_033','feature_064']

results = {}
for c in ord_cols:
    means = df.groupby(c)['pfs_months'].agg(['mean','count']).to_dict()
    X = sm.add_constant(df[c])
    m = sm.OLS(pfs, X).fit()
    results[c] = {'beta_lin': float(m.params[c]),
                  'p_lin': float(m.pvalues[c]),
                  'levels': {int(k):int(v) for k,v in df[c].value_counts().to_dict().items()},
                  'mean_pfs_by_level': {int(k):float(v) for k,v in df.groupby(c)['pfs_months'].mean().to_dict().items()}}

print(json.dumps(results, indent=2))
