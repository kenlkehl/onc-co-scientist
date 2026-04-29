import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))

# Test all binary features after adjusting for known predictors
int_cols = df.select_dtypes(include='int64').columns.tolist()
binary_cols = [c for c in int_cols if df[c].nunique()==2]

results = {}
for c in binary_cols:
    formula = f'pfs_months ~ {c} + feature_078 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)'
    try:
        m = smf.ols(formula, data=df).fit()
        results[c] = {'beta_adj': float(m.params[c]),
                      'p_adj': float(m.pvalues[c]),
                      'prevalence': float(df[c].mean())}
    except Exception as e:
        continue

ranked = sorted(results.items(), key=lambda x: x[1]['p_adj'])
print('Top 30 binary features adjusted for PS/age/labs:')
for f, r in ranked[:30]:
    print(f"{f}: beta_adj={r['beta_adj']:+.4f}, p={r['p_adj']:.2e}, prev={r['prevalence']:.3f}")

with open('_work/binary_adj_results.json','w') as fp:
    json.dump(results, fp, indent=2)
