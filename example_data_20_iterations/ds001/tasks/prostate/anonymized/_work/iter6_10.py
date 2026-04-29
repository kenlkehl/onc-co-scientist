import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_parquet('dataset.parquet')

results = {}
pfs = df['pfs_months']

# Iter 6: lab markers - cont features
# All major lab markers as univariate predictors
lab_features = ['feature_006','feature_099','feature_055','feature_014','feature_070',
                'feature_044','feature_084','feature_028','feature_065','feature_117',
                'feature_056','feature_103','feature_094','feature_120','feature_019',
                'feature_059','feature_003','feature_010','feature_101','feature_020',
                'feature_090','feature_118','feature_054','feature_062','feature_031',
                'feature_082','feature_075','feature_122','feature_005','feature_009']

for f in lab_features:
    X = sm.add_constant(df[f])
    m = sm.OLS(pfs, X).fit()
    results[f] = {'beta': float(m.params[f]),
                  'p': float(m.pvalues[f]),
                  'mean': float(df[f].mean()),
                  'std': float(df[f].std())}

# Sort by absolute t-statistic for ranking
ranked = sorted(results.items(), key=lambda x: x[1]['p'])
print('Top 15 univariate lab predictors:')
for f, r in ranked[:15]:
    print(f"{f}: beta={r['beta']:.4f}, p={r['p']:.6f}, std={r['std']:.2f}")
print()
print('Bottom 10 (least significant):')
for f, r in ranked[-10:]:
    print(f"{f}: beta={r['beta']:.4f}, p={r['p']:.4f}")

# Save full
with open('_work/lab_results.json','w') as fp:
    json.dump(results, fp, indent=2)
