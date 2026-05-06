"""Main analysis script: runs many tests and records results."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT_PATH = "C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/work/results.json"

df = pd.read_parquet("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/dataset.parquet")
print(df.shape)

results = {}

# Identify column types
binary_cols = [c for c in df.columns if df[c].dtype in ['int64'] and df[c].nunique() == 2]
multi_int_cols = [c for c in df.columns if df[c].dtype in ['int64'] and 2 < df[c].nunique() <= 5]
float_cols = [c for c in df.columns if df[c].dtype == 'float64' and c != 'pfs_months']
obj_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'patient_id']

print('binary:', binary_cols)
print('multi_int:', multi_int_cols)
print('float:', float_cols)
print('obj:', obj_cols)

# === Iter 1: main effects of binary features on pfs_months (t-test) ===
binary_main = []
for c in binary_cols:
    g0 = df.loc[df[c] == 0, 'pfs_months']
    g1 = df.loc[df[c] == 1, 'pfs_months']
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    diff = g1.mean() - g0.mean()
    binary_main.append({'col': c, 'mean1': g1.mean(), 'mean0': g0.mean(),
                        'diff': diff, 'n1': len(g1), 'n0': len(g0),
                        'p': float(p), 't': float(t)})
binary_main_df = pd.DataFrame(binary_main).sort_values('p')
print('\n=== Binary main effects ===')
print(binary_main_df.to_string())
results['binary_main'] = binary_main

# === Categorical (object) ===
cat_main = []
for c in obj_cols:
    groups = [df.loc[df[c] == lvl, 'pfs_months'].values for lvl in df[c].unique()]
    if len(groups) == 2:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        cat_main.append({'col': c, 'levels': list(df[c].unique()),
                         'means': [g.mean() for g in groups],
                         'p': float(p), 'test': 't'})
    else:
        f, p = stats.f_oneway(*groups)
        cat_main.append({'col': c, 'levels': list(df[c].unique()),
                         'means': [g.mean() for g in groups],
                         'p': float(p), 'test': 'anova'})
print('\n=== Categorical main effects ===')
for r in cat_main:
    print(r)
results['cat_main'] = cat_main

# === Multi-int (e.g. feature_014: 0,1,2) ===
mint_main = []
for c in multi_int_cols:
    groups = [df.loc[df[c] == lvl, 'pfs_months'].values for lvl in sorted(df[c].unique())]
    f, p = stats.f_oneway(*groups)
    mint_main.append({'col': c, 'levels': sorted(df[c].unique().tolist()),
                      'means': [g.mean() for g in groups],
                      'ns': [len(g) for g in groups],
                      'p': float(p)})
print('\n=== Multi-int main effects ===')
for r in mint_main:
    print(r)
results['mint_main'] = mint_main

# === Continuous: Pearson and Spearman correlation ===
cont_main = []
for c in float_cols:
    r, p = stats.pearsonr(df[c], df['pfs_months'])
    rs, ps = stats.spearmanr(df[c], df['pfs_months'])
    cont_main.append({'col': c, 'pearson_r': float(r), 'pearson_p': float(p),
                      'spearman_r': float(rs), 'spearman_p': float(ps)})
cont_main_df = pd.DataFrame(cont_main).sort_values('pearson_p')
print('\n=== Continuous main effects ===')
print(cont_main_df.to_string())
results['cont_main'] = cont_main

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, default=str, indent=2)

print('\nSaved to', OUT_PATH)
