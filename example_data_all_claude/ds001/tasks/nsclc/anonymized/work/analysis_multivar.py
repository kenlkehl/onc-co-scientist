"""Multivariable model and interaction screens."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT_PATH = "C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/work/multivar.json"

df = pd.read_parquet("C:/Users/klkehl/are_llms_biased/data/ds001/tasks/nsclc/anonymized/dataset.parquet")

binary_cols = [c for c in df.columns if df[c].dtype in ['int64'] and df[c].nunique() == 2]
float_cols = [c for c in df.columns if df[c].dtype == 'float64' and c != 'pfs_months']
results = {}

# encode obj cols as dummies
df['smoke_current'] = (df['feature_001'] == 'current').astype(int)
df['smoke_former'] = (df['feature_001'] == 'former').astype(int)
df['hist_squamous'] = (df['feature_006'] == 'squamous').astype(int)

# Multivariable OLS: PFS ~ all features
X_cols = (binary_cols + ['feature_014'] + float_cols +
          ['smoke_current', 'smoke_former', 'hist_squamous'])
X = df[X_cols].copy()
X = sm.add_constant(X)
y = df['pfs_months']
m = sm.OLS(y, X).fit()
print(m.summary())

multivar = []
for name, b, p, lo, hi in zip(m.params.index, m.params.values, m.pvalues.values,
                               m.conf_int()[0].values, m.conf_int()[1].values):
    multivar.append({'var': name, 'beta': float(b), 'p': float(p), 'ci_lo': float(lo), 'ci_hi': float(hi)})
results['multivar'] = multivar
results['rsquared'] = float(m.rsquared)
print('\nR^2:', m.rsquared)

# Interaction screens — pairwise binary x binary on PFS via OLS with interaction term
print('\n=== Interaction screen: binary x binary ===')
inter_results = []
for i, c1 in enumerate(binary_cols):
    for c2 in binary_cols[i+1:]:
        d = df[[c1, c2, 'pfs_months']].copy()
        d['inter'] = d[c1] * d[c2]
        Xi = sm.add_constant(d[[c1, c2, 'inter']])
        mi = sm.OLS(d['pfs_months'], Xi).fit()
        inter_results.append({'a': c1, 'b': c2,
                              'beta_a': float(mi.params[c1]),
                              'beta_b': float(mi.params[c2]),
                              'beta_inter': float(mi.params['inter']),
                              'p_inter': float(mi.pvalues['inter']),
                              'p_a': float(mi.pvalues[c1]),
                              'p_b': float(mi.pvalues[c2])})
inter_df = pd.DataFrame(inter_results).sort_values('p_inter')
print(inter_df.head(20).to_string())
results['inter_bb'] = inter_results

# === Treatment candidates: feature_023, feature_011, feature_016, feature_018, feature_022 ===
# Assume each is a candidate "treatment" -> screen interactions with every other feature
print('\n=== Heterogeneity screen for candidate treatment features ===')
candidate_tx = ['feature_023', 'feature_011', 'feature_016', 'feature_018', 'feature_022', 'feature_028']

all_other_cols = [c for c in (binary_cols + ['feature_014'] + float_cols +
                              ['smoke_current', 'smoke_former', 'hist_squamous'])]

hetero = []
for tx in candidate_tx:
    for mod in all_other_cols:
        if mod == tx:
            continue
        d = df[[tx, mod, 'pfs_months']].copy()
        d['inter'] = d[tx] * d[mod]
        Xi = sm.add_constant(d[[tx, mod, 'inter']])
        try:
            mi = sm.OLS(d['pfs_months'], Xi).fit()
            hetero.append({'tx': tx, 'mod': mod,
                           'beta_tx': float(mi.params[tx]),
                           'beta_mod': float(mi.params[mod]),
                           'beta_inter': float(mi.params['inter']),
                           'p_inter': float(mi.pvalues['inter'])})
        except Exception as e:
            print('skip', tx, mod, e)
hetero_df = pd.DataFrame(hetero).sort_values('p_inter')
print(hetero_df.head(40).to_string())
results['hetero'] = hetero

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, default=str, indent=2)
print('\nSaved to', OUT_PATH)
