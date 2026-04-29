"""Refine: stratified analyses around feature_092 biomarker and feature_006/007/039 treatment markers."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

# 1) Are features 006, 007, 039 correlated?
print("=== Cross-tabs of suspected treatment-like binary vars ===")
for a, b in [('feature_006', 'feature_007'), ('feature_006', 'feature_039'),
             ('feature_007', 'feature_039'), ('feature_006', 'feature_013'),
             ('feature_007', 'feature_013'), ('feature_039', 'feature_013')]:
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"{a} x {b}: chi2 p={p:.2e}")
    print(ct)
    print()

# 2) Distribution of feature_092 conditional on feature_006/007/039
print("=== feature_092 distribution by 006, 007, 039 ===")
for g in ['feature_006', 'feature_007', 'feature_039', 'feature_013']:
    print(f"{g}: 092 mean(0)={df.loc[df[g]==0,'feature_092'].mean():.3f} mean(1)={df.loc[df[g]==1,'feature_092'].mean():.3f}")
print()

# 3) Response rate by feature_092 quartiles, stratified by feature_006
print("=== Response rate by feature_092 tertile and feature_006 ===")
df['_092_tert'] = pd.qcut(df['feature_092'], 3, labels=['low', 'mid', 'high'])
print(df.groupby(['_092_tert', 'feature_006'], observed=False)['objective_response'].agg(['mean', 'count']))
print()

# 4) Is f013 a treatment? Check response rate stratified
print("=== Response rate by feature_013 ===")
print(df.groupby('feature_013')['objective_response'].agg(['mean', 'count']))
print()

# 5) Does feature_013 modify feature_006?
m = smf.logit("objective_response ~ feature_013 * feature_006 + feature_051 + feature_011 + feature_067 + feature_092", data=df).fit(disp=0)
print('=== f013 x f006 interaction ===')
print(m.summary().tables[1])
print()

# 6) Check if continuous features ranked by main-effect strength have non-linear effects
# feature_011 (0-28): try as quartile
df['_011_q'] = pd.qcut(df['feature_011'], 4, labels=False, duplicates='drop')
print('=== feature_011 quartile rates ===')
print(df.groupby('_011_q')['objective_response'].agg(['mean', 'count']))
print()

# 7) Correlation of feature_092 with feature_011 (might both be treatment-related)
print('feature_092 vs feature_011 (Spearman):', stats.spearmanr(df['feature_092'], df['feature_011']))
print('feature_011 vs feature_006 (mean by group):')
print(df.groupby('feature_006')['feature_011'].describe()[['mean', 'std', 'count']])
print()

# 8) Stage-like: feature_051 (ECOG-like 0/1/2) — interaction with feature_006/092
print('=== feature_051 stratified response rates ===')
print(df.groupby('feature_051')['objective_response'].agg(['mean', 'count']))
print('=== feature_051 x feature_006 ===')
print(df.groupby(['feature_051', 'feature_006'])['objective_response'].agg(['mean', 'count']))
print()

# 9) Three-way: feature_006 + feature_092 jointly
df['_092_hi'] = (df['feature_092'] > df['feature_092'].quantile(0.5)).astype(int)
print('=== response rate by feature_006 x feature_092 hi/lo ===')
print(df.groupby(['feature_006', '_092_hi'])['objective_response'].agg(['mean', 'count']))
