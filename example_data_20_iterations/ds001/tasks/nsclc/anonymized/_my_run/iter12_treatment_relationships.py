"""Iteration 12: explore relationships among 006/007/039.
Are they correlated? Mutually exclusive? Additive on response? Possibly distinct treatments."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)

# Correlations
print('=== Pairwise correlations & chi-square (006/007/039) ===')
for a, b in [('feature_006','feature_007'),('feature_006','feature_039'),('feature_007','feature_039')]:
    ct = pd.crosstab(df[a], df[b])
    chi2, p, _, _ = chi2_contingency(ct)
    corr = df[a].corr(df[b])
    print(f'  {a} vs {b}: chi-sq p={p:.3g}, phi-corr={corr:.3f}')
    print(ct.to_string())
    print()

# Joint count by (006, 007, 039)
print('=== Joint count + ORR by treatment combo ===')
df['combo'] = (df['feature_006'].astype(str) + '|' +
               df['feature_007'].astype(str) + '|' +
               df['feature_039'].astype(str))
tab = df.groupby('combo')['objective_response'].agg(['mean','count'])
print(tab.to_string())

# Within biomarker-high subgroup
print('\n=== Within feature_092_hi=1 ===')
hi = df[df['f092_hi']==1]
tab_hi = hi.groupby('combo')['objective_response'].agg(['mean','count'])
print(tab_hi.to_string())

# Are 006/007/039 effects additive in biomarker-high?
print('\n=== Logistic model in biomarker-high: 006 + 007 + 039 main effects (additive) ===')
y_hi = hi['objective_response'].values
X = pd.DataFrame({
    'f006': hi['feature_006'].astype(float),
    'f007': hi['feature_007'].astype(float),
    'f039': hi['feature_039'].astype(float),
})
X = sm.add_constant(X).astype(float)
res = sm.Logit(y_hi, X).fit(disp=False, maxiter=100)
print(res.summary())

# Add pairwise interactions among treatments in biomarker-high
print('\n=== With pairwise treatment interactions ===')
X['f006xf007'] = X['f006']*X['f007']
X['f006xf039'] = X['f006']*X['f039']
X['f007xf039'] = X['f007']*X['f039']
res2 = sm.Logit(y_hi, X).fit(disp=False, maxiter=100)
print(res2.summary())
