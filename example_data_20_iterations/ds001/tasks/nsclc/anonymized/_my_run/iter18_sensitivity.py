"""Iteration 18: sensitivity analyses.
- Alternative biomarker thresholds for triple-combo effect.
- Verify that the (006/007/039) triple is the unique synergistic triplet (test other triples).
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)

# 1) Alternative biomarker thresholds
print('=== Triple-combo benefit at different feature_092 thresholds ===')
for th in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    hi = (df['feature_092'] >= th).astype(int)
    n_hi = hi.sum()
    if n_hi < 100: continue
    tab = df[df['triple']==1].assign(hi=hi[df['triple']==1].values).groupby('hi')['objective_response'].agg(['mean','count'])
    untreated = df[df['triple']==0].assign(hi=hi[df['triple']==0].values).groupby('hi')['objective_response'].agg(['mean','count'])
    if 1 in tab.index and 1 in untreated.index:
        diff_hi = tab.loc[1,'mean'] - untreated.loc[1,'mean']
        diff_lo = tab.loc[0,'mean'] - untreated.loc[0,'mean']
    else:
        diff_hi = diff_lo = np.nan
    # Test interaction
    data = pd.DataFrame({'hi': hi.astype(float), 'tr': df['triple'].astype(float)})
    data['hi_tr'] = data['hi']*data['tr']
    X = sm.add_constant(data).astype(float)
    res = sm.Logit(y, X).fit(disp=False, maxiter=100)
    print(f'  thresh={th:.2f}, n_hi={n_hi}: diff_lo={diff_lo:+.3f}, diff_hi={diff_hi:+.3f}, '
          f'interaction_OR={np.exp(res.params["hi_tr"]):.2f}, p={res.pvalues["hi_tr"]:.3g}')

# 2) Search all triplet combinations among top binaries to confirm 006/007/039 is unique
print('\n=== Other triple combinations among top binaries with biomarker_hi (4-way interaction) ===')
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)
candidates = ['feature_006','feature_007','feature_039','feature_013','feature_067',
              'feature_007','feature_115','feature_086','feature_039','feature_021']
candidates = list(dict.fromkeys(candidates))  # unique

results = []
for a, b, c in combinations(candidates, 3):
    trip = ((df[a]==1)&(df[b]==1)&(df[c]==1)).astype(int)
    if trip.sum() < 100: continue
    data = pd.DataFrame({'tr': trip.astype(float), 'hi': df['f092_hi'].astype(float)})
    data['tr_hi'] = data['tr']*data['hi']
    X = sm.add_constant(data).astype(float)
    try:
        res = sm.Logit(y, X).fit(disp=False, maxiter=100)
        results.append({
            'triple': f'{a}|{b}|{c}', 'n_triple': int(trip.sum()),
            'inter_coef': res.params['tr_hi'], 'inter_p': res.pvalues['tr_hi'],
        })
    except Exception:
        pass

res_df = pd.DataFrame(results).sort_values('inter_p')
print(res_df.head(10).to_string(index=False))

# 3) Logistic with separately-included terms 006/007/039 and their pairwise + 3-way interaction
# Already done. Confirm.

# 4) Predicted vs observed ORR validation in a few stratified bins
print('\n=== Cross-tab: triple-combo × biomarker_hi (sanity check) ===')
ct = df.groupby(['triple','f092_hi'])['objective_response'].agg(['mean','count']).unstack('f092_hi')
print(ct.to_string())
