"""Iteration 13: 3-way interaction test feature_006 × feature_007 × feature_039.
Also test 4-way: × feature_092_hi.
The triple combination cell shows striking ORR=45% in biomarker-high patients."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)

# Three-way interaction in full data (biomarker-blind)
print('=== Three-way interaction (006 × 007 × 039), all patients ===')
data = pd.DataFrame({
    'f006': df['feature_006'].astype(float),
    'f007': df['feature_007'].astype(float),
    'f039': df['feature_039'].astype(float),
})
data['f06_07'] = data['f006']*data['f007']
data['f06_39'] = data['f006']*data['f039']
data['f07_39'] = data['f007']*data['f039']
data['f06_07_39'] = data['f006']*data['f007']*data['f039']
X_full = sm.add_constant(data).astype(float)
X_no3 = X_full.drop(columns=['f06_07_39'])
res_full = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
res_no3 = sm.Logit(y, X_no3).fit(disp=False, maxiter=200)
lr = 2*(res_full.llf - res_no3.llf)
p = 1 - chi2_dist.cdf(lr, df=1)
print(f'3-way coef={res_full.params["f06_07_39"]:.3f}, LR p={p:.3g}')
print(res_full.summary())

# Now test 4-way: triple × biomarker_hi
print('\n=== Four-way: 006 × 007 × 039 × f092_hi ===')
data4 = data.copy()
data4['hi'] = df['f092_hi'].astype(float)
# All 2-way, 3-way and 4-way
for a, b in [('f006','hi'), ('f007','hi'), ('f039','hi')]:
    data4[f'{a}_{b}'] = data4[a]*data4[b]
data4['f06_07_hi'] = data4['f006']*data4['f007']*data4['hi']
data4['f06_39_hi'] = data4['f006']*data4['f039']*data4['hi']
data4['f07_39_hi'] = data4['f007']*data4['f039']*data4['hi']
data4['f06_07_39_hi'] = data4['f06_07_39']*data4['hi']
X4 = sm.add_constant(data4).astype(float)
res4 = sm.Logit(y, X4).fit(disp=False, maxiter=300)
print(res4.summary())

# LR test for 4-way
X4_no = X4.drop(columns=['f06_07_39_hi'])
res4_no = sm.Logit(y, X4_no).fit(disp=False, maxiter=200)
lr4 = 2*(res4.llf - res4_no.llf)
p4 = 1 - chi2_dist.cdf(lr4, df=1)
print(f'\n4-way (triple-combo × biomarker) LR p={p4:.3g}, coef={res4.params["f06_07_39_hi"]:.3f}')

# Predicted and observed ORR by combo × biomarker
print('\n=== Observed ORR by combo × biomarker ===')
combo_full = (df['feature_006'].astype(str) + df['feature_007'].astype(str) + df['feature_039'].astype(str))
ct = df.assign(combo=combo_full).groupby(['f092_hi','combo'])['objective_response'].agg(['mean','count']).unstack(0)
print(ct.to_string())
