"""Iteration 8: subgroup analyses.
Test whether biomarker-treatment interactions are modified by histology, smoking, ECOG (f051),
race, sex (likely binary)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values

# Sex column? Find a binary that resembles sex (~50/50). Look at p1 values for binaries:
binary_cols = [c for c in df.columns if df[c].dtype != 'object' and df[c].nunique() == 2 and c not in ('patient_id','objective_response')]
sex_cands = [(c, (df[c]==1).mean()) for c in binary_cols if 0.4 < (df[c]==1).mean() < 0.6]
print('Binary cols with prevalence 40-60% (candidate sex/race):')
for c, p in sorted(sex_cands, key=lambda x: -abs(x[1]-0.5)):
    print(f'  {c}: prev={p:.3f}')
print()

# Look at histology effect on treatment-biomarker interaction
print('\n=== ORR by histology × feature_006 × feature_092_high ===')
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)
for trt in ['feature_006', 'feature_007', 'feature_039']:
    print(f'\n--- {trt} ---')
    tab = df.groupby(['feature_043', 'f092_hi', trt])['objective_response'].agg(['mean','count']).unstack(trt)
    print(tab.to_string())

# 3-way interaction: feature_043 × feature_092_high × treatment
# Use logistic regression with interaction terms
print('\n=== 3-way interaction tests (treatment × f092_hi × histology) ===')
df['squamous'] = (df['feature_043'] == 'squamous').astype(int)
for trt in ['feature_006', 'feature_007', 'feature_039']:
    data = pd.DataFrame({
        'sq': df['squamous'].values,
        'hi': df['f092_hi'].values,
        'trt': df[trt].values,
    })
    data['sq_hi'] = data['sq']*data['hi']
    data['sq_trt'] = data['sq']*data['trt']
    data['hi_trt'] = data['hi']*data['trt']
    data['sq_hi_trt'] = data['sq']*data['hi']*data['trt']
    X_full = sm.add_constant(data).astype(float)
    res_full = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
    # Without 3-way
    X_no3 = X_full.drop(columns=['sq_hi_trt'])
    res_no3 = sm.Logit(y, X_no3).fit(disp=False, maxiter=200)
    lr = 2*(res_full.llf - res_no3.llf)
    p = 1 - chi2_dist.cdf(lr, df=1)
    print(f'  {trt}: 3-way coef={res_full.params["sq_hi_trt"]:.3f}, LR p={p:.3g}')
