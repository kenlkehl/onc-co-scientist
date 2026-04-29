"""Iteration 9: subgroup analyses by smoking and feature_051 (ordinal, looks like ECOG)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)

# Smoking × treatment × biomarker
print('=== Smoking × biomarker × feature_006 ===')
for c in ['feature_006', 'feature_007', 'feature_039']:
    print(f'\n--- {c} (rows = smoking × biomarker_hi, cols = treatment) ---')
    tab = df.groupby(['feature_057', 'f092_hi', c])['objective_response'].agg(['mean','count']).unstack(c)
    print(tab.to_string())

# feature_051 (ordinal 0/1/2) × treatment_006/007/039 — ECOG subgroup
print('\n=== feature_051 (ordinal) × treatments × biomarker ===')
for c in ['feature_006', 'feature_007', 'feature_039']:
    print(f'\n--- {c} ---')
    tab = df.groupby(['feature_051', 'f092_hi', c])['objective_response'].agg(['mean','count']).unstack(c)
    print(tab.to_string())

# Test interactions: feature_051 × treatment in feature_092_hi subset
print('\n=== feature_051 × treatment in biomarker-high subset ===')
hi = df[df['f092_hi']==1].copy()
y_hi = hi['objective_response'].values
for trt in ['feature_006', 'feature_007', 'feature_039']:
    data = pd.DataFrame({
        'f051': (hi['feature_051'] - hi['feature_051'].mean()) / hi['feature_051'].std(),
        'trt': hi[trt].astype(float),
    })
    data['f051_x_trt'] = data['f051'] * data['trt']
    X = sm.add_constant(data).astype(float)
    res = sm.Logit(y_hi, X).fit(disp=False, maxiter=100)
    print(f'  {trt}: f051_x_trt coef={res.params["f051_x_trt"]:.3f}, p={res.pvalues["f051_x_trt"]:.3g}')

# Test smoking × treatment interaction in biomarker-high subset
print('\n=== smoking × treatment in biomarker-high subset (chi-sq trend) ===')
for trt in ['feature_006', 'feature_007', 'feature_039']:
    print(f'\n--- {trt} ---')
    sub = df[df['f092_hi']==1]
    for sm_status in ['current', 'former', 'never']:
        s = sub[sub['feature_057']==sm_status]
        n0 = (s[trt]==0).sum()
        n1 = (s[trt]==1).sum()
        if n0>20 and n1>20:
            r0 = s.loc[s[trt]==0, 'objective_response'].mean()
            r1 = s.loc[s[trt]==1, 'objective_response'].mean()
            print(f'  {sm_status}: trt0 ORR={r0:.3f} (n={n0}), trt1 ORR={r1:.3f} (n={n1}), diff={r1-r0:+.3f}')
