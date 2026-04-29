"""Iteration 6: dissect feature_092 × {006,007,039} synergy.
Hypothesis: feature_092 is a continuous biomarker (e.g., PD-L1) and 006/007/039 may be IO treatments
whose efficacy depends on biomarker level."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

# Stratified ORR by feature_092 quartile, separately for feature_006/007/039 = 0 vs 1
df['f092_q'] = pd.qcut(df['feature_092'], 4, labels=['Q1','Q2','Q3','Q4'])

for trt in ['feature_006', 'feature_007', 'feature_039']:
    print(f'\n=== feature_092 quartile × {trt} ===')
    tab = df.groupby(['f092_q', trt], observed=True)['objective_response'].agg(['mean','count']).unstack(trt)
    print(tab.to_string())

# Check: maybe these binary features are mutually exclusive treatments?
print('\n=== Joint table 006/007/039 ===')
tab = df.groupby(['feature_006', 'feature_007', 'feature_039']).size()
print(tab.to_string())
print(f'\nTotal: {tab.sum()}')

# Check: relationship between 006/007/039 and feature_013/067 (the negative binaries) -- mutually exclusive?
print('\n=== feature_013 × feature_006 ===')
print(pd.crosstab(df['feature_013'], df['feature_006']))
print('\n=== feature_067 × feature_006 ===')
print(pd.crosstab(df['feature_067'], df['feature_006']))

# Look at feature_092 distribution overall and within key subgroups
print('\n=== feature_092 overall stats ===')
print(df['feature_092'].describe())
# Is it bimodal?
hist, edges = np.histogram(df['feature_092'], bins=20)
for h, e1, e2 in zip(hist, edges[:-1], edges[1:]):
    print(f'  [{e1:.3f}, {e2:.3f}): {h}')
