"""Iteration 19: check selection patterns -- is triple-combo more often given to biomarker-high?"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

df = pd.read_parquet('../dataset.parquet')
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)
df['f092_hi'] = (df['feature_092']>=0.5).astype(int)

# Triple-combo prevalence by biomarker
print('=== Triple-combo receipt by biomarker_hi ===')
print(pd.crosstab(df['f092_hi'], df['triple'], margins=True))
print(df.groupby('f092_hi')['triple'].agg(['mean','count']).to_string())
ct = pd.crosstab(df['f092_hi'], df['triple'])
chi2, p, _, _ = chi2_contingency(ct)
print(f'Chi-square p={p:.3g}')

# Continuous biomarker by triple receipt
print('\n=== feature_092 mean by triple ===')
print(df.groupby('triple')['feature_092'].agg(['mean','std','count']).to_string())
t, p = ttest_ind(df.loc[df['triple']==1,'feature_092'], df.loc[df['triple']==0,'feature_092'])
print(f't-test p={p:.3g}')

# Each individual treatment by biomarker
print('\n=== Each treatment receipt by biomarker_hi ===')
for c in ['feature_006','feature_007','feature_039']:
    rate_lo = df.loc[df['f092_hi']==0, c].mean()
    rate_hi = df.loc[df['f092_hi']==1, c].mean()
    print(f'  {c}: hi={rate_hi:.3f}, lo={rate_lo:.3f}')

# By feature_051 (ECOG)
print('\n=== Triple receipt by feature_051 (ECOG-like) ===')
print(df.groupby('feature_051')['triple'].agg(['mean','count']).to_string())

# By feature_013 (poor predictor)
print('\n=== Triple receipt by feature_013 ===')
print(df.groupby('feature_013')['triple'].agg(['mean','count']).to_string())

# By age
df['age_bin'] = pd.cut(df['feature_078'], bins=[30,55,65,75,90], include_lowest=True)
print('\n=== Triple receipt by age band ===')
print(df.groupby('age_bin', observed=True)['triple'].agg(['mean','count']).to_string())
