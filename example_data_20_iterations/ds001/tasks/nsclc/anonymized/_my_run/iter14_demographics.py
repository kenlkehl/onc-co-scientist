"""Iteration 14: demographic disparities in (a) overall ORR, (b) likelihood of receiving triple combo,
(c) treatment effect.
Race (feature_123), insurance (feature_005), age (feature_078)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)
df['triple'] = ((df['feature_006']==1) & (df['feature_007']==1) & (df['feature_039']==1)).astype(int)

# 1) Triple combo prevalence by race
print('=== Triple combo receipt by race ===')
tab = df.groupby('feature_123')['triple'].agg(['mean','count'])
print(tab.to_string())
ct = pd.crosstab(df['feature_123'], df['triple'])
chi2, p, _, _ = chi2_contingency = __import__('scipy.stats',fromlist=['chi2_contingency']).chi2_contingency(ct)
print(f'Chi-square p={p:.3g}')

# 2) Triple combo prevalence by insurance
print('\n=== Triple combo receipt by insurance ===')
tab = df.groupby('feature_005')['triple'].agg(['mean','count'])
print(tab.to_string())
ct = pd.crosstab(df['feature_005'], df['triple'])
chi2, p, _, _ = __import__('scipy.stats',fromlist=['chi2_contingency']).chi2_contingency(ct)
print(f'Chi-square p={p:.3g}')

# 3) Biomarker-high prevalence by race
print('\n=== feature_092 (biomarker) mean by race ===')
print(df.groupby('feature_123')['feature_092'].agg(['mean','count']).to_string())
print('\n=== feature_092_hi prevalence by race ===')
print(df.groupby('feature_123')['f092_hi'].agg(['mean','count']).to_string())
ct = pd.crosstab(df['feature_123'], df['f092_hi'])
chi2, p, _, _ = __import__('scipy.stats',fromlist=['chi2_contingency']).chi2_contingency(ct)
print(f'Chi-square p={p:.3g}')

# 4) ORR adjusted by race within triple combo + biomarker high
print('\n=== ORR among triple-combo + biomarker-high by race ===')
sub = df[(df['triple']==1) & (df['f092_hi']==1)]
print(f'Subgroup size: {len(sub)}')
print(sub.groupby('feature_123')['objective_response'].agg(['mean','count']).to_string())

# 5) Race adjusted multivariable
print('\n=== Multivariable: ORR ~ triple + biomarker_hi + triple*biomarker_hi + race + insurance + age ===')
df_m = df.copy()
df_m['age_z'] = (df_m['feature_078']-df_m['feature_078'].mean())/df_m['feature_078'].std()
race_d = pd.get_dummies(df_m['feature_123'], prefix='race', drop_first=True).astype(int)
ins_d = pd.get_dummies(df_m['feature_005'], prefix='ins', drop_first=True).astype(int)
X = pd.concat([df_m[['triple','f092_hi','age_z']].astype(float), race_d, ins_d], axis=1)
X['triple_x_hi'] = X['triple']*X['f092_hi']
X = sm.add_constant(X).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(res.summary())

# 6) Test interaction race × treatment
print('\n=== Race × triple × biomarker_hi interaction ===')
df_m['white'] = (df_m['feature_123']=='white').astype(int)
data = pd.DataFrame({
    'white': df_m['white'].astype(float),
    'triple': df_m['triple'].astype(float),
    'hi': df_m['f092_hi'].astype(float),
})
data['triple_hi'] = data['triple']*data['hi']
data['white_triple'] = data['white']*data['triple']
data['white_hi'] = data['white']*data['hi']
data['white_triple_hi'] = data['white']*data['triple']*data['hi']
X = sm.add_constant(data).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
X_no = X.drop(columns=['white_triple_hi'])
res_no = sm.Logit(y, X_no).fit(disp=False, maxiter=200)
lr = 2*(res.llf - res_no.llf)
p = 1 - chi2_dist.cdf(lr, df=1)
print(f'3-way interaction: coef={res.params["white_triple_hi"]:.3f}, LR p={p:.3g}')
