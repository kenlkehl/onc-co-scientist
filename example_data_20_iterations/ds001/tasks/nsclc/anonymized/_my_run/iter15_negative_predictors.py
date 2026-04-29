"""Iteration 15: characterize negative predictors feature_011, feature_013, feature_067, feature_051.
Test whether they are independent or correlated with each other (e.g., overlapping prognostic markers)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values

# Distributions
print('=== feature_011 distribution ===')
print(df['feature_011'].describe())
print('\nfeature_011 frequencies:')
print(df['feature_011'].value_counts().sort_index().head(15))
print(f'Pct == 0: {(df["feature_011"]==0).mean():.3f}')

# Correlation matrix
print('\n=== Pairwise correlations among neg predictors ===')
sub = df[['feature_011','feature_013','feature_067','feature_051']].astype(float)
print(sub.corr().to_string())

# feature_013 × feature_067 — co-occurrence
print('\n=== feature_013 × feature_067 ===')
print(pd.crosstab(df['feature_013'], df['feature_067']))

# Joint logistic with all 4 to see independent contributions
print('\n=== Joint logistic ===')
X = pd.DataFrame({
    'f011_z': (df['feature_011']-df['feature_011'].mean())/df['feature_011'].std(),
    'f013': df['feature_013'].astype(float),
    'f067': df['feature_067'].astype(float),
    'f051_z': (df['feature_051']-df['feature_051'].mean())/df['feature_051'].std(),
})
X = sm.add_constant(X).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=200)
print(res.summary())

# Test: do feature_013/067 modify the triple-combo benefit?
print('\n=== feature_013 × triple × biomarker_hi ===')
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)
df['f092_hi'] = (df['feature_092']>=0.5).astype(int)
for v in ['feature_013', 'feature_067']:
    data = pd.DataFrame({
        'v': df[v].astype(float),
        'triple': df['triple'].astype(float),
        'hi': df['f092_hi'].astype(float),
    })
    data['triple_hi'] = data['triple']*data['hi']
    data['v_triple'] = data['v']*data['triple']
    data['v_hi'] = data['v']*data['hi']
    data['v_triple_hi'] = data['v']*data['triple']*data['hi']
    X_full = sm.add_constant(data).astype(float)
    X_no = X_full.drop(columns=['v_triple_hi'])
    res_full = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
    res_no = sm.Logit(y, X_no).fit(disp=False, maxiter=200)
    lr = 2*(res_full.llf - res_no.llf)
    p = 1 - chi2_dist.cdf(lr, df=1)
    print(f'  {v}: 3-way coef={res_full.params["v_triple_hi"]:.3f}, p={p:.3g}')

# Stratified ORR
print('\n=== ORR by feature_013 × triple × biomarker_hi ===')
print(df.groupby(['feature_013','triple','f092_hi'])['objective_response'].agg(['mean','count']).unstack(['triple','f092_hi']).to_string())
