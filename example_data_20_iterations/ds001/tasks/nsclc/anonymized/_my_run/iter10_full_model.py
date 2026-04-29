"""Iteration 10: comprehensive multivariable logistic model with biomarker × treatment interactions
controlling for all top predictors. Also test feature_011 non-linearity (quartile model)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['f092_hi'] = (df['feature_092'] >= 0.5).astype(int)

# Build dataset: standardized continuous, dummy categoricals
def std(s):
    return (s - s.mean()) / s.std()

X_data = pd.DataFrame({
    'f013': df['feature_013'].astype(float),
    'f067': df['feature_067'].astype(float),
    'f006': df['feature_006'].astype(float),
    'f007': df['feature_007'].astype(float),
    'f039': df['feature_039'].astype(float),
    'f011': std(df['feature_011']),
    'f099': std(df['feature_099']),
    'f063': std(df['feature_063']),
    'f092': std(df['feature_092']),
    'f051': std(df['feature_051']),
    'f092_hi': df['f092_hi'].astype(float),
})
# Drug-biomarker interactions
X_data['f006_x_hi'] = X_data['f006'] * X_data['f092_hi']
X_data['f007_x_hi'] = X_data['f007'] * X_data['f092_hi']
X_data['f039_x_hi'] = X_data['f039'] * X_data['f092_hi']
X_data['squamous'] = (df['feature_043']=='squamous').astype(float)

# Reduce collinearity: drop f092 main (since f092_hi is its binarization)
X_full = sm.add_constant(X_data).astype(float)
res = sm.Logit(y, X_full).fit(disp=False, maxiter=300)
print(res.summary())

# Save
out = pd.DataFrame({'coef': res.params, 'se': res.bse, 'p_value': res.pvalues, 'or': np.exp(res.params)})
out.to_csv('iter10_full_model.csv')

# Test feature_011 non-linearity: compare linear vs quartile parameterization
print('\n=== feature_011 non-linearity test ===')
df['f011_q'] = pd.qcut(df['feature_011'], 4, labels=False, duplicates='drop')
nq = df['f011_q'].nunique()
print(f'Quartiles created: {nq}')
print(df.groupby('f011_q')['objective_response'].agg(['mean','count']))

# Linear model
data_lin = X_data.copy()
res_lin = sm.Logit(y, sm.add_constant(data_lin).astype(float)).fit(disp=False, maxiter=200)
ll_lin = res_lin.llf

# Quartile model -- replace f011 with dummies
data_q = X_data.drop(columns=['f011']).copy()
for q in range(1, nq):
    data_q[f'f011_q{q}'] = (df['f011_q']==q).astype(float)
res_q = sm.Logit(y, sm.add_constant(data_q).astype(float)).fit(disp=False, maxiter=200)
ll_q = res_q.llf
lr = 2*(ll_q - ll_lin)
df_diff = nq - 1 - 1  # k-1 dummies vs 1 linear
p = 1 - chi2_dist.cdf(lr, df=df_diff)
print(f'\nLinear vs quartile: LR={lr:.2f}, df={df_diff}, p={p:.4g}')
print('Quartile dummies:')
for k, v in res_q.params.items():
    if 'f011_q' in k:
        print(f'  {k}: coef={v:.3f}, p={res_q.pvalues[k]:.3g}')
