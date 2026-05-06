"""Iteration 6: Check feature_001 (ordinal 0/1/2) as a possible 5th modifier."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
T = 'feature_008'

# Within full cohort, ORR by T x feature_001
print('=== Full cohort: ORR by T x feature_001 ===')
g = df.groupby(['feature_001', T])['objective_response'].agg(['mean','count']).unstack()
g.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
g['risk_diff'] = g['ORR_T1'] - g['ORR_T0']
g['rel_risk'] = g['ORR_T1']/g['ORR_T0']
print(g.to_string(float_format=lambda x: f'{x:8.4f}'))
print()

# Within favorable subgroup
fav = ((df['feature_013']==0) & (df['feature_015']==0) &
       (df['feature_021']==0) & (df['feature_027']==0))
print('=== Favorable subgroup: ORR by T x feature_001 ===')
g = df[fav].groupby(['feature_001', T])['objective_response'].agg(['mean','count']).unstack()
g.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
g['risk_diff'] = g['ORR_T1'] - g['ORR_T0']
g['rel_risk'] = g['ORR_T1']/g['ORR_T0']
print(g.to_string(float_format=lambda x: f'{x:8.4f}'))
print()

# Treat feature_001 as ordinal continuous within full cohort
print('=== Logistic, feature_001 as ordinal x T (full cohort) ===')
df['Tx_f001'] = df[T] * df['feature_001']
X = sm.add_constant(df[[T,'feature_001','Tx_f001']].astype(float))
m = sm.Logit(df['objective_response'].astype(int), X).fit(disp=False)
print(m.summary())
or_table = pd.DataFrame({'beta': m.params,'OR': np.exp(m.params),'p': m.pvalues})
print(or_table.round(4))
print()

# Testing whether including feature_001 as a 5th modifier improves the joint subgroup model
print('=== Adding feature_001 to the joint modifier model ===')
modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']
X = df[[T] + modifiers + ['feature_001']].copy()
for m_ in modifiers:
    X[f'T_x_{m_}'] = X[T] * X[m_]
X['T_x_f001'] = X[T] * X['feature_001']
X = sm.add_constant(X.astype(float))
m = sm.Logit(df['objective_response'].astype(int), X).fit(disp=False)
print(m.summary())
print()

# Within favorable, does feature_001 still modify T?
print('=== Favorable: T x feature_001 LRT ===')
sub = df[fav].copy()
sub['Tx_f001'] = sub[T] * sub['feature_001']
Xf = sm.add_constant(sub[[T,'feature_001','Tx_f001']].astype(float))
Xr = sm.add_constant(sub[[T,'feature_001']].astype(float))
mf = sm.Logit(sub['objective_response'].astype(int), Xf).fit(disp=False)
mr = sm.Logit(sub['objective_response'].astype(int), Xr).fit(disp=False)
lrt = 2*(mf.llf - mr.llf)
p_int = stats.chi2.sf(lrt, 1)
print(f'LRT for T x feature_001 in favorable subgroup: chi2={lrt:.3f}, p={p_int:.3e}')
print('Coefficients of full model:')
print(pd.DataFrame({'beta':mf.params,'OR':np.exp(mf.params),'p':mf.pvalues}).round(4))
