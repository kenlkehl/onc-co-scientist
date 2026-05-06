"""Iteration 4: Joint subgroup definition for treatment-effect heterogeneity.
- Estimate treatment (feature_008) effect within combinations of modifiers.
- Look for the largest 'sweet spot' subgroup with biggest treatment effect.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
T = 'feature_008'
y = df['objective_response'].astype(int)

modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']
print('=== Joint multivariable model: T x each modifier (all in one model) ===')
X = df[[T] + modifiers].copy()
for m in modifiers:
    X[f'T_x_{m}'] = X[T] * X[m]
X = sm.add_constant(X.astype(float))
model = sm.Logit(y, X).fit(disp=False)
print(model.summary())
print()
print('=== ORs ===')
or_table = pd.DataFrame({'beta': model.params, 'OR': np.exp(model.params),
                         'OR_LCL': np.exp(model.conf_int()[0]),
                         'OR_UCL': np.exp(model.conf_int()[1]),
                         'p': model.pvalues})
print(or_table.round(4))
print()

# ===== Stratified ORR by combinations of all 4 binary modifiers (16 cells) =====
print('=== ORR by treatment status x combination of binary modifiers ===')
df['mod_pattern'] = (df['feature_013'].astype(str) +
                    df['feature_015'].astype(str) +
                    df['feature_021'].astype(str) +
                    df['feature_027'].astype(str))
g = df.groupby(['mod_pattern', T])['objective_response'].agg(['mean','count']).unstack()
g.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
g['risk_diff'] = g['ORR_T1'] - g['ORR_T0']
g['rel_risk'] = g['ORR_T1'] / g['ORR_T0'].replace(0, np.nan)
g = g.sort_values('rel_risk', ascending=False)
print(g.to_string(float_format=lambda x: f'{x:8.4f}'))
print()

# ===== Define 'favorable' subgroup: all 4 modifiers = 0 + low feature_022 =====
fav_binary = ((df['feature_013']==0) & (df['feature_015']==0) &
              (df['feature_021']==0) & (df['feature_027']==0))
print(f'Favorable on 4 binaries: n={fav_binary.sum()} ({fav_binary.mean()*100:.1f}%)')
sub = df[fav_binary]
ct = pd.crosstab(sub[T], sub['objective_response'])
print(ct)
p0 = sub.loc[sub[T]==0,'objective_response'].mean()
p1 = sub.loc[sub[T]==1,'objective_response'].mean()
chi2, pv, _, _ = stats.chi2_contingency(ct)
print(f'ORR T=0: {p0:.4f}  T=1: {p1:.4f}  diff: {p1-p0:+.4f}  chi-square p={pv:.3e}')
print()

# ===== Within favorable binary subgroup, does feature_022 still modify? =====
print('=== Within favorable binary subgroup, T x log(feature_022) interaction ===')
sub2 = df[fav_binary].copy()
sub2['log_f022'] = np.log1p(sub2['feature_022'])
sub2['log_f022_z'] = (sub2['log_f022'] - sub2['log_f022'].mean()) / sub2['log_f022'].std()
sub2['Tx'] = sub2[T] * sub2['log_f022_z']
Xs = sm.add_constant(sub2[[T, 'log_f022_z', 'Tx']].astype(float))
ms = sm.Logit(sub2['objective_response'].astype(int), Xs).fit(disp=False)
print(ms.summary())
print(f"OR(T) at -1 SD log_f022: {np.exp(ms.params[T] - ms.params['Tx']):.3f}")
print(f"OR(T) at +1 SD log_f022: {np.exp(ms.params[T] + ms.params['Tx']):.3f}")
print()

# Try cuts of feature_022 within favorable binary subgroup
print('=== ORR by T x feature_022 quartile within favorable subgroup ===')
sub2['f022_q'] = pd.qcut(sub2['feature_022'], 4, labels=['Q1(low)','Q2','Q3','Q4(high)'])
gg = sub2.groupby(['f022_q', T])['objective_response'].agg(['mean','count']).unstack()
gg.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
gg['risk_diff'] = gg['ORR_T1'] - gg['ORR_T0']
gg['rel_risk'] = gg['ORR_T1'] / gg['ORR_T0'].replace(0, np.nan)
print(gg.to_string(float_format=lambda x: f'{x:8.4f}'))
