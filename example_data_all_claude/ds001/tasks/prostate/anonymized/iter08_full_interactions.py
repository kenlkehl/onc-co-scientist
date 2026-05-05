"""Iteration 8: Full saturated model — does an additive (linear) interaction
with the 4 modifiers fit as well as the saturated cell-by-cell model?
Also: are the modifiers themselves prognostic (apart from treatment effect)?
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
T = 'feature_008'
modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']
y = df['objective_response'].astype(int)

# Reduced (additive) model: T + 4 modifiers + 4 (T*modifier)
X_red = df[[T] + modifiers].copy()
for m in modifiers:
    X_red[f'T_x_{m}'] = X_red[T] * X_red[m]
X_red = sm.add_constant(X_red.astype(float))
m_red = sm.Logit(y, X_red).fit(disp=False)

# Full saturated model on the 32 (= 2 x 2^4) cells: every interaction
# Construct dummies for each cell pattern of (T, m1, m2, m3, m4)
df['cell'] = (df[T].astype(str) + '_' +
              df['feature_013'].astype(str) +
              df['feature_015'].astype(str) +
              df['feature_021'].astype(str) +
              df['feature_027'].astype(str))
print('Cell counts:')
print(df['cell'].value_counts().sort_index().head(40))
print()
# Saturated logistic with all-pairs and higher-order interactions among the 5 binary
# Quick way: fit cell means and compare deviance
n_per = df.groupby('cell')['objective_response'].count()
y_per = df.groupby('cell')['objective_response'].sum()
p_per = y_per / n_per
ll_sat = (y_per * np.log(p_per.replace(0, 1)) + (n_per - y_per)*np.log((1-p_per).replace(0, 1))).sum()
print(f'Log-likelihood, saturated: {ll_sat:.2f}')
print(f'Log-likelihood, additive interaction model: {m_red.llf:.2f}')
print(f'LL difference (saturated - additive): {ll_sat - m_red.llf:.2f}')
df_diff = (2**5 - 1) - (1 + 4 + 1 + 4)  # 31 - 10 = 21 extra cell parameters
print(f'df: {df_diff}')
chi2_diff = 2*(ll_sat - m_red.llf)
print(f'LRT chi2={chi2_diff:.2f}, p={stats.chi2.sf(chi2_diff, df_diff):.3e}')
print()

# ===== Check: are modifiers prognostic in T=0 group? =====
print('=== Prognostic effect of modifiers in T=0 group (no treatment) ===')
sub = df[df[T]==0].copy()
print(f'n_T0 = {len(sub)}')
for m in modifiers:
    p0 = sub.loc[sub[m]==0,'objective_response'].mean()
    p1 = sub.loc[sub[m]==1,'objective_response'].mean()
    ct = pd.crosstab(sub[m], sub['objective_response'])
    chi2, pv, _, _ = stats.chi2_contingency(ct)
    print(f'  {m}: ORR(=0)={p0:.4f}, ORR(=1)={p1:.4f}, diff={p1-p0:+.4f}, p={pv:.3e}')
print()

# In T=1 group:
print('=== Effect of modifiers in T=1 group (treated) ===')
sub = df[df[T]==1].copy()
print(f'n_T1 = {len(sub)}')
for m in modifiers:
    p0 = sub.loc[sub[m]==0,'objective_response'].mean()
    p1 = sub.loc[sub[m]==1,'objective_response'].mean()
    ct = pd.crosstab(sub[m], sub['objective_response'])
    chi2, pv, _, _ = stats.chi2_contingency(ct)
    print(f'  {m}: ORR(=0)={p0:.4f}, ORR(=1)={p1:.4f}, diff={p1-p0:+.4f}, p={pv:.3e}')
