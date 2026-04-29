"""Stratified and interaction analyses for top features."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

# ---- Look at feature_057 levels and feature_035 ----
print('feature_057 distribution & response by level:')
for lvl in sorted(df['feature_057'].unique()):
    m = df['feature_057']==lvl
    print(f'  level {lvl}: n={m.sum()}, response_rate={y[m].mean():.4f}')

print('\nfeature_035 (binary) response:')
for lvl in [0,1]:
    m = df['feature_035']==lvl
    print(f'  {lvl}: n={m.sum()}, response_rate={y[m].mean():.4f}')

# ---- feature_057 x feature_035 interaction ----
print('\n=== feature_057 x feature_035 interaction ===')
ct = df.groupby(['feature_057','feature_035'])['objective_response'].agg(['count','mean']).reset_index()
print(ct.to_string())

# Logistic with interaction
X = pd.DataFrame({
    'f057': df['feature_057'].astype(float),
    'f035': df['feature_035'].astype(float),
    'f057_x_f035': df['feature_057'].astype(float) * df['feature_035'].astype(float),
})
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
print('\nMain+interaction model:')
print(m.summary().tables[1])

# ---- All race/ethnicity main effects ----
print('\n=== feature_005 (race/ethnicity) response rates ===')
for lvl in df['feature_005'].unique():
    msk = df['feature_005']==lvl
    print(f'  {lvl}: n={msk.sum()}, response_rate={y[msk].mean():.4f}')

# ---- feature_087 (insurance) ----
print('\n=== feature_087 (insurance) response rates ===')
for lvl in df['feature_087'].unique():
    msk = df['feature_087']==lvl
    print(f'  {lvl}: n={msk.sum()}, response_rate={y[msk].mean():.4f}')

# ---- Multivariable model with top features ----
print('\n=== Multivariable model with top features ===')
X = pd.DataFrame({
    'f057': df['feature_057'].astype(float),
    'f035': df['feature_035'].astype(float),
    'f011': (df['feature_011']-df['feature_011'].mean())/df['feature_011'].std(),
    'f006': (df['feature_006']-df['feature_006'].mean())/df['feature_006'].std(),
    'f099': (df['feature_099']-df['feature_099'].mean())/df['feature_099'].std(),
    'f092': (df['feature_092']-df['feature_092'].mean())/df['feature_092'].std(),
    'f063': (df['feature_063']-df['feature_063'].mean())/df['feature_063'].std(),
})
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
print(m.summary().tables[1])

# ---- Examine f057 effect within strata of f035 ----
print('\n=== f057 effect stratified by f035 ===')
for g in [0,1]:
    sub = df[df['feature_035']==g]
    yy = sub['objective_response']
    xx = sub['feature_057'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'  f035={g}: OR_per_unit_f057 = {OR:.3f}, p = {mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# ---- f035 effect stratified by f057 levels ----
print('\n=== f035 effect stratified by f057 ===')
for lvl in sorted(df['feature_057'].unique()):
    sub = df[df['feature_057']==lvl]
    yy = sub['objective_response']
    xx = sub['feature_035'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'  f057={lvl}: OR_f035 = {OR:.3f}, p = {mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# ---- f035 effect by race ----
print('\n=== f035 effect within race groups ===')
for r in df['feature_005'].unique():
    sub = df[df['feature_005']==r]
    yy = sub['objective_response']
    xx = sub['feature_035'].astype(float)
    if xx.nunique() < 2 or yy.sum() == 0:
        continue
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'  race={r}: OR_f035 = {OR:.3f}, p = {mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# ---- f057 effect by race ----
print('\n=== f057 effect within race groups ===')
for r in df['feature_005'].unique():
    sub = df[df['feature_005']==r]
    yy = sub['objective_response']
    xx = sub['feature_057'].astype(float)
    if xx.nunique() < 2 or yy.sum() == 0:
        continue
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'  race={r}: OR_f057 = {OR:.3f}, p = {mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# ---- Race main effect adjusted for f057 ----
print('\n=== Race effect adjusted for f057 ===')
race_d = pd.get_dummies(df['feature_005'], prefix='race', drop_first=True).astype(float)
X = pd.concat([race_d, df['feature_057'].astype(float).rename('f057')], axis=1)
m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
print(m.summary().tables[1])
