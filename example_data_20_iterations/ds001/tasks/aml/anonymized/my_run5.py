"""Drill into the f035 interactions."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

# f035 x feature_027 interaction
print('=== f035 x feature_027 cross-tab ===')
tab = df.groupby(['feature_027','feature_035'])['objective_response'].agg(['count','mean']).round(4)
print(tab)

# Stratified analysis
for f027 in [0,1]:
    sub = df[df['feature_027']==f027]
    yy = sub['objective_response']
    xx = sub['feature_035'].astype(float)
    if xx.nunique() < 2:
        continue
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'feature_027={f027}: f035 effect OR={OR:.3f}, p={mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# f035 x feature_014
print('\n=== f035 x feature_014 cross-tab ===')
tab = df.groupby(['feature_014','feature_035'])['objective_response'].agg(['count','mean']).round(4)
print(tab)
for f014 in [0,1]:
    sub = df[df['feature_014']==f014]
    yy = sub['objective_response']
    xx = sub['feature_035'].astype(float)
    if xx.nunique() < 2:
        continue
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    OR = np.exp(mm.params.iloc[1])
    print(f'feature_014={f014}: f035 effect OR={OR:.3f}, p={mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# Joint feature_027 + feature_035: response in 4 cells
print('\n=== Joint f027+f035 cells ===')
df['joint'] = df['feature_027'].astype(str) + '_' + df['feature_035'].astype(str)
print(df.groupby('joint')['objective_response'].agg(['count','mean']))

# f035 x feature_058
print('\n=== f035 x feature_058 cross-tab ===')
tab = df.groupby(['feature_058','feature_035'])['objective_response'].agg(['count','mean']).round(4)
print(tab)

# f035 x feature_124
print('\n=== f035 x feature_124 cross-tab ===')
tab = df.groupby(['feature_124','feature_035'])['objective_response'].agg(['count','mean']).round(4)
print(tab)

# f057 dose-response was linear (p=0.86), good. So treat as linear/ordinal.

# Check feature_027 main effect
print('\n=== feature_027 main effect ===')
m = sm.Logit(y, sm.add_constant(df['feature_027'].astype(float))).fit(disp=0)
print(f'OR={np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')
# stratified by f035
for f035 in [0,1]:
    sub = df[df['feature_035']==f035]
    yy = sub['objective_response']
    xx = sub['feature_027'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    print(f'  f035={f035}: f027 OR={np.exp(mm.params.iloc[1]):.3f}, p={mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# Check feature_014 main effect by f035
print('\n=== feature_014 stratified by f035 ===')
for f035 in [0,1]:
    sub = df[df['feature_035']==f035]
    yy = sub['objective_response']
    xx = sub['feature_014'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    print(f'  f035={f035}: f014 OR={np.exp(mm.params.iloc[1]):.3f}, p={mm.pvalues.iloc[1]:.3g}, n={len(sub)}')

# Look at interaction: continuous-feature interactions screening
print('\n=== Continuous-feature interactions screening ===')
cont_cols = [c for c in df.columns if c.startswith('feature_') and df[c].dtype.kind == 'f']
# Standardize all
zs = {}
for c in cont_cols:
    zs[c] = (df[c]-df[c].mean())/df[c].std()
for c1 in ['feature_011','feature_006','feature_099','feature_063','feature_092']:
    for c2 in cont_cols:
        if c2 == c1:
            continue
        # f057 x cont, but here we test pairwise continuous interaction beyond main
        X = pd.DataFrame({
            'a': zs[c1],
            'b': zs[c2],
            'inter': zs[c1]*zs[c2],
        })
        try:
            mm = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
            p = mm.pvalues['inter']
            if p < 0.001:
                b = mm.params['inter']
                print(f'  {c1} x {c2}: interaction beta={b:+.4f}, p={p:.4g}')
        except Exception:
            pass

# f057 x f035 x f027 three-way? Maybe later
# Confirm: 3-level f057 dose response by f011 strata
print('\n=== f057 effect by f011 quartiles ===')
df['f011_q'] = pd.qcut(df['feature_011'], 4, duplicates='drop')
qs = df['f011_q'].cat.categories
for q in qs:
    sub = df[df['f011_q']==q]
    yy = sub['objective_response']
    xx = sub['feature_057'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    print(f'  f011 in {q}: f057 OR_per_unit={np.exp(mm.params.iloc[1]):.3f}, p={mm.pvalues.iloc[1]:.3g}')

# f057 dose-response with categorical to verify
print('\n=== f057 effect by f006 quartiles ===')
df['f006_q'] = pd.qcut(df['feature_006'], 4, duplicates='drop')
for q in df['f006_q'].cat.categories:
    sub = df[df['f006_q']==q]
    yy = sub['objective_response']
    xx = sub['feature_057'].astype(float)
    mm = sm.Logit(yy, sm.add_constant(xx)).fit(disp=0)
    print(f'  f006 in {q}: f057 OR={np.exp(mm.params.iloc[1]):.3f}, p={mm.pvalues.iloc[1]:.3g}')
