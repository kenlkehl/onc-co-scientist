"""Additional targeted iterations: more interactions, dose-response, demographic effects."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

results = {}

# 1. Dose-response curve for feature_057 (test linearity)
print('=== feature_057 dose-response ===')
rates = []
for lvl in sorted(df['feature_057'].unique()):
    msk = df['feature_057']==lvl
    rates.append((lvl, msk.sum(), y[msk].mean()))
print(rates)
# Linear vs categorical fit
X1 = sm.add_constant(df['feature_057'].astype(float))
m1 = sm.Logit(y, X1).fit(disp=0)
ll1 = m1.llf
X2 = sm.add_constant(pd.get_dummies(df['feature_057'], drop_first=True).astype(float))
m2 = sm.Logit(y, X2).fit(disp=0)
ll2 = m2.llf
lrt = 2*(ll2 - ll1)
p_nonlin = 1 - stats.chi2.cdf(lrt, 1)
print(f'LRT linear vs categorical: chi2={lrt:.3f}, df=1, p={p_nonlin:.4f}')

# 2. Test feature_092 — extreme skew, log scale?
print('\n=== feature_092 transformation ===')
print('Range:', df['feature_092'].min(), '-', df['feature_092'].max())
log92 = np.log1p(df['feature_092'])
m = sm.Logit(y, sm.add_constant(log92)).fit(disp=0)
print(f'log(1+feature_092): OR={np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')

# 3. Test feature_063 — also skewed
print('\n=== feature_063 transformation ===')
log63 = np.log1p(df['feature_063'])
m = sm.Logit(y, sm.add_constant(log63)).fit(disp=0)
print(f'log(1+feature_063): OR={np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')

# 4. Sex-like features (look at binary features that have ~50/50 distribution)
print('\n=== Binary features near 50/50 split (check if any is a sex indicator) ===')
binary_cols = [c for c in df.columns if c.startswith('feature_') and df[c].dtype != object and df[c].nunique()==2]
balanced = []
for c in binary_cols:
    p = df[c].mean()
    if 0.3 < p < 0.7:
        balanced.append((c, p, y[df[c]==1].mean(), y[df[c]==0].mean()))
for c, p, r1, r0 in sorted(balanced, key=lambda x: abs(x[1]-0.5))[:15]:
    print(f'  {c}: prev={p:.3f}, resp1={r1:.4f}, resp0={r0:.4f}, diff={r1-r0:+.4f}')

# 5. Race/sex stratified f057 effect
print('\n=== f057 effect within race x sex-like binaries (compute heterogeneity) ===')
# Test if any binary feature acts as effect modifier for f057
print('Binary features modifying f057 effect (interaction p<0.01):')
for c in binary_cols:
    X = pd.DataFrame({
        'f057': df['feature_057'].astype(float),
        'b': df[c].astype(float),
        'inter': df['feature_057'].astype(float)*df[c].astype(float),
    })
    try:
        mm = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        p = mm.pvalues['inter']
        if p < 0.01:
            b = mm.params['inter']
            print(f'  f057 x {c}: interaction beta={b:+.4f}, p={p:.4g}')
    except Exception as e:
        pass

# 6. Binary features modifying f035 effect
print('\n=== Binary features modifying f035 effect (interaction p<0.01) ===')
for c in binary_cols:
    if c == 'feature_035':
        continue
    X = pd.DataFrame({
        'f035': df['feature_035'].astype(float),
        'b': df[c].astype(float),
        'inter': df['feature_035'].astype(float)*df[c].astype(float),
    })
    try:
        mm = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
        p = mm.pvalues['inter']
        if p < 0.01:
            b = mm.params['inter']
            print(f'  f035 x {c}: interaction beta={b:+.4f}, p={p:.4g}')
    except Exception as e:
        pass

# 7. Joint effect of f057 and f011 — additive vs multiplicative
print('\n=== Joint effect: f057 levels x f011 quartiles ===')
df['f011_q'] = pd.qcut(df['feature_011'], 4, duplicates='drop', labels=['Q1','Q2','Q3','Q4'])
ct = df.groupby(['feature_057','f011_q'], observed=True)['objective_response'].agg(['count','mean']).round(4)
print(ct)

# 8. Race-adjusted (maybe race loses significance?)
print('\n=== race main effect (not adjusted) chi-square ===')
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df['feature_005'], y))
print(f'chi2={chi2:.3f}, p={p:.4g}')

# 9. Insurance (feature_087) main effect
print('\n=== insurance (feature_087) chi-square ===')
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df['feature_087'], y))
print(f'chi2={chi2:.3f}, p={p:.4g}')

# 10. f099 x f057 — check if any interesting
print('\n=== Specific tests: response rate by combinations ===')
df['f057_f035'] = df['feature_057'].astype(str) + '_' + df['feature_035'].astype(str)
ct2 = df.groupby('f057_f035')['objective_response'].agg(['count','mean']).round(4)
print(ct2)

# 11. Test 5-class feature_096 effect
print('\n=== feature_096 (5-level) ===')
for lvl in sorted(df['feature_096'].unique()):
    m = df['feature_096']==lvl
    print(f'  {lvl}: n={m.sum()}, response_rate={y[m].mean():.4f}')
m = sm.Logit(y, sm.add_constant(df['feature_096'].astype(float))).fit(disp=0)
print(f'OR per unit: {np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')

# 12. feature_064 (5-level)
print('\n=== feature_064 (5-level) ===')
for lvl in sorted(df['feature_064'].unique()):
    m = df['feature_064']==lvl
    print(f'  {lvl}: n={m.sum()}, response_rate={y[m].mean():.4f}')
m = sm.Logit(y, sm.add_constant(df['feature_064'].astype(float))).fit(disp=0)
print(f'OR per unit: {np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')

# 13. feature_018 (11-level — maybe count?)
print('\n=== feature_018 (11-level) ===')
m = sm.Logit(y, sm.add_constant(df['feature_018'].astype(float))).fit(disp=0)
print(f'OR per unit: {np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')
df['f018_q'] = pd.qcut(df['feature_018'], 4, duplicates='drop')
print(df.groupby('f018_q', observed=True)['objective_response'].agg(['count','mean']))

# 14. feature_042 / feature_125 / feature_045
for c in ['feature_042','feature_125','feature_045']:
    m = sm.Logit(y, sm.add_constant(df[c].astype(float))).fit(disp=0)
    print(f'{c}: OR per unit: {np.exp(m.params.iloc[1]):.4f}, p={m.pvalues.iloc[1]:.4g}')

# 15. Bonferroni-significant binary list
print('\n=== Binary features Bonferroni-significant ===')
from scipy import stats
for c in binary_cols:
    p1 = y[df[c]==1].mean()
    p0 = y[df[c]==0].mean()
    n1 = (df[c]==1).sum()
    n0 = (df[c]==0).sum()
    # 2x2
    n11 = (df.loc[df[c]==1,'objective_response']==1).sum()
    n10 = n1 - n11
    n01 = (df.loc[df[c]==0,'objective_response']==1).sum()
    n00 = n0 - n01
    try:
        chi2, p, _, _ = stats.chi2_contingency([[n11,n10],[n01,n00]])
    except Exception:
        continue
    if p < 0.05/124:
        print(f'  {c}: rate1={p1:.4f}, rate0={p0:.4f}, diff={p1-p0:+.4f}, p={p:.3g}')
