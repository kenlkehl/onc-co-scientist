"""Iteration 5: Are there ANY other treatment modifiers besides the 4 binaries?
- Within favorable subgroup, screen every other feature for residual T-effect heterogeneity.
- Within unfavorable subgroup, check whether any feature subset 'rescues' a treatment effect.
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
fav = ((df['feature_013']==0) & (df['feature_015']==0) &
       (df['feature_021']==0) & (df['feature_027']==0))
unfav = ~fav

binary_other = [c for c in df.columns if c.startswith('feature_') and df[c].nunique()==2 and c != T and c not in modifiers]
multi_other = ['feature_001', 'feature_010']
cont_other = [c for c in df.columns if c.startswith('feature_') and df[c].nunique() > 10]

def test_interaction(sub, feature, T='feature_008'):
    """LRT for T x feature interaction in subgroup; return p, OR(T) at low/high."""
    s = sub.copy()
    if s[feature].nunique() <= 5:
        s['mod'] = s[feature].astype(float)
        scale_low, scale_high = 0, 1
    else:
        s['mod'] = (s[feature] - s[feature].mean()) / s[feature].std()
        scale_low, scale_high = -1, 1
    s['Tx'] = s[T] * s['mod']
    Xf = sm.add_constant(s[[T,'mod','Tx']].astype(float))
    Xr = sm.add_constant(s[[T,'mod']].astype(float))
    try:
        mf = sm.Logit(s['objective_response'].astype(int), Xf).fit(disp=False)
        mr = sm.Logit(s['objective_response'].astype(int), Xr).fit(disp=False)
        lrt = 2*(mf.llf - mr.llf)
        p = stats.chi2.sf(lrt, 1)
        or_lo = np.exp(mf.params[T] + mf.params['Tx']*scale_low)
        or_hi = np.exp(mf.params[T] + mf.params['Tx']*scale_high)
        return p, or_lo, or_hi
    except Exception:
        return np.nan, np.nan, np.nan

print(f'=== FAVORABLE subgroup (n={fav.sum()}): residual T-effect modifiers ===')
print(f'{"feature":<14}{"OR(T)@low":>12}{"OR(T)@high":>14}{"int p":>14}')
results_fav = []
for c in binary_other + multi_other + cont_other:
    p, lo, hi = test_interaction(df[fav], c, T)
    results_fav.append((c, lo, hi, p))
    print(f'{c:<14}{lo:>12.3f}{hi:>14.3f}{p:>14.3e}')
print()

# Sort by p
print('Top by smallest p in favorable subgroup:')
fav_df = pd.DataFrame(results_fav, columns=['feature','OR_T_lo','OR_T_hi','p']).sort_values('p')
print(fav_df.head(10).to_string(index=False))
print()

print(f'=== UNFAVORABLE subgroup (n={unfav.sum()}): is there any subgroup where T works? ===')
print(f'{"feature":<14}{"OR(T)@low":>12}{"OR(T)@high":>14}{"int p":>14}')
results_unf = []
for c in binary_other + multi_other + cont_other:
    p, lo, hi = test_interaction(df[unfav], c, T)
    results_unf.append((c, lo, hi, p))
    print(f'{c:<14}{lo:>12.3f}{hi:>14.3f}{p:>14.3e}')
print()
print('Top by smallest p in unfavorable subgroup:')
unf_df = pd.DataFrame(results_unf, columns=['feature','OR_T_lo','OR_T_hi','p']).sort_values('p')
print(unf_df.head(10).to_string(index=False))
print()

# ===== Single-feature stratification within unfavorable: any subgroup w/ T effect? =====
print('=== Within unfavorable: ORR by T x each binary (top 5 by interaction p) ===')
for c in unf_df.head(5)['feature'].tolist():
    if df[c].nunique() <= 5:
        sub = df[unfav]
        g = sub.groupby([c, T])['objective_response'].agg(['mean','count']).unstack()
        g.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
        g['risk_diff'] = g['ORR_T1'] - g['ORR_T0']
        print(f'\n{c}:')
        print(g.to_string(float_format=lambda x: f'{x:8.4f}'))
