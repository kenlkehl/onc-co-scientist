"""Iterations 16-21: Final explorations for hidden patterns.
- Within unfavorable subgroup, exhaustive 2-feature combinations as 'rescue' subgroups.
- Risk-difference scale modifier check by feature_001 within responsive subgroup.
- Test continuous laboratory features non-linearly (quadratic) on outcome.
- Continuous feature 'cutoff' search for any treatment modifier.
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
fav = ((df['feature_013']==0)&(df['feature_015']==0)&
       (df['feature_021']==0)&(df['feature_027']==0))

# ===== Within unfavorable, search for any 2-binary subgroup with rescued T effect =====
print('=== Within UNFAVORABLE: 2-feature subgroups (binary off + binary off) with T effect ===')
unfav_df = df[~fav].copy()
binary_other = [c for c in df.columns if c.startswith('feature_') and df[c].nunique()==2 and c not in modifiers + [T]]
multi_other = ['feature_001', 'feature_010']
print(f'n_unfav = {len(unfav_df)}')

best_results = []
# Single binary subgroup conditions within unfavorable
for c in binary_other:
    for v in [0, 1]:
        ssub = unfav_df[unfav_df[c]==v]
        if len(ssub) < 200: continue
        p0 = ssub.loc[ssub[T]==0,'objective_response'].mean()
        p1 = ssub.loc[ssub[T]==1,'objective_response'].mean()
        rd = p1 - p0
        n0 = (ssub[T]==0).sum(); n1 = (ssub[T]==1).sum()
        # binomial test
        if n0 > 0 and n1 > 0:
            cnt0 = ssub.loc[ssub[T]==0,'objective_response'].sum()
            cnt1 = ssub.loc[ssub[T]==1,'objective_response'].sum()
            ct = np.array([[cnt0, n0-cnt0],[cnt1, n1-cnt1]])
            try:
                _, pv, _, _ = stats.chi2_contingency(ct)
            except Exception:
                pv = np.nan
            best_results.append((f'{c}={v}', len(ssub), n0, n1, p0, p1, rd, pv))

# Pairs of binary
for a, b in combinations(binary_other, 2):
    for va in [0,1]:
        for vb in [0,1]:
            ssub = unfav_df[(unfav_df[a]==va) & (unfav_df[b]==vb)]
            if len(ssub) < 200: continue
            n0 = (ssub[T]==0).sum(); n1 = (ssub[T]==1).sum()
            if n0 < 50 or n1 < 50: continue
            p0 = ssub.loc[ssub[T]==0,'objective_response'].mean()
            p1 = ssub.loc[ssub[T]==1,'objective_response'].mean()
            rd = p1 - p0
            cnt0 = ssub.loc[ssub[T]==0,'objective_response'].sum()
            cnt1 = ssub.loc[ssub[T]==1,'objective_response'].sum()
            ct = np.array([[cnt0, n0-cnt0],[cnt1, n1-cnt1]])
            try:
                _, pv, _, _ = stats.chi2_contingency(ct)
            except Exception:
                pv = np.nan
            best_results.append((f'{a}={va} & {b}={vb}', len(ssub), n0, n1, p0, p1, rd, pv))

br = pd.DataFrame(best_results, columns=['subgroup','n','n_T0','n_T1','ORR_T0','ORR_T1','RD','p'])
print('\nTop 10 by RD (positive direction):')
print(br.sort_values('RD', ascending=False).head(10).to_string(index=False))
print('\nMax positive RD (any subgroup of size >= 200): ', br['RD'].max())
print()

# ===== Risk-difference scale: T effect by feature_001 within responsive subgroup =====
print('=== Within RESPONSIVE subgroup: ORR by T x feature_001 (RD scale) ===')
sub_fav = df[fav]
g = sub_fav.groupby(['feature_001', T])['objective_response'].agg(['mean','count']).unstack()
g.columns = ['ORR_T0','ORR_T1','n_T0','n_T1']
g['RD'] = g['ORR_T1'] - g['ORR_T0']
print(g.to_string(float_format=lambda x: f'{x:8.4f}'))
print()

# ===== Cutoff search for continuous features within responsive subgroup =====
print('=== Cutoff search: does any continuous feature modify T effect in responsive subgroup? ===')
cont_feats = ['feature_022', 'feature_020', 'feature_024', 'feature_002', 'feature_018',
              'feature_026', 'feature_009', 'feature_029', 'feature_003', 'feature_012',
              'feature_025', 'feature_028', 'feature_007', 'feature_014', 'feature_032',
              'feature_031', 'feature_016']
sub_fav = df[fav].copy()
for c in cont_feats:
    # split at median
    med = sub_fav[c].median()
    s_low = sub_fav[sub_fav[c] <= med]
    s_high = sub_fav[sub_fav[c] > med]
    rd_low = s_low.loc[s_low[T]==1,'objective_response'].mean() - s_low.loc[s_low[T]==0,'objective_response'].mean()
    rd_high = s_high.loc[s_high[T]==1,'objective_response'].mean() - s_high.loc[s_high[T]==0,'objective_response'].mean()
    print(f'{c}: RD low_half={rd_low:+.4f}  high_half={rd_high:+.4f}  diff={rd_high-rd_low:+.4f}')
print()

# ===== Quadratic effect of continuous features in untreated (prognostic) =====
print('=== Quadratic effect check (T=0, untreated) ===')
sub_t0 = df[df[T]==0].copy()
y_t0 = sub_t0['objective_response'].astype(int)
for c in ['feature_022','feature_020','feature_024','feature_002','feature_018','feature_026']:
    x = (sub_t0[c] - sub_t0[c].mean()) / sub_t0[c].std()
    X = pd.DataFrame({'x': x, 'x2': x**2})
    X = sm.add_constant(X.astype(float))
    m = sm.Logit(y_t0, X).fit(disp=False)
    print(f'{c}: x-coef={m.params["x"]:+.4f} (p={m.pvalues["x"]:.3e})  x^2-coef={m.params["x2"]:+.4f} (p={m.pvalues["x2"]:.3e})')
