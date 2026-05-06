"""Iteration 11-13: Test additional candidate treatments and look for any missed signal.

Test feature_022, feature_002, feature_021, feature_005 as potential treatment-like variables
(any binary that could plausibly represent a treatment beyond the obvious 4).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)

# Re-screen for interactions on more candidate Ts
extra_treats = ['feature_022','feature_002','feature_021','feature_011','feature_028','feature_023']
all_X = ['feature_031','feature_023','feature_011','feature_022','feature_016',
         'feature_028','feature_002','feature_005','feature_021','feature_018',
         'feature_020','feature_027','feature_012','hist_squam','smk_current','smk_former',
         'feature_014','feature_015','feature_019','feature_025','feature_032','feature_017',
         'feature_024','feature_010','feature_003','feature_030','feature_008',
         'feature_004','feature_013','feature_026','feature_029','feature_007',
         'feature_033','feature_009']

results = []
cont_set = {'feature_015','feature_019','feature_025','feature_032','feature_017',
            'feature_024','feature_010','feature_003','feature_030','feature_008',
            'feature_004','feature_013','feature_026','feature_029','feature_007',
            'feature_033','feature_009','feature_014'}

for T in extra_treats:
    for X in all_X:
        if X == T: continue
        if X in cont_set:
            xv = (df[X] - df[X].mean()) / df[X].std()
        else:
            xv = df[X].astype(float)
        d = pd.DataFrame({'y': df['pfs_months'].values,
                          'T': df[T].astype(float).values, 'X': xv.values})
        m = smf.ols("y ~ T*X", data=d).fit()
        results.append({'T': T, 'X': X,
                        'beta_T': m.params.get('T'),
                        'beta_TX': m.params.get('T:X'),
                        'p_TX': m.pvalues.get('T:X')})

R = pd.DataFrame(results).sort_values('p_TX')
print("=== Top 20 interactions across extra T candidates ===")
print(R.head(20).to_string(index=False))

# Specifically: any treatment-by-feature interaction with magnitude > 1.0 month?
print("\n=== Large effect-modifier candidates (|beta_TX| > 1) ===")
print(R[abs(R['beta_TX']) > 1].head(20).to_string(index=False))

# Look at feature_015 modulation of feature_018 across the FULL cohort
print("\n=== feature_018 effect by feature_015 quartile (full cohort) ===")
df['f015_q'] = pd.qcut(df['feature_015'], 4, labels=['Q1','Q2','Q3','Q4'])
for q in ['Q1','Q2','Q3','Q4']:
    sub = df[df['f015_q']==q]
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    print(f"  {q}: range=[{sub['feature_015'].min():.1f},{sub['feature_015'].max():.1f}], n={len(sub)}, "
          f"eff={a.mean()-b.mean():+.4f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# feature_018 effect within active subgroup by feature_015 quartile
print("\n=== feature_018 effect by feature_015 quartile WITHIN SG_018=1 ===")
df['SG_018'] = ((df['feature_016']==1) & (df['feature_031']==0) &
                (df['feature_028']==0) & (df['feature_005']==0)).astype(int)
sg = df[df['SG_018']==1]
for q in ['Q1','Q2','Q3','Q4']:
    sub = sg[sg['f015_q']==q]
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    if len(a) > 5 and len(b) > 5:
        print(f"  {q}: n={len(sub)}, mean_T1={a.mean():.3f}, mean_T0={b.mean():.3f}, "
              f"eff={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")

# Comprehensive subgroup search for feature_012 using small subgroups defined by 2 binary gates
print("\n=== Best feature_012 subgroup via 2-binary-gate search ===")
T = 'feature_012'
binaries = ['feature_031','feature_023','feature_011','feature_022','feature_016',
            'feature_028','feature_002','feature_005','feature_021','feature_018',
            'feature_020','feature_027','hist_squam','smk_current','smk_former']
import itertools
best = []
for a, b in itertools.combinations(binaries, 2):
    for va in [0,1]:
        for vb in [0,1]:
            mask = (df[a]==va) & (df[b]==vb)
            if mask.sum() < 200: continue
            sub = df[mask]
            t1 = sub.loc[sub[T]==1, 'pfs_months']
            t0 = sub.loc[sub[T]==0, 'pfs_months']
            if len(t1) < 50 or len(t0) < 50: continue
            eff = t1.mean() - t0.mean()
            p = stats.ttest_ind(t1, t0).pvalue
            best.append((a, va, b, vb, mask.sum(), eff, p))
best.sort(key=lambda x: -abs(x[5]))
for entry in best[:8]:
    print(f"  {entry[0]}={entry[1]} & {entry[2]}={entry[3]}: n={entry[4]}, eff={entry[5]:+.3f}, p={entry[6]:.3e}")
