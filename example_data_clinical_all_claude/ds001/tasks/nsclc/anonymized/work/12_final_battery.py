"""Iteration 12-15: Final analyses to lock in findings.

(a) Adjusted models within SG_018 to confirm subgroup effect with covariates.
(b) Confirm feature_012 weak negative effect in feature_016=1 patients.
(c) Search for any tertiary treatment effect.
(d) Histology and stage interactions.
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

# Distribution of treatments by candidate biomarker
print("=== feature_018 prevalence by (feature_016, feature_031) ===")
print(df.groupby(['feature_016','feature_031'])['feature_018'].agg(['mean','count']))
print("\n(should be ~0.35 everywhere if randomly allocated)")

# (a) Adjusted feature_018 effect within SG_018=1 with full covariates
print("\n=== Adjusted feature_018 effect WITHIN SG_018=1 ===")
df['SG_018'] = ((df['feature_016']==1) & (df['feature_031']==0) &
                (df['feature_028']==0) & (df['feature_005']==0)).astype(int)
sg = df[df['SG_018']==1].copy()
m = smf.ols("pfs_months ~ feature_018 + feature_015 + feature_014 + feature_023 + feature_011 + "
            "feature_017 + feature_025 + hist_squam + smk_current + smk_former + feature_021",
            data=sg).fit()
print(m.summary().tables[1])

# (b) feature_012 in feature_016=1 patients (and subgroup of f016=1, f023=0)
print("\n=== feature_012 effect in select subgroups (adjusted) ===")
for desc, mask in [('f016=1', df['feature_016']==1),
                   ('f016=1 & f023=0', (df['feature_016']==1) & (df['feature_023']==0)),
                   ('f016=1 & f031=0', (df['feature_016']==1) & (df['feature_031']==0))]:
    sub = df[mask].copy()
    m = smf.ols("pfs_months ~ feature_012 + feature_015 + feature_014 + feature_023 + feature_011 + "
                "feature_017 + feature_025 + feature_018 + hist_squam + smk_current + smk_former",
                data=sub).fit()
    p = m.params.get('feature_012')
    pval = m.pvalues.get('feature_012')
    se = m.bse.get('feature_012')
    print(f"  {desc}: n={len(sub)}, beta_012={p:+.4f} (SE {se:.4f}), p={pval:.3e}")

# (c) Tertiary search: among continuous lab values, do any modify the effect of feature_018 in SG_018=0?
print("\n=== Looking for hidden interaction outside SG_018 ===")
nonsg = df[df['SG_018']==0]
cont_set = ['feature_015','feature_019','feature_025','feature_032','feature_017',
            'feature_024','feature_010','feature_003','feature_030','feature_008',
            'feature_004','feature_013','feature_026','feature_029','feature_007',
            'feature_033','feature_009']
results = []
for X in cont_set:
    xv = (nonsg[X] - nonsg[X].mean()) / nonsg[X].std()
    d = pd.DataFrame({'y': nonsg['pfs_months'].values,
                      'T': nonsg['feature_018'].astype(float).values, 'X': xv.values})
    m = smf.ols("y ~ T*X", data=d).fit()
    results.append({'X': X, 'beta_T': m.params.get('T'), 'beta_TX': m.params.get('T:X'),
                    'p_TX': m.pvalues.get('T:X')})
print(pd.DataFrame(results).sort_values('p_TX').head(8).to_string(index=False))

# (d) Histology and stage interactions for feature_018 within active subgroup
print("\n=== feature_018 effect within SG_018=1 by histology and stage ===")
sg = df[df['SG_018']==1]
for f06 in ['adenocarcinoma','squamous']:
    sub = sg[sg['feature_006']==f06]
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    if len(a)>10 and len(b)>10:
        print(f"  {f06}: n={len(sub)}, eff={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")
for f23 in [0,1]:
    sub = sg[sg['feature_023']==f23]
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    if len(a)>10 and len(b)>10:
        print(f"  f023={f23}: n={len(sub)}, eff={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")
for sm in ['never','former','current']:
    sub = sg[sg['feature_001']==sm]
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    if len(a)>10 and len(b)>10:
        print(f"  smk={sm}: n={len(sub)}, eff={a.mean()-b.mean():+.3f}, p={stats.ttest_ind(a,b).pvalue:.3e}")
