"""Iteration 5: Refine enzalutamide-responsive subgroup definition.

Examine joint modifiers within mcrpc=0 strata, and full multi-feature subgroup test.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')

print("=== Stratified enzalutamide effect within mcrpc=0 subgroup ===")
sub = df[df['mcrpc']==0].copy()
print(f"n(mcrpc=0)={len(sub)}")
for f in ['ar_v7_positive','brca2_mutation','msi_high','psma_high','visceral_mets']:
    for v in [0,1]:
        s = sub[sub[f]==v]
        on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
        off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
        if len(on) and len(off):
            print(f"  {f}={v}: n={len(s)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")

print("\n=== Stratified enzalutamide effect within mcrpc=0 AND ar_v7=0 ===")
sub = df[(df['mcrpc']==0) & (df['ar_v7_positive']==0)].copy()
print(f"n(mcrpc=0, ar_v7=0)={len(sub)}")
for f in ['brca2_mutation','msi_high','psma_high','visceral_mets']:
    for v in [0,1]:
        s = sub[sub[f]==v]
        on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
        off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
        if len(on) and len(off):
            print(f"  {f}={v}: n={len(s)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")

print("\n=== mcrpc=0, ar_v7=0, brca2=0 ===")
sub = df[(df['mcrpc']==0) & (df['ar_v7_positive']==0) & (df['brca2_mutation']==0)].copy()
print(f"n={len(sub)}")
for f in ['msi_high','psma_high']:
    for v in [0,1]:
        s = sub[sub[f]==v]
        on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
        off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
        if len(on) and len(off):
            print(f"  {f}={v}: n={len(s)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")

print("\n=== mcrpc=0, ar_v7=0, brca2=0, msi_high=0 ===")
sub = df[(df['mcrpc']==0) & (df['ar_v7_positive']==0) & (df['brca2_mutation']==0) & (df['msi_high']==0)].copy()
print(f"n={len(sub)}")
on = sub.loc[sub['treatment_enzalutamide']==1,'objective_response']
off = sub.loc[sub['treatment_enzalutamide']==0,'objective_response']
print(f"  enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")
chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(sub['treatment_enzalutamide'], sub['objective_response']))
print(f"  chi2 p={p:.3g}")

print("\n=== Same subgroup, by PSA quartile ===")
sub['psa_q'] = pd.qcut(sub['psa_ng_ml'], 4, duplicates='drop')
for q, g in sub.groupby('psa_q', observed=True):
    on = g.loc[g['treatment_enzalutamide']==1,'objective_response']
    off = g.loc[g['treatment_enzalutamide']==0,'objective_response']
    if len(on) and len(off):
        print(f"  psa_q={q}: n={len(g)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")

print("\n=== Same subgroup, by ECOG ===")
for v in [0,1,2]:
    s = sub[sub['ecog_ps']==v]
    on = s.loc[s['treatment_enzalutamide']==1,'objective_response']
    off = s.loc[s['treatment_enzalutamide']==0,'objective_response']
    if len(on) and len(off):
        print(f"  ecog_ps={v}: n={len(s)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")

print("\n=== mcrpc=0, ar_v7=0, brca2=0, msi_high=0, ecog<=1, psa<median ===")
sub2 = sub[(sub['ecog_ps']<=1) & (sub['psa_ng_ml'] < sub['psa_ng_ml'].median())]
on = sub2.loc[sub2['treatment_enzalutamide']==1,'objective_response']
off = sub2.loc[sub2['treatment_enzalutamide']==0,'objective_response']
print(f"  n={len(sub2)}, enza on rate={on.mean():.3f} (n={len(on)})  off rate={off.mean():.3f} (n={len(off)})  diff={on.mean()-off.mean():+.3f}")
