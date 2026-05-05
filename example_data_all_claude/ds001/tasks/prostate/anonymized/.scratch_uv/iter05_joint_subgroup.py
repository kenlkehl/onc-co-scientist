"""Iteration 5: characterize the feature_008 treatment-effect subgroup.

We saw strong interactions between feature_008 and feature_013, _015, _021, _027.
We also saw feature_008 x feature_022 (continuous) suppression.

Step 1: stratified response rates.
Step 2: joint subgroup definition; test the simultaneous predicates.
Step 3: see whether the residual heterogeneity vanishes once we
        condition on all the modifiers.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response']

print('=== feature_008 stratified by feature_013 ===')
for v13 in [0, 1]:
    sub = df[df.feature_013 == v13]
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    n1 = (sub.feature_008==1).sum(); n0 = (sub.feature_008==0).sum()
    print(f' feature_013={v13}: n8=1: {n1}, RR={rr1:.3f} | n8=0: {n0}, RR={rr0:.3f} | diff={rr1-rr0:+.3f}')

print()
print('=== feature_008 x feature_015 ===')
for v in [0,1]:
    sub = df[df.feature_015 == v]
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    print(f' feature_015={v}: RR(8=1)={rr1:.3f}, RR(8=0)={rr0:.3f}, diff={rr1-rr0:+.3f}')

print()
print('=== feature_008 x feature_021 ===')
for v in [0,1]:
    sub = df[df.feature_021 == v]
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    print(f' feature_021={v}: RR(8=1)={rr1:.3f}, RR(8=0)={rr0:.3f}, diff={rr1-rr0:+.3f}')

print()
print('=== feature_008 x feature_027 ===')
for v in [0,1]:
    sub = df[df.feature_027 == v]
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    print(f' feature_027={v}: RR(8=1)={rr1:.3f}, RR(8=0)={rr0:.3f}, diff={rr1-rr0:+.3f}')

print()
print('=== Joint "favorable" subgroup: 13=0 AND 15=0 AND 21=0 AND 27=0 ===')
fav = (df.feature_013==0) & (df.feature_015==0) & (df.feature_021==0) & (df.feature_027==0)
print(f' n_favorable={fav.sum()}')
rr1 = df.loc[fav & (df.feature_008==1), 'objective_response'].mean()
rr0 = df.loc[fav & (df.feature_008==0), 'objective_response'].mean()
n1 = (fav & (df.feature_008==1)).sum(); n0 = (fav & (df.feature_008==0)).sum()
print(f' favorable & 8=1: n={n1}, RR={rr1:.3f}')
print(f' favorable & 8=0: n={n0}, RR={rr0:.3f}')
print(f' diff = {rr1-rr0:+.3f}')

print()
print('=== "unfavorable" subgroup: 13=1 OR 15=1 OR 21=1 OR 27=1 ===')
unfav = ~fav
rr1 = df.loc[unfav & (df.feature_008==1), 'objective_response'].mean()
rr0 = df.loc[unfav & (df.feature_008==0), 'objective_response'].mean()
n1 = (unfav & (df.feature_008==1)).sum(); n0 = (unfav & (df.feature_008==0)).sum()
print(f' unfavorable & 8=1: n={n1}, RR={rr1:.3f}')
print(f' unfavorable & 8=0: n={n0}, RR={rr0:.3f}')
print(f' diff = {rr1-rr0:+.3f}')

print()
print('=== Stratify by ALL FOUR + feature_022 quartiles ===')
for v22q in range(4):
    q_low = df.feature_022.quantile(v22q/4)
    q_hi = df.feature_022.quantile((v22q+1)/4)
    in_q = (df.feature_022 >= q_low) & (df.feature_022 <= q_hi if v22q==3 else df.feature_022 < q_hi)
    sub = df[fav & in_q]
    if len(sub) < 10: continue
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    n1 = (sub.feature_008==1).sum(); n0 = (sub.feature_008==0).sum()
    print(f' favorable & feature_022 Q{v22q+1} [{q_low:.1f}, {q_hi:.1f}]: n8=1={n1} RR={rr1:.3f} | n8=0={n0} RR={rr0:.3f} | diff={rr1-rr0:+.3f}')

print()
print('=== Logit: feature_008 effect within favorable subgroup adjusted for feature_022, _001 ===')
sub = df[fav]
X = sub[['feature_008', 'feature_022', 'feature_001']].astype(float).copy()
X['feature_022'] = (X['feature_022'] - X['feature_022'].mean()) / X['feature_022'].std()
X['feature_022_x_8'] = X['feature_022'] * X['feature_008']
X['feature_001_x_8'] = X['feature_001'] * X['feature_008']
X = sm.add_constant(X)
mod = sm.Logit(sub['objective_response'].astype(int).values, X).fit(disp=0, maxiter=100)
print(mod.summary())
