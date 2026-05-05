"""Iteration 6: necessity test for each predicate of the favorable subgroup.

For each combination of 13/15/21/27, compute the feature_008 effect.
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

mods = ['feature_013', 'feature_015', 'feature_021', 'feature_027']

print('=== feature_008 effect across all 16 combinations of (13,15,21,27) ===')
print(f'{"v13":>4} {"v15":>4} {"v21":>4} {"v27":>4} | n_total | n_8=1 RR1 | n_8=0 RR0 | diff      p')
for combo in itertools.product([0,1], repeat=4):
    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(mods, combo):
        mask &= (df[c] == v).values
    sub = df[mask]
    if len(sub) < 5:
        continue
    n8_1 = (sub.feature_008==1).sum()
    n8_0 = (sub.feature_008==0).sum()
    if n8_1 < 5 or n8_0 < 5:
        continue
    rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
    rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
    # Fisher exact (or chi square)
    a = (sub.loc[sub.feature_008==1, 'objective_response']==1).sum()
    b = (sub.loc[sub.feature_008==1, 'objective_response']==0).sum()
    c_ = (sub.loc[sub.feature_008==0, 'objective_response']==1).sum()
    d = (sub.loc[sub.feature_008==0, 'objective_response']==0).sum()
    table = [[int(a), int(b)], [int(c_), int(d)]]
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
    except Exception:
        p = np.nan
    print(f'{combo[0]:>4} {combo[1]:>4} {combo[2]:>4} {combo[3]:>4} | {len(sub):>7} | n={n8_1} RR={rr1:.3f} | n={n8_0} RR={rr0:.3f} | {rr1-rr0:+.3f}  p={p:.2e}')

print()
print('=== Test: is each predicate INDIVIDUALLY necessary? ===')
print('     Define base = all four = 0; relax one predicate at a time; effect should drop only when relaxed predicate flips')
fav_full = (df.feature_013==0) & (df.feature_015==0) & (df.feature_021==0) & (df.feature_027==0)
for relax in mods:
    others = [m for m in mods if m != relax]
    for relax_val in [0, 1]:
        mask = (df[relax] == relax_val).values
        for o in others:
            mask &= (df[o]==0).values
        sub = df[mask]
        n8_1 = (sub.feature_008==1).sum()
        n8_0 = (sub.feature_008==0).sum()
        rr1 = sub.loc[sub.feature_008==1, 'objective_response'].mean()
        rr0 = sub.loc[sub.feature_008==0, 'objective_response'].mean()
        diff = rr1 - rr0
        print(f'  others=0, {relax}={relax_val}: n8=1={n8_1} RR={rr1:.3f} | n8=0={n8_0} RR={rr0:.3f} | diff={diff:+.3f}')
    print()
