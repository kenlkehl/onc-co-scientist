"""Examine joint subgroups for feature_008 treatment effect."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import product

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
T = df['feature_008'].astype(int).values

# Top modifiers (binary): 013, 015, 021, 027
# Top continuous modifier: feature_022
# Joint subgroup based on binary
binary_mods = ['feature_013', 'feature_015', 'feature_021', 'feature_027']

print('Treatment effect (response rate diff) within all 16 binary combinations of top modifiers:')
print(f'{"f013":<6}{"f015":<6}{"f021":<6}{"f027":<6}{"n":>8}{"R|T0":>10}{"R|T1":>10}{"diff":>10}{"OR":>10}')
res_rows = []
for v13, v15, v21, v27 in product([0,1], repeat=4):
    mask = (df['feature_013']==v13) & (df['feature_015']==v15) & (df['feature_021']==v21) & (df['feature_027']==v27)
    n = int(mask.sum())
    if n < 10:
        continue
    yy = y[mask]
    tt = T[mask]
    n0 = (tt==0).sum(); n1 = (tt==1).sum()
    if n0==0 or n1==0:
        continue
    r0 = yy[tt==0].mean()
    r1 = yy[tt==1].mean()
    # Compute OR
    a = ((tt==1) & (yy==1)).sum() + 0.5
    b = ((tt==1) & (yy==0)).sum() + 0.5
    c = ((tt==0) & (yy==1)).sum() + 0.5
    d = ((tt==0) & (yy==0)).sum() + 0.5
    OR = (a*d)/(b*c)
    res_rows.append((v13,v15,v21,v27,n,r0,r1,r1-r0,OR))
    print(f'{v13:<6}{v15:<6}{v21:<6}{v27:<6}{n:>8}{r0:>10.3f}{r1:>10.3f}{r1-r0:>10.3f}{OR:>10.3f}')

print('\nResponse in "favorable" (all four = 0) vs other combos:')
fav = (df['feature_013']==0) & (df['feature_015']==0) & (df['feature_021']==0) & (df['feature_027']==0)
print(f'  Favorable n={fav.sum()}  T=1 resp={y[fav & (T==1)].mean():.3f}  T=0 resp={y[fav & (T==0)].mean():.3f}  diff={(y[fav & (T==1)].mean()-y[fav & (T==0)].mean()):.3f}')
print(f'  Other     n={(~fav).sum()}  T=1 resp={y[~fav & (T==1)].mean():.3f}  T=0 resp={y[~fav & (T==0)].mean():.3f}  diff={(y[~fav & (T==1)].mean()-y[~fav & (T==0)].mean()):.3f}')

# Also include feature_022 (continuous, top modifier). Check by terciles or by median.
print('\nNow add feature_022 modifier — within favorable (0,0,0,0), split by feature_022 median:')
m22 = df['feature_022'].median()
print(f'  feature_022 median: {m22}')
favlow = fav & (df['feature_022'] <= m22)
favhigh = fav & (df['feature_022'] > m22)
for grp_name, grp_mask in [('fav-low_f022', favlow), ('fav-high_f022', favhigh)]:
    n = grp_mask.sum()
    r0 = y[grp_mask & (T==0)].mean() if (grp_mask & (T==0)).sum() else np.nan
    r1 = y[grp_mask & (T==1)].mean() if (grp_mask & (T==1)).sum() else np.nan
    print(f'  {grp_name}: n={n} T=0 resp={r0:.3f} T=1 resp={r1:.3f} diff={r1-r0:.3f}')
