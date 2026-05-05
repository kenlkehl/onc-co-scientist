"""Test the necessity of each component of the favorable subgroup definition."""
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].astype(int).values
T = df['feature_008'].astype(int).values

# Drop each excluder one at a time, and look at treatment effect
mods = ['feature_013','feature_015','feature_021','feature_027']

print('Drop each excluder, see treatment effect WITHIN combos:')
print('===  All four required (favorable= all 0):')
fav_all = (df['feature_013']==0)&(df['feature_015']==0)&(df['feature_021']==0)&(df['feature_027']==0)
print(f'  fav n={fav_all.sum()}, T-effect: {y[fav_all & (T==1)].mean():.3f} - {y[fav_all & (T==0)].mean():.3f} = {y[fav_all & (T==1)].mean()-y[fav_all & (T==0)].mean():.3f}')
nonfav_all = ~fav_all
print(f'  non-fav n={nonfav_all.sum()}, T-effect: {y[nonfav_all & (T==1)].mean():.3f} - {y[nonfav_all & (T==0)].mean():.3f} = {y[nonfav_all & (T==1)].mean()-y[nonfav_all & (T==0)].mean():.3f}')

print('\n=== Drop one at a time:')
for drop in mods:
    mods_keep = [m for m in mods if m != drop]
    fav_mask = pd.Series([True]*len(df))
    for m in mods_keep:
        fav_mask = fav_mask & (df[m]==0)
    fav_mask = fav_mask.values
    print(f'\n  Without "{drop}=0", fav defined as {[m+"=0" for m in mods_keep]}')
    n = fav_mask.sum()
    if (fav_mask & (T==1)).sum()>0 and (fav_mask & (T==0)).sum()>0:
        print(f'    fav n={n}: T1={y[fav_mask & (T==1)].mean():.3f}, T0={y[fav_mask & (T==0)].mean():.3f}, diff={y[fav_mask & (T==1)].mean()-y[fav_mask & (T==0)].mean():.3f}')
    nonfav = ~fav_mask
    print(f'    non-fav n={nonfav.sum()}: T1={y[nonfav & (T==1)].mean():.3f}, T0={y[nonfav & (T==0)].mean():.3f}, diff={y[nonfav & (T==1)].mean()-y[nonfav & (T==0)].mean():.3f}')

# Now look at the reverse: only keep the strict definition, but stratify by other binary features
print('\n=== Within favorable (all 4 = 0), heterogeneity by other binaries:')
fav = fav_all
otherbin = ['feature_006','feature_005','feature_023','feature_011','feature_017','feature_019','feature_004']
for ob in otherbin:
    for v in [0,1]:
        m = fav & (df[ob]==v)
        if m.sum() < 50: continue
        if (m & (T==1)).sum()==0 or (m & (T==0)).sum()==0: continue
        d = y[m & (T==1)].mean() - y[m & (T==0)].mean()
        print(f'  fav AND {ob}={v}: n={m.sum()}, T-eff={d:.3f}')

# Also stratify by feature_001 (3 levels) and feature_010 (5 levels)
print('\n=== Within favorable, by feature_001 levels:')
for v in [0,1,2]:
    m = fav & (df['feature_001']==v)
    if m.sum() < 50: continue
    d = y[m & (T==1)].mean() - y[m & (T==0)].mean()
    print(f'  fav AND feature_001={v}: n={m.sum()}, T-eff={d:.3f}')

print('\n=== Within favorable, by feature_010 levels:')
for v in [6,7,8,9,10]:
    m = fav & (df['feature_010']==v)
    if m.sum() < 50: continue
    if (m & (T==1)).sum()==0 or (m & (T==0)).sum()==0: continue
    d = y[m & (T==1)].mean() - y[m & (T==0)].mean()
    print(f'  fav AND feature_010={v}: n={m.sum()}, T-eff={d:.3f}')
