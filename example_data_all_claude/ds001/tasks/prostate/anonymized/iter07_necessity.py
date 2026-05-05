"""Iteration 7: Necessity check — does dropping any one of the 4 modifiers degrade the subgroup?
Compute ORR by T within each progressively relaxed subgroup definition.
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

def orr_table(mask, label):
    sub = df[mask]
    p0 = sub.loc[sub[T]==0,'objective_response'].mean()
    p1 = sub.loc[sub[T]==1,'objective_response'].mean()
    ct = pd.crosstab(sub[T], sub['objective_response'])
    if ct.shape == (2,2):
        chi2, pv, _, _ = stats.chi2_contingency(ct)
    else:
        pv = np.nan
    n = mask.sum()
    return label, n, (sub[T]==0).sum(), (sub[T]==1).sum(), p0, p1, p1-p0, pv

print(f"{'subgroup definition':<60}{'n':>8}{'n0':>7}{'n1':>7}{'ORR0':>8}{'ORR1':>8}{'RD':>8}{'p':>10}")
# All 4 = 0 (full favorable)
mask_full = ((df['feature_013']==0)&(df['feature_015']==0)&
             (df['feature_021']==0)&(df['feature_027']==0))
r = orr_table(mask_full, 'all 4 = 0 (favorable)')
print(f"{r[0]:<60}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>+8.4f}{r[7]:>10.2e}")
# Drop one constraint at a time
for drop in modifiers:
    others = [m for m in modifiers if m != drop]
    mask = pd.Series(True, index=df.index)
    for m in others:
        mask &= (df[m]==0)
    r = orr_table(mask, f'all except {drop} = 0')
    print(f"{r[0]:<60}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>+8.4f}{r[7]:>10.2e}")
# 3-of-4 = 0 by every pair
print()
print("=== Subgroup defined by combinations of 0-3 of the 4 modifiers ===")
for k in range(0, 5):
    for cmb in combinations(modifiers, k):
        mask = pd.Series(True, index=df.index)
        for m in cmb:
            mask &= (df[m]==0)
        r = orr_table(mask, '+'.join(cmb)+' = 0' if k > 0 else 'no constraint')
        print(f"{r[0]:<60}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>+8.4f}{r[7]:>10.2e}")

print()
# Cells with exactly k modifiers = 1
print("=== ORR by T within cells defined by # modifiers = 1 ===")
n_mods = df[modifiers].sum(axis=1)
for k in range(0, 5):
    mask = (n_mods == k)
    r = orr_table(mask, f'exactly {k} of 4 modifiers ON')
    print(f"{r[0]:<60}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>+8.4f}{r[7]:>10.2e}")
