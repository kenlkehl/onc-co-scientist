"""Iteration 9-10: Are there OTHER treatment-like features (balanced + ORR effect + heterogeneous)?
Also: pairwise interactions among the 4 modifiers themselves.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].astype(int)

# ===== Test each binary feature as a hypothetical treatment =====
binary_features = [c for c in df.columns if c.startswith('feature_') and df[c].nunique()==2]
print('=== Each binary feature as candidate treatment ===')
print(f'{"feature":<14}{"prev":>8}{"OR":>8}{"diff":>8}{"avg_balance_p":>16}{"max_intxn_chi2":>18}')
other_binaries = [c for c in df.columns if c.startswith('feature_') and df[c].nunique()==2]

for cand in binary_features:
    others = [c for c in other_binaries if c != cand]
    # Balance: average chi-square p across other binaries
    bal_ps = []
    for o in others:
        ct = pd.crosstab(df[cand], df[o])
        if ct.shape == (2,2):
            _, p, _, _ = stats.chi2_contingency(ct)
            bal_ps.append(p)
    avg_bal_p = np.median(bal_ps)
    # Effect on outcome
    p0 = df.loc[df[cand]==0,'objective_response'].mean()
    p1 = df.loc[df[cand]==1,'objective_response'].mean()
    or_ = (p1/(1-p1)) / (p0/(1-p0))
    # Max interaction chi2 with any other feature
    max_chi2 = 0
    for o in others:
        try:
            X = df[[cand, o]].copy()
            X['Tx'] = X[cand]*X[o]
            Xf = sm.add_constant(X.astype(float))
            Xr = sm.add_constant(df[[cand, o]].astype(float))
            mf = sm.Logit(y, Xf).fit(disp=False)
            mr = sm.Logit(y, Xr).fit(disp=False)
            chi2 = 2*(mf.llf - mr.llf)
            if chi2 > max_chi2:
                max_chi2 = chi2
        except Exception:
            pass
    print(f'{cand:<14}{(df[cand]==1).mean():>8.4f}{or_:>8.3f}{p1-p0:>+8.4f}{avg_bal_p:>16.3f}{max_chi2:>18.2f}')

print()
# Within full cohort, test pairwise interaction among the 4 modifiers (not with T) as a tx model
# (do they synergize on prognosis?)
T = 'feature_008'
modifiers = ['feature_013', 'feature_015', 'feature_021', 'feature_027']
print('=== Pairwise interactions among the 4 modifiers (within T=0 only, on outcome) ===')
sub_t0 = df[df[T]==0]
y0 = sub_t0['objective_response'].astype(int)
print(f'{"pair":<30}{"interaction p":>16}')
for a, b in combinations(modifiers, 2):
    X = sub_t0[[a,b]].copy()
    X['ab'] = X[a]*X[b]
    Xf = sm.add_constant(X.astype(float))
    Xr = sm.add_constant(sub_t0[[a,b]].astype(float))
    try:
        mf = sm.Logit(y0, Xf).fit(disp=False)
        mr = sm.Logit(y0, Xr).fit(disp=False)
        p = stats.chi2.sf(2*(mf.llf-mr.llf), 1)
    except Exception:
        p = np.nan
    print(f'{a} x {b}{"":<5}{p:>16.3e}')
print()

# Within T=1, do modifiers act synergistically on suppressing ORR?
print('=== Pairwise interactions among the 4 modifiers (within T=1 only, on outcome) ===')
sub_t1 = df[df[T]==1]
y1 = sub_t1['objective_response'].astype(int)
print(f'{"pair":<30}{"interaction p":>16}{"OR_interaction":>16}')
for a, b in combinations(modifiers, 2):
    X = sub_t1[[a,b]].copy()
    X['ab'] = X[a]*X[b]
    Xf = sm.add_constant(X.astype(float))
    Xr = sm.add_constant(sub_t1[[a,b]].astype(float))
    try:
        mf = sm.Logit(y1, Xf).fit(disp=False)
        mr = sm.Logit(y1, Xr).fit(disp=False)
        p = stats.chi2.sf(2*(mf.llf-mr.llf), 1)
        or_ab = np.exp(mf.params['ab'])
    except Exception:
        p = np.nan; or_ab = np.nan
    print(f'{a} x {b}{"":<5}{p:>16.3e}{or_ab:>16.3f}')
