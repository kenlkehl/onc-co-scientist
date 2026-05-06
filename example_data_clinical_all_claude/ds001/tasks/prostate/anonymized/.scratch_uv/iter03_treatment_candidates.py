"""Iteration 3: identify candidate 'treatment' features.

A treatment-like feature should be a binary feature that is reasonably
balanced across the cohort and across other features (i.e. its assignment
is roughly random or guided by clinical decision rather than determined
by an underlying biology). We:
  1. Compute correlation of each binary feature with all the strong
     prognostic features.
  2. Compute univariate response association.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
features = [c for c in df.columns if c not in ('patient_id', 'objective_response', 'feature_030')]
binaries = [c for c in features if df[c].nunique() == 2]
print('binary features:', binaries)
print()

# Correlations among binaries
print('--- chi-square / phi between every pair of binaries ---')
for i, a in enumerate(binaries):
    row = []
    for b in binaries:
        if a == b: continue
        ct = pd.crosstab(df[a], df[b])
        if (ct < 5).any().any():
            row.append((b, np.nan))
            continue
        chi2, p, _, _ = stats.chi2_contingency(ct)
        # phi
        n = ct.values.sum()
        phi = np.sqrt(chi2 / n)
        sign = np.sign(np.corrcoef(df[a], df[b])[0,1])
        row.append((b, sign * phi, p))
    row.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 1)
    print(f'{a}: top corr ->', [(b, round(phi,3), f'{p:.1e}') for b,phi,p in row[:6]])
print()

# How does each binary's association with response change with adjustment?
print('--- binaries: univariate vs adjusted (adj for top prognostics) ---')
adj_features = ['feature_001', 'feature_010', 'feature_022', 'feature_020', 'feature_002', 'feature_016',
                'feature_026', 'feature_009', 'feature_018', 'feature_024']
adj_features = [a for a in adj_features if a in features]
y = df['objective_response'].astype(int).values
for b in binaries:
    others = [a for a in adj_features if a != b]
    Xb = df[[b] + others].astype(float).copy()
    Xb = sm.add_constant(Xb)
    try:
        mod = sm.Logit(y, Xb).fit(disp=0, maxiter=100)
        beta_adj = mod.params[b]; p_adj = mod.pvalues[b]
    except Exception as e:
        beta_adj, p_adj = np.nan, np.nan
    # Univariate
    Xu = sm.add_constant(df[[b]].astype(float))
    mod_u = sm.Logit(y, Xu).fit(disp=0, maxiter=100)
    beta_u = mod_u.params[b]; p_u = mod_u.pvalues[b]
    print(f'{b} | rate={df[b].mean():.3f} | univar beta={beta_u:+.3f} (p={p_u:.1e}) | adj beta={beta_adj:+.3f} (p={p_adj:.1e})')
