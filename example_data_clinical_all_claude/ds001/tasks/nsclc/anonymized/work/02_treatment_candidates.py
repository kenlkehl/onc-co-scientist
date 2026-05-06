"""Iteration 2: identify candidate treatment indicators.

Heuristic: 'treatment' candidates often look like:
  - Binary or low-prevalence with allocation that looks deliberate (50/50, 30/70...)
  - Weak or null main effect on PFS BUT strong heterogeneity (interaction with biomarkers)
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

# 1. Cross-tabulate the candidate treatment-like binaries
candidates = ['feature_012', 'feature_018', 'feature_020', 'feature_027']
print("=== Candidate treatments distribution ===")
print(df[candidates].sum() / len(df))

# 2. Correlation among candidates (mutual exclusivity?)
print("\n=== Correlation among candidates ===")
print(df[candidates].corr())

# 3. Joint cross-tab of candidates (any patient receiving multiple?)
print("\n=== Joint distribution (sum across candidates per patient) ===")
ct = df[candidates].sum(axis=1).value_counts().sort_index()
print(ct)

# 4. Cross-tab pairs
import itertools
print("\n=== Pairwise cross-tabs ===")
for a, b in itertools.combinations(candidates, 2):
    ct = pd.crosstab(df[a], df[b])
    print(f"\n{a} vs {b}:")
    print(ct)

# 5. Conditional means of PFS by each candidate
print("\n=== Mean PFS by candidate ===")
for c in candidates:
    means = df.groupby(c)['pfs_months'].agg(['mean', 'count'])
    print(f"{c}: {means.to_dict()}")

# 6. Quick interaction check: for each candidate, does the effect differ across feature_006 (histology)?
print("\n=== Stratified PFS by candidate × histology ===")
for c in candidates:
    sub = df.groupby(['feature_006', c])['pfs_months'].mean().unstack()
    print(f"\n{c}:")
    print(sub)
