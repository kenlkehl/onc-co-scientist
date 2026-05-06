"""Iteration 8: tree-based subgroup discovery.

For each candidate treatment T, fit a regression tree on (pfs_months, T, X)
that explicitly looks for interaction by computing the difference in mean PFS between
T=1 and T=0 within each leaf, then identifies the leaf with the most extreme effect.

Approach: fit a sklearn DecisionTreeRegressor on (X) with response = (T - pT) * (Y - pY)
which is the IPW-style transformation. Or simpler: fit two trees (one per arm) and
compare.

Easier: use a custom 'effect tree' approach: split on X to maximize the contrast between
T=1 mean and T=0 mean.
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)
y = df['pfs_months'].values

X_cols = ['feature_015','feature_031','feature_014','feature_023','feature_011',
          'feature_022','feature_016','feature_028','feature_002','feature_005',
          'feature_019','feature_021','feature_025','feature_032','feature_017',
          'feature_024','feature_010','feature_003','feature_030','feature_008',
          'feature_004','feature_013','feature_026','feature_029','feature_007',
          'feature_033','feature_009','hist_squam','smk_current','smk_former']

for T in ['feature_012','feature_018','feature_020','feature_027']:
    t = df[T].values
    # Centered: each unit's contribution to ATE is (Y * (T - pT)/(pT*(1-pT)))
    pT = t.mean()
    z = (t - pT)/(pT * (1-pT)) * y  # IPTW transformation
    Xmat = df[X_cols].values
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=200).fit(Xmat, z)
    leaves = tree.apply(Xmat)
    # Compute true effect within each leaf
    leaf_effects = []
    for leaf_id in np.unique(leaves):
        mask = (leaves == leaf_id)
        n = mask.sum()
        a = y[mask & (t==1)].mean() if (mask & (t==1)).sum() > 0 else np.nan
        b = y[mask & (t==0)].mean() if (mask & (t==0)).sum() > 0 else np.nan
        leaf_effects.append((leaf_id, n, a, b, a - b, (mask & (t==1)).sum(), (mask & (t==0)).sum()))
    leaf_effects.sort(key=lambda x: -abs(x[4]) if not np.isnan(x[4]) else 0)
    print(f"\n=== Tree-based leaf effects for T={T} ===")
    print(f"{'leaf':<8}{'n':<10}{'mean_T1':<12}{'mean_T0':<12}{'effect':<10}{'n1':<8}{'n0':<8}")
    for le, n, a, b, e, n1, n0 in leaf_effects[:8]:
        if not np.isnan(e):
            print(f"{le:<8}{n:<10}{a:<12.3f}{b:<12.3f}{e:<+10.3f}{n1:<8}{n0:<8}")

    # Print decision rules
    from sklearn.tree import export_text
    print(f"\n--- Tree decision rules for T={T} ---")
    text = export_text(tree, feature_names=X_cols, max_depth=4)
    print(text[:3000])
