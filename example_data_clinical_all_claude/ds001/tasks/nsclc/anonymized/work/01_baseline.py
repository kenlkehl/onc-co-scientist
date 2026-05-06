"""Iteration 1: baseline associations between every feature and pfs_months."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values
features = [c for c in df.columns if c not in ('patient_id', 'pfs_months')]

print(f"N={len(df)}, mean PFS={y.mean():.3f}, median={np.median(y):.3f}")
print(f"\n--- Feature association with PFS ---")
print(f"{'feature':<14}{'kind':<10}{'effect':>12}{'p':>14}{'note':<40}")

results = {}
for col in features:
    s = df[col]
    if s.dtype == 'object':
        # Categorical -> ANOVA
        groups = [df.loc[s == lvl, 'pfs_months'].values for lvl in s.unique()]
        F, p = stats.f_oneway(*groups)
        # report mean by level
        means = df.groupby(col)['pfs_months'].mean().to_dict()
        # signed effect: max-min of group means
        eff = max(means.values()) - min(means.values())
        results[col] = {'kind': 'anova', 'effect': eff, 'p': p, 'means': means}
        print(f"{col:<14}{'cat':<10}{eff:>12.4f}{p:>14.3e}{str(means)[:40]:<40}")
    elif s.nunique() == 2:
        a = df.loc[s == s.unique()[0], 'pfs_months'].values
        b = df.loc[s == s.unique()[1], 'pfs_months'].values
        t, p = stats.ttest_ind(a, b)
        # signed effect: mean(group=1) - mean(group=0)
        eff = df.loc[s == 1, 'pfs_months'].mean() - df.loc[s == 0, 'pfs_months'].mean()
        results[col] = {'kind': 'binary', 'effect': eff, 'p': p}
        print(f"{col:<14}{'bin':<10}{eff:>12.4f}{p:>14.3e}{'':<40}")
    elif s.nunique() <= 5:
        # ordinal/few levels: ANOVA
        groups = [df.loc[s == lvl, 'pfs_months'].values for lvl in sorted(s.unique())]
        F, p = stats.f_oneway(*groups)
        means = df.groupby(col)['pfs_months'].mean().to_dict()
        eff = max(means.values()) - min(means.values())
        results[col] = {'kind': 'ord', 'effect': eff, 'p': p, 'means': means}
        print(f"{col:<14}{'ord':<10}{eff:>12.4f}{p:>14.3e}{str(means)[:40]:<40}")
    else:
        r, p = stats.pearsonr(s.values, y)
        results[col] = {'kind': 'cont', 'effect': r, 'p': p}
        print(f"{col:<14}{'cont':<10}{r:>12.4f}{p:>14.3e}{'':<40}")

# Sort by p-value
print("\n--- Sorted by significance ---")
sorted_r = sorted(results.items(), key=lambda kv: kv[1]['p'])
for col, r in sorted_r:
    print(f"{col:<14}{r['kind']:<10}eff={r['effect']:>10.4f}  p={r['p']:.3e}")

import json
out = {col: {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
            for k, v in r.items()} for col, r in results.items()}
with open('work/01_baseline_results.json', 'w') as f:
    json.dump(out, f, indent=2, default=str)
