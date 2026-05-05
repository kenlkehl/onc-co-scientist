"""Iteration 10-12: Validation, cross-validation, and treatment-combination analyses."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)

# Define the subgroup
df['SG_018'] = ((df['feature_016']==1) & (df['feature_031']==0) &
                (df['feature_028']==0) & (df['feature_005']==0)).astype(int)

print("=== Subgroup membership counts ===")
print(df['SG_018'].value_counts())

# Adjusted model with subgroup-specific treatment effect
print("\n=== Joint adjusted model: feature_018 main + SG_018 main + interaction ===")
m = smf.ols("pfs_months ~ feature_018 * SG_018 + feature_015 + feature_014 + feature_023 + feature_011 + "
            "feature_017 + feature_025 + hist_squam + smk_current + smk_former",
            data=df).fit()
print(m.summary().tables[1])

# Cross-validation: random 50% split, fit subgroup on half, test treatment effect on other half
np.random.seed(42)
idx = np.random.permutation(len(df))
half = len(df) // 2
train_idx, test_idx = idx[:half], idx[half:]
train, test = df.iloc[train_idx], df.iloc[test_idx]
print("\n=== 50/50 split cross-check ===")
for name, sub in [('train', train), ('test', test)]:
    sg = sub[sub['SG_018']==1]
    nsg = sub[sub['SG_018']==0]
    a = sg.loc[sg['feature_018']==1, 'pfs_months']
    b = sg.loc[sg['feature_018']==0, 'pfs_months']
    eff_in = a.mean() - b.mean()
    p_in = stats.ttest_ind(a,b).pvalue
    a2 = nsg.loc[nsg['feature_018']==1, 'pfs_months']
    b2 = nsg.loc[nsg['feature_018']==0, 'pfs_months']
    eff_out = a2.mean() - b2.mean()
    p_out = stats.ttest_ind(a2,b2).pvalue
    print(f"  {name}: in-subgroup eff={eff_in:+.3f} p={p_in:.2e}  out-subgroup eff={eff_out:+.3f} p={p_out:.2e}")

# Treatment combinations: does feature_018 + feature_012 add anything within the active subgroup?
print("\n=== Treatment combinations within active subgroup (SG_018=1) ===")
sg = df[df['SG_018']==1]
combo = sg.groupby(['feature_012','feature_018'])['pfs_months'].agg(['mean','std','count']).round(3)
print(combo)
# 2x2 ANOVA
print("\n2x2 ANOVA inside SG_018=1:")
m2 = smf.ols("pfs_months ~ feature_012 * feature_018", data=sg).fit()
print(m2.summary().tables[1])

# Same for feature_020, feature_027
print("\n=== feature_018 × feature_020 within active subgroup ===")
combo2 = sg.groupby(['feature_020','feature_018'])['pfs_months'].agg(['mean','count']).round(3)
print(combo2)
m3 = smf.ols("pfs_months ~ feature_020 * feature_018", data=sg).fit()
print(m3.summary().tables[1])

print("\n=== feature_018 × feature_027 within active subgroup ===")
combo3 = sg.groupby(['feature_027','feature_018'])['pfs_months'].agg(['mean','count']).round(3)
print(combo3)
m4 = smf.ols("pfs_months ~ feature_027 * feature_018", data=sg).fit()
print(m4.summary().tables[1])

# Sensitivity: which gate matters most? Test removing each gate
print("\n=== Sensitivity: which gate matters most? ===")
gates = ['feature_031', 'feature_028', 'feature_005']
for skip in gates:
    other = [g for g in gates if g != skip]
    mask = (df['feature_016']==1)
    for g in other:
        mask = mask & (df[g]==0)
    sub = df[mask]
    # Within this looser subgroup, test feature_018 effect
    a = sub.loc[sub['feature_018']==1, 'pfs_months']
    b = sub.loc[sub['feature_018']==0, 'pfs_months']
    eff = a.mean() - b.mean()
    p = stats.ttest_ind(a,b).pvalue
    print(f"  Drop gate {skip}: subgroup f016=1 & gates(other)=0, n={len(sub)}, eff={eff:+.3f}, p={p:.3e}")
