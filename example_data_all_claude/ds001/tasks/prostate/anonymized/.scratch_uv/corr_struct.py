import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

bin_feats = ['feature_006','feature_013','feature_021','feature_015','feature_027','feature_005',
             'feature_008','feature_023','feature_011','feature_017','feature_019','feature_004']

# Pairwise prevalence — look for mutually exclusive (treatment-arm-like) groups
print("Joint prevalence among binary features (cross-tab proportions):")
for i, a in enumerate(bin_feats):
    for b in bin_feats[i+1:]:
        ct = pd.crosstab(df[a], df[b])
        n11 = ct.loc[1,1] if (1 in ct.index and 1 in ct.columns) else 0
        n10 = ct.loc[1,0] if (1 in ct.index and 0 in ct.columns) else 0
        n01 = ct.loc[0,1] if (0 in ct.index and 1 in ct.columns) else 0
        n00 = ct.loc[0,0] if (0 in ct.index and 0 in ct.columns) else 0
        # phi = correlation
        phi = (n11*n00 - n10*n01) / np.sqrt(max(1,(n11+n10)*(n01+n00)*(n11+n01)*(n10+n00)))
        if abs(phi) > 0.05:
            print(f"  {a} x {b}: 11={n11}, 10={n10}, 01={n01}, 00={n00}, phi={phi:.3f}")

print("\n\nAre any binary features perfectly partitioned (e.g., treatment arms summing to all patients)?")
combos = ['feature_005','feature_008','feature_011','feature_023','feature_004','feature_017','feature_019']
for c in combos:
    print(f"  {c}: prevalence={df[c].mean():.3f}")
sum_combo = df[['feature_008','feature_011','feature_023']].sum(axis=1).value_counts().sort_index()
print(f"  sum(8+11+23): {dict(sum_combo)}")
sum_combo2 = df[['feature_017','feature_019','feature_004']].sum(axis=1).value_counts().sort_index()
print(f"  sum(17+19+4): {dict(sum_combo2)}")
sum_all = df[bin_feats].sum(axis=1).value_counts().sort_index()
print(f"  sum(all bin): {dict(sum_all)}")

# Continuous features – look at distributions for a hint of identity
print("\n\nContinuous: pairwise correlations (|r|>0.10 only)")
cont_feats = ['feature_016','feature_022','feature_002','feature_018','feature_020','feature_024','feature_031',
              'feature_026','feature_009','feature_029','feature_003','feature_012','feature_025','feature_028',
              'feature_007','feature_014','feature_032']
cm = df[cont_feats].corr()
for i, a in enumerate(cont_feats):
    for b in cont_feats[i+1:]:
        if abs(cm.loc[a,b]) > 0.10:
            print(f"  {a} ~ {b}: r={cm.loc[a,b]:.3f}")

# Check relationship of feature_001 levels vs feature_022 (PSA) and feature_010 (Gleason candidate)
print("\nFeature_001 levels vs continuous lab medians:")
print(df.groupby('feature_001')[['feature_022','feature_018','feature_026','feature_002']].median())

print("\nResponse rate by feature_010 (Gleason?):")
print(df.groupby('feature_010')['objective_response'].agg(['mean','count']))

print("\nResponse rate joint table feature_008 x feature_013:")
print(df.groupby(['feature_008','feature_013'])['objective_response'].agg(['mean','count']))

print("\nResponse rate joint table feature_008 x feature_001:")
print(df.groupby(['feature_008','feature_001'])['objective_response'].agg(['mean','count']))

print("\nResponse rate joint table feature_008 x feature_015:")
print(df.groupby(['feature_008','feature_015'])['objective_response'].agg(['mean','count']))

print("\nResponse rate joint table feature_008 x feature_021:")
print(df.groupby(['feature_008','feature_021'])['objective_response'].agg(['mean','count']))
