"""Iteration 1: Univariate main effects of each feature on objective_response."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_parquet('dataset.parquet')
y = df['objective_response'].values
print(f'n={len(df)}, ORR={y.mean():.4f}')
print()

binary_cols = [c for c in df.columns if c.startswith('feature_') and df[c].nunique() == 2]
multi_cols = [c for c in df.columns if c.startswith('feature_') and 2 < df[c].nunique() <= 10]
cont_cols = [c for c in df.columns if c.startswith('feature_') and df[c].nunique() > 10]
zero_cols = [c for c in df.columns if c.startswith('feature_') and df[c].nunique() == 1]
print('binary:', binary_cols)
print('few-level (3-10):', multi_cols)
print('continuous:', cont_cols)
print('zero variance:', zero_cols)
print()

print('=== BINARY FEATURES vs ORR (chi-square) ===')
print(f"{'feature':<14}{'n=1':>8}{'ORR(0)':>10}{'ORR(1)':>10}{'diff':>10}{'OR':>10}{'p':>12}")
for c in binary_cols:
    g0 = y[df[c] == 0]
    g1 = y[df[c] == 1]
    if len(g0) == 0 or len(g1) == 0:
        continue
    p0 = g0.mean(); p1 = g1.mean()
    a = (df[c]==1).sum() * p1
    b = (df[c]==1).sum() * (1-p1)
    cc = (df[c]==0).sum() * p0
    d = (df[c]==0).sum() * (1-p0)
    or_ = (a*d)/(b*cc) if (b*cc) > 0 else np.nan
    ct = pd.crosstab(df[c], df['objective_response'])
    chi2, pval, _, _ = stats.chi2_contingency(ct)
    print(f'{c:<14}{(df[c]==1).sum():>8}{p0:>10.4f}{p1:>10.4f}{p1-p0:>+10.4f}{or_:>10.3f}{pval:>12.3e}')
print()

print('=== FEW-LEVEL FEATURES (3-10) vs ORR ===')
for c in multi_cols:
    g = df.groupby(c)['objective_response'].agg(['mean', 'count'])
    print(f'\n{c}:')
    print(g)
    ct = pd.crosstab(df[c], df['objective_response'])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, pval, _, _ = stats.chi2_contingency(ct)
        print(f'chi-square p = {pval:.3e}')
print()

print('=== CONTINUOUS FEATURES vs ORR (mean by ORR group, t-test) ===')
print(f"{'feature':<14}{'mean@0':>14}{'mean@1':>14}{'diff':>14}{'p':>12}")
for c in cont_cols:
    g0 = df.loc[df['objective_response']==0, c].values
    g1 = df.loc[df['objective_response']==1, c].values
    t, p = stats.ttest_ind(g0, g1, equal_var=False)
    print(f'{c:<14}{g0.mean():>14.4f}{g1.mean():>14.4f}{g1.mean()-g0.mean():>+14.4f}{p:>12.3e}')
