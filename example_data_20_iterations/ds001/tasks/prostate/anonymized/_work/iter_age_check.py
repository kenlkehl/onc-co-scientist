import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('dataset.parquet')
df['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))

# Examine feature_078 more carefully — is it age?
# Quartile analysis
df['f078_q'] = pd.qcut(df['feature_078'], 4, labels=['Q1','Q2','Q3','Q4'])
print('PFS by feature_078 quartile:')
print(df.groupby('f078_q')['pfs_months'].agg(['mean','std','count']))

# Correlation with PS
print('\nf078 by PS:')
print(df.groupby('feature_057')['feature_078'].mean())

# Does f078 modify treatment effect?
print('\nFeature_078 x feature_109 stratified by quartile:')
for q in ['Q1','Q2','Q3','Q4']:
    sub = df[df['f078_q']==q]
    m = smf.ols('pfs_months ~ feature_109 + feature_009 + feature_006 + log_f013 + C(feature_057)', data=sub).fit()
    print(f"  {q}: f078 range {sub['feature_078'].min():.0f}-{sub['feature_078'].max():.0f}, beta_109={m.params['feature_109']:.4f}, p={m.pvalues['feature_109']:.2e}")

# Consider: could feature_009 be age? Range 1.5-5.5, mean 3.8 — too narrow for age.
# feature_038: 6-18 mean 12.5 — that's hemoglobin
# feature_005: 6-22 mean 13 — too narrow, also could be hgb-related

# Look for association of f078 with other features
print('\nLooking at pairwise associations with feature_078:')
features_to_check = ['feature_038','feature_009','feature_006','feature_028','feature_120','feature_019','feature_092']
for f in features_to_check:
    r = stats.pearsonr(df['feature_078'], df[f])
    print(f"  feature_078 vs {f}: r={r.statistic:.3f}, p={r.pvalue:.2e}")

