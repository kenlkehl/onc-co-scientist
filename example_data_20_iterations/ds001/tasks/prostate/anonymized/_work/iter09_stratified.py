"""Iteration 9: stratified subgroup analyses to characterize heterogeneity."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

# (A) feature_051 effect within tertiles of feature_078 (do treatment-related signals
#     emerge among older vs younger / higher vs lower performance patients?)
df['t78_tertile'] = pd.qcut(df['feature_078'], 3, labels=['low','mid','high'])
print('=== feature_051 effect within feature_078 tertiles (Welch t-test) ===')
for t in ['low','mid','high']:
    sub = df[df['t78_tertile']==t]
    g1 = sub.loc[sub['feature_051']==1,'pfs_months']
    g0 = sub.loc[sub['feature_051']==0,'pfs_months']
    diff = g1.mean() - g0.mean()
    t_stat, p = stats.ttest_ind(g1, g0, equal_var=False)
    print(f'  feature_078 {t}: n1={len(g1)}, n0={len(g0)}, diff={diff:+.3f}, t={t_stat:.2f}, p={p:.3e}')

# (B) feature_078 effect (Spearman) within race subgroups
print('\n=== feature_078 vs pfs_months Spearman by race ===')
for r in df['feature_018'].unique():
    sub = df[df['feature_018']==r]
    rho, p = stats.spearmanr(sub['feature_078'], sub['pfs_months'])
    # also linear slope
    slope, intercept, _, plin, _ = stats.linregress(sub['feature_078'], sub['pfs_months'])
    print(f'  {r} (n={len(sub)}): rho={rho:.3f} p={p:.3e}; slope={slope:.4f}/unit')

# (C) Equity: do residuals from the main model differ by race/insurance?
# Fit model WITHOUT race/insurance, then check if residuals correlate with them.
fit_terms = ['feature_078','log_feature_013','feature_006','feature_009','feature_092',
             'feature_109','feature_039','feature_027','feature_043']
formula = 'pfs_months ~ ' + ' + '.join(fit_terms)
m = smf.ols(formula, data=df).fit()
df['resid'] = m.resid

print('\n=== Residuals by race ===')
for r in df['feature_018'].unique():
    g = df.loc[df['feature_018']==r,'resid']
    print(f'  {r}: n={len(g)}, mean={g.mean():+.4f}, std={g.std():.4f}')
F, p = stats.f_oneway(*[df.loc[df['feature_018']==r,'resid'] for r in df['feature_018'].unique()])
print(f'  ANOVA across races: F={F:.3f}, p={p:.3e}')

print('\n=== Residuals by insurance ===')
for r in df['feature_045'].unique():
    g = df.loc[df['feature_045']==r,'resid']
    print(f'  {r}: n={len(g)}, mean={g.mean():+.4f}, std={g.std():.4f}')
F, p = stats.f_oneway(*[df.loc[df['feature_045']==r,'resid'] for r in df['feature_045'].unique()])
print(f'  ANOVA across insurance: F={F:.3f}, p={p:.3e}')
