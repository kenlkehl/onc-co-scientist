"""Iteration 10-12: extended interaction scans and final model."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

# ITERATION 10: scan ALL binary features for interaction with feature_078
# (does any binary feature modify the strong feature_078 -> pfs effect?)
nunique = df.nunique()
binary_cols = [c for c in df.columns if c not in ['patient_id','pfs_months']
               and nunique[c]==2 and df[c].dtype != object]

print('=== Scan: binary x feature_078 interactions on pfs_months ===')
results = []
for c in binary_cols:
    formula = f'pfs_months ~ feature_078 + {c} + feature_078:{c}'
    m = smf.ols(formula, data=df).fit()
    inter = f'feature_078:{c}'
    if inter in m.params.index:
        results.append({'feature': c, 'interaction_coef': m.params[inter],
                        'p': m.pvalues[inter]})
res = pd.DataFrame(results).sort_values('p')
print(res.head(20).to_string(index=False))
print(f'\nTotal scanned: {len(res)}; nominal p<0.05: {(res["p"]<0.05).sum()}')
print(f'Bonferroni p<0.05/{len(res)} = {0.05/len(res):.5f}; significant: {(res["p"]<0.05/len(res)).sum()}')

# ITERATION 11: Best parsimonious model with key non-linearities + interactions
print('\n\n=== Final parsimonious model ===')
formula = ('pfs_months ~ feature_078 + I(feature_078**2) '
           '+ log_feature_013 + feature_006 + feature_009 + feature_092 '
           '+ I(feature_092**2) + feature_109 + feature_039 + feature_027 + feature_043 '
           '+ feature_078:log_feature_013 + feature_078:feature_006 + feature_078:feature_009')
m = smf.ols(formula, data=df).fit()
print(f'R² = {m.rsquared:.4f}, Adj R² = {m.rsquared_adj:.4f}')
print(m.summary().tables[1])

# ITERATION 12: Sensitivity — does the headline feature_078 effect hold under quantile regression?
print('\n=== Sensitivity: feature_078 slope by tertile of log_feature_013 ===')
df['l13_tertile'] = pd.qcut(df['log_feature_013'], 3, labels=['low','mid','high'])
for t in ['low','mid','high']:
    sub = df[df['l13_tertile']==t]
    slope, intercept, _, plin, _ = stats.linregress(sub['feature_078'], sub['pfs_months'])
    print(f'  log_feature_013 {t}: n={len(sub)}, slope={slope:.4f}/unit, p={plin:.3e}')

print('\n=== Sensitivity: outcome means at extreme feature_078 values ===')
print(df.groupby(pd.qcut(df['feature_078'], 5))['pfs_months'].agg(['count','mean','std']))
