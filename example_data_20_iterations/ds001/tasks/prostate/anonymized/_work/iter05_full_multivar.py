"""Iteration 5: full multivariable OLS with ALL features (continuous, binary, categorical)."""
import pandas as pd, numpy as np
import statsmodels.api as sm

df = pd.read_parquet('../dataset.parquet')
df['log_feature_013'] = np.log1p(df['feature_013'])

nunique = df.nunique()
exclude = {'patient_id','pfs_months','feature_013'}  # use log version for 013
cont_cols = [c for c in df.columns if c not in exclude and nunique[c] > 10 and df[c].dtype != object]
bin_cols = [c for c in df.columns if c not in exclude and nunique[c] == 2 and df[c].dtype != object]

# add log_feature_013 to cont_cols, drop original
cont_cols.append('log_feature_013')

# build design matrix
X_cont = df[cont_cols].copy()
X_bin = df[bin_cols].copy()
X_race = pd.get_dummies(df['feature_018'], prefix='race', drop_first=True)
X_ins = pd.get_dummies(df['feature_045'], prefix='ins', drop_first=True)

X = pd.concat([X_cont, X_bin, X_race, X_ins], axis=1).astype(float)
X = sm.add_constant(X)
y = df['pfs_months'].values

m = sm.OLS(y, X).fit()
# Save full summary
with open('iter05_full_multivar_summary.txt','w') as f:
    f.write(str(m.summary()))

# Print sorted by abs(t)
res = pd.DataFrame({'coef': m.params, 'se': m.bse, 't': m.tvalues, 'p': m.pvalues})
res['abs_t'] = res['t'].abs()
res = res.sort_values('abs_t', ascending=False)
print('R^2:', m.rsquared, 'Adj R^2:', m.rsquared_adj, 'N:', int(m.nobs))
print('\nTop 30 predictors by |t|:')
print(res.head(30).to_string())
print('\nSignificant at p<0.001 (after Bonferroni for ~123 tests, p<4e-4):')
res_sig = res[res['p'] < 0.001].sort_values('p')
print(res_sig.to_string())
res.to_csv('iter05_full_multivar_coefs.csv')
