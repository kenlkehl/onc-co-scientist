"""Iteration 4: multivariable OLS using top continuous + binary predictors + race/insurance."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

df = pd.read_parquet('../dataset.parquet')

# log-transform highly skewed feature_013 (PSA-like)
df['log_feature_013'] = np.log1p(df['feature_013'])

# Top continuous predictors from iter01
formula = ('pfs_months ~ feature_078 + log_feature_013 + feature_006 + feature_009 '
           '+ feature_051 + feature_109 + feature_039 + feature_105 '
           '+ C(feature_018, Treatment(reference="white")) '
           '+ C(feature_045, Treatment(reference="private"))')

m = smf.ols(formula, data=df).fit()
print(m.summary())
with open('iter04_multivar_summary.txt','w') as f:
    f.write(str(m.summary()))
