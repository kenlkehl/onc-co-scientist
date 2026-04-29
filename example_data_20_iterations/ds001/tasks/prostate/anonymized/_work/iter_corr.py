import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')

# Examine feature_051 — strongest binary predictor
# What's it correlated with?
print("=== feature_051 correlations ===")
log_psa = np.log(df['feature_013'].clip(lower=0.01))
log_alp = np.log(df['feature_092'].clip(lower=0.01))

# By feature_051
for f in ['feature_078','feature_038','feature_009','feature_006']:
    a = df.loc[df['feature_051']==1, f].mean()
    b = df.loc[df['feature_051']==0, f].mean()
    print(f"  {f}: f051=1 mean={a:.3f}, f051=0 mean={b:.3f}")

print(f"  log_PSA: f051=1 mean={log_psa[df['feature_051']==1].mean():.3f}, f051=0 mean={log_psa[df['feature_051']==0].mean():.3f}")
print(f"  log_ALP: f051=1 mean={log_alp[df['feature_051']==1].mean():.3f}, f051=0 mean={log_alp[df['feature_051']==0].mean():.3f}")

# By PS
print('PS vs f051:')
print(pd.crosstab(df['feature_057'], df['feature_051'], normalize='columns'))

# Compute multivariable adjusted effect of f051
df2 = df.copy()
df2['log_f013'] = np.log(df['feature_013'].clip(lower=0.01))
df2['log_f092'] = np.log(df['feature_092'].clip(lower=0.01))
m = smf.ols('pfs_months ~ feature_051 + feature_078 + feature_038 + feature_009 + feature_006 + log_f013 + log_f092 + C(feature_057)', data=df2).fit()
print('\n=== Multivariable model ===')
print(m.summary().tables[1])

# Effect of f051 after adjusting
print(f"\nAdjusted f051 beta: {m.params['feature_051']:.4f}, p={m.pvalues['feature_051']:.2e}")
