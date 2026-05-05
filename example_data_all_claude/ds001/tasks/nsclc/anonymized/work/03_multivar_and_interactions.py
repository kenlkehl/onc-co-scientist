"""Iteration 3-4: multivariable prognostic model + treatment-by-feature interaction screen.

For each treatment candidate (T in feature_012, feature_018, feature_020, feature_027),
fit OLS: pfs_months ~ T*X + Z, where X cycles through all other features and Z is a fixed
covariate set (top prognostic features).
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet').copy()

# Encode categoricals
df['hist_squam'] = (df['feature_006'] == 'squamous').astype(int)
df['smk_current'] = (df['feature_001'] == 'current').astype(int)
df['smk_former'] = (df['feature_001'] == 'former').astype(int)
df['smk_never'] = (df['feature_001'] == 'never').astype(int)

all_features = [c for c in df.columns if c not in ('patient_id', 'pfs_months', 'feature_001', 'feature_006')]
all_features += ['hist_squam', 'smk_current', 'smk_former']  # never is reference

# Multivariable prognostic OLS with all features
formula = "pfs_months ~ " + " + ".join(all_features)
m = smf.ols(formula, data=df).fit()
print("=== Multivariable OLS (all features) ===")
print(f"R-squared: {m.rsquared:.4f}")
# Print top features by t-stat
coefs = m.params.drop('Intercept')
ses = m.bse.drop('Intercept')
ps = m.pvalues.drop('Intercept')
ord_idx = (coefs/ses).abs().sort_values(ascending=False).index
print(f"\n{'feature':<18}{'beta':>10}{'se':>10}{'t':>10}{'p':>14}")
for f in ord_idx[:25]:
    print(f"{f:<18}{coefs[f]:>10.4f}{ses[f]:>10.4f}{coefs[f]/ses[f]:>10.2f}{ps[f]:>14.3e}")

# Save params for later use
import json
with open('work/03_full_model.json', 'w') as fh:
    json.dump({'r2': m.rsquared, 'coefs': coefs.to_dict(), 'ses': ses.to_dict(), 'p': ps.to_dict()}, fh, indent=2, default=str)
