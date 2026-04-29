"""More targeted refinements: dose-response, more interactions, alt subgroups."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_parquet('../dataset.parquet')

# 1. feature_011 dose-response (ordinal-ish 0-28)
print("=== feature_011 deciles vs response ===")
df['_011_d'] = pd.qcut(df['feature_011'], 5, labels=False, duplicates='drop')
print(df.groupby('_011_d')['objective_response'].agg(['mean', 'count']))
print()

# 2. feature_064 (0-10 ordinal) main effect
m = smf.logit("objective_response ~ feature_064", data=df).fit(disp=0)
print("feature_064 main: coef=%.4f p=%.4g OR=%.4f" % (m.params['feature_064'], m.pvalues['feature_064'], np.exp(m.params['feature_064'])))
print(df.groupby('feature_064')['objective_response'].agg(['mean', 'count']))
print()

# 3. Check feature_092 quartile dose-response
df['_092_q'] = pd.qcut(df['feature_092'], 4, labels=False, duplicates='drop')
print("=== feature_092 quartile rates ===")
print(df.groupby('_092_q')['objective_response'].agg(['mean', 'count']))
print()

# 4. Extra interaction: feature_011 x feature_006
m = smf.logit("objective_response ~ feature_011 * feature_006 + feature_051 + feature_013 + feature_067 + feature_092", data=df).fit(disp=0)
print("=== feature_011 x feature_006 ===")
print(m.summary().tables[1])
print()

# 5. Three-way: ECOG x feature_006
m = smf.logit("objective_response ~ feature_051 * feature_006 + feature_013 + feature_011 + feature_067 + feature_092", data=df).fit(disp=0)
print("=== feature_051 x feature_006 ===")
print(m.summary().tables[1])
print()

# 6. Histology and smoking interactions
m = smf.logit("objective_response ~ C(feature_043) * feature_006 + feature_051 + feature_013 + feature_011 + feature_067 + feature_092", data=df).fit(disp=0)
print("=== histology x feature_006 ===")
print(m.summary().tables[1])
print()

# 7. Insurance disparity in OUTCOME, after adjusting
m = smf.logit("objective_response ~ C(feature_005) + feature_051 + feature_013 + feature_011 + feature_067 + feature_006 + feature_092", data=df).fit(disp=0)
print("=== insurance (feature_005) ===")
print(m.summary().tables[1])
print()

# 8. Race disparity in feature_006 receipt (proxy for treatment access)?
ct = pd.crosstab(df['feature_123'], df['feature_006'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"race x feature_006 chi2 p={p:.3g}")
print(ct)
print(ct.div(ct.sum(axis=1), axis=0).round(3))
print()

# 9. Insurance x feature_006 receipt
ct = pd.crosstab(df['feature_005'], df['feature_006'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"insurance x feature_006 chi2 p={p:.3g}")
print(ct.div(ct.sum(axis=1), axis=0).round(3))
print()

# 10. feature_092 quintile interaction with feature_006: average response within (006, 092 quintile) cells
df['_092_q5'] = pd.qcut(df['feature_092'], 5, labels=False, duplicates='drop')
print('=== response by feature_006 x feature_092 quintile ===')
print(df.groupby(['feature_006', '_092_q5'])['objective_response'].agg(['mean', 'count']))
