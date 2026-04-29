"""Iteration 17: final consolidated multivariable logistic model with all confirmed effects.
Estimate adjusted OR and predicted ORR for key patient profiles."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)
df['f092_hi'] = (df['feature_092']>=0.5).astype(int)

def std(s): return (s - s.mean()) / s.std()

X = pd.DataFrame({
    # confirmed prognostic / negative predictors
    'f013': df['feature_013'].astype(float),
    'f067': df['feature_067'].astype(float),
    'f011_z': std(df['feature_011']),
    'f051_z': std(df['feature_051']),
    'f099_z': std(df['feature_099']),
    'f063_z': std(df['feature_063']),
    # treatments and biomarker
    'f006': df['feature_006'].astype(float),
    'f007': df['feature_007'].astype(float),
    'f039': df['feature_039'].astype(float),
    'f092_hi': df['f092_hi'].astype(float),
    'triple': df['triple'].astype(float),
    'triple_x_hi': df['triple']*df['f092_hi'],
})
# Drop the redundant individual treatment terms when triple=1 captures all 3 together
# Actually, keep them all -- triple represents the supra-additive 3-way effect

X = sm.add_constant(X).astype(float)
res = sm.Logit(y, X).fit(disp=False, maxiter=300)
print(res.summary())

# Save
out = pd.DataFrame({'coef': res.params, 'se': res.bse, 'p_value': res.pvalues, 'or': np.exp(res.params)})
out.to_csv('iter17_final_model.csv')
print(out.to_string())

# Predicted ORR for key profiles
print('\n=== Predicted ORR by treatment × biomarker (mean covariate values) ===')
profiles = [
    ('No treatment, biomarker low',  {'f006':0,'f007':0,'f039':0,'f092_hi':0,'triple':0,'triple_x_hi':0}),
    ('No treatment, biomarker high', {'f006':0,'f007':0,'f039':0,'f092_hi':1,'triple':0,'triple_x_hi':0}),
    ('Single (006), biomarker low',  {'f006':1,'f007':0,'f039':0,'f092_hi':0,'triple':0,'triple_x_hi':0}),
    ('Single (006), biomarker high', {'f006':1,'f007':0,'f039':0,'f092_hi':1,'triple':0,'triple_x_hi':0}),
    ('Triple, biomarker low',        {'f006':1,'f007':1,'f039':1,'f092_hi':0,'triple':1,'triple_x_hi':0}),
    ('Triple, biomarker high',       {'f006':1,'f007':1,'f039':1,'f092_hi':1,'triple':1,'triple_x_hi':1}),
]
mean_row = {k: (X[k].mean() if k != 'const' else 1.0) for k in X.columns}
for name, override in profiles:
    row = mean_row.copy()
    row.update(override)
    logit = sum(res.params[k]*row[k] for k in X.columns)
    p = 1/(1+np.exp(-logit))
    print(f'  {name:35s}: predicted ORR = {p:.3f}')
