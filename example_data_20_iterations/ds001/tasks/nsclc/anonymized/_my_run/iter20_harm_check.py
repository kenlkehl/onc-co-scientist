"""Iteration 20: check for negative (harm) interactions or surprises.
Are there features that REDUCE the triple-combo benefit, or treatments that hurt response?"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist

df = pd.read_parquet('../dataset.parquet')
y = df['objective_response'].values
df['triple'] = ((df['feature_006']==1)&(df['feature_007']==1)&(df['feature_039']==1)).astype(int)
df['f092_hi'] = (df['feature_092']>=0.5).astype(int)
df['triple_hi'] = df['triple']*df['f092_hi']

# Scan all binaries: do any modify the triple_hi effect (3-way interactions)?
print('=== Scan: binary × triple_hi interaction (effect modifiers) ===')
binary_cols = [c for c in df.columns if c not in ('patient_id','objective_response','triple','f092_hi','triple_hi')
               and df[c].dtype != 'object' and df[c].nunique() == 2]

results = []
for c in binary_cols:
    data = pd.DataFrame({
        'mod': df[c].astype(float),
        'tr': df['triple'].astype(float),
        'hi': df['f092_hi'].astype(float),
    })
    data['tr_hi'] = data['tr']*data['hi']
    data['mod_tr'] = data['mod']*data['tr']
    data['mod_hi'] = data['mod']*data['hi']
    data['mod_tr_hi'] = data['mod']*data['tr']*data['hi']
    X_full = sm.add_constant(data).astype(float)
    X_no = X_full.drop(columns=['mod_tr_hi'])
    try:
        rf = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
        rn = sm.Logit(y, X_no).fit(disp=False, maxiter=200)
        lr = 2*(rf.llf - rn.llf)
        p = 1 - chi2_dist.cdf(lr, df=1)
        results.append({'feature': c, 'coef': rf.params['mod_tr_hi'], 'p_value': p})
    except Exception:
        pass

res_df = pd.DataFrame(results).sort_values('p_value')
res_df.to_csv('iter20_harm_check.csv', index=False)
print(res_df.head(15).to_string(index=False))
print(f'\n# Significant (p<0.05/77 = {0.05/77:.4f}): {(res_df.p_value<0.05/77).sum()}')

# Continuous predictors × triple_hi
print('\n=== Scan: continuous × triple_hi interaction ===')
cont_cols = [c for c in df.columns if c not in ('patient_id','objective_response') and df[c].dtype in ('float64','int64')
             and df[c].nunique() > 10]
results_c = []
for c in cont_cols:
    z = (df[c]-df[c].mean())/df[c].std()
    data = pd.DataFrame({
        'mod': z.values,
        'tr': df['triple'].astype(float),
        'hi': df['f092_hi'].astype(float),
    })
    data['tr_hi'] = data['tr']*data['hi']
    data['mod_tr'] = data['mod']*data['tr']
    data['mod_hi'] = data['mod']*data['hi']
    data['mod_tr_hi'] = data['mod']*data['tr']*data['hi']
    X_full = sm.add_constant(data).astype(float)
    X_no = X_full.drop(columns=['mod_tr_hi'])
    try:
        rf = sm.Logit(y, X_full).fit(disp=False, maxiter=200)
        rn = sm.Logit(y, X_no).fit(disp=False, maxiter=200)
        lr = 2*(rf.llf - rn.llf)
        p = 1 - chi2_dist.cdf(lr, df=1)
        results_c.append({'feature': c, 'coef': rf.params['mod_tr_hi'], 'p_value': p})
    except Exception:
        pass
res_c = pd.DataFrame(results_c).sort_values('p_value')
res_c.to_csv('iter20_harm_check_continuous.csv', index=False)
print(res_c.head(10).to_string(index=False))
print(f'\n# Continuous significant (p<0.05/36 = {0.05/36:.4f}): {(res_c.p_value<0.05/36).sum()}')
