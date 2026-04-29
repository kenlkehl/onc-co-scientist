import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')

results = {}

# Iteration 1: Demographics - race & insurance
# H1: Black race is associated with shorter PFS than white
# H2: Hispanic race is associated with shorter PFS than white
# H3: Patients with medicaid have shorter PFS than private
# H4: Uninsured patients have shorter PFS than private

# Race effects
race = df['feature_018']
pfs = df['pfs_months']

# H1 black vs white
b = pfs[race=='black']
w = pfs[race=='white']
t1 = stats.ttest_ind(b, w, equal_var=False)
results['h1'] = {'mean_black': float(b.mean()), 'mean_white': float(w.mean()),
                 'diff': float(b.mean()-w.mean()), 'p': float(t1.pvalue), 't': float(t1.statistic)}

# H2 hispanic vs white
h = pfs[race=='hispanic']
t2 = stats.ttest_ind(h, w, equal_var=False)
results['h2'] = {'mean_hispanic': float(h.mean()), 'mean_white': float(w.mean()),
                 'diff': float(h.mean()-w.mean()), 'p': float(t2.pvalue), 't': float(t2.statistic)}

# H3 medicaid vs private
ins = df['feature_045']
m = pfs[ins=='medicaid']
p = pfs[ins=='private']
t3 = stats.ttest_ind(m, p, equal_var=False)
results['h3'] = {'mean_medicaid': float(m.mean()), 'mean_private': float(p.mean()),
                 'diff': float(m.mean()-p.mean()), 'p': float(t3.pvalue), 't': float(t3.statistic)}

# H4 uninsured vs private
u = pfs[ins=='uninsured']
t4 = stats.ttest_ind(u, p, equal_var=False)
results['h4'] = {'mean_uninsured': float(u.mean()), 'mean_private': float(p.mean()),
                 'diff': float(u.mean()-p.mean()), 'p': float(t4.pvalue), 't': float(t4.statistic)}

# H5 medicare vs private  
me = pfs[ins=='medicare']
t5 = stats.ttest_ind(me, p, equal_var=False)
results['h5'] = {'mean_medicare': float(me.mean()), 'mean_private': float(p.mean()),
                 'diff': float(me.mean()-p.mean()), 'p': float(t5.pvalue), 't': float(t5.statistic)}

# Iteration 2: Age (feature_078, range 30-90 mean 65)
# H6: Older age associated with shorter PFS
# H7: Per 10-year increase, PFS decreases linearly

X = sm.add_constant(df['feature_078'])
m6 = sm.OLS(df['pfs_months'], X).fit()
results['h6'] = {'beta_age': float(m6.params['feature_078']),
                 'p': float(m6.pvalues['feature_078']),
                 'beta_per10': float(m6.params['feature_078']*10)}

# Iteration 3: Hemoglobin (feature_038, range 6-18, mean 12.5)
# H8: Lower hemoglobin associated with shorter PFS
X = sm.add_constant(df['feature_038'])
m8 = sm.OLS(df['pfs_months'], X).fit()
results['h8'] = {'beta_hgb': float(m8.params['feature_038']),
                 'p': float(m8.pvalues['feature_038'])}

# Iteration 4: PSA-like marker (feature_013, 0.12-3622 mean 42, log skewed)
# H9: Log(feature_013) negatively associated with PFS
log13 = np.log(df['feature_013'].clip(lower=0.01))
X = sm.add_constant(log13)
X.columns = ['const', 'log_f013']
m9 = sm.OLS(df['pfs_months'], X).fit()
results['h9'] = {'beta_log_f013': float(m9.params['log_f013']),
                 'p': float(m9.pvalues['log_f013'])}

# Iteration 5: ALP-like (feature_092)
log92 = np.log(df['feature_092'].clip(lower=0.01))
X = sm.add_constant(log92)
X.columns = ['const', 'log_f092']
m10 = sm.OLS(df['pfs_months'], X).fit()
results['h10'] = {'beta_log_f092': float(m10.params['log_f092']),
                  'p': float(m10.pvalues['log_f092'])}

# H11: Performance status (feature_057, 0-2)
# Higher PS = shorter PFS?
groups = [df.loc[df['feature_057']==v,'pfs_months'] for v in [0,1,2]]
f57 = stats.f_oneway(*groups)
mean_by = df.groupby('feature_057')['pfs_months'].mean().to_dict()
results['h11'] = {'mean_by_f057': {int(k):float(v) for k,v in mean_by.items()},
                  'p': float(f57.pvalue)}

# H12: Gleason-like (feature_067, 6-10)
mean_by = df.groupby('feature_067')['pfs_months'].mean().to_dict()
groups = [df.loc[df['feature_067']==v,'pfs_months'] for v in sorted(df['feature_067'].unique())]
f67 = stats.f_oneway(*groups)
results['h12'] = {'mean_by_f067': {int(k):float(v) for k,v in mean_by.items()},
                  'p': float(f67.pvalue)}

print(json.dumps(results, indent=2))
