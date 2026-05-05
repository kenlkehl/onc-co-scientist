"""Iterations 1-5: main effects of treatments and key features on pfs_months."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_parquet('dataset.parquet')

results = {}

# Iteration 1: each treatment unadjusted main effect on pfs
treatments = [
    'treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
    'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab'
]
for tx in treatments:
    a = df.loc[df[tx]==1,'pfs_months']
    b = df.loc[df[tx]==0,'pfs_months']
    t,p = stats.ttest_ind(a,b,equal_var=False)
    results[f'unadj_{tx}'] = {
        'mean_on': float(a.mean()), 'mean_off': float(b.mean()),
        'diff': float(a.mean()-b.mean()), 'p': float(p), 'n_on': int(len(a))
    }

# Iteration 2: clinical features main effects
for f in ['stage_iv','has_brain_mets','node_positive','postmenopausal',
          'er_positive','pr_positive','her2_positive','her2_low',
          'brca1_mutation','brca2_mutation','pik3ca_mutation','sex_female']:
    a = df.loc[df[f]==1,'pfs_months']; b = df.loc[df[f]==0,'pfs_months']
    if len(a) and len(b):
        t,p = stats.ttest_ind(a,b,equal_var=False)
        results[f'unadj_{f}'] = {
            'mean_yes': float(a.mean()), 'mean_no': float(b.mean()),
            'diff': float(a.mean()-b.mean()), 'p': float(p), 'n_yes': int(len(a))
        }

# Iteration 3: ECOG (multilevel) and continuous predictors
for f in ['ecog_ps']:
    grp = df.groupby(f)['pfs_months'].agg(['mean','std','count']).to_dict()
    results[f'group_{f}'] = grp

# Continuous predictors via Pearson correlation
for f in ['age_years','ki67_pct','tumor_size_cm','albumin_g_dl','ldh_u_l',
         'weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
         'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
         'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']:
    r,p = stats.pearsonr(df[f], df['pfs_months'])
    results[f'corr_{f}'] = {'r': float(r), 'p': float(p)}

with open('out_main.json','w') as fp:
    json.dump(results, fp, indent=2, default=str)
print(json.dumps(results, indent=2, default=str))
