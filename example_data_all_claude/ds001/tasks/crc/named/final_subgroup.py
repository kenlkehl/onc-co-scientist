"""Final regorafenib subgroup verification."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
df['cea_low'] = (df['cea_ng_ml'] <= df['cea_ng_ml'].median()).astype(int)
df['left_sided'] = (df['right_sided_primary']==0).astype(int)
df['kras_wt'] = (df['kras_mutation']==0).astype(int)
df['braf_wt'] = (df['braf_v600e']==0).astype(int)
df['nras_wt'] = (df['nras_mutation']==0).astype(int)

def subgroup_tx(mask, tx, df=df):
    sub = df[mask]
    t = sub[sub[tx]==1]['pfs_months']
    u = sub[sub[tx]==0]['pfs_months']
    if len(t)<10 or len(u)<10:
        return None
    eff = float(t.mean() - u.mean())
    _, pv = stats.ttest_ind(t, u, equal_var=False)
    return {'effect': eff, 'p': float(pv),
            'n_t': int(len(t)), 'n_u': int(len(u)),
            'mean_t': float(t.mean()), 'mean_u': float(u.mean())}

results = {}
# Final big subgroup: regorafenib in left-sided + KRAS WT + BRAF WT + (low CEA?)
for sub_def, mask in [
    ('left_kraswt_brafwt', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)),
    ('left_kraswt_brafwt_cealow', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['cea_low']==1)),
    ('left_kraswt_brafwt_ceahigh', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['cea_low']==0)),
    ('left_kraswt_brafwt_nraswt', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['nras_wt']==1)),
    ('left_kraswt_brafwt_nraswt_cealow', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['nras_wt']==1)&(df['cea_low']==1)),
    ('rightsided_only', df['left_sided']==0),
    ('kras_mut_only', df['kras_wt']==0),
    ('braf_v600e_only', df['braf_wt']==0),
    ('cea_high_only', df['cea_low']==0),
    # Each "unfavorable" combined to demonstrate suppressor effects
    ('left_kraswt_brafwt_ceahigh_check', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['cea_low']==0)),
    ('right_kraswt_brafwt_cealow_check', (df['left_sided']==0)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['cea_low']==1)),
    ('left_krasmut_brafwt_cealow_check', (df['left_sided']==1)&(df['kras_wt']==0)&(df['braf_wt']==1)&(df['cea_low']==1)),
    ('left_kraswt_brafmut_cealow_check', (df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==0)&(df['cea_low']==1)),
]:
    results[sub_def] = subgroup_tx(mask, 'treatment_regorafenib')

# Multivariable model with the four-way subgroup
df['rego_subgroup'] = ((df['left_sided']==1)&(df['kras_wt']==1)&(df['braf_wt']==1)&(df['cea_low']==1)).astype(int)
m = smf.ols('pfs_months ~ treatment_regorafenib * rego_subgroup', df).fit()
results['interaction_test'] = {
    'rego_main_outside': float(m.params['treatment_regorafenib']),
    'rego_main_outside_p': float(m.pvalues['treatment_regorafenib']),
    'subgroup_main': float(m.params['rego_subgroup']),
    'interaction_coef': float(m.params['treatment_regorafenib:rego_subgroup']),
    'interaction_p': float(m.pvalues['treatment_regorafenib:rego_subgroup']),
    'rego_in_subgroup': float(m.params['treatment_regorafenib'] + m.params['treatment_regorafenib:rego_subgroup']),
}

# Adjusted multivariable
m2 = smf.ols('pfs_months ~ treatment_regorafenib * rego_subgroup + age_years + ecog_ps + stage_iv + albumin_g_dl + weight_loss_pct_6mo + ldh_u_l', df).fit()
results['interaction_adj'] = {
    'rego_main_outside': float(m2.params['treatment_regorafenib']),
    'rego_main_outside_p': float(m2.pvalues['treatment_regorafenib']),
    'interaction_coef': float(m2.params['treatment_regorafenib:rego_subgroup']),
    'interaction_p': float(m2.pvalues['treatment_regorafenib:rego_subgroup']),
    'rego_in_subgroup': float(m2.params['treatment_regorafenib'] + m2.params['treatment_regorafenib:rego_subgroup']),
}

with open('final_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print(json.dumps(results, indent=2, default=str))
