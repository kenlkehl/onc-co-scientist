"""Iterative analysis of ds001_crc dataset.
Outputs JSON results that will be assembled into transcript.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
print(df.shape)

results = {}

# ---------- Iteration 1: main effects of clinical features on PFS ----------
def lin_main(y, x):
    X = sm.add_constant(x.astype(float))
    m = sm.OLS(y, X).fit()
    return float(m.params.iloc[1]), float(m.pvalues.iloc[1])

iter1 = {}
for col in ['ecog_ps','stage_iv','age_years','albumin_g_dl','ldh_u_l',
            'cea_ng_ml','weight_loss_pct_6mo','crp_mg_l','nlr',
            'hemoglobin_g_dl','alkaline_phosphatase_u_l','sex_female',
            'right_sided_primary']:
    eff, pv = lin_main(df['pfs_months'], df[col])
    iter1[col] = {'effect': eff, 'p': pv}
results['iter1_main_effects'] = iter1
print('iter1 done')

# ---------- Iteration 2: biomarker main effects on PFS ----------
iter2 = {}
for col in ['kras_mutation','nras_mutation','braf_v600e','msi_high',
            'her2_amplified','ntrk_fusion']:
    eff, pv = lin_main(df['pfs_months'], df[col])
    iter2[col] = {'effect': eff, 'p': pv,
                  'mean_pos': float(df.loc[df[col]==1,'pfs_months'].mean()),
                  'mean_neg': float(df.loc[df[col]==0,'pfs_months'].mean()),
                  'n_pos': int(df[col].sum())}
results['iter2_biomarker_main'] = iter2
print('iter2 done')

# ---------- Iteration 3: treatment main effects on PFS (unadjusted) ----------
iter3 = {}
for tx in ['treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
           'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']:
    eff, pv = lin_main(df['pfs_months'], df[tx])
    iter3[tx] = {'effect': eff, 'p': pv,
                 'mean_treated': float(df.loc[df[tx]==1,'pfs_months'].mean()),
                 'mean_untreated': float(df.loc[df[tx]==0,'pfs_months'].mean()),
                 'n_treated': int(df[tx].sum())}
results['iter3_tx_main'] = iter3
print('iter3 done')

# ---------- Iteration 4: ECOG ordinal ----------
iter4 = {}
for k in [0,1,2]:
    sub = df[df['ecog_ps']==k]['pfs_months']
    iter4[f'ecog_{k}'] = {'mean': float(sub.mean()), 'n': int(len(sub))}
# ANOVA across ecog
g = [df.loc[df['ecog_ps']==k,'pfs_months'].values for k in [0,1,2]]
f, pv = stats.f_oneway(*g)
iter4['anova'] = {'F': float(f), 'p': float(pv)}
results['iter4_ecog_ordinal'] = iter4
print('iter4 done')

# ---------- Iteration 5: pembrolizumab × MSI interaction ----------
def interact(formula, data):
    m = smf.ols(formula, data=data).fit()
    return m

m5 = interact('pfs_months ~ treatment_pembrolizumab * msi_high', df)
iter5 = {'params': m5.params.to_dict(),
         'pvalues': m5.pvalues.to_dict()}
# simple subgroup means
for msi in [0,1]:
    for tx in [0,1]:
        sub = df[(df['msi_high']==msi)&(df['treatment_pembrolizumab']==tx)]
        iter5[f'msi{msi}_pembro{tx}'] = {'mean': float(sub['pfs_months'].mean()), 'n': int(len(sub))}
results['iter5_pembro_x_msi'] = iter5
print('iter5 done')

# ---------- Iteration 6: encorafenib × BRAF V600E interaction ----------
m6 = interact('pfs_months ~ treatment_encorafenib * braf_v600e', df)
iter6 = {'params': m6.params.to_dict(),
         'pvalues': m6.pvalues.to_dict()}
for b in [0,1]:
    for tx in [0,1]:
        sub = df[(df['braf_v600e']==b)&(df['treatment_encorafenib']==tx)]
        iter6[f'braf{b}_enco{tx}'] = {'mean': float(sub['pfs_months'].mean()), 'n': int(len(sub))}
results['iter6_enco_x_braf'] = iter6
print('iter6 done')

# ---------- Iteration 7: trastuzumab_tucatinib × HER2 ----------
m7 = interact('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', df)
iter7 = {'params': m7.params.to_dict(),
         'pvalues': m7.pvalues.to_dict()}
for h in [0,1]:
    for tx in [0,1]:
        sub = df[(df['her2_amplified']==h)&(df['treatment_trastuzumab_tucatinib']==tx)]
        iter7[f'her2_{h}_tt{tx}'] = {'mean': float(sub['pfs_months'].mean()), 'n': int(len(sub))}
results['iter7_tt_x_her2'] = iter7
print('iter7 done')

# ---------- Iteration 8: cetuximab × KRAS interaction ----------
m8 = interact('pfs_months ~ treatment_cetuximab * kras_mutation', df)
iter8 = {'params': m8.params.to_dict(),
         'pvalues': m8.pvalues.to_dict()}
for k in [0,1]:
    for tx in [0,1]:
        sub = df[(df['kras_mutation']==k)&(df['treatment_cetuximab']==tx)]
        iter8[f'kras{k}_cet{tx}'] = {'mean': float(sub['pfs_months'].mean()), 'n': int(len(sub))}
results['iter8_cet_x_kras'] = iter8
print('iter8 done')

# ---------- Iteration 9: cetuximab × RAS WT × BRAF WT × left-sided ----------
df['all_wt'] = ((df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)).astype(int)
df['left_sided'] = (df['right_sided_primary']==0).astype(int)
df['cet_eligible'] = (df['all_wt']==1)&(df['left_sided']==1)

iter9 = {}
# overall cetuximab in all_wt + left_sided
for sub_def, mask in [
    ('all_wt_left_sided', (df['all_wt']==1)&(df['left_sided']==1)),
    ('all_wt_right_sided', (df['all_wt']==1)&(df['right_sided_primary']==1)),
    ('any_ras_or_braf_mut', df['all_wt']==0),
]:
    sub = df[mask]
    if len(sub) < 50: continue
    treated = sub[sub['treatment_cetuximab']==1]['pfs_months']
    untreated = sub[sub['treatment_cetuximab']==0]['pfs_months']
    if len(treated)<10 or len(untreated)<10: continue
    eff = float(treated.mean() - untreated.mean())
    t, pv = stats.ttest_ind(treated, untreated, equal_var=False)
    iter9[sub_def] = {'effect': eff, 'p': float(pv),
                     'mean_treated': float(treated.mean()),
                     'mean_untreated': float(untreated.mean()),
                     'n_treated': int(len(treated)), 'n_untreated': int(len(untreated))}
results['iter9_cet_subgroups'] = iter9
print('iter9 done')

# ---------- Iteration 10: bevacizumab main and × subgroups ----------
iter10 = {}
for sub_name, mask in [
    ('all', np.ones(len(df), dtype=bool)),
    ('right_sided', df['right_sided_primary']==1),
    ('left_sided', df['right_sided_primary']==0),
    ('kras_mut', df['kras_mutation']==1),
    ('kras_wt', df['kras_mutation']==0),
    ('stage_iv', df['stage_iv']==1),
    ('not_stage_iv', df['stage_iv']==0),
    ('msi_high', df['msi_high']==1),
    ('msi_low', df['msi_high']==0),
]:
    sub = df[mask]
    treated = sub[sub['treatment_bevacizumab']==1]['pfs_months']
    untreated = sub[sub['treatment_bevacizumab']==0]['pfs_months']
    if len(treated)<10 or len(untreated)<10: continue
    eff = float(treated.mean() - untreated.mean())
    t, pv = stats.ttest_ind(treated, untreated, equal_var=False)
    iter10[sub_name] = {'effect': eff, 'p': float(pv),
                       'n_t': int(len(treated)), 'n_u': int(len(untreated))}
results['iter10_bev_subgroups'] = iter10
print('iter10 done')

# ---------- Iteration 11: regorafenib subgroups ----------
iter11 = {}
for sub_name, mask in [
    ('all', np.ones(len(df), dtype=bool)),
    ('ecog_0', df['ecog_ps']==0),
    ('ecog_1', df['ecog_ps']==1),
    ('ecog_2', df['ecog_ps']==2),
    ('alb_low', df['albumin_g_dl']<3.5),
    ('alb_high', df['albumin_g_dl']>=3.5),
    ('high_ldh', df['ldh_u_l']>df['ldh_u_l'].median()),
    ('low_ldh', df['ldh_u_l']<=df['ldh_u_l'].median()),
    ('high_cea', df['cea_ng_ml']>df['cea_ng_ml'].median()),
    ('low_cea', df['cea_ng_ml']<=df['cea_ng_ml'].median()),
    ('high_crp', df['crp_mg_l']>df['crp_mg_l'].median()),
    ('low_crp', df['crp_mg_l']<=df['crp_mg_l'].median()),
]:
    sub = df[mask]
    treated = sub[sub['treatment_regorafenib']==1]['pfs_months']
    untreated = sub[sub['treatment_regorafenib']==0]['pfs_months']
    if len(treated)<20 or len(untreated)<20: continue
    eff = float(treated.mean() - untreated.mean())
    t, pv = stats.ttest_ind(treated, untreated, equal_var=False)
    iter11[sub_name] = {'effect': eff, 'p': float(pv),
                       'n_t': int(len(treated)), 'n_u': int(len(untreated))}
results['iter11_rego_subgroups'] = iter11
print('iter11 done')

# Save
with open('iter_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('saved')
