"""More iterations: heterogeneity searches and refined subgroups."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
results = {}

# helper: subgroup t-test on PFS by treatment
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

# ---------- Iteration 12: regorafenib × CEA (continuous and tertile) ----------
m12 = smf.ols('pfs_months ~ treatment_regorafenib * cea_ng_ml', df).fit()
iter12 = {'params': m12.params.to_dict(), 'pvalues': m12.pvalues.to_dict()}
# tertiles
df['cea_tertile'] = pd.qcut(df['cea_ng_ml'], 3, labels=['low','mid','high'])
for t in ['low','mid','high']:
    iter12[f'rego_in_cea_{t}'] = subgroup_tx(df['cea_tertile']==t, 'treatment_regorafenib')
results['iter12_rego_cea'] = iter12
print('iter12 done')

# ---------- Iteration 13: regorafenib × biomarker screen ----------
iter13 = {}
for bm in ['kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified',
           'ntrk_fusion','right_sided_primary','stage_iv','sex_female']:
    f = f'pfs_months ~ treatment_regorafenib * {bm}'
    m = smf.ols(f, df).fit()
    iter13[bm] = {
        'interaction_coef': float(m.params[f'treatment_regorafenib:{bm}']),
        'interaction_p': float(m.pvalues[f'treatment_regorafenib:{bm}']),
        'rego_main_in_bm0': subgroup_tx(df[bm]==0, 'treatment_regorafenib'),
        'rego_main_in_bm1': subgroup_tx(df[bm]==1, 'treatment_regorafenib'),
    }
results['iter13_rego_x_biomarkers'] = iter13
print('iter13 done')

# ---------- Iteration 14: regorafenib × continuous features ----------
iter14 = {}
for ft in ['age_years','ecog_ps','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo',
           'crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l',
           'ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl',
           'bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']:
    f = f'pfs_months ~ treatment_regorafenib * {ft}'
    m = smf.ols(f, df).fit()
    iter14[ft] = {
        'interaction_coef': float(m.params[f'treatment_regorafenib:{ft}']),
        'interaction_p': float(m.pvalues[f'treatment_regorafenib:{ft}']),
    }
results['iter14_rego_continuous'] = iter14
print('iter14 done')

# ---------- Iteration 15: multivariable model — features + Tx, with CEA-tx interaction ----------
covars = ['age_years','sex_female','ecog_ps','stage_iv','right_sided_primary',
          'kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified',
          'ntrk_fusion','cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo',
          'crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l',
          'ast_u_l','alt_u_l','total_bilirubin_mg_dl','creatinine_mg_dl',
          'bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
txs = ['treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
       'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']
formula = 'pfs_months ~ ' + ' + '.join(covars + txs)
m15 = smf.ols(formula, df).fit()
iter15 = {'params': m15.params.to_dict(), 'pvalues': m15.pvalues.to_dict(),
          'rsq': float(m15.rsquared), 'n': int(m15.nobs)}
results['iter15_multivariable'] = iter15
print('iter15 done')

# ---------- Iteration 16: refined regorafenib subgroups (low CEA × other) ----------
df['cea_low'] = (df['cea_ng_ml'] <= df['cea_ng_ml'].median()).astype(int)
df['ldh_low'] = (df['ldh_u_l'] <= df['ldh_u_l'].median()).astype(int)
df['ldh_high'] = (df['ldh_u_l'] > df['ldh_u_l'].median()).astype(int)

iter16 = {}
for sub_def, mask in [
    ('cea_low', df['cea_low']==1),
    ('cea_high', df['cea_low']==0),
    ('cea_low_alb_high', (df['cea_low']==1)&(df['albumin_g_dl']>=3.5)),
    ('cea_low_ldh_low', (df['cea_low']==1)&(df['ldh_low']==1)),
    ('cea_low_ecog_lt2', (df['cea_low']==1)&(df['ecog_ps']<2)),
    ('cea_low_stage4', (df['cea_low']==1)&(df['stage_iv']==1)),
    ('cea_low_not_stage4', (df['cea_low']==1)&(df['stage_iv']==0)),
    ('cea_low_left_sided', (df['cea_low']==1)&(df['right_sided_primary']==0)),
    ('cea_low_right_sided', (df['cea_low']==1)&(df['right_sided_primary']==1)),
    ('cea_low_kras_wt', (df['cea_low']==1)&(df['kras_mutation']==0)),
    ('cea_low_kras_mut', (df['cea_low']==1)&(df['kras_mutation']==1)),
    ('cea_low_braf_wt', (df['cea_low']==1)&(df['braf_v600e']==0)),
    ('cea_low_msi_low', (df['cea_low']==1)&(df['msi_high']==0)),
]:
    iter16[sub_def] = subgroup_tx(mask, 'treatment_regorafenib')
results['iter16_rego_refined'] = iter16
print('iter16 done')

# ---------- Iteration 17: cetuximab refined ----------
df['ras_wt'] = ((df['kras_mutation']==0)&(df['nras_mutation']==0)).astype(int)
df['triple_wt'] = ((df['kras_mutation']==0)&(df['nras_mutation']==0)&(df['braf_v600e']==0)).astype(int)

iter17 = {}
for sub_def, mask in [
    ('all', np.ones(len(df), dtype=bool)),
    ('ras_wt', df['ras_wt']==1),
    ('ras_mut', df['ras_wt']==0),
    ('triple_wt', df['triple_wt']==1),
    ('triple_wt_left', (df['triple_wt']==1)&(df['right_sided_primary']==0)),
    ('triple_wt_right', (df['triple_wt']==1)&(df['right_sided_primary']==1)),
    ('triple_wt_left_msi_low', (df['triple_wt']==1)&(df['right_sided_primary']==0)&(df['msi_high']==0)),
    ('triple_wt_left_msi_high', (df['triple_wt']==1)&(df['right_sided_primary']==0)&(df['msi_high']==1)),
    ('triple_wt_left_her2_neg', (df['triple_wt']==1)&(df['right_sided_primary']==0)&(df['her2_amplified']==0)),
    ('left_sided', df['right_sided_primary']==0),
    ('right_sided', df['right_sided_primary']==1),
]:
    iter17[sub_def] = subgroup_tx(mask, 'treatment_cetuximab')
results['iter17_cet_refined'] = iter17
print('iter17 done')

# ---------- Iteration 18: pembrolizumab × MSI revisited with multivariable adjustment ----------
m18 = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high + age_years + sex_female + ecog_ps + stage_iv + right_sided_primary + kras_mutation + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + cea_ng_ml', df).fit()
iter18 = {
    'pembro_main': float(m18.params['treatment_pembrolizumab']),
    'pembro_main_p': float(m18.pvalues['treatment_pembrolizumab']),
    'msi_main': float(m18.params['msi_high']),
    'msi_main_p': float(m18.pvalues['msi_high']),
    'interaction': float(m18.params['treatment_pembrolizumab:msi_high']),
    'interaction_p': float(m18.pvalues['treatment_pembrolizumab:msi_high']),
}
# also alt interactions: pembro × tumor side, pembro × ECOG
for v in ['right_sided_primary','ecog_ps','kras_mutation','braf_v600e','her2_amplified','stage_iv']:
    f = f'pfs_months ~ treatment_pembrolizumab * {v}'
    m = smf.ols(f, df).fit()
    iter18[f'pembro_x_{v}'] = {
        'coef': float(m.params[f'treatment_pembrolizumab:{v}']),
        'p': float(m.pvalues[f'treatment_pembrolizumab:{v}']),
    }
results['iter18_pembro_revisit'] = iter18
print('iter18 done')

# ---------- Iteration 19: encorafenib × BRAF revisited with adjustment, also × MEK partner candidates (cetuximab) ----------
m19 = smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e + ecog_ps + stage_iv + albumin_g_dl + weight_loss_pct_6mo', df).fit()
iter19 = {
    'enco_braf_int': float(m19.params['treatment_encorafenib:braf_v600e']),
    'enco_braf_int_p': float(m19.pvalues['treatment_encorafenib:braf_v600e']),
}
# encorafenib + cetuximab combo subgroup
for sub_def, mask in [
    ('braf_v600e', df['braf_v600e']==1),
    ('braf_v600e_with_cet', (df['braf_v600e']==1)&(df['treatment_cetuximab']==1)),
    ('braf_v600e_no_cet', (df['braf_v600e']==1)&(df['treatment_cetuximab']==0)),
    ('braf_wt', df['braf_v600e']==0),
]:
    iter19[sub_def] = subgroup_tx(mask, 'treatment_encorafenib')
# triple interaction
m19b = smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e * treatment_cetuximab', df).fit()
iter19['triple_int_params'] = m19b.params.to_dict()
iter19['triple_int_pvals'] = m19b.pvalues.to_dict()
results['iter19_enco_revisit'] = iter19
print('iter19 done')

# ---------- Iteration 20: trastuzumab/tucatinib × HER2 × RAS WT ----------
iter20 = {}
for sub_def, mask in [
    ('her2_amp', df['her2_amplified']==1),
    ('her2_amp_ras_wt', (df['her2_amplified']==1)&(df['ras_wt']==1)),
    ('her2_amp_ras_mut', (df['her2_amplified']==1)&(df['ras_wt']==0)),
    ('her2_neg', df['her2_amplified']==0),
    ('her2_amp_braf_wt', (df['her2_amplified']==1)&(df['braf_v600e']==0)),
    ('her2_amp_left', (df['her2_amplified']==1)&(df['right_sided_primary']==0)),
]:
    iter20[sub_def] = subgroup_tx(mask, 'treatment_trastuzumab_tucatinib')
m20 = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified * ras_wt', df).fit()
iter20['triple_params'] = m20.params.to_dict()
iter20['triple_pvals'] = m20.pvalues.to_dict()
results['iter20_tt_her2_raswt'] = iter20
print('iter20 done')

# ---------- Iteration 21: bevacizumab subgroup screen ----------
iter21 = {}
for sub_name, mask in [
    ('all', np.ones(len(df), dtype=bool)),
    ('ecog_0', df['ecog_ps']==0),
    ('ecog_1', df['ecog_ps']==1),
    ('ecog_2', df['ecog_ps']==2),
    ('right_sided_only', df['right_sided_primary']==1),
    ('high_crp', df['crp_mg_l']>df['crp_mg_l'].median()),
    ('low_crp', df['crp_mg_l']<=df['crp_mg_l'].median()),
    ('high_ldh', df['ldh_u_l']>df['ldh_u_l'].median()),
]:
    iter21[sub_name] = subgroup_tx(mask, 'treatment_bevacizumab')
results['iter21_bev_screen'] = iter21
print('iter21 done')

# ---------- Iteration 22: regorafenib heterogeneity by ALL clinical features (interaction screen, multivariable) ----------
# Use stratified low-CEA × ECOG × albumin combinations to find best-supported subgroup
iter22 = {}
df['ecog_low'] = (df['ecog_ps'] <= 1).astype(int)
df['alb_high_'] = (df['albumin_g_dl'] >= 3.5).astype(int)
df['wl_low'] = (df['weight_loss_pct_6mo'] < df['weight_loss_pct_6mo'].median()).astype(int)
df['hb_high'] = (df['hemoglobin_g_dl'] >= 12).astype(int)
df['nlr_low'] = (df['nlr'] <= df['nlr'].median()).astype(int)

for sub_def, mask in [
    ('cea_low_ecog_low', (df['cea_low']==1)&(df['ecog_low']==1)),
    ('cea_low_alb_high', (df['cea_low']==1)&(df['alb_high_']==1)),
    ('cea_low_ecog_low_alb_high', (df['cea_low']==1)&(df['ecog_low']==1)&(df['alb_high_']==1)),
    ('cea_low_alb_high_wl_low', (df['cea_low']==1)&(df['alb_high_']==1)&(df['wl_low']==1)),
    ('cea_low_alb_high_hb_high', (df['cea_low']==1)&(df['alb_high_']==1)&(df['hb_high']==1)),
    ('cea_low_alb_high_nlr_low', (df['cea_low']==1)&(df['alb_high_']==1)&(df['nlr_low']==1)),
]:
    iter22[sub_def] = subgroup_tx(mask, 'treatment_regorafenib')
results['iter22_rego_multimod'] = iter22
print('iter22 done')

# ---------- Iteration 23: regorafenib in unfavorable subgroups (whose unfavorable status suppresses tx effect) ----------
# Test if regorafenib effect is suppressed in high-CEA + other unfavorable
iter23 = {}
for sub_def, mask in [
    ('cea_high', df['cea_low']==0),
    ('cea_high_alb_low', (df['cea_low']==0)&(df['albumin_g_dl']<3.5)),
    ('cea_high_ecog_2', (df['cea_low']==0)&(df['ecog_ps']==2)),
    ('alb_low_only', df['albumin_g_dl']<3.5),
    ('ecog_2_only', df['ecog_ps']==2),
]:
    iter23[sub_def] = subgroup_tx(mask, 'treatment_regorafenib')
# also, fit interactions of regorafenib with composite "unfavorable" score
df['unfavorable_score'] = (
    (df['cea_low']==0).astype(int)
    + (df['albumin_g_dl']<3.5).astype(int)
    + (df['ecog_ps']==2).astype(int)
    + (df['weight_loss_pct_6mo']>=df['weight_loss_pct_6mo'].median()).astype(int)
    + (df['ldh_high']==1).astype(int)
)
m23 = smf.ols('pfs_months ~ treatment_regorafenib * unfavorable_score', df).fit()
iter23['unfavorable_int'] = {
    'rego_main_at_score0': float(m23.params['treatment_regorafenib']),
    'rego_int_per_unit': float(m23.params['treatment_regorafenib:unfavorable_score']),
    'p_int': float(m23.pvalues['treatment_regorafenib:unfavorable_score']),
}
for s in [0,1,2,3,4,5]:
    iter23[f'unfavorable_{s}'] = subgroup_tx(df['unfavorable_score']==s, 'treatment_regorafenib')
results['iter23_rego_unfavorable'] = iter23
print('iter23 done')

# ---------- Iteration 24: cetuximab × NRAS, BRAF, side adjusted; final cetuximab subgroup hypothesis ----------
iter24 = {}
m24 = smf.ols('pfs_months ~ treatment_cetuximab * triple_wt + age_years + ecog_ps + stage_iv + right_sided_primary + albumin_g_dl + weight_loss_pct_6mo + cea_ng_ml', df).fit()
iter24['cet_x_triplewt_adj'] = {
    'coef': float(m24.params['treatment_cetuximab:triple_wt']),
    'p': float(m24.pvalues['treatment_cetuximab:triple_wt']),
    'cet_main_in_triplewt': float(m24.params['treatment_cetuximab'] + m24.params['treatment_cetuximab:triple_wt']),
}
# Test side-by-side cetuximab × side
for sub_def, mask in [
    ('triple_wt_left_sided_msi_low_her2_neg', (df['triple_wt']==1)&(df['right_sided_primary']==0)&(df['msi_high']==0)&(df['her2_amplified']==0)),
    ('triple_wt_left_sided_msi_low_her2_neg_ecog_low',
     (df['triple_wt']==1)&(df['right_sided_primary']==0)&(df['msi_high']==0)&(df['her2_amplified']==0)&(df['ecog_ps']<2)),
]:
    iter24[sub_def] = subgroup_tx(mask, 'treatment_cetuximab')
results['iter24_cet_final'] = iter24
print('iter24 done')

# ---------- Iteration 25: final pembrolizumab subgroups with continuous adjustment, regorafenib final hypothesis ----------
iter25 = {}
# pembro × MSI in young, low ECOG patients
for sub_def, mask in [
    ('msi_high_ecog_low', (df['msi_high']==1)&(df['ecog_ps']<2)),
    ('msi_high_left_ecog_low', (df['msi_high']==1)&(df['right_sided_primary']==0)&(df['ecog_ps']<2)),
    ('msi_high_right', (df['msi_high']==1)&(df['right_sided_primary']==1)),
]:
    iter25[sub_def] = subgroup_tx(mask, 'treatment_pembrolizumab')

# Final regorafenib adjusted model
m25 = smf.ols('pfs_months ~ treatment_regorafenib * cea_ng_ml + age_years + ecog_ps + stage_iv + right_sided_primary + albumin_g_dl + weight_loss_pct_6mo + kras_mutation + braf_v600e', df).fit()
iter25['rego_x_cea_adjusted'] = {
    'coef': float(m25.params['treatment_regorafenib:cea_ng_ml']),
    'p': float(m25.pvalues['treatment_regorafenib:cea_ng_ml']),
    'rego_main': float(m25.params['treatment_regorafenib']),
    'rego_main_p': float(m25.pvalues['treatment_regorafenib']),
}

# best-supported regorafenib subgroup hypothesis: CEA in lowest tertile
df['cea_tertile_v'] = pd.qcut(df['cea_ng_ml'], 3, labels=['low','mid','high'])
iter25['rego_in_cea_lowest'] = subgroup_tx(df['cea_tertile_v']=='low', 'treatment_regorafenib')
iter25['rego_in_cea_mid'] = subgroup_tx(df['cea_tertile_v']=='mid', 'treatment_regorafenib')
iter25['rego_in_cea_highest'] = subgroup_tx(df['cea_tertile_v']=='high', 'treatment_regorafenib')

# treatment-by-feature exhaustive screen for each treatment and continuous feature
final_screen = {}
for tx in txs:
    for ft in ['cea_ng_ml','ldh_u_l','crp_mg_l','albumin_g_dl','weight_loss_pct_6mo','nlr','age_years','ecog_ps']:
        f = f'pfs_months ~ {tx} * {ft}'
        m = smf.ols(f, df).fit()
        final_screen[f'{tx}__x__{ft}'] = {
            'coef': float(m.params[f'{tx}:{ft}']),
            'p': float(m.pvalues[f'{tx}:{ft}']),
        }
iter25['exhaustive_screen'] = final_screen
results['iter25_final'] = iter25
print('iter25 done')

with open('iter2_results.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('saved')
