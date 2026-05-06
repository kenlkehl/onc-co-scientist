"""Iter 11-20: investigate joint subgroups and clinically-expected subgroup effects."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

with open('results_v2.json') as f:
    results = json.load(f)

binary_features = [
    'sex_female','stage_iv','has_brain_mets','node_positive','postmenopausal',
    'er_positive','pr_positive','her2_positive','her2_low',
    'brca1_mutation','brca2_mutation','pik3ca_mutation',
]
treatments = [
    'treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
    'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab',
]

def subgroup_effect(treat, mask, label):
    """Compute treatment effect within mask subset."""
    sub = df.loc[mask]
    n_treated = int((sub[treat]==1).sum())
    n_untreated = int((sub[treat]==0).sum())
    if n_treated < 50 or n_untreated < 50:
        return {'label': label, 'n_treated': n_treated, 'n_untreated': n_untreated, 'note': 'too small'}
    yt = y[mask & (df[treat]==1).values]
    yu = y[mask & (df[treat]==0).values]
    diff = float(np.mean(yt) - np.mean(yu))
    t, p = stats.ttest_ind(yt, yu, equal_var=False)
    return {
        'label': label,
        'n_treated': n_treated,
        'n_untreated': n_untreated,
        'mean_treated': float(np.mean(yt)),
        'mean_untreated': float(np.mean(yu)),
        'diff': diff,
        'p': float(p),
    }

results['subgroup_effects'] = {}

# --- Palbociclib joint subgroup (ER+ AND HER2- AND PIK3CA-wt) ---
results['subgroup_effects']['palbo_joint'] = {}
mask_all = np.ones(len(df), dtype=bool)
results['subgroup_effects']['palbo_joint']['overall'] = subgroup_effect('treatment_palbociclib', mask_all, 'overall')
m = (df['er_positive']==1).values
results['subgroup_effects']['palbo_joint']['ER+'] = subgroup_effect('treatment_palbociclib', m, 'ER+')
m = (df['er_positive']==1).values & (df['her2_positive']==0).values
results['subgroup_effects']['palbo_joint']['ER+_HER2-'] = subgroup_effect('treatment_palbociclib', m, 'ER+ HER2-')
m = (df['er_positive']==1).values & (df['her2_positive']==0).values & (df['pik3ca_mutation']==0).values
results['subgroup_effects']['palbo_joint']['ER+_HER2-_PIK3CAwt'] = subgroup_effect('treatment_palbociclib', m, 'ER+ HER2- PIK3CAwt')
m = (df['er_positive']==1).values & (df['her2_positive']==0).values & (df['pik3ca_mutation']==0).values & (df['pr_positive']==1).values
results['subgroup_effects']['palbo_joint']['ER+_HER2-_PIK3CAwt_PR+'] = subgroup_effect('treatment_palbociclib', m, 'ER+ HER2- PIK3CAwt PR+')
# Restrict ki67
m_full = m & (df['ki67_pct'] < df['ki67_pct'].median()).values
results['subgroup_effects']['palbo_joint']['ER+_HER2-_PIK3CAwt_PR+_lowKi67'] = subgroup_effect('treatment_palbociclib', m_full, 'ER+ HER2- PIK3CAwt PR+ lowKi67')

# --- Sacituzumab x has_brain_mets ---
results['subgroup_effects']['sacit_brain'] = {}
results['subgroup_effects']['sacit_brain']['overall'] = subgroup_effect('treatment_sacituzumab_govitecan', mask_all, 'overall')
m = (df['has_brain_mets']==1).values
results['subgroup_effects']['sacit_brain']['brain_mets'] = subgroup_effect('treatment_sacituzumab_govitecan', m, 'has brain mets')
m = (df['has_brain_mets']==0).values
results['subgroup_effects']['sacit_brain']['no_brain_mets'] = subgroup_effect('treatment_sacituzumab_govitecan', m, 'no brain mets')

# --- Trastuzumab x HER2+ (clinically expected) ---
results['subgroup_effects']['tras_her2'] = {}
results['subgroup_effects']['tras_her2']['overall'] = subgroup_effect('treatment_trastuzumab', mask_all, 'overall')
results['subgroup_effects']['tras_her2']['HER2+'] = subgroup_effect('treatment_trastuzumab', (df['her2_positive']==1).values, 'HER2+')
results['subgroup_effects']['tras_her2']['HER2-'] = subgroup_effect('treatment_trastuzumab', (df['her2_positive']==0).values, 'HER2-')
results['subgroup_effects']['tras_her2']['HER2+_no_brain'] = subgroup_effect(
    'treatment_trastuzumab', ((df['her2_positive']==1)&(df['has_brain_mets']==0)).values, 'HER2+ no brain')
results['subgroup_effects']['tras_her2']['HER2_low'] = subgroup_effect(
    'treatment_trastuzumab', (df['her2_low']==1).values, 'HER2-low')

# --- Olaparib x BRCA1/BRCA2 ---
results['subgroup_effects']['ola_brca'] = {}
results['subgroup_effects']['ola_brca']['overall'] = subgroup_effect('treatment_olaparib', mask_all, 'overall')
results['subgroup_effects']['ola_brca']['BRCA1'] = subgroup_effect('treatment_olaparib', (df['brca1_mutation']==1).values, 'BRCA1+')
results['subgroup_effects']['ola_brca']['BRCA2'] = subgroup_effect('treatment_olaparib', (df['brca2_mutation']==1).values, 'BRCA2+')
results['subgroup_effects']['ola_brca']['BRCA_any'] = subgroup_effect(
    'treatment_olaparib', ((df['brca1_mutation']==1)|(df['brca2_mutation']==1)).values, 'BRCA1+ or BRCA2+')
results['subgroup_effects']['ola_brca']['BRCA_none'] = subgroup_effect(
    'treatment_olaparib', ((df['brca1_mutation']==0)&(df['brca2_mutation']==0)).values, 'BRCA wt')

# --- Pembrolizumab x TNBC (ER- PR- HER2-) ---
results['subgroup_effects']['pembro_tnbc'] = {}
results['subgroup_effects']['pembro_tnbc']['overall'] = subgroup_effect('treatment_pembrolizumab', mask_all, 'overall')
m_tnbc = ((df['er_positive']==0)&(df['pr_positive']==0)&(df['her2_positive']==0)).values
results['subgroup_effects']['pembro_tnbc']['TNBC'] = subgroup_effect('treatment_pembrolizumab', m_tnbc, 'TNBC')
results['subgroup_effects']['pembro_tnbc']['ER+'] = subgroup_effect('treatment_pembrolizumab', (df['er_positive']==1).values, 'ER+')
results['subgroup_effects']['pembro_tnbc']['HER2+'] = subgroup_effect('treatment_pembrolizumab', (df['her2_positive']==1).values, 'HER2+')
results['subgroup_effects']['pembro_tnbc']['high_ki67'] = subgroup_effect(
    'treatment_pembrolizumab', (df['ki67_pct']>=df['ki67_pct'].median()).values, 'high ki67')
results['subgroup_effects']['pembro_tnbc']['high_pdl1_proxy_TNBC_highKi67'] = subgroup_effect(
    'treatment_pembrolizumab', m_tnbc & (df['ki67_pct']>=df['ki67_pct'].median()).values, 'TNBC + high ki67')

# --- Tamoxifen x HR+/postmenopausal ---
results['subgroup_effects']['tam_hr'] = {}
results['subgroup_effects']['tam_hr']['overall'] = subgroup_effect('treatment_tamoxifen', mask_all, 'overall')
results['subgroup_effects']['tam_hr']['ER+'] = subgroup_effect('treatment_tamoxifen', (df['er_positive']==1).values, 'ER+')
results['subgroup_effects']['tam_hr']['ER+_premeno'] = subgroup_effect(
    'treatment_tamoxifen', ((df['er_positive']==1)&(df['postmenopausal']==0)).values, 'ER+ premenopausal')
results['subgroup_effects']['tam_hr']['ER+_postmeno'] = subgroup_effect(
    'treatment_tamoxifen', ((df['er_positive']==1)&(df['postmenopausal']==1)).values, 'ER+ postmenopausal')
results['subgroup_effects']['tam_hr']['HR-'] = subgroup_effect(
    'treatment_tamoxifen', ((df['er_positive']==0)&(df['pr_positive']==0)).values, 'HR-')

with open('results_v2.json', 'w') as fp:
    json.dump(results, fp, indent=2)

# Print
for k, sub in results['subgroup_effects'].items():
    print(f'\n=== {k} ===')
    for label, v in sub.items():
        if 'note' in v:
            print(f'  {label}: too small (n_t={v["n_treated"]}, n_u={v["n_untreated"]})')
        else:
            print(f'  {label}: diff={v["diff"]:+.3f} (p={v["p"]:.2e}), n_t={v["n_treated"]}, n_u={v["n_untreated"]}')
