"""Iter 23-25: confirmation of best subgroup definitions."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

df = pd.read_parquet('dataset.parquet')
y = df['pfs_months'].values

with open('results_v2.json') as f:
    results = json.load(f)

results['final_confirmations'] = {}

def confirm(treat, mask, label):
    yt = y[mask & (df[treat]==1).values]
    yu = y[mask & (df[treat]==0).values]
    diff = float(np.mean(yt) - np.mean(yu))
    t, p = stats.ttest_ind(yt, yu, equal_var=False)
    return {'label': label, 'diff': diff, 'p': float(p),
            'n_t': int(len(yt)), 'n_u': int(len(yu))}

# Palbociclib — most-refined subgroup
m = ((df['er_positive']==1)&(df['her2_positive']==0)&(df['pik3ca_mutation']==0)&(df['pr_positive']==1)&(df['ki67_pct']<df['ki67_pct'].median())).values
results['final_confirmations']['palbo_best'] = confirm(
    'treatment_palbociclib', m, 'ER+/HER2-/PIK3CAwt/PR+/lowKi67')

# Compare to complement
results['final_confirmations']['palbo_complement'] = confirm(
    'treatment_palbociclib', ~m, 'NOT(ER+/HER2-/PIK3CAwt/PR+/lowKi67)')

# Olaparib — BRCA1 OR BRCA2
m = ((df['brca1_mutation']==1)|(df['brca2_mutation']==1)).values
results['final_confirmations']['ola_brca_any'] = confirm(
    'treatment_olaparib', m, 'BRCA1+ or BRCA2+')
results['final_confirmations']['ola_brca_none'] = confirm(
    'treatment_olaparib', ~m, 'BRCA wt')

# Sacituzumab — brain mets + low ki67
m = ((df['has_brain_mets']==1)&(df['ki67_pct']<df['ki67_pct'].median())).values
results['final_confirmations']['sacit_brain_lowki67'] = confirm(
    'treatment_sacituzumab_govitecan', m, 'brain_mets + low_ki67')

# Pembrolizumab — explicitly check for any positive subgroup
# Try TNBC + brain mets
m = ((df['er_positive']==0)&(df['pr_positive']==0)&(df['her2_positive']==0)&(df['has_brain_mets']==1)).values
results['final_confirmations']['pembro_tnbc_brain'] = confirm(
    'treatment_pembrolizumab', m, 'TNBC + brain_mets')

# Pembrolizumab — TNBC with high tumor burden / high ki67 / high LDH
m = ((df['er_positive']==0)&(df['pr_positive']==0)&(df['her2_positive']==0)&
     (df['ki67_pct']>=df['ki67_pct'].median())&(df['ldh_u_l']>=df['ldh_u_l'].median())).values
results['final_confirmations']['pembro_tnbc_highki67_highldh'] = confirm(
    'treatment_pembrolizumab', m, 'TNBC + high ki67 + high LDH')

# Trastuzumab — HER2+ + ER- (HER2-driven subtype)
m = ((df['her2_positive']==1)&(df['er_positive']==0)).values
results['final_confirmations']['tras_her2_er_neg'] = confirm(
    'treatment_trastuzumab', m, 'HER2+ ER-')
m = ((df['her2_positive']==1)&(df['er_positive']==1)).values
results['final_confirmations']['tras_her2_er_pos'] = confirm(
    'treatment_trastuzumab', m, 'HER2+ ER+')

# Tamoxifen — BRCA1 subgroup (top from search)
m = ((df['brca1_mutation']==1)&(df['er_positive']==1)).values
results['final_confirmations']['tam_brca1_er_pos'] = confirm(
    'treatment_tamoxifen', m, 'BRCA1+ ER+')

# Sacituzumab adjusted regression interaction (control for other treatments and covariates)
def adjusted_interaction(treat, mod_mask_col_expr, label):
    """Run y ~ const + treat + mod + treat*mod + other covars"""
    mod = mod_mask_col_expr.astype(float)
    treat_v = df[treat].astype(float)
    inter = treat_v * mod
    other_treats = [t for t in [
        'treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
        'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab',
    ] if t != treat]
    covars = [
        'age_years','ecog_ps','stage_iv','er_positive','her2_positive','pik3ca_mutation',
        'ki67_pct','albumin_g_dl','weight_loss_pct_6mo','ldh_u_l',
    ]
    X = pd.DataFrame({'treat': treat_v, 'mod': mod, 'inter': inter})
    for c in other_treats + covars:
        X[c] = df[c].astype(float)
    Xv = sm.add_constant(X.values)
    m = sm.OLS(y, Xv).fit()
    return {'label': label,
            'inter_coef': float(m.params[3]),
            'inter_p': float(m.pvalues[3]),
            'treat_coef_mod0': float(m.params[1])}

results['final_confirmations']['sacit_brain_adj_inter'] = adjusted_interaction(
    'treatment_sacituzumab_govitecan',
    (df['has_brain_mets']==1).astype(float),
    'sacit x has_brain_mets, adjusted')
results['final_confirmations']['sacit_brain_lowki67_adj_inter'] = adjusted_interaction(
    'treatment_sacituzumab_govitecan',
    ((df['has_brain_mets']==1)&(df['ki67_pct']<df['ki67_pct'].median())).astype(float),
    'sacit x (brain_mets & low_ki67), adjusted')
results['final_confirmations']['ola_brca_adj_inter'] = adjusted_interaction(
    'treatment_olaparib',
    ((df['brca1_mutation']==1)|(df['brca2_mutation']==1)).astype(float),
    'ola x (BRCA1 or BRCA2), adjusted')
results['final_confirmations']['palbo_subgroup_adj_inter'] = adjusted_interaction(
    'treatment_palbociclib',
    ((df['er_positive']==1)&(df['her2_positive']==0)&(df['pik3ca_mutation']==0)).astype(float),
    'palbo x (ER+/HER2-/PIK3CAwt), adjusted')

with open('results_v2.json', 'w') as fp:
    json.dump(results, fp, indent=2)

for k, v in results['final_confirmations'].items():
    if 'inter_coef' in v:
        print(f'{k}: inter_coef={v["inter_coef"]:+.3f}, p={v["inter_p"]:.2e}, treat_coef_mod0={v["treat_coef_mod0"]:+.3f}')
    else:
        print(f'{k} ({v["label"]}): diff={v["diff"]:+.3f}, p={v["p"]:.2e}, n_t={v["n_t"]}, n_u={v["n_u"]}')
