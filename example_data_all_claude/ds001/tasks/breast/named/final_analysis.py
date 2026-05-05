"""Comprehensive analysis of ds001_breast for transcript.json.
Outputs final_results.json with structured numeric results per iteration."""

import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')

OUT = {}

def reg_ols(formula, label):
    m = smf.ols(formula, data=df).fit()
    return m

def linreg_one(y, x_name):
    x = df[x_name].astype(float)
    yv = df[y].astype(float)
    slope, intercept, r, p, se = stats.linregress(x, yv)
    return {'feature': x_name, 'slope': float(slope), 'intercept': float(intercept),
            'r': float(r), 'p_value': float(p), 'n': int(len(x))}

def ttest_binary(y, group_col):
    g1 = df.loc[df[group_col]==1, y].dropna()
    g0 = df.loc[df[group_col]==0, y].dropna()
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    return {'feature': group_col, 'mean_1': float(g1.mean()), 'mean_0': float(g0.mean()),
            'diff': float(g1.mean() - g0.mean()), 't': float(t), 'p_value': float(p),
            'n_1': int(len(g1)), 'n_0': int(len(g0))}

# ===== ITERATION 1: descriptive checks =====
OUT['iter1_descriptive'] = {
    'pfs_overall_mean': float(df['pfs_months'].mean()),
    'pfs_overall_median': float(df['pfs_months'].median()),
    'pfs_overall_sd': float(df['pfs_months'].std()),
    'sex_female_pct': float(df['sex_female'].mean()),
    'er_positive_pct': float(df['er_positive'].mean()),
    'her2_positive_pct': float(df['her2_positive'].mean()),
    'brca1_pct': float(df['brca1_mutation'].mean()),
    'brca2_pct': float(df['brca2_mutation'].mean()),
    'stage_iv_pct': float(df['stage_iv'].mean()),
    'brain_mets_pct': float(df['has_brain_mets'].mean()),
    'tnbc_pct': float(((df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)).mean()),
    'hr_pos_her2_neg_pct': float((((df['er_positive']==1) | (df['pr_positive']==1)) & (df['her2_positive']==0)).mean()),
    'treatment_use': {c: float(df[c].mean()) for c in df.columns if c.startswith('treatment_')},
}
print('Iter1 done')

# ===== ITERATION 2: clinical/staging features (binary) vs PFS =====
binary_features_clinical = ['sex_female', 'stage_iv', 'has_brain_mets', 'node_positive', 'postmenopausal']
OUT['iter2_clinical_binary_vs_pfs'] = [ttest_binary('pfs_months', c) for c in binary_features_clinical]
print('Iter2 done')

# ===== ITERATION 3: continuous demographic/clinical vs PFS =====
cont_clinical = ['age_years', 'ecog_ps', 'tumor_size_cm', 'ki67_pct', 'weight_loss_pct_6mo']
OUT['iter3_clinical_cont_vs_pfs'] = [linreg_one('pfs_months', c) for c in cont_clinical]
print('Iter3 done')

# ===== ITERATION 4: biomarkers (binary) vs PFS =====
biomarker_binary = ['er_positive', 'pr_positive', 'her2_positive', 'her2_low',
                    'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation']
OUT['iter4_biomarker_vs_pfs'] = [ttest_binary('pfs_months', c) for c in biomarker_binary]
print('Iter4 done')

# ===== ITERATION 5: lab tests vs PFS =====
labs = ['albumin_g_dl', 'ldh_u_l', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
        'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l', 'total_bilirubin_mg_dl',
        'creatinine_mg_dl', 'bun_mg_dl', 'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
OUT['iter5_labs_vs_pfs'] = [linreg_one('pfs_months', c) for c in labs]
print('Iter5 done')

# ===== ITERATION 6: treatment main effects (unadjusted) =====
treatments = ['treatment_tamoxifen','treatment_palbociclib','treatment_trastuzumab',
              'treatment_olaparib','treatment_sacituzumab_govitecan','treatment_pembrolizumab']
OUT['iter6_tx_unadj'] = [ttest_binary('pfs_months', t) for t in treatments]
print('Iter6 done')

# ===== ITERATION 7: treatment effects adjusted for prognostics =====
adj_covs = ['age_years','ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','ldh_u_l',
            'crp_mg_l','nlr','weight_loss_pct_6mo','tumor_size_cm']
adj_results = {}
for t in treatments:
    formula = 'pfs_months ~ ' + t + ' + ' + ' + '.join(adj_covs)
    m = smf.ols(formula, data=df).fit()
    adj_results[t] = {
        'coef': float(m.params[t]), 'se': float(m.bse[t]),
        'p_value': float(m.pvalues[t]), 'ci_low': float(m.conf_int().loc[t,0]),
        'ci_high': float(m.conf_int().loc[t,1])
    }
OUT['iter7_tx_adjusted'] = adj_results
print('Iter7 done')

# ===== ITERATION 8: tamoxifen x ER+ interaction =====
def interaction_test(tx, modifier, base_covs=adj_covs):
    formula = f'pfs_months ~ {tx} * {modifier} + ' + ' + '.join(base_covs)
    m = smf.ols(formula, data=df).fit()
    inter_term = f'{tx}:{modifier}'
    if inter_term not in m.params:
        # statsmodels may name differently; check
        for k in m.params.index:
            if tx in k and modifier in k and ':' in k:
                inter_term = k; break
    return {
        'tx': tx, 'modifier': modifier,
        'main_tx': float(m.params[tx]),
        'main_mod': float(m.params[modifier]),
        'interaction': float(m.params[inter_term]),
        'p_interaction': float(m.pvalues[inter_term]),
        'p_main_tx': float(m.pvalues[tx]),
    }

def stratified_tx_effect(tx, modifier, base_covs=adj_covs):
    res = {}
    for val in [0,1]:
        sub = df[df[modifier]==val]
        if len(sub) < 50:
            continue
        formula = f'pfs_months ~ {tx} + ' + ' + '.join(base_covs)
        m = smf.ols(formula, data=sub).fit()
        res[f'modifier={val}'] = {
            'n': int(len(sub)),
            'coef': float(m.params[tx]),
            'se': float(m.bse[tx]),
            'p_value': float(m.pvalues[tx])
        }
    return res

OUT['iter8_tamoxifen_x_er'] = {
    'interaction_test': interaction_test('treatment_tamoxifen', 'er_positive'),
    'stratified': stratified_tx_effect('treatment_tamoxifen', 'er_positive'),
}
OUT['iter8_tamoxifen_x_pr'] = {
    'interaction_test': interaction_test('treatment_tamoxifen', 'pr_positive'),
    'stratified': stratified_tx_effect('treatment_tamoxifen', 'pr_positive'),
}
print('Iter8 done')

# ===== ITERATION 9: trastuzumab x HER2 =====
OUT['iter9_trastuzumab_x_her2pos'] = {
    'interaction_test': interaction_test('treatment_trastuzumab', 'her2_positive'),
    'stratified': stratified_tx_effect('treatment_trastuzumab', 'her2_positive'),
}
OUT['iter9_trastuzumab_x_her2low'] = {
    'interaction_test': interaction_test('treatment_trastuzumab', 'her2_low'),
    'stratified': stratified_tx_effect('treatment_trastuzumab', 'her2_low'),
}
print('Iter9 done')

# ===== ITERATION 10: olaparib x BRCA =====
df['brca_any'] = ((df['brca1_mutation']==1) | (df['brca2_mutation']==1)).astype(int)
OUT['iter10_olaparib_x_brca1'] = {
    'interaction_test': interaction_test('treatment_olaparib', 'brca1_mutation'),
    'stratified': stratified_tx_effect('treatment_olaparib', 'brca1_mutation'),
}
OUT['iter10_olaparib_x_brca2'] = {
    'interaction_test': interaction_test('treatment_olaparib', 'brca2_mutation'),
    'stratified': stratified_tx_effect('treatment_olaparib', 'brca2_mutation'),
}
OUT['iter10_olaparib_x_brca_any'] = {
    'interaction_test': interaction_test('treatment_olaparib', 'brca_any'),
    'stratified': stratified_tx_effect('treatment_olaparib', 'brca_any'),
}
print('Iter10 done')

# ===== ITERATION 11: palbociclib x ER+, x HER2-, x postmenopausal =====
df['er_pos_her2_neg'] = ((df['er_positive']==1) & (df['her2_positive']==0)).astype(int)
df['hr_pos'] = ((df['er_positive']==1) | (df['pr_positive']==1)).astype(int)
OUT['iter11_palbo_x_er'] = {
    'interaction_test': interaction_test('treatment_palbociclib', 'er_positive'),
    'stratified': stratified_tx_effect('treatment_palbociclib', 'er_positive'),
}
OUT['iter11_palbo_x_her2neg'] = {
    'interaction_test': interaction_test('treatment_palbociclib', 'her2_positive'),
    'stratified': stratified_tx_effect('treatment_palbociclib', 'her2_positive'),
}
OUT['iter11_palbo_x_postmeno'] = {
    'interaction_test': interaction_test('treatment_palbociclib', 'postmenopausal'),
    'stratified': stratified_tx_effect('treatment_palbociclib', 'postmenopausal'),
}
OUT['iter11_palbo_x_er_pos_her2_neg'] = {
    'interaction_test': interaction_test('treatment_palbociclib', 'er_pos_her2_neg'),
    'stratified': stratified_tx_effect('treatment_palbociclib', 'er_pos_her2_neg'),
}
print('Iter11 done')

# ===== ITERATION 12: sacituzumab govitecan x TNBC =====
df['tnbc'] = ((df['er_positive']==0) & (df['pr_positive']==0) & (df['her2_positive']==0)).astype(int)
OUT['iter12_saci_x_tnbc'] = {
    'interaction_test': interaction_test('treatment_sacituzumab_govitecan', 'tnbc'),
    'stratified': stratified_tx_effect('treatment_sacituzumab_govitecan', 'tnbc'),
}
OUT['iter12_saci_x_er_neg'] = {
    'interaction_test': interaction_test('treatment_sacituzumab_govitecan', 'er_positive'),
    'stratified': stratified_tx_effect('treatment_sacituzumab_govitecan', 'er_positive'),
}
OUT['iter12_saci_x_her2_low'] = {
    'interaction_test': interaction_test('treatment_sacituzumab_govitecan', 'her2_low'),
    'stratified': stratified_tx_effect('treatment_sacituzumab_govitecan', 'her2_low'),
}
print('Iter12 done')

# ===== ITERATION 13: pembrolizumab x TNBC, stage_iv, etc =====
OUT['iter13_pembro_x_tnbc'] = {
    'interaction_test': interaction_test('treatment_pembrolizumab', 'tnbc'),
    'stratified': stratified_tx_effect('treatment_pembrolizumab', 'tnbc'),
}
OUT['iter13_pembro_x_stage_iv'] = {
    'interaction_test': interaction_test('treatment_pembrolizumab', 'stage_iv'),
    'stratified': stratified_tx_effect('treatment_pembrolizumab', 'stage_iv'),
}
OUT['iter13_pembro_x_pdl1_unavail_proxy_tnbc_brain'] = {
    'interaction_test': interaction_test('treatment_pembrolizumab', 'has_brain_mets'),
    'stratified': stratified_tx_effect('treatment_pembrolizumab', 'has_brain_mets'),
}
print('Iter13 done')

# ===== ITERATION 14: full treatment x ALL biomarker interaction screen =====
modifiers = ['er_positive','pr_positive','her2_positive','her2_low','brca1_mutation',
             'brca2_mutation','brca_any','pik3ca_mutation','postmenopausal','stage_iv',
             'has_brain_mets','node_positive','tnbc','hr_pos','er_pos_her2_neg']

screen = []
for t in treatments:
    for m_ in modifiers:
        try:
            r = interaction_test(t, m_)
            screen.append(r)
        except Exception as e:
            screen.append({'tx': t, 'modifier': m_, 'error': str(e)})
OUT['iter14_screen'] = sorted(screen, key=lambda x: x.get('p_interaction', 1.0))
print('Iter14 done')

# ===== ITERATION 15: refined subgroup definitions for each treatment =====
# For each treatment, fit Tx coefficient inside the most plausible subgroup (combo)
def stratified_tx_in_subgroup(tx, predicate_str, label, base_covs=adj_covs):
    sub = df.query(predicate_str)
    if len(sub) < 50:
        return {'label': label, 'predicate': predicate_str, 'n': int(len(sub)),
                'note': 'too small'}
    formula = f'pfs_months ~ {tx} + ' + ' + '.join(base_covs)
    m = smf.ols(formula, data=sub).fit()
    return {'label': label, 'predicate': predicate_str, 'n': int(len(sub)),
            'coef': float(m.params[tx]), 'se': float(m.bse[tx]),
            'p_value': float(m.pvalues[tx])}

OUT['iter15_subgroup_definitions'] = {
    'tamoxifen_in_ER+': stratified_tx_in_subgroup('treatment_tamoxifen','er_positive==1','ER+ only'),
    'tamoxifen_in_ER-': stratified_tx_in_subgroup('treatment_tamoxifen','er_positive==0','ER- only'),
    'tamoxifen_in_ER+PR+': stratified_tx_in_subgroup('treatment_tamoxifen','er_positive==1 and pr_positive==1','ER+/PR+'),
    'trastuzumab_in_HER2+': stratified_tx_in_subgroup('treatment_trastuzumab','her2_positive==1','HER2+ only'),
    'trastuzumab_in_HER2-': stratified_tx_in_subgroup('treatment_trastuzumab','her2_positive==0','HER2- only'),
    'olaparib_in_BRCAany': stratified_tx_in_subgroup('treatment_olaparib','brca1_mutation==1 or brca2_mutation==1','BRCA1/2 mutation'),
    'olaparib_in_BRCA-wt': stratified_tx_in_subgroup('treatment_olaparib','brca1_mutation==0 and brca2_mutation==0','BRCA wild-type'),
    'palbo_in_HRposHER2neg': stratified_tx_in_subgroup('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0','HR+/HER2-'),
    'palbo_in_HRposHER2neg_postmeno': stratified_tx_in_subgroup('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0 and postmenopausal==1','HR+/HER2- postmenopausal'),
    'palbo_in_HRneg_or_HER2pos': stratified_tx_in_subgroup('treatment_palbociclib','(er_positive==0 and pr_positive==0) or her2_positive==1','HR- or HER2+'),
    'saci_in_TNBC': stratified_tx_in_subgroup('treatment_sacituzumab_govitecan','er_positive==0 and pr_positive==0 and her2_positive==0','TNBC'),
    'saci_in_nonTNBC': stratified_tx_in_subgroup('treatment_sacituzumab_govitecan','er_positive==1 or pr_positive==1 or her2_positive==1','non-TNBC'),
    'pembro_in_TNBC': stratified_tx_in_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0','TNBC'),
    'pembro_in_nonTNBC': stratified_tx_in_subgroup('treatment_pembrolizumab','er_positive==1 or pr_positive==1 or her2_positive==1','non-TNBC'),
}
print('Iter15 done')

# ===== ITERATION 16: explore additional modifiers within positive subgroups =====
# Check if PFS benefit of tamoxifen in ER+ is further modified by postmenopausal status
def threeway(tx, mod1, mod2):
    formula = f'pfs_months ~ {tx} * {mod1} * {mod2} + ' + ' + '.join(adj_covs)
    m = smf.ols(formula, data=df).fit()
    coefs = {k: float(m.params[k]) for k in m.params.index if k.count(':')>=1 and tx in k}
    pvals = {k: float(m.pvalues[k]) for k in m.params.index if k.count(':')>=1 and tx in k}
    return {'tx': tx, 'mod1': mod1, 'mod2': mod2, 'coefs': coefs, 'pvals': pvals}

OUT['iter16_threeway'] = {
    'tamoxifen_er_postmeno': threeway('treatment_tamoxifen', 'er_positive', 'postmenopausal'),
    'tamoxifen_er_pr': threeway('treatment_tamoxifen', 'er_positive', 'pr_positive'),
    'palbo_hrpos_postmeno': threeway('treatment_palbociclib', 'hr_pos', 'postmenopausal'),
    'palbo_er_her2neg_postmeno': threeway('treatment_palbociclib', 'er_pos_her2_neg', 'postmenopausal'),
    'trastuzumab_her2pos_node': threeway('treatment_trastuzumab', 'her2_positive', 'node_positive'),
}
print('Iter16 done')

# ===== ITERATION 17: ECOG/age/stage modifiers of treatment effect =====
OUT['iter17_clinical_modifiers'] = {}
for tx in treatments:
    OUT['iter17_clinical_modifiers'][tx] = {
        'x_stage_iv': interaction_test(tx, 'stage_iv'),
        'x_brain_mets': interaction_test(tx, 'has_brain_mets'),
        'x_node_positive': interaction_test(tx, 'node_positive'),
    }
print('Iter17 done')

# ===== ITERATION 18: continuous biomarker modifiers (ki67_pct, tumor_size_cm) =====
def cont_modifier(tx, mod):
    formula = f'pfs_months ~ {tx} * {mod} + ' + ' + '.join([c for c in adj_covs if c != mod])
    m = smf.ols(formula, data=df).fit()
    inter = f'{tx}:{mod}'
    return {'tx': tx, 'modifier': mod,
            'tx_main': float(m.params[tx]), 'p_tx_main': float(m.pvalues[tx]),
            'mod_main': float(m.params[mod]), 'p_mod_main': float(m.pvalues[mod]),
            'interaction': float(m.params[inter]), 'p_interaction': float(m.pvalues[inter])}

OUT['iter18_continuous_modifiers'] = []
for tx in treatments:
    for mod in ['ki67_pct','tumor_size_cm','age_years','ecog_ps','albumin_g_dl','ldh_u_l','crp_mg_l','nlr']:
        try:
            OUT['iter18_continuous_modifiers'].append(cont_modifier(tx, mod))
        except Exception as e:
            OUT['iter18_continuous_modifiers'].append({'tx':tx,'mod':mod,'error':str(e)})
OUT['iter18_continuous_modifiers'].sort(key=lambda r: r.get('p_interaction', 1))
print('Iter18 done')

# ===== ITERATION 19: best joint subgroups - try defining each tx subgroup with a 2-3 var combo  =====
def fit_joint_subgroup(tx, predicate, base_covs=adj_covs):
    sub = df.query(predicate)
    comp = df.query(f'not ({predicate})')
    if len(sub) < 50: return None
    formula = f'pfs_months ~ {tx} + ' + ' + '.join(base_covs)
    m_in = smf.ols(formula, data=sub).fit()
    m_out = smf.ols(formula, data=comp).fit()
    return {
        'predicate': predicate,
        'n_in': int(len(sub)), 'n_out': int(len(comp)),
        'tx_effect_in': float(m_in.params[tx]), 'p_in': float(m_in.pvalues[tx]),
        'tx_effect_out': float(m_out.params[tx]), 'p_out': float(m_out.pvalues[tx]),
        'difference': float(m_in.params[tx] - m_out.params[tx])
    }

OUT['iter19_joint_subgroups'] = {
    # Tamoxifen
    'tam_ER+': fit_joint_subgroup('treatment_tamoxifen','er_positive==1'),
    'tam_ER+PR+': fit_joint_subgroup('treatment_tamoxifen','er_positive==1 and pr_positive==1'),
    'tam_ER+PR+postmeno': fit_joint_subgroup('treatment_tamoxifen','er_positive==1 and pr_positive==1 and postmenopausal==1'),
    'tam_ER+postmeno': fit_joint_subgroup('treatment_tamoxifen','er_positive==1 and postmenopausal==1'),
    # Trastuzumab
    'tras_HER2+': fit_joint_subgroup('treatment_trastuzumab','her2_positive==1'),
    'tras_HER2+_nodeP': fit_joint_subgroup('treatment_trastuzumab','her2_positive==1 and node_positive==1'),
    'tras_HER2+_StageIV': fit_joint_subgroup('treatment_trastuzumab','her2_positive==1 and stage_iv==1'),
    # Olaparib
    'ola_BRCA1': fit_joint_subgroup('treatment_olaparib','brca1_mutation==1'),
    'ola_BRCA2': fit_joint_subgroup('treatment_olaparib','brca2_mutation==1'),
    'ola_BRCAany': fit_joint_subgroup('treatment_olaparib','brca1_mutation==1 or brca2_mutation==1'),
    # Palbociclib
    'palbo_HR+HER2-': fit_joint_subgroup('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0'),
    'palbo_HR+HER2-_postmeno': fit_joint_subgroup('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0 and postmenopausal==1'),
    'palbo_ER+HER2-_postmeno': fit_joint_subgroup('treatment_palbociclib','er_positive==1 and her2_positive==0 and postmenopausal==1'),
    # Sacituzumab
    'saci_TNBC': fit_joint_subgroup('treatment_sacituzumab_govitecan','er_positive==0 and pr_positive==0 and her2_positive==0'),
    'saci_TNBC_StageIV': fit_joint_subgroup('treatment_sacituzumab_govitecan','er_positive==0 and pr_positive==0 and her2_positive==0 and stage_iv==1'),
    # Pembrolizumab
    'pembro_TNBC': fit_joint_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0'),
    'pembro_TNBC_StageIV': fit_joint_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0 and stage_iv==1'),
    'pembro_TNBC_PIK3CA-': fit_joint_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0 and pik3ca_mutation==0'),
}
print('Iter19 done')

# ===== ITERATION 20: prognostic-only model (sanity, multivariable) =====
formula = 'pfs_months ~ ' + ' + '.join(adj_covs)
m = smf.ols(formula, data=df).fit()
OUT['iter20_prognostic_model'] = {
    'r_squared': float(m.rsquared),
    'coefs': {k: float(v) for k,v in m.params.items()},
    'pvals': {k: float(v) for k,v in m.pvalues.items()}
}
print('Iter20 done')

# ===== ITERATION 21: full model with all treatments and key biomarker interactions =====
core_inter = ('treatment_tamoxifen*er_positive + treatment_trastuzumab*her2_positive + '
              'treatment_olaparib*brca_any + treatment_palbociclib*er_pos_her2_neg + '
              'treatment_sacituzumab_govitecan*tnbc + treatment_pembrolizumab*tnbc')
formula = 'pfs_months ~ ' + core_inter + ' + ' + ' + '.join(adj_covs)
m = smf.ols(formula, data=df).fit()
OUT['iter21_full_model'] = {
    'r_squared': float(m.rsquared),
    'coefs': {k: float(v) for k,v in m.params.items()},
    'pvals': {k: float(v) for k,v in m.pvalues.items()}
}
print('Iter21 done')

# ===== ITERATION 22: search secondary modifiers within each "indication" subgroup =====
# Within HER2+, does trastuzumab effect vary by ECOG, stage_iv, brain_mets, age, ki67?
def secondary_screen(tx, base_predicate, candidates):
    sub = df.query(base_predicate).copy()
    out = []
    for cand in candidates:
        try:
            formula = f'pfs_months ~ {tx} * {cand} + ' + ' + '.join([c for c in adj_covs if c != cand])
            m = smf.ols(formula, data=sub).fit()
            inter = f'{tx}:{cand}'
            out.append({'cand': cand, 'tx_main': float(m.params[tx]),
                        'p_tx_main': float(m.pvalues[tx]),
                        'interaction': float(m.params[inter]), 'p_interaction': float(m.pvalues[inter])})
        except Exception as e:
            out.append({'cand': cand, 'error': str(e)})
    return sorted(out, key=lambda r: r.get('p_interaction', 1))

candidates_all = ['age_years','ecog_ps','stage_iv','has_brain_mets','node_positive',
                  'postmenopausal','ki67_pct','tumor_size_cm','albumin_g_dl','ldh_u_l',
                  'crp_mg_l','nlr','weight_loss_pct_6mo','pik3ca_mutation','her2_low']
OUT['iter22_secondary_modifiers'] = {
    'trastuzumab_in_HER2pos': secondary_screen('treatment_trastuzumab','her2_positive==1', candidates_all),
    'tamoxifen_in_ERpos': secondary_screen('treatment_tamoxifen','er_positive==1', candidates_all),
    'olaparib_in_BRCAany': secondary_screen('treatment_olaparib','brca1_mutation==1 or brca2_mutation==1', candidates_all),
    'palbo_in_HRposHER2neg': secondary_screen('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0', candidates_all),
    'saci_in_TNBC': secondary_screen('treatment_sacituzumab_govitecan','er_positive==0 and pr_positive==0 and her2_positive==0', candidates_all),
    'pembro_in_TNBC': secondary_screen('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0', candidates_all),
}
print('Iter22 done')

# ===== ITERATION 23: refine subgroups by adding the strongest secondary modifier (cut at median) =====
# Now define complete subgroup hypotheses including the unfavorable secondary feature whose value suppresses the effect
def median_cut(col):
    return float(df[col].median())

def fit_combined_subgroup(tx, predicate):
    sub = df.query(predicate)
    if len(sub) < 50:
        return None
    formula = f'pfs_months ~ {tx} + ' + ' + '.join(adj_covs)
    m = smf.ols(formula, data=sub).fit()
    return {'predicate': predicate, 'n': int(len(sub)),
            'tx_coef': float(m.params[tx]), 'se': float(m.bse[tx]),
            'p_value': float(m.pvalues[tx])}

ki67_med = median_cut('ki67_pct')
tumor_med = median_cut('tumor_size_cm')
ldh_med = median_cut('ldh_u_l')
crp_med = median_cut('crp_mg_l')
albumin_med = median_cut('albumin_g_dl')
OUT['iter23_refined_combined_subgroups'] = {
    'tam_ER+_ECOG=0': fit_combined_subgroup('treatment_tamoxifen','er_positive==1 and ecog_ps==0'),
    'tam_ER+_lowKi67': fit_combined_subgroup('treatment_tamoxifen', f'er_positive==1 and ki67_pct < {ki67_med}'),
    'tam_ER+PR+_postmeno': fit_combined_subgroup('treatment_tamoxifen','er_positive==1 and pr_positive==1 and postmenopausal==1'),
    'tras_HER2+_ECOG<2': fit_combined_subgroup('treatment_trastuzumab','her2_positive==1 and ecog_ps < 2'),
    'tras_HER2+_noBrainMets': fit_combined_subgroup('treatment_trastuzumab','her2_positive==1 and has_brain_mets==0'),
    'ola_BRCA_ECOG<2': fit_combined_subgroup('treatment_olaparib','(brca1_mutation==1 or brca2_mutation==1) and ecog_ps < 2'),
    'palbo_HR+HER2-_postmeno_ECOG<2': fit_combined_subgroup('treatment_palbociclib','(er_positive==1 or pr_positive==1) and her2_positive==0 and postmenopausal==1 and ecog_ps < 2'),
    'saci_TNBC_StageIV': fit_combined_subgroup('treatment_sacituzumab_govitecan','er_positive==0 and pr_positive==0 and her2_positive==0 and stage_iv==1'),
    'pembro_TNBC_StageIV': fit_combined_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0 and stage_iv==1'),
    'pembro_TNBC_StageIV_noBrain': fit_combined_subgroup('treatment_pembrolizumab','er_positive==0 and pr_positive==0 and her2_positive==0 and stage_iv==1 and has_brain_mets==0'),
}
print('Iter23 done')

# ===== ITERATION 24: confirm final hypotheses with formal interaction tests =====
def final_subgroup_indicator(name, predicate):
    df[name] = df.eval(predicate).astype(int)

final_subgroup_indicator('sub_tam', 'er_positive==1')
final_subgroup_indicator('sub_tras', 'her2_positive==1')
final_subgroup_indicator('sub_ola', 'brca1_mutation==1 or brca2_mutation==1')
final_subgroup_indicator('sub_palbo', '(er_positive==1 or pr_positive==1) and her2_positive==0')
final_subgroup_indicator('sub_saci', 'er_positive==0 and pr_positive==0 and her2_positive==0')
final_subgroup_indicator('sub_pembro', 'er_positive==0 and pr_positive==0 and her2_positive==0')

OUT['iter24_final_subgroup_interactions'] = {
    'tamoxifen_x_sub': interaction_test('treatment_tamoxifen', 'sub_tam'),
    'trastuzumab_x_sub': interaction_test('treatment_trastuzumab', 'sub_tras'),
    'olaparib_x_sub': interaction_test('treatment_olaparib', 'sub_ola'),
    'palbociclib_x_sub': interaction_test('treatment_palbociclib', 'sub_palbo'),
    'sacituzumab_x_sub': interaction_test('treatment_sacituzumab_govitecan', 'sub_saci'),
    'pembrolizumab_x_sub': interaction_test('treatment_pembrolizumab', 'sub_pembro'),
}
print('Iter24 done')

# ===== ITERATION 25: combined-treatment view — does benefit accrue when "matched"? =====
# For each treatment-indication "match", compute mean PFS in matched-on, matched-off,
# unmatched-on, unmatched-off
def four_cell(tx, sub_col):
    res = {}
    for s in [0,1]:
        for t in [0,1]:
            mask = (df[sub_col]==s) & (df[tx]==t)
            res[f's{s}_t{t}'] = {'n': int(mask.sum()), 'mean_pfs': float(df.loc[mask,'pfs_months'].mean())}
    return res

OUT['iter25_four_cell_views'] = {
    'tamoxifen_x_sub_tam': four_cell('treatment_tamoxifen','sub_tam'),
    'trastuzumab_x_sub_tras': four_cell('treatment_trastuzumab','sub_tras'),
    'olaparib_x_sub_ola': four_cell('treatment_olaparib','sub_ola'),
    'palbociclib_x_sub_palbo': four_cell('treatment_palbociclib','sub_palbo'),
    'sacituzumab_x_sub_saci': four_cell('treatment_sacituzumab_govitecan','sub_saci'),
    'pembrolizumab_x_sub_pembro': four_cell('treatment_pembrolizumab','sub_pembro'),
}
print('Iter25 done')

with open('final_results.json','w') as f:
    json.dump(OUT, f, indent=2, default=str)
print('Saved final_results.json')
