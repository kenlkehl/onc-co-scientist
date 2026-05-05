"""Comprehensive analysis of ds001_nsclc."""
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT = {}

df = pd.read_parquet('dataset.parquet')
print('shape', df.shape)

# helper
def linreg(formula, data=df):
    m = smf.ols(formula, data=data).fit()
    return m

def report(label, m, term):
    coef = m.params[term]
    p = m.pvalues[term]
    print(f"{label} | {term} = {coef:.4f}, p = {p:.3e}")
    return float(coef), float(p)

# encode categoricals
df['hist_sq'] = (df['histology']=='squamous').astype(int)
df['smk_never'] = (df['smoking_status']=='never').astype(int)
df['smk_current'] = (df['smoking_status']=='current').astype(int)
df['smk_former'] = (df['smoking_status']=='former').astype(int)

###############################################################################
# Iter 1 — main effects of treatments on pfs
###############################################################################
res = {}
for tx in ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']:
    g1 = df.loc[df[tx]==1,'pfs_months']
    g0 = df.loc[df[tx]==0,'pfs_months']
    t = stats.ttest_ind(g1, g0, equal_var=False)
    res[tx] = {
        'mean_on': float(g1.mean()),
        'mean_off': float(g0.mean()),
        'mean_diff': float(g1.mean()-g0.mean()),
        'p': float(t.pvalue),
        'n_on': int(g1.size),
        'n_off': int(g0.size),
    }
OUT['it1_treatment_main'] = res
print('it1', res)

###############################################################################
# Iter 2 — clinical features main effect (univariate regression on pfs)
###############################################################################
clin_cols = ['age_years','sex_female','ecog_ps','stage_iv','has_brain_mets',
             'albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr',
             'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
             'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
             'sodium_meq_l','potassium_meq_l','calcium_mg_dl',
             'hist_sq','smk_never','smk_current']
res = {}
for c in clin_cols:
    m = linreg(f'pfs_months ~ {c}')
    res[c] = {'beta': float(m.params[c]), 'p': float(m.pvalues[c])}
OUT['it2_clin_uni'] = res
print('it2 done')

###############################################################################
# Iter 3 — biomarker main effects on pfs
###############################################################################
bio = ['egfr_mutation','kras_g12c','alk_fusion','stk11_mutation','brca2_mutation','pdl1_tps','tmb_high']
res = {}
for c in bio:
    m = linreg(f'pfs_months ~ {c}')
    res[c] = {'beta': float(m.params[c]), 'p': float(m.pvalues[c])}
OUT['it3_bio_uni'] = res
print('it3', res)

###############################################################################
# Iter 4 — multivariable model
###############################################################################
mv_cols = clin_cols + bio + ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']
formula = 'pfs_months ~ ' + ' + '.join(mv_cols)
m = linreg(formula)
res = {c: {'beta': float(m.params[c]), 'p': float(m.pvalues[c])} for c in mv_cols}
OUT['it4_mv'] = res
print('it4 done')

###############################################################################
# Iter 5 — Pembrolizumab x pdl1, tmb_high, smoking, histology, stk11
###############################################################################
# pdl1_tps is on a 0-1 scale.
res = {}
for thr_label, thr in [('0.01', 0.01), ('0.5', 0.5)]:
    sub_high = df[df['pdl1_tps']>=thr]
    sub_low = df[df['pdl1_tps']<thr]
    a = sub_high.loc[sub_high['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub_high.loc[sub_high['treatment_pembrolizumab']==0,'pfs_months'].mean()
    b = sub_low.loc[sub_low['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub_low.loc[sub_low['treatment_pembrolizumab']==0,'pfs_months'].mean()
    res[f'pdl1_ge_{thr_label}'] = {'effect_high': float(a), 'n_high': int(sub_high.shape[0]),
                                    'effect_low': float(b), 'n_low': int(sub_low.shape[0])}
# Interaction p-value via regression
m = linreg('pfs_months ~ treatment_pembrolizumab * pdl1_tps')
res['inter_pembro_pdl1'] = {'beta': float(m.params['treatment_pembrolizumab:pdl1_tps']), 'p': float(m.pvalues['treatment_pembrolizumab:pdl1_tps'])}
m = linreg('pfs_months ~ treatment_pembrolizumab * tmb_high')
res['inter_pembro_tmb'] = {'beta': float(m.params['treatment_pembrolizumab:tmb_high']), 'p': float(m.pvalues['treatment_pembrolizumab:tmb_high'])}
m = linreg('pfs_months ~ treatment_pembrolizumab * stk11_mutation')
res['inter_pembro_stk11'] = {'beta': float(m.params['treatment_pembrolizumab:stk11_mutation']), 'p': float(m.pvalues['treatment_pembrolizumab:stk11_mutation'])}
m = linreg('pfs_months ~ treatment_pembrolizumab * hist_sq')
res['inter_pembro_squamous'] = {'beta': float(m.params['treatment_pembrolizumab:hist_sq']), 'p': float(m.pvalues['treatment_pembrolizumab:hist_sq'])}
m = linreg('pfs_months ~ treatment_pembrolizumab * smk_never')
res['inter_pembro_never_smk'] = {'beta': float(m.params['treatment_pembrolizumab:smk_never']), 'p': float(m.pvalues['treatment_pembrolizumab:smk_never'])}
OUT['it5_pembro_subgroup'] = res
print('it5', res)

###############################################################################
# Iter 6 — sotorasib x kras_g12c, plus stk11 interaction
###############################################################################
res = {}
for sub_name, sub_filter in [('kras_g12c=1', df['kras_g12c']==1), ('kras_g12c=0', df['kras_g12c']==0)]:
    sub = df[sub_filter]
    a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
    res[f'effect_in_{sub_name}'] = float(a)
m = linreg('pfs_months ~ treatment_sotorasib * kras_g12c')
res['inter_sot_kras'] = {'beta': float(m.params['treatment_sotorasib:kras_g12c']), 'p': float(m.pvalues['treatment_sotorasib:kras_g12c'])}
m = linreg('pfs_months ~ treatment_sotorasib * stk11_mutation')
res['inter_sot_stk11'] = {'beta': float(m.params['treatment_sotorasib:stk11_mutation']), 'p': float(m.pvalues['treatment_sotorasib:stk11_mutation'])}
# 3-way: among kras+ patients, does stk11 modify sotorasib effect?
sub = df[df['kras_g12c']==1]
m = linreg('pfs_months ~ treatment_sotorasib * stk11_mutation', data=sub)
res['inter_sot_stk11_in_krasg12c'] = {'beta': float(m.params['treatment_sotorasib:stk11_mutation']), 'p': float(m.pvalues['treatment_sotorasib:stk11_mutation'])}
OUT['it6_sotorasib_subgroup'] = res
print('it6', res)

###############################################################################
# Iter 7 — olaparib x brca2
###############################################################################
res = {}
for sub_name, sub_filter in [('brca2=1', df['brca2_mutation']==1), ('brca2=0', df['brca2_mutation']==0)]:
    sub = df[sub_filter]
    a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
    res[f'effect_in_{sub_name}'] = {'effect': float(a), 'n': int(sub.shape[0])}
m = linreg('pfs_months ~ treatment_olaparib * brca2_mutation')
res['inter_ola_brca2'] = {'beta': float(m.params['treatment_olaparib:brca2_mutation']), 'p': float(m.pvalues['treatment_olaparib:brca2_mutation'])}
OUT['it7_olaparib_subgroup'] = res
print('it7', res)

###############################################################################
# Iter 8 — osimertinib x egfr_mutation, x alk_fusion
###############################################################################
res = {}
for sub_name, sub_filter in [('egfr=1', df['egfr_mutation']==1), ('egfr=0', df['egfr_mutation']==0),
                               ('alk=1', df['alk_fusion']==1), ('alk=0', df['alk_fusion']==0)]:
    sub = df[sub_filter]
    a = sub.loc[sub['treatment_osimertinib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_osimertinib']==0,'pfs_months'].mean()
    res[f'effect_in_{sub_name}'] = {'effect': float(a), 'n': int(sub.shape[0])}
m = linreg('pfs_months ~ treatment_osimertinib * egfr_mutation')
res['inter_osi_egfr'] = {'beta': float(m.params['treatment_osimertinib:egfr_mutation']), 'p': float(m.pvalues['treatment_osimertinib:egfr_mutation'])}
m = linreg('pfs_months ~ treatment_osimertinib * alk_fusion')
res['inter_osi_alk'] = {'beta': float(m.params['treatment_osimertinib:alk_fusion']), 'p': float(m.pvalues['treatment_osimertinib:alk_fusion'])}
OUT['it8_osi_subgroup'] = res
print('it8', res)

###############################################################################
# Iter 9 — systematic interaction screen of each treatment x feature
###############################################################################
modifier_cols = ['age_years','sex_female','ecog_ps','stage_iv','has_brain_mets',
                 'egfr_mutation','kras_g12c','alk_fusion','stk11_mutation','brca2_mutation',
                 'pdl1_tps','tmb_high',
                 'albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr',
                 'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
                 'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
                 'sodium_meq_l','potassium_meq_l','calcium_mg_dl',
                 'hist_sq','smk_never','smk_current']
screen = {}
for tx in ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']:
    rows = []
    for mod in modifier_cols:
        try:
            m = linreg(f'pfs_months ~ {tx} * {mod}')
            term = f'{tx}:{mod}'
            rows.append({'modifier': mod, 'beta': float(m.params[term]), 'p': float(m.pvalues[term])})
        except Exception as e:
            rows.append({'modifier': mod, 'error': str(e)})
    rows.sort(key=lambda r: r.get('p', 1.0))
    screen[tx] = rows
OUT['it9_screen'] = screen
print('it9 screen complete')

###############################################################################
# Iter 10 — Joint pembro subgroup: PDL1 high, smoker, non-squamous, no STK11
###############################################################################
res = {}
# PDL1 >=50 + tmb_high
for tps_thr in [1,50]:
    for tmb_req in [0,1]:
        sub = df[(df['pdl1_tps']>=tps_thr) & (df['tmb_high']>=tmb_req)]
        if sub.shape[0] < 100: continue
        a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
        res[f'pdl1>={tps_thr}, tmb_high>={tmb_req}'] = {'effect': float(a), 'n': int(sub.shape[0])}

# In PDL1 high subgroup, does stk11 suppress?
sub_high = df[df['pdl1_tps']>=0.5]
res['n_pdl1_ge0.5'] = int(sub_high.shape[0])
for stk in [0,1]:
    s = sub_high[sub_high['stk11_mutation']==stk]
    a = s.loc[s['treatment_pembrolizumab']==1,'pfs_months'].mean() - s.loc[s['treatment_pembrolizumab']==0,'pfs_months'].mean()
    res[f'pdl1ge0.5_stk11={stk}_effect'] = {'effect': float(a), 'n': int(s.shape[0])}
m = linreg('pfs_months ~ treatment_pembrolizumab * stk11_mutation', data=sub_high)
res['inter_pembro_stk11_in_pdl1ge50'] = {'beta': float(m.params['treatment_pembrolizumab:stk11_mutation']), 'p': float(m.pvalues['treatment_pembrolizumab:stk11_mutation'])}

# In PDL1 high + STK11=0
sub2 = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0)]
a = sub2.loc[sub2['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub2.loc[sub2['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pdl1ge0.5_stk11=0_effect'] = {'effect': float(a), 'n': int(sub2.shape[0])}
# PDL1>=50 + STK11=0 + smoker
sub3 = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0) & (df['smoking_status']!='never')]
a = sub3.loc[sub3['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub3.loc[sub3['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pdl1ge0.5_stk11=0_smoker_effect'] = {'effect': float(a), 'n': int(sub3.shape[0])}
# PDL1>=50 + STK11=0 + non-squamous
sub4 = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0) & (df['hist_sq']==0)]
a = sub4.loc[sub4['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub4.loc[sub4['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pdl1ge0.5_stk11=0_nonsq_effect'] = {'effect': float(a), 'n': int(sub4.shape[0])}
OUT['it10_pembro_joint'] = res
print('it10', res)

###############################################################################
# Iter 11 — Joint sotorasib subgroup: KRAS G12C + STK11 modifier
###############################################################################
res = {}
for kras in [0,1]:
    for stk in [0,1]:
        sub = df[(df['kras_g12c']==kras) & (df['stk11_mutation']==stk)]
        a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
        res[f'kras={kras}_stk11={stk}'] = {'effect': float(a), 'n': int(sub.shape[0])}
# 3-way regression
m = linreg('pfs_months ~ treatment_sotorasib * kras_g12c * stk11_mutation')
for t in m.params.index:
    if 'treatment_sotorasib' in t:
        res[t] = {'beta': float(m.params[t]), 'p': float(m.pvalues[t])}
OUT['it11_sotorasib_joint'] = res
print('it11', res)

###############################################################################
# Iter 12 — Olaparib + brca2 subgroup
###############################################################################
res = {}
sub = df[df['brca2_mutation']==1]
a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
res['brca2=1_effect'] = {'effect': float(a), 'n': int(sub.shape[0])}
sub = df[df['brca2_mutation']==0]
a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
res['brca2=0_effect'] = {'effect': float(a), 'n': int(sub.shape[0])}

# In brca2+, does any feature suppress?
for mod in ['ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','nlr','crp_mg_l','stk11_mutation']:
    try:
        m = linreg(f'pfs_months ~ treatment_olaparib * {mod}', data=df[df['brca2_mutation']==1])
        term = f'treatment_olaparib:{mod}'
        res[f'inter_olaparib_{mod}_in_brca2pos'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
    except Exception as e:
        res[f'inter_olaparib_{mod}_in_brca2pos'] = {'error': str(e)}
OUT['it12_olaparib_joint'] = res
print('it12', res)

###############################################################################
# Iter 13 — Osimertinib in EGFR+, modifiers among that subgroup
###############################################################################
res = {}
sub_egfr = df[df['egfr_mutation']==1]
res['n_egfr_pos'] = int(sub_egfr.shape[0])
a = sub_egfr.loc[sub_egfr['treatment_osimertinib']==1,'pfs_months'].mean() - sub_egfr.loc[sub_egfr['treatment_osimertinib']==0,'pfs_months'].mean()
res['osi_effect_in_egfrpos'] = {'effect': float(a), 'n': int(sub_egfr.shape[0])}
for mod in ['ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','ldh_u_l','smk_never','hist_sq','stk11_mutation','tmb_high']:
    try:
        m = linreg(f'pfs_months ~ treatment_osimertinib * {mod}', data=sub_egfr)
        term = f'treatment_osimertinib:{mod}'
        res[f'inter_osi_{mod}_in_egfrpos'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
    except Exception as e:
        res[f'inter_osi_{mod}_in_egfrpos'] = {'error': str(e)}
OUT['it13_osimertinib_joint'] = res
print('it13', res)

###############################################################################
# Iter 14 — Lab-based modifiers of pembrolizumab in PDL1>=50, STK11=0
###############################################################################
res = {}
base = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0)]
res['n_base'] = int(base.shape[0])
for mod in ['albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl']:
    m = linreg(f'pfs_months ~ treatment_pembrolizumab * {mod}', data=base)
    term = f'treatment_pembrolizumab:{mod}'
    res[f'inter_pembro_{mod}_in_pdl1ge50_stk11=0'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
OUT['it14_pembro_lab_modifiers'] = res
print('it14', res)

###############################################################################
# Iter 15 — Sotorasib effect in KRAS+ further modifiers
###############################################################################
res = {}
base = df[df['kras_g12c']==1]
res['n_base'] = int(base.shape[0])
a = base.loc[base['treatment_sotorasib']==1,'pfs_months'].mean() - base.loc[base['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_effect_in_kraspos'] = float(a)
# stk11 modifier among kras+ already tested. Also check liver function etc.
for mod in ['ecog_ps','stage_iv','has_brain_mets','albumin_g_dl','ldh_u_l','nlr','stk11_mutation','hist_sq']:
    m = linreg(f'pfs_months ~ treatment_sotorasib * {mod}', data=base)
    term = f'treatment_sotorasib:{mod}'
    res[f'inter_sot_{mod}_in_kraspos'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
# kras+ AND stk11=0 specifically
sub = df[(df['kras_g12c']==1) & (df['stk11_mutation']==0)]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_effect_in_kraspos_stk11=0'] = {'effect': float(a), 'n': int(sub.shape[0])}
sub = df[(df['kras_g12c']==1) & (df['stk11_mutation']==1)]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_effect_in_kraspos_stk11=1'] = {'effect': float(a), 'n': int(sub.shape[0])}
OUT['it15_sot_modifiers'] = res
print('it15', res)

###############################################################################
# Iter 16 — Brain mets effect on PFS overall and as effect modifier
###############################################################################
res = {}
m = linreg('pfs_months ~ has_brain_mets')
res['main_brain_mets'] = {'beta': float(m.params['has_brain_mets']), 'p': float(m.pvalues['has_brain_mets'])}
for tx in ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']:
    m = linreg(f'pfs_months ~ {tx} * has_brain_mets')
    term = f'{tx}:has_brain_mets'
    res[f'inter_{tx}_brain'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
OUT['it16_brain_mets'] = res
print('it16', res)

###############################################################################
# Iter 17 — Sex / age / ECOG interactions with each treatment
###############################################################################
res = {}
for tx in ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']:
    for mod in ['sex_female','age_years','ecog_ps']:
        m = linreg(f'pfs_months ~ {tx} * {mod}')
        term = f'{tx}:{mod}'
        res[f'{tx}_x_{mod}'] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
OUT['it17_demo_modifiers'] = res
print('it17', res)

###############################################################################
# Iter 18 — Final candidate subgroups for each treatment
###############################################################################
res = {}
# pembro: pdl1>=50 + stk11=0
sub = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0)]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1ge50_stk11=0'] = {'effect': float(a), 'n': int(sub.shape[0]),
                                    't_p': float(stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'], sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']).pvalue)}
# pembro: pdl1>=50 alone
sub = df[df['pdl1_tps']>=0.5]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1ge50'] = {'effect': float(a), 'n': int(sub.shape[0]),
                           't_p': float(stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'], sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']).pvalue)}

# sotorasib: kras_g12c=1
sub = df[df['kras_g12c']==1]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_kras1'] = {'effect': float(a), 'n': int(sub.shape[0]),
                    't_p': float(stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months']).pvalue)}
# kras=1, stk11=0
sub = df[(df['kras_g12c']==1) & (df['stk11_mutation']==0)]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_kras1_stk11=0'] = {'effect': float(a), 'n': int(sub.shape[0]),
                              't_p': float(stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months']).pvalue)}

# olaparib: brca2=1
sub = df[df['brca2_mutation']==1]
a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
res['ola_brca2'] = {'effect': float(a), 'n': int(sub.shape[0]),
                    't_p': float(stats.ttest_ind(sub.loc[sub['treatment_olaparib']==1,'pfs_months'], sub.loc[sub['treatment_olaparib']==0,'pfs_months']).pvalue)}

# osimertinib: egfr=1
sub = df[df['egfr_mutation']==1]
a = sub.loc[sub['treatment_osimertinib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_osimertinib']==0,'pfs_months'].mean()
res['osi_egfr1'] = {'effect': float(a), 'n': int(sub.shape[0]),
                    't_p': float(stats.ttest_ind(sub.loc[sub['treatment_osimertinib']==1,'pfs_months'], sub.loc[sub['treatment_osimertinib']==0,'pfs_months']).pvalue)}
OUT['it18_final_subgroups'] = res
print('it18', res)

###############################################################################
# Iter 19 — Gradient boosting / regression tree-style: brute search subgroups for each treatment
###############################################################################
# Use 2-way feature combos for each treatment with binary modifiers
binary_mods = ['sex_female','stage_iv','has_brain_mets','egfr_mutation','kras_g12c','alk_fusion',
               'stk11_mutation','brca2_mutation','tmb_high','hist_sq','smk_never','smk_current']
def best_subgroup(tx):
    rows = []
    # single feature
    for f in binary_mods:
        for v in [0,1]:
            sub = df[df[f]==v]
            if sub.shape[0] < 200: continue
            a = sub.loc[sub[tx]==1,'pfs_months'].mean() - sub.loc[sub[tx]==0,'pfs_months'].mean()
            t = stats.ttest_ind(sub.loc[sub[tx]==1,'pfs_months'], sub.loc[sub[tx]==0,'pfs_months'], equal_var=False)
            rows.append({'sub': f'{f}={v}', 'n': int(sub.shape[0]), 'effect': float(a), 'p': float(t.pvalue)})
    # 2-feature combos
    import itertools
    for f1,f2 in itertools.combinations(binary_mods,2):
        for v1 in [0,1]:
            for v2 in [0,1]:
                sub = df[(df[f1]==v1) & (df[f2]==v2)]
                if sub.shape[0] < 200: continue
                a = sub.loc[sub[tx]==1,'pfs_months'].mean() - sub.loc[sub[tx]==0,'pfs_months'].mean()
                t = stats.ttest_ind(sub.loc[sub[tx]==1,'pfs_months'], sub.loc[sub[tx]==0,'pfs_months'], equal_var=False)
                rows.append({'sub': f'{f1}={v1}, {f2}={v2}', 'n': int(sub.shape[0]), 'effect': float(a), 'p': float(t.pvalue)})
    rows.sort(key=lambda r: r['effect'], reverse=True)
    return rows[:20]
res = {}
for tx in ['treatment_pembrolizumab','treatment_sotorasib','treatment_olaparib','treatment_osimertinib']:
    res[tx] = best_subgroup(tx)
OUT['it19_brute'] = res
print('it19 done')

###############################################################################
# Iter 20 — confirm null effect for olaparib and overall pembro outside PDL1
###############################################################################
res = {}
sub = df[df['pdl1_tps']<0.01]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1<0.01'] = {'effect': float(a), 'n': int(sub.shape[0]),
                         't_p': float(stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'], sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']).pvalue)}
sub = df[(df['pdl1_tps']>=0.01)&(df['pdl1_tps']<0.5)]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
res['pembro_pdl1_0.01_0.49'] = {'effect': float(a), 'n': int(sub.shape[0]),
                            't_p': float(stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'], sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']).pvalue)}
sub = df[df['kras_g12c']==0]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
res['sot_kras0'] = {'effect': float(a), 'n': int(sub.shape[0]),
                     't_p': float(stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months']).pvalue)}
sub = df[df['brca2_mutation']==0]
a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
res['ola_brca2_0'] = {'effect': float(a), 'n': int(sub.shape[0]),
                       't_p': float(stats.ttest_ind(sub.loc[sub['treatment_olaparib']==1,'pfs_months'], sub.loc[sub['treatment_olaparib']==0,'pfs_months']).pvalue)}
sub = df[df['egfr_mutation']==0]
a = sub.loc[sub['treatment_osimertinib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_osimertinib']==0,'pfs_months'].mean()
res['osi_egfr_0'] = {'effect': float(a), 'n': int(sub.shape[0]),
                      't_p': float(stats.ttest_ind(sub.loc[sub['treatment_osimertinib']==1,'pfs_months'], sub.loc[sub['treatment_osimertinib']==0,'pfs_months']).pvalue)}
OUT['it20_marker_neg_subgroups'] = res
print('it20', res)

###############################################################################
# Iter 21 — TMB high effect within PDL1 high
###############################################################################
res = {}
for tps in [0.01,0.5]:
    for tmb in [0,1]:
        sub = df[(df['pdl1_tps']>=tps) & (df['tmb_high']==tmb) & (df['stk11_mutation']==0)]
        if sub.shape[0]<100: continue
        a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
        res[f'pdl1ge{tps}_tmb={tmb}_stk11=0'] = {'effect': float(a), 'n': int(sub.shape[0])}
OUT['it21_pembro_tmb'] = res
print('it21', res)

###############################################################################
# Iter 22 — sotorasib and other co-mutations among kras+
###############################################################################
res = {}
sub = df[df['kras_g12c']==1]
# full reg
m = smf.ols('pfs_months ~ treatment_sotorasib + stk11_mutation + tmb_high + ecog_ps + albumin_g_dl + ldh_u_l + has_brain_mets + treatment_sotorasib:stk11_mutation + treatment_sotorasib:tmb_high', data=sub).fit()
for term in m.params.index:
    if 'treatment' in term:
        res[term] = {'beta': float(m.params[term]), 'p': float(m.pvalues[term])}
OUT['it22_sot_full'] = res
print('it22', res)

###############################################################################
# Iter 23 — osi in egfr+ refinement (smoking, alk co-occurrence, brain mets)
###############################################################################
res = {}
sub_egfr = df[df['egfr_mutation']==1]
for mod in ['has_brain_mets','smk_never','tmb_high','stk11_mutation']:
    s1 = sub_egfr[sub_egfr[mod]==1]
    s0 = sub_egfr[sub_egfr[mod]==0]
    a1 = s1.loc[s1['treatment_osimertinib']==1,'pfs_months'].mean() - s1.loc[s1['treatment_osimertinib']==0,'pfs_months'].mean()
    a0 = s0.loc[s0['treatment_osimertinib']==1,'pfs_months'].mean() - s0.loc[s0['treatment_osimertinib']==0,'pfs_months'].mean()
    res[mod] = {'effect_when_1': float(a1), 'n_when_1': int(s1.shape[0]),
                'effect_when_0': float(a0), 'n_when_0': int(s0.shape[0])}
OUT['it23_osi_modifier_in_egfrpos'] = res
print('it23', res)

###############################################################################
# Iter 24 — confirm pembro has zero effect in pdl1<1
###############################################################################
res = {}
# 3-way: pembro x pdl1 x stk11
m = smf.ols('pfs_months ~ treatment_pembrolizumab * pdl1_tps + treatment_pembrolizumab * stk11_mutation + pdl1_tps:stk11_mutation', data=df).fit()
for t in m.params.index:
    if 'treatment' in t:
        res[t] = {'beta': float(m.params[t]), 'p': float(m.pvalues[t])}

# pembro effect at PDL1 quintiles in stk11=0
for lo,hi in [(0.0,0.05),(0.05,0.2),(0.2,0.4),(0.4,0.6),(0.6,1.01)]:
    sub = df[(df['pdl1_tps']>=lo) & (df['pdl1_tps']<hi) & (df['stk11_mutation']==0)]
    if sub.shape[0]<50: continue
    a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
    res[f'pdl1_{lo}_{hi}_stk11=0'] = {'effect': float(a), 'n': int(sub.shape[0])}
OUT['it24_pembro_continuous_pdl1'] = res
print('it24', res)

###############################################################################
# Iter 25 — Final consolidated treatment-effect subgroup hypotheses
###############################################################################
res = {}
# Final pembro best subgroup
sub = df[(df['pdl1_tps']>=0.5) & (df['stk11_mutation']==0)]
a = sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'].mean() - sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months'].mean()
n = int(sub.shape[0])
t = stats.ttest_ind(sub.loc[sub['treatment_pembrolizumab']==1,'pfs_months'], sub.loc[sub['treatment_pembrolizumab']==0,'pfs_months']).pvalue
res['pembro_FINAL'] = {'subgroup': 'pdl1_tps>=50 AND stk11_mutation=0', 'effect': float(a), 'n': n, 't_p': float(t)}

# sotorasib
sub = df[(df['kras_g12c']==1) & (df['stk11_mutation']==0)]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
n = int(sub.shape[0])
t = stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months']).pvalue
res['sot_FINAL'] = {'subgroup': 'kras_g12c=1 AND stk11_mutation=0', 'effect': float(a), 'n': n, 't_p': float(t)}

sub = df[df['kras_g12c']==1]
a = sub.loc[sub['treatment_sotorasib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_sotorasib']==0,'pfs_months'].mean()
n = int(sub.shape[0])
t = stats.ttest_ind(sub.loc[sub['treatment_sotorasib']==1,'pfs_months'], sub.loc[sub['treatment_sotorasib']==0,'pfs_months']).pvalue
res['sot_simple'] = {'subgroup': 'kras_g12c=1', 'effect': float(a), 'n': n, 't_p': float(t)}

# olaparib
sub = df[df['brca2_mutation']==1]
a = sub.loc[sub['treatment_olaparib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_olaparib']==0,'pfs_months'].mean()
n = int(sub.shape[0])
t = stats.ttest_ind(sub.loc[sub['treatment_olaparib']==1,'pfs_months'], sub.loc[sub['treatment_olaparib']==0,'pfs_months']).pvalue
res['ola_FINAL'] = {'subgroup': 'brca2_mutation=1', 'effect': float(a), 'n': n, 't_p': float(t)}

# osimertinib
sub = df[df['egfr_mutation']==1]
a = sub.loc[sub['treatment_osimertinib']==1,'pfs_months'].mean() - sub.loc[sub['treatment_osimertinib']==0,'pfs_months'].mean()
n = int(sub.shape[0])
t = stats.ttest_ind(sub.loc[sub['treatment_osimertinib']==1,'pfs_months'], sub.loc[sub['treatment_osimertinib']==0,'pfs_months']).pvalue
res['osi_FINAL'] = {'subgroup': 'egfr_mutation=1', 'effect': float(a), 'n': n, 't_p': float(t)}
OUT['it25_final'] = res
print('it25', res)

with open('fresh_results.json','w') as f:
    json.dump(OUT, f, indent=2, default=str)
print('saved fresh_results.json')
