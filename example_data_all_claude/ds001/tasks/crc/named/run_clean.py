"""Iterative analysis of ds001_crc cohort. Writes raw results to clean_results.json."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
out = {}

def fit_ols(formula, label):
    m = smf.ols(formula, data=df).fit()
    return m, label

def record(key, **kw):
    out[key] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in kw.items()}

# ---------- Iteration 1: demographics / baseline clinical ----------
# 1a age main effect
m = smf.ols('pfs_months ~ age_years', data=df).fit()
record('i1_age', beta=m.params['age_years'], p=m.pvalues['age_years'], n=len(df),
       result=f"PFS ~ age_years OLS: beta={m.params['age_years']:.5f} mo per year, p={m.pvalues['age_years']:.4g}")

# 1b sex main effect
m = smf.ols('pfs_months ~ sex_female', data=df).fit()
record('i1_sex', beta=m.params['sex_female'], p=m.pvalues['sex_female'],
       result=f"PFS ~ sex_female OLS: beta={m.params['sex_female']:.4f} mo, p={m.pvalues['sex_female']:.4g}")

# 1c ECOG
m = smf.ols('pfs_months ~ C(ecog_ps)', data=df).fit()
m_lin = smf.ols('pfs_months ~ ecog_ps', data=df).fit()
ec_means = df.groupby('ecog_ps')['pfs_months'].mean().to_dict()
record('i1_ecog', beta=m_lin.params['ecog_ps'], p=m_lin.pvalues['ecog_ps'],
       means=ec_means,
       result=f"PFS by ECOG: {ec_means}; linear beta={m_lin.params['ecog_ps']:.4f}, p={m_lin.pvalues['ecog_ps']:.4g}")

# 1d Stage IV
m = smf.ols('pfs_months ~ stage_iv', data=df).fit()
record('i1_stage', beta=m.params['stage_iv'], p=m.pvalues['stage_iv'],
       result=f"PFS ~ stage_iv: beta={m.params['stage_iv']:.4f}, p={m.pvalues['stage_iv']:.4g}")

# ---------- Iteration 2: labs / inflammatory markers ----------
labs = ['cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr',
        'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l',
        'total_bilirubin_mg_dl','creatinine_mg_dl','bun_mg_dl',
        'sodium_meq_l','potassium_meq_l','calcium_mg_dl']
lab_results = {}
for c in labs:
    m = smf.ols(f'pfs_months ~ {c}', data=df).fit()
    lab_results[c] = {'beta': float(m.params[c]), 'p': float(m.pvalues[c])}
out['i2_labs'] = {'results': lab_results,
                  'result_summary': '; '.join(f"{c}: beta={v['beta']:.4g}, p={v['p']:.3g}" for c,v in lab_results.items())}

# ---------- Iteration 3: tumor location + RAS/RAF main effects ----------
for var in ['right_sided_primary','kras_mutation','nras_mutation','braf_v600e','msi_high','her2_amplified','ntrk_fusion']:
    m = smf.ols(f'pfs_months ~ {var}', data=df).fit()
    means = df.groupby(var)['pfs_months'].mean().to_dict()
    record(f'i3_{var}', beta=m.params[var], p=m.pvalues[var], means=means,
           result=f"PFS by {var}: {means}; beta={m.params[var]:.4f}, p={m.pvalues[var]:.4g}")

# ---------- Iteration 4: marginal treatment main effects ----------
treatments = ['treatment_cetuximab','treatment_bevacizumab','treatment_pembrolizumab',
              'treatment_encorafenib','treatment_trastuzumab_tucatinib','treatment_regorafenib']
for t in treatments:
    m = smf.ols(f'pfs_months ~ {t}', data=df).fit()
    means = df.groupby(t)['pfs_months'].mean().to_dict()
    record(f'i4_{t}', beta=m.params[t], p=m.pvalues[t], means=means,
           result=f"PFS by {t}: {means}; beta={m.params[t]:.4f} mo, p={m.pvalues[t]:.4g}")

# ---------- Iteration 5: cetuximab x KRAS interaction ----------
m = smf.ols('pfs_months ~ treatment_cetuximab * kras_mutation', data=df).fit()
record('i5_cetux_x_kras', beta=m.params['treatment_cetuximab:kras_mutation'],
       p=m.pvalues['treatment_cetuximab:kras_mutation'],
       cetux_main=float(m.params['treatment_cetuximab']),
       kras_main=float(m.params['kras_mutation']),
       result=f"Cetuximab x KRAS interaction: beta={m.params['treatment_cetuximab:kras_mutation']:.4f}, p={m.pvalues['treatment_cetuximab:kras_mutation']:.4g}")

# stratified
for k in [0,1]:
    sub = df[df['kras_mutation']==k]
    m = smf.ols('pfs_months ~ treatment_cetuximab', data=sub).fit()
    record(f'i5_cetux_in_kras{k}', beta=m.params['treatment_cetuximab'], p=m.pvalues['treatment_cetuximab'],
           n=len(sub),
           result=f"Cetuximab effect in KRAS={k} (n={len(sub)}): beta={m.params['treatment_cetuximab']:.4f}, p={m.pvalues['treatment_cetuximab']:.4g}")

# ---------- Iteration 6: cetuximab x NRAS, x BRAF, x right_sided ----------
for mod in ['nras_mutation','braf_v600e','right_sided_primary']:
    m = smf.ols(f'pfs_months ~ treatment_cetuximab * {mod}', data=df).fit()
    interaction = f'treatment_cetuximab:{mod}'
    record(f'i6_cetux_x_{mod}', beta=m.params[interaction], p=m.pvalues[interaction],
           result=f"Cetuximab x {mod} interaction beta={m.params[interaction]:.4f}, p={m.pvalues[interaction]:.4g}")
    for v in [0,1]:
        sub = df[df[mod]==v]
        if (sub['treatment_cetuximab']==1).sum() < 10 or (sub['treatment_cetuximab']==0).sum() < 10:
            continue
        m2 = smf.ols('pfs_months ~ treatment_cetuximab', data=sub).fit()
        record(f'i6_cetux_in_{mod}{v}', beta=m2.params['treatment_cetuximab'], p=m2.pvalues['treatment_cetuximab'],
               n=len(sub),
               result=f"Cetuximab effect in {mod}={v} (n={len(sub)}): beta={m2.params['treatment_cetuximab']:.4f}, p={m2.pvalues['treatment_cetuximab']:.4g}")

# ---------- Iteration 7: cetuximab in pan-wt, left-sided population ----------
wt = (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0)
left = df['right_sided_primary']==0
sub = df[wt & left]
m = smf.ols('pfs_months ~ treatment_cetuximab', data=sub).fit()
record('i7_cetux_panwt_left', beta=m.params['treatment_cetuximab'], p=m.pvalues['treatment_cetuximab'],
       n=len(sub),
       n_treated=int((sub['treatment_cetuximab']==1).sum()),
       result=f"Cetuximab in pan-RAS/BRAF wt + left-sided (n={len(sub)}): beta={m.params['treatment_cetuximab']:.4f}, p={m.pvalues['treatment_cetuximab']:.4g}")

# right-sided pan-wt for contrast
sub2 = df[wt & ~left]
m = smf.ols('pfs_months ~ treatment_cetuximab', data=sub2).fit()
record('i7_cetux_panwt_right', beta=m.params['treatment_cetuximab'], p=m.pvalues['treatment_cetuximab'],
       n=len(sub2),
       result=f"Cetuximab in pan-RAS/BRAF wt + right-sided (n={len(sub2)}): beta={m.params['treatment_cetuximab']:.4f}, p={m.pvalues['treatment_cetuximab']:.4g}")

# mutant pop
sub3 = df[~wt]
m = smf.ols('pfs_months ~ treatment_cetuximab', data=sub3).fit()
record('i7_cetux_mut', beta=m.params['treatment_cetuximab'], p=m.pvalues['treatment_cetuximab'],
       n=len(sub3),
       result=f"Cetuximab in any KRAS/NRAS/BRAF mutant (n={len(sub3)}): beta={m.params['treatment_cetuximab']:.4f}, p={m.pvalues['treatment_cetuximab']:.4g}")

# ---------- Iteration 8: pembrolizumab x MSI-high ----------
m = smf.ols('pfs_months ~ treatment_pembrolizumab * msi_high', data=df).fit()
record('i8_pembro_x_msi', beta=m.params['treatment_pembrolizumab:msi_high'],
       p=m.pvalues['treatment_pembrolizumab:msi_high'],
       result=f"Pembro x MSI interaction beta={m.params['treatment_pembrolizumab:msi_high']:.4f}, p={m.pvalues['treatment_pembrolizumab:msi_high']:.4g}")
for v in [0,1]:
    sub = df[df['msi_high']==v]
    m2 = smf.ols('pfs_months ~ treatment_pembrolizumab', data=sub).fit()
    record(f'i8_pembro_in_msi{v}', beta=m2.params['treatment_pembrolizumab'], p=m2.pvalues['treatment_pembrolizumab'],
           n=len(sub),
           result=f"Pembro effect in msi_high={v} (n={len(sub)}): beta={m2.params['treatment_pembrolizumab']:.4f}, p={m2.pvalues['treatment_pembrolizumab']:.4g}")

# ---------- Iteration 9: encorafenib x BRAF V600E ----------
m = smf.ols('pfs_months ~ treatment_encorafenib * braf_v600e', data=df).fit()
record('i9_enco_x_braf', beta=m.params['treatment_encorafenib:braf_v600e'],
       p=m.pvalues['treatment_encorafenib:braf_v600e'],
       result=f"Encorafenib x BRAF interaction beta={m.params['treatment_encorafenib:braf_v600e']:.4f}, p={m.pvalues['treatment_encorafenib:braf_v600e']:.4g}")
for v in [0,1]:
    sub = df[df['braf_v600e']==v]
    m2 = smf.ols('pfs_months ~ treatment_encorafenib', data=sub).fit()
    record(f'i9_enco_in_braf{v}', beta=m2.params['treatment_encorafenib'], p=m2.pvalues['treatment_encorafenib'],
           n=len(sub),
           result=f"Encorafenib effect in braf_v600e={v} (n={len(sub)}): beta={m2.params['treatment_encorafenib']:.4f}, p={m2.pvalues['treatment_encorafenib']:.4g}")

# ---------- Iteration 10: trastuzumab+tucatinib x HER2 ----------
m = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib * her2_amplified', data=df).fit()
ix = 'treatment_trastuzumab_tucatinib:her2_amplified'
record('i10_trastuc_x_her2', beta=m.params[ix], p=m.pvalues[ix],
       result=f"Trastuzumab+tucatinib x HER2 interaction beta={m.params[ix]:.4f}, p={m.pvalues[ix]:.4g}")
for v in [0,1]:
    sub = df[df['her2_amplified']==v]
    m2 = smf.ols('pfs_months ~ treatment_trastuzumab_tucatinib', data=sub).fit()
    record(f'i10_trastuc_in_her2{v}', beta=m2.params['treatment_trastuzumab_tucatinib'], p=m2.pvalues['treatment_trastuzumab_tucatinib'],
           n=len(sub),
           result=f"Trastuzumab+tucatinib effect in her2_amplified={v} (n={len(sub)}): beta={m2.params['treatment_trastuzumab_tucatinib']:.4f}, p={m2.pvalues['treatment_trastuzumab_tucatinib']:.4g}")

# ---------- Iteration 11: bevacizumab and regorafenib by all biomarkers ----------
for t in ['treatment_bevacizumab','treatment_regorafenib']:
    for mod in ['kras_mutation','braf_v600e','msi_high','her2_amplified','right_sided_primary','stage_iv']:
        m = smf.ols(f'pfs_months ~ {t} * {mod}', data=df).fit()
        ix = f'{t}:{mod}'
        record(f'i11_{t}_x_{mod}', beta=m.params[ix], p=m.pvalues[ix],
               result=f"{t} x {mod} interaction beta={m.params[ix]:.4f}, p={m.pvalues[ix]:.4g}")

# ---------- Iteration 12: treatment x continuous-feature interaction screen ----------
cont = ['age_years','ecog_ps','cea_ng_ml','albumin_g_dl','ldh_u_l','weight_loss_pct_6mo',
        'crp_mg_l','nlr','hemoglobin_g_dl','alkaline_phosphatase_u_l']
screen = {}
for t in treatments:
    screen[t] = {}
    for f in cont:
        m = smf.ols(f'pfs_months ~ {t} * {f}', data=df).fit()
        ix = f'{t}:{f}'
        screen[t][f] = {'beta': float(m.params[ix]), 'p': float(m.pvalues[ix])}
out['i12_tx_x_cont_screen'] = screen

# ---------- Iteration 13: treatment x binary-feature interaction screen (full) ----------
bins = ['sex_female','stage_iv','right_sided_primary','kras_mutation','nras_mutation',
        'braf_v600e','msi_high','her2_amplified','ntrk_fusion']
screen_bin = {}
for t in treatments:
    screen_bin[t] = {}
    for f in bins:
        m = smf.ols(f'pfs_months ~ {t} * {f}', data=df).fit()
        ix = f'{t}:{f}'
        screen_bin[t][f] = {'beta': float(m.params[ix]), 'p': float(m.pvalues[ix])}
out['i13_tx_x_bin_screen'] = screen_bin

# ---------- Iteration 14: refine cetuximab subgroup -- joint definition ----------
# Try cetuximab in left-sided + RAS-wt + BRAF-wt
for left_v in [0,1]:
    for kras_v in [0,1]:
        for nras_v in [0,1]:
            for braf_v in [0,1]:
                sub = df[(df['right_sided_primary']==(0 if left_v else 1))]  # kludgy; rewrite
# (Easier path) compute key cells
def cetux_in(mask, label):
    sub = df[mask]
    if (sub['treatment_cetuximab']==1).sum() < 30 or (sub['treatment_cetuximab']==0).sum() < 30:
        return None
    m = smf.ols('pfs_months ~ treatment_cetuximab', data=sub).fit()
    return {'beta': float(m.params['treatment_cetuximab']), 'p': float(m.pvalues['treatment_cetuximab']), 'n': len(sub),
            'n_treated': int((sub['treatment_cetuximab']==1).sum()), 'label': label}

cetux_subs = {}
combos = [
    ('left_panwt', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0)),
    ('left_panwt_lowecog', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0) & (df['ecog_ps']<=1)),
    ('left_kraswt_only', (df['right_sided_primary']==0) & (df['kras_mutation']==0)),
    ('left_kraswt_nraswt', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0)),
    ('right_panwt', (df['right_sided_primary']==1) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0)),
    ('left_kras_only', (df['right_sided_primary']==0) & (df['kras_mutation']==1)),
    ('left_braf_only', (df['right_sided_primary']==0) & (df['braf_v600e']==1)),
    ('left_panwt_msi_low', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0) & (df['msi_high']==0)),
    ('left_panwt_msi_high', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0) & (df['msi_high']==1)),
    ('left_kraswt_braf_wt', (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['braf_v600e']==0)),
]
for label, mask in combos:
    r = cetux_in(mask, label)
    if r: cetux_subs[label] = r
out['i14_cetux_subs'] = cetux_subs

# ---------- Iteration 15: encorafenib joint subgroup ----------
def trt_in(treatment, mask, label):
    sub = df[mask]
    if (sub[treatment]==1).sum() < 20 or (sub[treatment]==0).sum() < 20:
        return None
    m = smf.ols(f'pfs_months ~ {treatment}', data=sub).fit()
    return {'beta': float(m.params[treatment]), 'p': float(m.pvalues[treatment]), 'n': len(sub),
            'n_treated': int((sub[treatment]==1).sum()), 'label': label}

enco_subs = {}
combos = [
    ('braf_only', df['braf_v600e']==1),
    ('braf_kraswt', (df['braf_v600e']==1) & (df['kras_mutation']==0)),
    ('braf_left', (df['braf_v600e']==1) & (df['right_sided_primary']==0)),
    ('braf_right', (df['braf_v600e']==1) & (df['right_sided_primary']==1)),
    ('braf_msilow', (df['braf_v600e']==1) & (df['msi_high']==0)),
    ('braf_msihi', (df['braf_v600e']==1) & (df['msi_high']==1)),
    ('braf_lowecog', (df['braf_v600e']==1) & (df['ecog_ps']<=1)),
    ('nonbraf', df['braf_v600e']==0),
]
for label, mask in combos:
    r = trt_in('treatment_encorafenib', mask, label)
    if r: enco_subs[label] = r
out['i15_enco_subs'] = enco_subs

# ---------- Iteration 16: pembro subgroup refinement ----------
pembro_subs = {}
combos = [
    ('msi_only', df['msi_high']==1),
    ('msi_kraswt', (df['msi_high']==1) & (df['kras_mutation']==0)),
    ('msi_brafwt', (df['msi_high']==1) & (df['braf_v600e']==0)),
    ('msi_low', df['msi_high']==0),
    ('msi_lowecog', (df['msi_high']==1) & (df['ecog_ps']<=1)),
    ('msi_left', (df['msi_high']==1) & (df['right_sided_primary']==0)),
    ('msi_right', (df['msi_high']==1) & (df['right_sided_primary']==1)),
]
for label, mask in combos:
    r = trt_in('treatment_pembrolizumab', mask, label)
    if r: pembro_subs[label] = r
out['i16_pembro_subs'] = pembro_subs

# ---------- Iteration 17: trastuzumab+tucatinib subgroup ----------
ttuc_subs = {}
combos = [
    ('her2_only', df['her2_amplified']==1),
    ('her2_kraswt', (df['her2_amplified']==1) & (df['kras_mutation']==0)),
    ('her2_brafwt', (df['her2_amplified']==1) & (df['braf_v600e']==0)),
    ('her2_left', (df['her2_amplified']==1) & (df['right_sided_primary']==0)),
    ('her2_right', (df['her2_amplified']==1) & (df['right_sided_primary']==1)),
    ('her2_lowecog', (df['her2_amplified']==1) & (df['ecog_ps']<=1)),
]
for label, mask in combos:
    r = trt_in('treatment_trastuzumab_tucatinib', mask, label)
    if r: ttuc_subs[label] = r
out['i17_ttuc_subs'] = ttuc_subs

# ---------- Iteration 18: bevacizumab subgroup ----------
bev_subs = {}
combos = [
    ('all', df.index==df.index),
    ('left', df['right_sided_primary']==0),
    ('right', df['right_sided_primary']==1),
    ('kraswt', df['kras_mutation']==0),
    ('krasmt', df['kras_mutation']==1),
    ('lowecog', df['ecog_ps']<=1),
    ('hiecog', df['ecog_ps']==2),
    ('stage4', df['stage_iv']==1),
    ('stagelt4', df['stage_iv']==0),
]
for label, mask in combos:
    r = trt_in('treatment_bevacizumab', mask, label)
    if r: bev_subs[label] = r
out['i18_bev_subs'] = bev_subs

# ---------- Iteration 19: regorafenib subgroup, particularly later-line / good prognostic ----------
rego_subs = {}
combos = [
    ('all', df.index==df.index),
    ('lowcea', df['cea_ng_ml'] < df['cea_ng_ml'].median()),
    ('hicea', df['cea_ng_ml'] >= df['cea_ng_ml'].median()),
    ('hialb', df['albumin_g_dl'] >= df['albumin_g_dl'].median()),
    ('loalb', df['albumin_g_dl'] < df['albumin_g_dl'].median()),
    ('lowecog', df['ecog_ps']<=1),
    ('hiecog', df['ecog_ps']==2),
    ('lowldh', df['ldh_u_l'] < df['ldh_u_l'].median()),
    ('hildh', df['ldh_u_l'] >= df['ldh_u_l'].median()),
    ('lo_wtloss', df['weight_loss_pct_6mo'] < df['weight_loss_pct_6mo'].median()),
    ('hi_wtloss', df['weight_loss_pct_6mo'] >= df['weight_loss_pct_6mo'].median()),
]
for label, mask in combos:
    r = trt_in('treatment_regorafenib', mask, label)
    if r: rego_subs[label] = r
out['i19_rego_subs'] = rego_subs

# ---------- Iteration 20: regorafenib triple-modifier (good prognostic) subgroup ----------
rego_combo_subs = {}
combos = [
    ('low_ecog_low_ldh', (df['ecog_ps']<=1) & (df['ldh_u_l']<df['ldh_u_l'].median())),
    ('low_ecog_hi_alb', (df['ecog_ps']<=1) & (df['albumin_g_dl']>=df['albumin_g_dl'].median())),
    ('low_ecog_hi_alb_low_ldh',
        (df['ecog_ps']<=1) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['ldh_u_l']<df['ldh_u_l'].median())),
    ('low_ecog_hi_alb_low_ldh_low_cea',
        (df['ecog_ps']<=1) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['ldh_u_l']<df['ldh_u_l'].median()) & (df['cea_ng_ml']<df['cea_ng_ml'].median())),
    ('hi_ecog_or_lo_alb_or_hi_ldh',
        (df['ecog_ps']==2) | (df['albumin_g_dl']<df['albumin_g_dl'].median()) | (df['ldh_u_l']>=df['ldh_u_l'].median())),
]
for label, mask in combos:
    r = trt_in('treatment_regorafenib', mask, label)
    if r: rego_combo_subs[label] = r
out['i20_rego_combo'] = rego_combo_subs

# ---------- Iteration 21: multivariable adjusted treatment effects ----------
covars = "age_years + C(ecog_ps) + stage_iv + right_sided_primary + cea_ng_ml + albumin_g_dl + ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + hemoglobin_g_dl"
adj_results = {}
for t in treatments:
    f = f"pfs_months ~ {t} + " + covars
    m = smf.ols(f, data=df).fit()
    adj_results[t] = {'beta': float(m.params[t]), 'p': float(m.pvalues[t])}
out['i21_adj_main_effects'] = adj_results

# ---------- Iteration 22: tree-like exploration via subgroup t-test screen ----------
# For each treatment, find feature whose value yields biggest treatment effect
import itertools
hetero_screen = {}
binary_modifiers = ['sex_female','stage_iv','right_sided_primary','kras_mutation','nras_mutation',
                    'braf_v600e','msi_high','her2_amplified','ntrk_fusion']
median_modifiers = {'ecog_ps_lo': df['ecog_ps']<=1,
                    'high_cea': df['cea_ng_ml']>=df['cea_ng_ml'].median(),
                    'low_albumin': df['albumin_g_dl']<df['albumin_g_dl'].median(),
                    'high_ldh': df['ldh_u_l']>=df['ldh_u_l'].median(),
                    'high_crp': df['crp_mg_l']>=df['crp_mg_l'].median(),
                    'high_nlr': df['nlr']>=df['nlr'].median(),
                    'high_wtloss': df['weight_loss_pct_6mo']>=df['weight_loss_pct_6mo'].median()}
for t in treatments:
    rs = {}
    # marginal
    rs['_overall'] = trt_in(t, df.index==df.index, 'overall')
    for b in binary_modifiers:
        for v in [0,1]:
            r = trt_in(t, df[b]==v, f'{b}={v}')
            if r: rs[f'{b}={v}'] = r
    for k, mask in median_modifiers.items():
        r = trt_in(t, mask, k)
        if r: rs[k] = r
        r = trt_in(t, ~mask, f'not_{k}')
        if r: rs[f'not_{k}'] = r
    hetero_screen[t] = rs
out['i22_hetero_screen'] = hetero_screen

# ---------- Iteration 23: cetuximab final subgroup -- explicit pan-RAS/BRAF-wt left-sided, low ECOG ----------
def trt_in_with_means(treatment, mask, label):
    sub = df[mask]
    if (sub[treatment]==1).sum() < 20 or (sub[treatment]==0).sum() < 20:
        return None
    m = smf.ols(f'pfs_months ~ {treatment}', data=sub).fit()
    means = sub.groupby(treatment)['pfs_months'].mean().to_dict()
    return {'beta': float(m.params[treatment]), 'p': float(m.pvalues[treatment]), 'n': len(sub),
            'n_treated': int((sub[treatment]==1).sum()), 'means': {str(k): float(v) for k,v in means.items()}, 'label': label}

# overall and progressively narrowed
final_cetux = {}
masks = {
    'all': df.index==df.index,
    'left': df['right_sided_primary']==0,
    'left_kraswt': (df['right_sided_primary']==0) & (df['kras_mutation']==0),
    'left_kraswt_nraswt': (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0),
    'left_panwt': (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0),
    'left_panwt_msilow': (df['right_sided_primary']==0) & (df['kras_mutation']==0) & (df['nras_mutation']==0) & (df['braf_v600e']==0) & (df['msi_high']==0),
}
for k, m in masks.items():
    r = trt_in_with_means('treatment_cetuximab', m, k)
    if r: final_cetux[k] = r
out['i23_final_cetux'] = final_cetux

# ---------- Iteration 24: encorafenib + pembrolizumab + trastuzumab final ----------
final_targeted = {}
final_targeted['enco_braf_only'] = trt_in_with_means('treatment_encorafenib', df['braf_v600e']==1, 'braf_v600e=1')
final_targeted['enco_braf_kraswt'] = trt_in_with_means('treatment_encorafenib', (df['braf_v600e']==1) & (df['kras_mutation']==0), 'braf_v600e=1, kras_wt')
final_targeted['enco_nonbraf'] = trt_in_with_means('treatment_encorafenib', df['braf_v600e']==0, 'braf_v600e=0')
final_targeted['pembro_msi_only'] = trt_in_with_means('treatment_pembrolizumab', df['msi_high']==1, 'msi_high=1')
final_targeted['pembro_nonmsi'] = trt_in_with_means('treatment_pembrolizumab', df['msi_high']==0, 'msi_high=0')
final_targeted['ttuc_her2'] = trt_in_with_means('treatment_trastuzumab_tucatinib', df['her2_amplified']==1, 'her2_amplified=1')
final_targeted['ttuc_nonher2'] = trt_in_with_means('treatment_trastuzumab_tucatinib', df['her2_amplified']==0, 'her2_amplified=0')
out['i24_final_targeted'] = final_targeted

# ---------- Iteration 25: bevacizumab + regorafenib best subgroups ----------
final_other = {}
# bev: try ECOG+stage IV combos
final_other['bev_overall'] = trt_in_with_means('treatment_bevacizumab', df.index==df.index, 'all')
final_other['bev_low_ecog'] = trt_in_with_means('treatment_bevacizumab', df['ecog_ps']<=1, 'ecog<=1')
final_other['bev_hi_ecog'] = trt_in_with_means('treatment_bevacizumab', df['ecog_ps']==2, 'ecog=2')
final_other['bev_lowcrp_loalb'] = trt_in_with_means('treatment_bevacizumab',
    (df['crp_mg_l']<df['crp_mg_l'].median()) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()), 'low_crp_high_alb')

final_other['rego_overall'] = trt_in_with_means('treatment_regorafenib', df.index==df.index, 'all')
final_other['rego_lowcea'] = trt_in_with_means('treatment_regorafenib', df['cea_ng_ml']<df['cea_ng_ml'].median(), 'low_cea')
final_other['rego_hialb'] = trt_in_with_means('treatment_regorafenib', df['albumin_g_dl']>=df['albumin_g_dl'].median(), 'high_alb')
final_other['rego_low_ecog'] = trt_in_with_means('treatment_regorafenib', df['ecog_ps']<=1, 'ecog<=1')
final_other['rego_lowcea_hialb'] = trt_in_with_means('treatment_regorafenib',
    (df['cea_ng_ml']<df['cea_ng_ml'].median()) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()), 'low_cea_high_alb')
final_other['rego_lowcea_hialb_lowldh'] = trt_in_with_means('treatment_regorafenib',
    (df['cea_ng_ml']<df['cea_ng_ml'].median()) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['ldh_u_l']<df['ldh_u_l'].median()), 'low_cea_high_alb_low_ldh')
final_other['rego_lowcea_hialb_lowecog'] = trt_in_with_means('treatment_regorafenib',
    (df['cea_ng_ml']<df['cea_ng_ml'].median()) & (df['albumin_g_dl']>=df['albumin_g_dl'].median()) & (df['ecog_ps']<=1), 'low_cea_high_alb_low_ecog')
out['i25_final_other'] = final_other

with open('clean_results.json', 'w') as fh:
    json.dump(out, fh, indent=2, default=str)
print('Done. Wrote clean_results.json')
print('Keys:', list(out.keys()))
