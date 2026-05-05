"""Iterations 16-25: refine and confirm VEN-AZA supergroup, suppressors, additional probes."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
with open('results_my.json') as f:
    results = json.load(f)

def store(key, eff, p, summary):
    results[key] = {
        'effect': None if eff is None else float(eff),
        'p': None if p is None else float(p),
        'summary': summary,
    }

def two_by_two(sub, tx, outcome='objective_response'):
    if len(sub)==0: return None
    p1 = sub.loc[sub[tx]==1, outcome].mean()
    p0 = sub.loc[sub[tx]==0, outcome].mean()
    n1 = int((sub[tx]==1).sum()); n0 = int((sub[tx]==0).sum())
    if n1==0 or n0==0: return None
    ct = pd.crosstab(sub[tx], sub[outcome])
    if ct.shape==(2,2):
        chi2, p, _, _ = stats.chi2_contingency(ct)
    else:
        p=None
    return float(p1-p0), float(p) if p is not None else None, float(p1), float(p0), n1, n0

covs = ['age_years','sex_female','ecog_ps','secondary_aml','unfit_for_intensive',
        'complex_karyotype','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
        'npm1_mutation','tp53_mutation','wbc_k_per_ul','blast_pct_marrow','albumin_g_dl',
        'ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl']

# ============================================================
# ITERATION 16: Drop each predicate of supergroup to see who matters
# ============================================================
print('\n=== Iteration 16: predicate ablation ===')
configs = {
    'unfit_npm1_tp53_ck':           {'unfit_for_intensive':1,'npm1_mutation':1,'tp53_mutation':0,'complex_karyotype':0},
    'npm1_tp53_ck':                 {'npm1_mutation':1,'tp53_mutation':0,'complex_karyotype':0},
    'unfit_tp53_ck':                {'unfit_for_intensive':1,'tp53_mutation':0,'complex_karyotype':0},
    'unfit_npm1_ck':                {'unfit_for_intensive':1,'npm1_mutation':1,'complex_karyotype':0},
    'unfit_npm1_tp53':              {'unfit_for_intensive':1,'npm1_mutation':1,'tp53_mutation':0},
    'unfit_npm1':                   {'unfit_for_intensive':1,'npm1_mutation':1},
    'unfit_npm1_tp53wt':            {'unfit_for_intensive':1,'npm1_mutation':1,'tp53_mutation':0},
    'unfit_npm1_ckabs':             {'unfit_for_intensive':1,'npm1_mutation':1,'complex_karyotype':0},
}
for name, conds in configs.items():
    mask = pd.Series([True]*len(df))
    for k,v in conds.items():
        mask &= (df[k]==v)
    sub = df[mask]
    r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i16_{name}', d2, p2,
              f'VEN-AZA in {conds}: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# Also: violate one predicate at a time
violation_configs = {
    'violate_unfit':       {'unfit_for_intensive':0,'npm1_mutation':1,'tp53_mutation':0,'complex_karyotype':0},
    'violate_npm1':        {'unfit_for_intensive':1,'npm1_mutation':0,'tp53_mutation':0,'complex_karyotype':0},
    'violate_tp53':        {'unfit_for_intensive':1,'npm1_mutation':1,'tp53_mutation':1,'complex_karyotype':0},
    'violate_ck':          {'unfit_for_intensive':1,'npm1_mutation':1,'tp53_mutation':0,'complex_karyotype':1},
}
for name, conds in violation_configs.items():
    mask = pd.Series([True]*len(df))
    for k,v in conds.items():
        mask &= (df[k]==v)
    sub = df[mask]
    r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i16_{name}', d2, p2,
              f'VEN-AZA when one predicate violated ({conds}): on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 17: 7+3 × complex_karyotype confirmation
# ============================================================
print('\n=== Iteration 17: 7+3 × complex_karyotype ===')
for v in [0,1]:
    sub = df[df['complex_karyotype']==v]
    r = two_by_two(sub, 'treatment_7plus3')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i17_7plus3_strat_ck_{v}', d2, p2,
              f'complex_karyotype={v}: 7+3 on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# 7+3 in tp53=0 & complex_karyotype=0 (favorable cytogenetics)
sub = df[(df['tp53_mutation']==0)&(df['complex_karyotype']==0)]
r = two_by_two(sub, 'treatment_7plus3')
d2,p2,a,b,n1,n0=r
store('i17_7plus3_tp53wt_ckabs', d2, p2,
      f'7+3 in tp53=0 & complex_karyotype=0: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 18: targeted Tx revisited - is benefit even for "right" patients null?
# ============================================================
print('\n=== Iteration 18: targeted Tx in marker-positive subset, multivariate ===')
# midostaurin in flt3_itd=1
sub = df[df['flt3_itd']==1].copy()
m = smf.logit('objective_response ~ treatment_midostaurin + age_years + ecog_ps + npm1_mutation + tp53_mutation + complex_karyotype', data=sub).fit(disp=0, maxiter=200)
store('i18_mido_in_flt3itd_adj', m.params['treatment_midostaurin'], m.pvalues['treatment_midostaurin'],
      f'within flt3_itd=1: adjusted logit beta(treatment_midostaurin)={m.params["treatment_midostaurin"]:.4f}, p={m.pvalues["treatment_midostaurin"]:.3g}')

# gilteritinib in flt3_itd=1
m = smf.logit('objective_response ~ treatment_gilteritinib + age_years + ecog_ps + npm1_mutation + tp53_mutation + complex_karyotype', data=sub).fit(disp=0, maxiter=200)
store('i18_gilt_in_flt3itd_adj', m.params['treatment_gilteritinib'], m.pvalues['treatment_gilteritinib'],
      f'within flt3_itd=1: adjusted logit beta(treatment_gilteritinib)={m.params["treatment_gilteritinib"]:.4f}, p={m.pvalues["treatment_gilteritinib"]:.3g}')

# ivosidenib in idh1=1
sub = df[df['idh1_mutation']==1].copy()
m = smf.logit('objective_response ~ treatment_ivosidenib + age_years + ecog_ps + npm1_mutation + tp53_mutation + complex_karyotype + unfit_for_intensive', data=sub).fit(disp=0, maxiter=200)
store('i18_ivo_in_idh1_adj', m.params['treatment_ivosidenib'], m.pvalues['treatment_ivosidenib'],
      f'within idh1=1: adjusted logit beta(treatment_ivosidenib)={m.params["treatment_ivosidenib"]:.4f}, p={m.pvalues["treatment_ivosidenib"]:.3g}')

# enasidenib in idh2=1
sub = df[df['idh2_mutation']==1].copy()
m = smf.logit('objective_response ~ treatment_enasidenib + age_years + ecog_ps + npm1_mutation + tp53_mutation + complex_karyotype + unfit_for_intensive', data=sub).fit(disp=0, maxiter=200)
store('i18_ena_in_idh2_adj', m.params['treatment_enasidenib'], m.pvalues['treatment_enasidenib'],
      f'within idh2=1: adjusted logit beta(treatment_enasidenib)={m.params["treatment_enasidenib"]:.4f}, p={m.pvalues["treatment_enasidenib"]:.3g}')

# ============================================================
# ITERATION 19: VEN-AZA x continuous lab values within npm1+/unfit subgroup
# ============================================================
print('\n=== Iteration 19: continuous modifiers within unfit/npm1=1 ===')
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)].copy()
# Does the effect attenuate by age, blast %, etc.?
m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine*age_years + tp53_mutation + complex_karyotype + ecog_ps + albumin_g_dl', data=sub).fit(disp=0, maxiter=200)
ik = [k for k in m.params.index if 'venetoclax' in k and ':' in k]
if ik:
    k=ik[0]
    store('i19_venaza_unfit_npm1_x_age', m.params[k], m.pvalues[k],
          f'within unfit=1&npm1=1: beta(treatment_venetoclax_azacitidine:age_years)={m.params[k]:.5f}, p={m.pvalues[k]:.3g}')

m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine*tp53_mutation + complex_karyotype + ecog_ps + albumin_g_dl', data=sub).fit(disp=0, maxiter=200)
ik = [k for k in m.params.index if 'venetoclax' in k and ':' in k]
if ik:
    k=ik[0]
    store('i19_venaza_unfit_npm1_x_tp53', m.params[k], m.pvalues[k],
          f'within unfit=1&npm1=1: beta(treatment_venetoclax_azacitidine:tp53_mutation)={m.params[k]:.4f}, p={m.pvalues[k]:.3g}')

m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine*complex_karyotype + tp53_mutation + ecog_ps + albumin_g_dl', data=sub).fit(disp=0, maxiter=200)
ik = [k for k in m.params.index if 'venetoclax' in k and ':' in k]
if ik:
    k=ik[0]
    store('i19_venaza_unfit_npm1_x_ck', m.params[k], m.pvalues[k],
          f'within unfit=1&npm1=1: beta(treatment_venetoclax_azacitidine:complex_karyotype)={m.params[k]:.4f}, p={m.pvalues[k]:.3g}')

# ============================================================
# ITERATION 20: ivosidenib × NPM1 marginal interaction (p=0.029) - explore
# ============================================================
print('\n=== Iteration 20: ivosidenib × npm1 interaction ===')
for v in [0,1]:
    sub = df[df['npm1_mutation']==v]
    r = two_by_two(sub, 'treatment_ivosidenib')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i20_ivo_strat_npm1{v}', d2, p2,
              f'npm1={v}: ivosidenib on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# enasidenib × sex_female - explore
for v in [0,1]:
    sub = df[df['sex_female']==v]
    r = two_by_two(sub, 'treatment_enasidenib')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i20_ena_strat_sex{v}', d2, p2,
              f'sex_female={v}: enasidenib on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 21: Comprehensive test of "VEN-AZA supergroup" vs no-VEN-AZA in supergroup using logistic with controls
# ============================================================
print('\n=== Iteration 21: supergroup confirmation adjusted ===')
df['supergroup'] = ((df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&
                    (df['tp53_mutation']==0)&(df['complex_karyotype']==0)).astype(int)
m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine*supergroup + ' + ' + '.join([c for c in covs if c not in ('unfit_for_intensive','npm1_mutation','tp53_mutation','complex_karyotype')]),
              data=df).fit(disp=0, maxiter=300)
print(m.summary().tables[1])
ik = [k for k in m.params.index if 'venetoclax' in k and 'supergroup' in k and ':' in k]
if ik:
    k=ik[0]
    store('i21_venaza_x_supergroup_adj', m.params[k], m.pvalues[k],
          f'logit beta({k})={m.params[k]:.4f}, p={m.pvalues[k]:.3g}; venaza main beta={m.params["treatment_venetoclax_azacitidine"]:.4f} (p={m.pvalues["treatment_venetoclax_azacitidine"]:.3g})')

# ============================================================
# ITERATION 22: Sex stratified VEN-AZA + final secondary AML check
# ============================================================
print('\n=== Iteration 22: misc checks ===')
# VEN-AZA × secondary AML
for v in [0,1]:
    sub = df[df['secondary_aml']==v]
    r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i22_venaza_secondary_aml_{v}', d2, p2,
              f'secondary_aml={v}: VEN-AZA on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 23: 7+3 outside complex karyotype: any benefit?
# ============================================================
print('\n=== Iteration 23: 7+3 outside complex_karyotype ===')
sub = df[df['complex_karyotype']==0]
m = smf.logit('objective_response ~ treatment_7plus3 + age_years + ecog_ps + unfit_for_intensive + npm1_mutation + tp53_mutation + secondary_aml + albumin_g_dl + blast_pct_marrow', data=sub).fit(disp=0, maxiter=200)
store('i23_7plus3_ckwt_adj', m.params['treatment_7plus3'], m.pvalues['treatment_7plus3'],
      f'within complex_karyotype=0: adjusted logit beta(treatment_7plus3)={m.params["treatment_7plus3"]:.4f}, p={m.pvalues["treatment_7plus3"]:.3g}')

# ============================================================
# ITERATION 24: Final supergroup OR estimate (logistic single-model in supergroup-only patients)
# ============================================================
print('\n=== Iteration 24: within supergroup adjusted OR ===')
sub = df[df['supergroup']==1].copy()
m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine + age_years + ecog_ps + sex_female + secondary_aml + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + albumin_g_dl + blast_pct_marrow + crp_mg_l', data=sub).fit(disp=0, maxiter=300)
print(m.summary().tables[1])
store('i24_venaza_within_supergroup_adj', m.params['treatment_venetoclax_azacitidine'], m.pvalues['treatment_venetoclax_azacitidine'],
      f'within supergroup (unfit=1&npm1=1&tp53=0&ck=0): adjusted logit beta(treatment_venetoclax_azacitidine)={m.params["treatment_venetoclax_azacitidine"]:.4f}, OR={np.exp(m.params["treatment_venetoclax_azacitidine"]):.3f}, p={m.pvalues["treatment_venetoclax_azacitidine"]:.3g}, n={len(sub)}')

# ============================================================
# ITERATION 25: Outside supergroup: does VEN-AZA still help? (final negative control)
# ============================================================
print('\n=== Iteration 25: VEN-AZA outside supergroup adjusted ===')
sub = df[df['supergroup']==0].copy()
m = smf.logit('objective_response ~ treatment_venetoclax_azacitidine + age_years + ecog_ps + sex_female + secondary_aml + unfit_for_intensive + flt3_itd + flt3_tkd + idh1_mutation + idh2_mutation + npm1_mutation + tp53_mutation + complex_karyotype + albumin_g_dl + blast_pct_marrow + crp_mg_l', data=sub).fit(disp=0, maxiter=300)
store('i25_venaza_outside_supergroup_adj', m.params['treatment_venetoclax_azacitidine'], m.pvalues['treatment_venetoclax_azacitidine'],
      f'outside supergroup: adjusted logit beta(treatment_venetoclax_azacitidine)={m.params["treatment_venetoclax_azacitidine"]:.4f}, OR={np.exp(m.params["treatment_venetoclax_azacitidine"]):.3f}, p={m.pvalues["treatment_venetoclax_azacitidine"]:.3g}, n={len(sub)}')

with open('results_my.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved with', len(results), 'entries')
