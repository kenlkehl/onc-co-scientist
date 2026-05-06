"""Iterations 11-18: deep VEN-AZA subgroup discovery, joint suppressor models, more screens."""
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
    if len(sub)==0:
        return None
    p1 = sub.loc[sub[tx]==1, outcome].mean()
    p0 = sub.loc[sub[tx]==0, outcome].mean()
    n1 = int((sub[tx]==1).sum()); n0 = int((sub[tx]==0).sum())
    if n1==0 or n0==0:
        return float(p1-p0) if not pd.isna(p1-p0) else None, None, p1, p0, n1, n0
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
# ITERATION 11: VEN-AZA combined modifier/suppressor subgroup
# ============================================================
print('\n=== Iteration 11: VEN-AZA joint subgroup ===')
# joint: unfit AND npm1 mutated, with TP53 wild-type and complex_karyotype = 0
df['venaza_supergroup'] = ((df['unfit_for_intensive']==1) & (df['npm1_mutation']==1) &
                          (df['tp53_mutation']==0) & (df['complex_karyotype']==0)).astype(int)

# Within supergroup
sub = df[df['venaza_supergroup']==1]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
if r:
    d2,p2,a,b,n1,n0 = r
    store('i11_venaza_supergroup', d2, p2,
          f'VEN-AZA in unfit=1 & npm1=1 & tp53=0 & complex_karyotype=0: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2}')
sub = df[df['venaza_supergroup']==0]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
if r:
    d2,p2,a,b,n1,n0 = r
    store('i11_venaza_outside_supergroup', d2, p2,
          f'VEN-AZA outside supergroup: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2}')

# Also test progressively
# A) unfit=1 only
sub = df[df['unfit_for_intensive']==1]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
d2,p2,a,b,n1,n0=r
store('i11_venaza_unfit', d2, p2, f'VEN-AZA in unfit=1: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# B) unfit=1 & npm1=1
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
d2,p2,a,b,n1,n0=r
store('i11_venaza_unfit_npm1', d2, p2, f'VEN-AZA in unfit=1 & npm1=1: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# C) unfit=1 & npm1=0
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==0)]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
d2,p2,a,b,n1,n0=r
store('i11_venaza_unfit_npm1neg', d2, p2, f'VEN-AZA in unfit=1 & npm1=0: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# D) unfit=0 & npm1=1
sub = df[(df['unfit_for_intensive']==0)&(df['npm1_mutation']==1)]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
d2,p2,a,b,n1,n0=r
store('i11_venaza_fit_npm1', d2, p2, f'VEN-AZA in unfit=0 & npm1=1: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# E) supergroup with TP53 mutation
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&(df['tp53_mutation']==1)]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
if r:
    d2,p2,a,b,n1,n0=r
    store('i11_venaza_unfit_npm1_tp53', d2, p2, f'VEN-AZA in unfit=1 & npm1=1 & tp53=1: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# F) supergroup with complex karyotype
sub = df[(df['unfit_for_intensive']==1)&(df['npm1_mutation']==1)&(df['complex_karyotype']==1)]
r = two_by_two(sub, 'treatment_venetoclax_azacitidine')
if r:
    d2,p2,a,b,n1,n0=r
    store('i11_venaza_unfit_npm1_ck', d2, p2, f'VEN-AZA in unfit=1 & npm1=1 & complex_karyotype=1: on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 12: Triple-interaction logistic model for VEN-AZA
# ============================================================
print('\n=== Iteration 12: triple interaction logistic model ===')
# venaza * unfit * npm1
f = ('objective_response ~ treatment_venetoclax_azacitidine*unfit_for_intensive*npm1_mutation + '
     + ' + '.join([c for c in covs if c not in ('unfit_for_intensive','npm1_mutation')]))
m = smf.logit(f, data=df).fit(disp=0, maxiter=300)
print(m.summary().tables[1])
for nm in m.params.index:
    if 'venetoclax' in nm and (':' in nm or 'venetoclax_azacitidine' == nm):
        store(f'i12_triple_{nm.replace(":","_x_")}', m.params[nm], m.pvalues[nm],
              f'triple-interaction logit beta({nm})={m.params[nm]:.4f}, p={m.pvalues[nm]:.3g}')

# ============================================================
# ITERATION 13: Other treatments x age tertile / unfit / fitness
# ============================================================
print('\n=== Iteration 13: other tx x fitness ===')
df['age_tert'] = pd.qcut(df['age_years'], 3, labels=[0,1,2]).astype(int)
for tx in ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib','treatment_enasidenib','treatment_7plus3']:
    for v in [0,1]:
        sub = df[df['unfit_for_intensive']==v]
        r = two_by_two(sub, tx)
        if r:
            d2,p2,a,b,n1,n0=r
            store(f'i13_{tx}_strat_unfit{v}', d2, p2,
                  f'unfit_for_intensive={v}: {tx} on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# 7+3 x age tertile
for v in [0,1,2]:
    sub = df[df['age_tert']==v]
    r = two_by_two(sub, 'treatment_7plus3')
    if r:
        d2,p2,a,b,n1,n0=r
        store(f'i13_7plus3_strat_age_tert{v}', d2, p2,
              f'age_tert={v}: 7+3 on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, p={p2}')

# ============================================================
# ITERATION 14: Systematic Tx x feature interaction screen for ALL treatments
# ============================================================
print('\n=== Iteration 14: full Tx x binary feature screen ===')
binary_feats = ['sex_female','secondary_aml','unfit_for_intensive','complex_karyotype',
                'flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
for tx in ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib','treatment_enasidenib',
           'treatment_venetoclax_azacitidine','treatment_7plus3']:
    for feat in binary_feats:
        if tx==feat: continue
        other_covs = [c for c in covs if c not in (tx, feat)]
        f = f'objective_response ~ {tx}*{feat} + ' + ' + '.join(other_covs)
        try:
            m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
            inter = [k for k in m.params.index if (':' in k and tx in k and feat in k)]
            if inter:
                ik = inter[0]
                store(f'i14_inter_{tx}_x_{feat}', m.params[ik], m.pvalues[ik],
                      f'logit interaction beta({ik})={m.params[ik]:.4f}, p={m.pvalues[ik]:.3g}')
        except Exception as e:
            pass

# ============================================================
# ITERATION 15: continuous features × VEN-AZA
# ============================================================
print('\n=== Iteration 15: VEN-AZA × continuous features ===')
for feat in ['blast_pct_marrow','wbc_k_per_ul','albumin_g_dl','ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl']:
    other_covs = [c for c in covs if c != feat]
    f = f'objective_response ~ treatment_venetoclax_azacitidine*{feat} + ' + ' + '.join(other_covs)
    try:
        m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
        inter = [k for k in m.params.index if (':' in k and 'venetoclax_azacitidine' in k and feat in k)]
        if inter:
            ik = inter[0]
            store(f'i15_venaza_x_{feat}', m.params[ik], m.pvalues[ik],
                  f'logit interaction beta({ik})={m.params[ik]:.4f}, p={m.pvalues[ik]:.3g}')
    except Exception as e:
        pass

with open('results_my.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved with', len(results), 'entries')
