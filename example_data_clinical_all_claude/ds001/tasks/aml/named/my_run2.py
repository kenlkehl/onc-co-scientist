"""Iterations 7-15. Read prior results, append more."""
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

def stratified(df, tx, modifier, mval, outcome='objective_response'):
    sub = df[df[modifier] == mval]
    p1 = sub.loc[sub[tx]==1, outcome].mean()
    p0 = sub.loc[sub[tx]==0, outcome].mean()
    n1 = int((sub[tx]==1).sum()); n0 = int((sub[tx]==0).sum())
    if n1==0 or n0==0:
        return None
    ct = pd.crosstab(sub[tx], sub[outcome])
    if ct.shape == (2,2):
        chi2, p, _, _ = stats.chi2_contingency(ct)
    else:
        p = None
    return float(p1-p0), p, float(p1), float(p0), n1, n0

covs = ['age_years','sex_female','ecog_ps','secondary_aml','unfit_for_intensive',
        'complex_karyotype','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
        'npm1_mutation','tp53_mutation','wbc_k_per_ul','blast_pct_marrow','albumin_g_dl',
        'ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl']
treatments = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib','treatment_enasidenib',
              'treatment_venetoclax_azacitidine','treatment_7plus3']

def fit_inter(df, tx, marker, all_covs):
    other_covs = [c for c in all_covs if c not in (tx, marker)]
    f = f'objective_response ~ {tx}*{marker} + ' + ' + '.join(other_covs)
    m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
    candidates = [k for k in m.params.index if (':' in k and tx in k and marker in k)]
    inter = candidates[0] if candidates else None
    return m, inter

# ============================================================
# ITERATION 7: VEN-AZA interactions with patient features
# ============================================================
print('\n=== Iteration 7: VEN-AZA modifier search ===')
modifiers_bin = ['unfit_for_intensive','secondary_aml','sex_female','complex_karyotype',
                 'flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']
for mod in modifiers_bin:
    m, inter = fit_inter(df, 'treatment_venetoclax_azacitidine', mod, covs)
    store(f'i7_venaza_x_{mod}', m.params[inter], m.pvalues[inter],
          f'logit beta(treatment_venetoclax_azacitidine:{mod})={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; venaza main={m.params["treatment_venetoclax_azacitidine"]:.4f} (p={m.pvalues["treatment_venetoclax_azacitidine"]:.3g})')
    for v in [0,1]:
        r = stratified(df, 'treatment_venetoclax_azacitidine', mod, v)
        if r:
            d2,p2,a,b,n1,n0 = r
            store(f'i7_venaza_strat_{mod}{v}', d2, p2,
                  f'{mod}={v}: venaza on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# Continuous mods: age, ecog (treat as continuous interaction); also age tertiles
for mod in ['age_years','ecog_ps']:
    m, inter = fit_inter(df, 'treatment_venetoclax_azacitidine', mod, covs)
    store(f'i7_venaza_x_{mod}', m.params[inter], m.pvalues[inter],
          f'logit beta(treatment_venetoclax_azacitidine:{mod})={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}')

# Age tertiles
df['age_tert'] = pd.qcut(df['age_years'], 3, labels=[0,1,2]).astype(int)
for v in [0,1,2]:
    r = stratified(df, 'treatment_venetoclax_azacitidine', 'age_tert', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i7_venaza_strat_age_tert{v}', d2, p2,
              f'age_tert={v}: venaza on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# ============================================================
# ITERATION 8: 7+3 interactions
# ============================================================
print('\n=== Iteration 8: 7+3 modifier search ===')
for mod in modifiers_bin + ['age_years','ecog_ps']:
    m, inter = fit_inter(df, 'treatment_7plus3', mod, covs)
    store(f'i8_7p3_x_{mod}', m.params[inter], m.pvalues[inter],
          f'logit beta(treatment_7plus3:{mod})={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; 7+3 main={m.params["treatment_7plus3"]:.4f}')
    if mod in modifiers_bin:
        for v in [0,1]:
            r = stratified(df, 'treatment_7plus3', mod, v)
            if r:
                d2,p2,a,b,n1,n0 = r
                store(f'i8_7p3_strat_{mod}{v}', d2, p2,
                      f'{mod}={v}: 7+3 on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# ============================================================
# ITERATION 9: midostaurin, gilteritinib joint with FLT3 and other features
# ============================================================
print('\n=== Iteration 9: targeted Tx subgroup search ===')
# Try midostaurin in flt3_itd=1 patients further stratified by NPM1, TP53
for npm in [0,1]:
    sub = df[(df['flt3_itd']==1) & (df['npm1_mutation']==npm)]
    if len(sub) > 100:
        r = stratified(sub, 'treatment_midostaurin', 'flt3_itd', 1)  # all flt3_itd=1
        # actually: simpler - just 2x2
        p1 = sub.loc[sub['treatment_midostaurin']==1,'objective_response'].mean()
        p0 = sub.loc[sub['treatment_midostaurin']==0,'objective_response'].mean()
        n1 = int((sub['treatment_midostaurin']==1).sum()); n0 = int((sub['treatment_midostaurin']==0).sum())
        ct = pd.crosstab(sub['treatment_midostaurin'], sub['objective_response'])
        if ct.shape == (2,2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
        else:
            p = None
        store(f'i9_mido_flt3itd_npm1_{npm}', float(p1-p0), float(p) if p else None,
              f'flt3_itd=1 & npm1={npm}: mido on={p1:.4f} (n={n1}) vs off={p0:.4f} (n={n0}), diff={p1-p0:.4f}, chi2 p={p}')

# Gilteritinib in flt3_itd OR flt3_tkd combined "FLT3-mutated"
df['flt3_any'] = ((df['flt3_itd']==1) | (df['flt3_tkd']==1)).astype(int)
for v in [0,1]:
    r = stratified(df, 'treatment_gilteritinib', 'flt3_any', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i9_gilt_strat_flt3any{v}', d2, p2,
              f'flt3_any={v}: gilt on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# Ivosidenib in IDH1+ further stratified by NPM1 / TP53 / unfit / age
for cofeat in ['npm1_mutation','tp53_mutation','complex_karyotype','unfit_for_intensive','secondary_aml']:
    for v in [0,1]:
        sub = df[(df['idh1_mutation']==1) & (df[cofeat]==v)]
        if len(sub) < 50: continue
        p1 = sub.loc[sub['treatment_ivosidenib']==1,'objective_response'].mean()
        p0 = sub.loc[sub['treatment_ivosidenib']==0,'objective_response'].mean()
        n1 = int((sub['treatment_ivosidenib']==1).sum()); n0 = int((sub['treatment_ivosidenib']==0).sum())
        if n1==0 or n0==0: continue
        ct = pd.crosstab(sub['treatment_ivosidenib'], sub['objective_response'])
        if ct.shape==(2,2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
        else:
            p=None
        store(f'i9_ivo_idh1_{cofeat}{v}', float(p1-p0), float(p) if p else None,
              f'idh1=1 & {cofeat}={v}: ivo on={p1:.4f} (n={n1}) vs off={p0:.4f} (n={n0}), diff={p1-p0:.4f}, chi2 p={p}')

# Enasidenib in IDH2+ further stratified
for cofeat in ['npm1_mutation','tp53_mutation','complex_karyotype','unfit_for_intensive','secondary_aml']:
    for v in [0,1]:
        sub = df[(df['idh2_mutation']==1) & (df[cofeat]==v)]
        if len(sub) < 50: continue
        p1 = sub.loc[sub['treatment_enasidenib']==1,'objective_response'].mean()
        p0 = sub.loc[sub['treatment_enasidenib']==0,'objective_response'].mean()
        n1 = int((sub['treatment_enasidenib']==1).sum()); n0 = int((sub['treatment_enasidenib']==0).sum())
        if n1==0 or n0==0: continue
        ct = pd.crosstab(sub['treatment_enasidenib'], sub['objective_response'])
        if ct.shape==(2,2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
        else:
            p=None
        store(f'i9_ena_idh2_{cofeat}{v}', float(p1-p0), float(p) if p else None,
              f'idh2=1 & {cofeat}={v}: ena on={p1:.4f} (n={n1}) vs off={p0:.4f} (n={n0}), diff={p1-p0:.4f}, chi2 p={p}')

# ============================================================
# ITERATION 10: Joint interactions with TP53 (suppressor candidate)
# ============================================================
print('\n=== Iteration 10: TP53 as effect suppressor ===')
# venaza: in tp53=0 vs tp53=1 ?
for tp in [0,1]:
    r = stratified(df, 'treatment_venetoclax_azacitidine', 'tp53_mutation', tp)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i10_venaza_strat_tp53_{tp}', d2, p2,
              f'tp53={tp}: venaza on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')
# venaza in tp53=0 + complex_karyotype=0
for ck in [0,1]:
    sub = df[(df['tp53_mutation']==0) & (df['complex_karyotype']==ck)]
    p1 = sub.loc[sub['treatment_venetoclax_azacitidine']==1,'objective_response'].mean()
    p0 = sub.loc[sub['treatment_venetoclax_azacitidine']==0,'objective_response'].mean()
    n1 = int((sub['treatment_venetoclax_azacitidine']==1).sum()); n0 = int((sub['treatment_venetoclax_azacitidine']==0).sum())
    ct = pd.crosstab(sub['treatment_venetoclax_azacitidine'], sub['objective_response'])
    if ct.shape==(2,2):
        chi2, p, _, _ = stats.chi2_contingency(ct)
    else:
        p=None
    store(f'i10_venaza_tp53_0_ck_{ck}', float(p1-p0), float(p) if p else None,
          f'tp53=0 & complex_karyotype={ck}: venaza on={p1:.4f} (n={n1}) vs off={p0:.4f} (n={n0}), diff={p1-p0:.4f}, chi2 p={p}')

with open('results_my.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved with', len(results), 'entries')
