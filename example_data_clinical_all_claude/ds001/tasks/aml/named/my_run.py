"""Iterative analysis script for ds001_aml. Outputs results to results_my.json."""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
print(f'Loaded {len(df)} rows, {df.shape[1]} cols')

results = {}

def chi2_or_fisher(a, b):
    ct = pd.crosstab(b, a)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    p1 = a[b == 1].mean()
    p0 = a[b == 0].mean()
    return float(p1 - p0), float(p), float(p1), float(p0)

def fmt(x, k=4):
    try:
        return f'{x:.{k}f}'
    except Exception:
        return str(x)

def store(key, eff, p, summary, sig=None):
    results[key] = {
        'effect': None if eff is None else float(eff),
        'p': None if p is None else float(p),
        'summary': summary,
    }
    if sig is not None:
        results[key]['significant'] = bool(sig)

# ============================================================
# ITERATION 1: Demographics
# ============================================================
print('\n=== Iteration 1: demographics ===')
out = df['objective_response']
mean_resp = df.loc[out==1,'age_years'].mean()
mean_non = df.loc[out==0,'age_years'].mean()
m_age = smf.logit('objective_response ~ age_years', data=df).fit(disp=0)
beta = m_age.params['age_years']; pval = m_age.pvalues['age_years']
store('i1_age', beta, pval,
      f'logit beta(age_years)={beta:.5f} (OR/yr={np.exp(beta):.4f}), p={pval:.3g}; mean age responders={mean_resp:.2f} vs non={mean_non:.2f}')

d, p, p1, p0 = chi2_or_fisher(out, df['sex_female'])
store('i1_sex', d, p, f'response rate female={p1:.4f} vs male={p0:.4f}, diff={d:.4f}, chi2 p={p:.3g}')

m_ecog = smf.logit('objective_response ~ ecog_ps', data=df).fit(disp=0)
ecog_rates = ', '.join([f'ecog={e}:{df.loc[df.ecog_ps==e,"objective_response"].mean():.4f}' for e in sorted(df.ecog_ps.unique())])
store('i1_ecog', m_ecog.params['ecog_ps'], m_ecog.pvalues['ecog_ps'],
      f'logit beta(ecog_ps)={m_ecog.params["ecog_ps"]:.4f}, p={m_ecog.pvalues["ecog_ps"]:.3g}; rates: ' + ecog_rates)

d, p, p1, p0 = chi2_or_fisher(out, df['secondary_aml'])
store('i1_secondary_aml', d, p, f'secondary_aml=1 response={p1:.4f} vs =0 {p0:.4f}, diff={d:.4f}, chi2 p={p:.3g}')

d, p, p1, p0 = chi2_or_fisher(out, df['unfit_for_intensive'])
store('i1_unfit', d, p, f'unfit_for_intensive=1 response={p1:.4f} vs =0 {p0:.4f}, diff={d:.4f}, chi2 p={p:.3g}')

# ============================================================
# ITERATION 2: Mutations / cytogenetics
# ============================================================
print('\n=== Iteration 2: mutations ===')
for marker in ['complex_karyotype','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation','npm1_mutation','tp53_mutation']:
    d, p, p1, p0 = chi2_or_fisher(out, df[marker])
    n_pos = int(df[marker].sum())
    store(f'i2_{marker}', d, p, f'{marker}=1 response={p1:.4f} (n={n_pos}) vs =0 {p0:.4f}, diff={d:.4f}, chi2 p={p:.3g}')

# ============================================================
# ITERATION 3: Continuous labs
# ============================================================
print('\n=== Iteration 3: labs ===')
labs = ['wbc_k_per_ul','blast_pct_marrow','albumin_g_dl','ldh_u_l','crp_mg_l','nlr',
        'hemoglobin_g_dl','alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
        'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl',
        'weight_loss_pct_6mo']
for lab in labs:
    m = smf.logit(f'objective_response ~ {lab}', data=df).fit(disp=0)
    beta = m.params[lab]; pv = m.pvalues[lab]
    mu_r = df.loc[out==1,lab].mean(); mu_n = df.loc[out==0,lab].mean()
    store(f'i3_{lab}', beta, pv,
          f'{lab}: logit beta={beta:.5f}, p={pv:.3g}; mean responders={mu_r:.3f}, non={mu_n:.3f}')

# ============================================================
# ITERATION 4: Treatment univariate main effects
# ============================================================
print('\n=== Iteration 4: tx main effects ===')
treatments = ['treatment_midostaurin','treatment_gilteritinib','treatment_ivosidenib','treatment_enasidenib',
              'treatment_venetoclax_azacitidine','treatment_7plus3']
for tx in treatments:
    d, p, p1, p0 = chi2_or_fisher(out, df[tx])
    n_pos = int(df[tx].sum())
    store(f'i4_{tx}', d, p, f'{tx}=1 (n={n_pos}) response={p1:.4f} vs =0 {p0:.4f}, diff={d:.4f}, chi2 p={p:.3g}')

# ============================================================
# ITERATION 5: Multivariable adjusted
# ============================================================
print('\n=== Iteration 5: multivariable adjusted ===')
covs = ['age_years','sex_female','ecog_ps','secondary_aml','unfit_for_intensive',
        'complex_karyotype','flt3_itd','flt3_tkd','idh1_mutation','idh2_mutation',
        'npm1_mutation','tp53_mutation','wbc_k_per_ul','blast_pct_marrow','albumin_g_dl',
        'ldh_u_l','crp_mg_l','nlr','hemoglobin_g_dl']
formula = 'objective_response ~ ' + ' + '.join(covs + treatments)
m_full = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
print(m_full.summary().tables[1])
for v in covs + treatments:
    if v in m_full.params:
        store(f'i5_full_{v}', m_full.params[v], m_full.pvalues[v],
              f'adjusted logit beta({v})={m_full.params[v]:.4f}, p={m_full.pvalues[v]:.3g}')

# ============================================================
# ITERATION 6: targeted Tx x mutation interactions
# ============================================================
print('\n=== Iteration 6: targeted Tx x mutation ===')

def fit_inter(df, tx, marker, covs):
    other_covs = [c for c in covs if c not in (tx, marker)]
    f = f'objective_response ~ {tx}*{marker} + ' + ' + '.join(other_covs)
    m = smf.logit(f, data=df).fit(disp=0, maxiter=200)
    candidates = [k for k in m.params.index if (':' in k and tx in k and marker in k)]
    inter = candidates[0] if candidates else None
    return m, inter

def stratified(df, tx, modifier, mval):
    sub = df[df[modifier] == mval]
    p1 = sub.loc[sub[tx]==1, 'objective_response'].mean()
    p0 = sub.loc[sub[tx]==0, 'objective_response'].mean()
    n1 = int((sub[tx]==1).sum()); n0 = int((sub[tx]==0).sum())
    if n1==0 or n0==0:
        return None
    # 2x2 chi2
    ct = pd.crosstab(sub[tx], sub['objective_response'])
    if ct.shape == (2,2):
        chi2, p, _, _ = stats.chi2_contingency(ct)
    else:
        p = None
    return p1-p0, p, p1, p0, n1, n0

# midostaurin x flt3_itd
m, inter = fit_inter(df, 'treatment_midostaurin', 'flt3_itd', covs)
store('i6_mido_x_flt3itd', m.params[inter], m.pvalues[inter],
      f'logit interaction beta(treatment_midostaurin:flt3_itd)={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; mido main={m.params["treatment_midostaurin"]:.4f} (p={m.pvalues["treatment_midostaurin"]:.3g})')
for v in [0,1]:
    r = stratified(df, 'treatment_midostaurin', 'flt3_itd', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i6_mido_strat_flt3itd{v}', d2, p2,
              f'flt3_itd={v}: mido on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# gilteritinib x flt3_itd
m, inter = fit_inter(df, 'treatment_gilteritinib', 'flt3_itd', covs)
store('i6_gilt_x_flt3itd', m.params[inter], m.pvalues[inter],
      f'logit beta(treatment_gilteritinib:flt3_itd)={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; gilt main={m.params["treatment_gilteritinib"]:.4f} (p={m.pvalues["treatment_gilteritinib"]:.3g})')
for v in [0,1]:
    r = stratified(df, 'treatment_gilteritinib', 'flt3_itd', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i6_gilt_strat_flt3itd{v}', d2, p2,
              f'flt3_itd={v}: gilt on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# gilteritinib x flt3_tkd
m, inter = fit_inter(df, 'treatment_gilteritinib', 'flt3_tkd', covs)
store('i6_gilt_x_flt3tkd', m.params[inter], m.pvalues[inter],
      f'logit beta(treatment_gilteritinib:flt3_tkd)={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}')
for v in [0,1]:
    r = stratified(df, 'treatment_gilteritinib', 'flt3_tkd', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i6_gilt_strat_flt3tkd{v}', d2, p2,
              f'flt3_tkd={v}: gilt on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# ivosidenib x idh1
m, inter = fit_inter(df, 'treatment_ivosidenib', 'idh1_mutation', covs)
store('i6_ivo_x_idh1', m.params[inter], m.pvalues[inter],
      f'logit beta(treatment_ivosidenib:idh1_mutation)={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; ivo main={m.params["treatment_ivosidenib"]:.4f} (p={m.pvalues["treatment_ivosidenib"]:.3g})')
for v in [0,1]:
    r = stratified(df, 'treatment_ivosidenib', 'idh1_mutation', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i6_ivo_strat_idh1{v}', d2, p2,
              f'idh1={v}: ivo on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

# enasidenib x idh2
m, inter = fit_inter(df, 'treatment_enasidenib', 'idh2_mutation', covs)
store('i6_ena_x_idh2', m.params[inter], m.pvalues[inter],
      f'logit beta(treatment_enasidenib:idh2_mutation)={m.params[inter]:.4f}, p={m.pvalues[inter]:.3g}; ena main={m.params["treatment_enasidenib"]:.4f} (p={m.pvalues["treatment_enasidenib"]:.3g})')
for v in [0,1]:
    r = stratified(df, 'treatment_enasidenib', 'idh2_mutation', v)
    if r:
        d2,p2,a,b,n1,n0 = r
        store(f'i6_ena_strat_idh2{v}', d2, p2,
              f'idh2={v}: ena on={a:.4f} (n={n1}) vs off={b:.4f} (n={n0}), diff={d2:.4f}, chi2 p={p2:.3g}')

with open('results_my.json','w') as f:
    json.dump(results, f, indent=2, default=str)
print('Saved results_my.json with', len(results), 'entries')
