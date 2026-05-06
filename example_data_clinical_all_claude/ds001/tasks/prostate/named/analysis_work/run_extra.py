"""Additional analyses focused on enzalutamide subgroups and refined predictive checks."""
import json, numpy as np, pandas as pd
from scipy import stats
from itertools import combinations
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('../dataset.parquet')

def diff_prop(g1, g0):
    n1, n0 = len(g1), len(g0)
    p1 = g1.mean() if n1>0 else np.nan
    p0 = g0.mean() if n0>0 else np.nan
    diff = p1 - p0
    if n1==0 or n0==0:
        return diff, np.nan, n1, n0
    table = np.array([[g1.sum(), n1-g1.sum()], [g0.sum(), n0-g0.sum()]])
    if (table==0).any() or n1<5 or n0<5:
        try:
            _, p = stats.fisher_exact(table)
        except Exception:
            p = np.nan
    else:
        _, p, _, _ = stats.chi2_contingency(table)
    return diff, p, n1, n0

extra = {}

# Detailed enzalutamide subgroup screen
# Enzalutamide x mcrpc, ar_v7, brca2, msi, visceral, high PSA
print("\n=== Enzalutamide stratified analyses ===")
res = []
for cond_name, cond in [
    ('all', pd.Series([True]*len(df))),
    ('mcrpc=0', df['mcrpc']==0),
    ('mcrpc=1', df['mcrpc']==1),
    ('ar_v7=0', df['ar_v7_positive']==0),
    ('ar_v7=1', df['ar_v7_positive']==1),
    ('brca2=0', df['brca2_mutation']==0),
    ('brca2=1', df['brca2_mutation']==1),
    ('msi=0', df['msi_high']==0),
    ('msi=1', df['msi_high']==1),
    ('visceral=0', df['visceral_mets']==0),
    ('visceral=1', df['visceral_mets']==1),
    ('mcrpc=0 & ar_v7=0', (df['mcrpc']==0)&(df['ar_v7_positive']==0)),
    ('mcrpc=0 & ar_v7=0 & brca2=0', (df['mcrpc']==0)&(df['ar_v7_positive']==0)&(df['brca2_mutation']==0)),
    ('mcrpc=0 & ar_v7=0 & brca2=0 & msi=0', (df['mcrpc']==0)&(df['ar_v7_positive']==0)&(df['brca2_mutation']==0)&(df['msi_high']==0)),
]:
    sub = df.loc[cond]
    g1 = sub.loc[sub['treatment_enzalutamide']==1, 'objective_response']
    g0 = sub.loc[sub['treatment_enzalutamide']==0, 'objective_response']
    diff, p, n1, n0 = diff_prop(g1, g0)
    rec = {'subgroup': cond_name, 'rr_enz': float(g1.mean()) if n1>0 else None,
           'rr_no_enz': float(g0.mean()) if n0>0 else None,
           'diff': float(diff) if not np.isnan(diff) else None,
           'p': float(p) if not np.isnan(p) else None,
           'n_enz': n1, 'n_no_enz': n0}
    res.append(rec)
    print(f"  {cond_name}: rr_enz={rec['rr_enz']}, rr_no={rec['rr_no_enz']}, diff={rec['diff']}, p={rec['p']}, n_enz={n1}")
extra['enz_strat'] = res

# Check: does the big enzalutamide effect persist in mcrpc=0, ar_v7=0, brca2=0, msi=0?
# This is "non-mCRPC and AR-V7 negative and BRCA2 wild-type and MSI-stable" = the "best responder" subgroup

# Now systematically check enzalutamide effect by visceral mets within mcrpc=0/ar_v7=0
print("\n=== Enzalutamide in mcrpc=0 by other modifiers ===")
sub_base = df.loc[(df['mcrpc']==0) & (df['ar_v7_positive']==0) & (df['brca2_mutation']==0) & (df['msi_high']==0)]
print(f"  Base subgroup size: {len(sub_base)}")
g1 = sub_base.loc[sub_base['treatment_enzalutamide']==1, 'objective_response']
g0 = sub_base.loc[sub_base['treatment_enzalutamide']==0, 'objective_response']
diff, p, n1, n0 = diff_prop(g1, g0)
print(f"  Enz effect in best subgroup: {g1.mean():.3f} vs {g0.mean():.3f}, diff={diff:.4f}, p={p:.2e}")
extra['enz_best_subgroup'] = {'rr_enz': float(g1.mean()), 'rr_no_enz': float(g0.mean()),
    'diff': float(diff), 'p': float(p), 'n_enz': n1, 'n_no_enz': n0}

# Test interactions of enzalutamide with all features in mcrpc=0 subset
print("\n=== Enzalutamide x feature interactions in mcrpc=0 ===")
ALL_FEATS = ['age_years','ecog_ps','visceral_mets','psa_ng_ml','gleason_score',
             'brca2_mutation','ar_v7_positive','msi_high','psma_high','albumin_g_dl',
             'ldh_u_l','weight_loss_pct_6mo','crp_mg_l','nlr','hemoglobin_g_dl',
             'alkaline_phosphatase_u_l','ast_u_l','alt_u_l','total_bilirubin_mg_dl',
             'creatinine_mg_dl','bun_mg_dl','sodium_meq_l','potassium_meq_l','calcium_mg_dl']
sub = df.loc[df['mcrpc']==0]
res = []
for f in ALL_FEATS:
    try:
        m = smf.logit(f"objective_response ~ treatment_enzalutamide*{f}", data=sub).fit(disp=0, maxiter=100)
        inter = f'treatment_enzalutamide:{f}'
        if inter in m.pvalues:
            res.append({'feature': f, 'coef': float(m.params[inter]), 'p': float(m.pvalues[inter])})
    except Exception:
        pass
res.sort(key=lambda x: x['p'])
for r in res[:8]:
    print(f"  {r['feature']}: coef={r['coef']:.4f}, p={r['p']:.2e}")
extra['enz_in_mcrpc0_inter'] = res

# Final check: in best-responder subgroup (mcrpc=0, ar_v7=0, brca2=0, msi=0), does the enz effect vary with anything?
print("\n=== Enz x feature in best-responder subgroup ===")
res = []
for f in ALL_FEATS:
    if f in ['mcrpc','ar_v7_positive','brca2_mutation','msi_high']:
        continue
    try:
        m = smf.logit(f"objective_response ~ treatment_enzalutamide*{f}", data=sub_base).fit(disp=0, maxiter=100)
        inter = f'treatment_enzalutamide:{f}'
        if inter in m.pvalues:
            res.append({'feature': f, 'coef': float(m.params[inter]), 'p': float(m.pvalues[inter])})
    except Exception:
        pass
res.sort(key=lambda x: x['p'])
for r in res[:8]:
    print(f"  {r['feature']}: coef={r['coef']:.4f}, p={r['p']:.2e}")
extra['enz_in_best_inter'] = res

# Verify joint interaction model
print("\n=== Enzalutamide adjusted joint interaction model ===")
m = smf.logit("objective_response ~ treatment_enzalutamide*(mcrpc + ar_v7_positive + brca2_mutation + msi_high) + ecog_ps + visceral_mets + albumin_g_dl + ldh_u_l + age_years", data=df).fit(disp=0, maxiter=200)
print(m.summary())
extra['enz_joint_model'] = {
    'enz_main_coef': float(m.params['treatment_enzalutamide']),
    'enz_main_p': float(m.pvalues['treatment_enzalutamide']),
    'inter_mcrpc_coef': float(m.params['treatment_enzalutamide:mcrpc']),
    'inter_mcrpc_p': float(m.pvalues['treatment_enzalutamide:mcrpc']),
    'inter_arv7_coef': float(m.params['treatment_enzalutamide:ar_v7_positive']),
    'inter_arv7_p': float(m.pvalues['treatment_enzalutamide:ar_v7_positive']),
    'inter_brca_coef': float(m.params['treatment_enzalutamide:brca2_mutation']),
    'inter_brca_p': float(m.pvalues['treatment_enzalutamide:brca2_mutation']),
    'inter_msi_coef': float(m.params['treatment_enzalutamide:msi_high']),
    'inter_msi_p': float(m.pvalues['treatment_enzalutamide:msi_high']),
}

# Check: Does the enzalutamide main effect remain in mcrpc=0 alone?
print("\n=== Enzalutamide effect by isolated single modifier ===")
sub = df.loc[df['mcrpc']==0]
g1 = sub.loc[sub['treatment_enzalutamide']==1, 'objective_response']
g0 = sub.loc[sub['treatment_enzalutamide']==0, 'objective_response']
diff, p, _, _ = diff_prop(g1, g0)
print(f"  mcrpc=0 only: diff={diff:.4f}, p={p:.2e}, n_enz={len(g1)}, rr_enz={g1.mean():.3f}, rr_no={g0.mean():.3f}")

sub = df.loc[df['ar_v7_positive']==0]
g1 = sub.loc[sub['treatment_enzalutamide']==1, 'objective_response']
g0 = sub.loc[sub['treatment_enzalutamide']==0, 'objective_response']
diff, p, _, _ = diff_prop(g1, g0)
print(f"  ar_v7=0 only: diff={diff:.4f}, p={p:.2e}, n_enz={len(g1)}, rr_enz={g1.mean():.3f}, rr_no={g0.mean():.3f}")

# Sample size in best subgroup
print(f"\n  BEST subgroup size: {len(sub_base)}, enz=1: {(sub_base['treatment_enzalutamide']==1).sum()}, enz=0: {(sub_base['treatment_enzalutamide']==0).sum()}")

# Enzalutamide x continuous PSA - what's the threshold?
print("\n=== Enzalutamide effect by PSA decile ===")
sub = df.loc[(df['mcrpc']==0) & (df['ar_v7_positive']==0)]
sub = sub.copy()
sub['psa_q'] = pd.qcut(sub['psa_ng_ml'], q=10, labels=False, duplicates='drop')
res = []
for q in range(10):
    s = sub.loc[sub['psa_q']==q]
    g1 = s.loc[s['treatment_enzalutamide']==1, 'objective_response']
    g0 = s.loc[s['treatment_enzalutamide']==0, 'objective_response']
    if len(g1)<10 or len(g0)<10: continue
    diff, p, _, _ = diff_prop(g1, g0)
    psa_lo = s['psa_ng_ml'].min()
    psa_hi = s['psa_ng_ml'].max()
    res.append({'q': int(q), 'psa_lo': float(psa_lo), 'psa_hi': float(psa_hi),
                'diff': float(diff), 'p': float(p) if not np.isnan(p) else None, 'n_enz': len(g1)})
    print(f"  PSA Q{q} ({psa_lo:.1f}-{psa_hi:.1f}): diff={diff:.3f}, p={p:.2e}, n_enz={len(g1)}")
extra['enz_psa_decile'] = res

# Save
with open('extra_results.json','w') as f:
    json.dump(extra, f, indent=2, default=str)
print("\nSaved extra_results.json")
