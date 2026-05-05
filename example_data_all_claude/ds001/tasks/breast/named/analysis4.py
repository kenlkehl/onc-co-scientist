"""Iter 21-25: confirm best subgroup hypotheses, multivariable interaction models."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

df = pd.read_parquet('dataset.parquet')
out = 'pfs_months'
df['brca_any'] = ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)).astype(int)
df['triple_neg'] = ((df['er_positive'] == 0) & (df['pr_positive'] == 0) & (df['her2_positive'] == 0)).astype(int)

results = {}

print('=== ITER 21: Confirm palbociclib best subgroup with 3-way interaction ===')
# Test that ER+ AND HER2- AND PIK3CA-mutation-negative jointly define the benefit
df['palbo_subgroup'] = ((df['er_positive'] == 1) & (df['her2_positive'] == 0)
                        & (df['pik3ca_mutation'] == 0)).astype(int)

m = smf.ols(f'{out} ~ treatment_palbociclib * palbo_subgroup', data=df).fit()
print(f"3-way subgroup (ER+/HER2-/PIK3CA-) interaction model:")
for k, v in m.params.items():
    print(f"  {k:50s} {v:+.4f}  p={m.pvalues[k]:.3g}")

# in subgroup
sub = df[df['palbo_subgroup'] == 1]
a = sub.loc[sub['treatment_palbociclib'] == 1, out]
b = sub.loc[sub['treatment_palbociclib'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
results['iter21_palbo_subgroup_in'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()),
    'n_tx': int(len(a)), 'n_ctrl': int(len(b)), 'p': float(p),
}
sub = df[df['palbo_subgroup'] == 0]
a = sub.loc[sub['treatment_palbociclib'] == 1, out]
b = sub.loc[sub['treatment_palbociclib'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
results['iter21_palbo_subgroup_out'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()),
    'n_tx': int(len(a)), 'n_ctrl': int(len(b)), 'p': float(p),
}
print(f"In subgroup: diff={results['iter21_palbo_subgroup_in']['diff']:+.3f}  p={results['iter21_palbo_subgroup_in']['p']:.3g}")
print(f"Out of subgroup: diff={results['iter21_palbo_subgroup_out']['diff']:+.3f}  p={results['iter21_palbo_subgroup_out']['p']:.3g}")

# Triple interaction (palbociclib x ER x HER2 x PIK3CA)
print('\nTriple interaction model:')
m = smf.ols(f'{out} ~ treatment_palbociclib * er_positive * her2_positive * pik3ca_mutation', data=df).fit()
print(f"  R^2 = {m.rsquared:.4f}")
key_terms = [k for k in m.params.index if 'treatment_palbociclib' in k]
for k in key_terms:
    print(f"  {k:80s} {m.params[k]:+.4f}  p={m.pvalues[k]:.3g}")

print('\n=== ITER 22: Olaparib best subgroup confirmation ===')
# BRCA+ /ER- has strongest effect; test
df['olap_subgroup'] = ((df['brca_any'] == 1) & (df['er_positive'] == 0)).astype(int)
sub = df[df['olap_subgroup'] == 1]
a = sub.loc[sub['treatment_olaparib'] == 1, out]
b = sub.loc[sub['treatment_olaparib'] == 0, out]
t, p = stats.ttest_ind(a, b, equal_var=False)
results['iter22_olap_brca_er_neg'] = {
    'mean_tx': float(a.mean()), 'mean_ctrl': float(b.mean()),
    'diff': float(a.mean() - b.mean()),
    'n_tx': int(len(a)), 'n_ctrl': int(len(b)), 'p': float(p),
}
print(f"Olaparib in BRCA+/ER-: diff={results['iter22_olap_brca_er_neg']['diff']:+.3f}  p={results['iter22_olap_brca_er_neg']['p']:.3g}")

# 3-way interaction
m = smf.ols(f'{out} ~ treatment_olaparib * brca_any * er_positive', data=df).fit()
print('Olaparib x BRCA x ER interaction:')
for k in [c for c in m.params.index if 'treatment_olaparib' in c]:
    print(f"  {k:60s} {m.params[k]:+.4f}  p={m.pvalues[k]:.3g}")

print('\n=== ITER 23: Adjusted model — palbociclib subgroup-specific effect adjusted for prognostics ===')
# Multivariable adjusted: palbociclib effect within ER+/HER2-/PIK3CA-, adjusting
adj_covars = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'ki67_pct',
              'albumin_g_dl', 'weight_loss_pct_6mo', 'ldh_u_l']
form = f"{out} ~ treatment_palbociclib + " + ' + '.join(adj_covars)
m = smf.ols(form, data=df[df['palbo_subgroup'] == 1]).fit()
results['iter23_palbo_adj_in_subgroup'] = {
    'palbo_coef': float(m.params['treatment_palbociclib']),
    'palbo_p': float(m.pvalues['treatment_palbociclib']),
    'r_squared': float(m.rsquared),
    'n': int(m.nobs),
}
print(f"Adjusted palbo effect within ER+/HER2-/PIK3CA-: coef={results['iter23_palbo_adj_in_subgroup']['palbo_coef']:+.4f}  p={results['iter23_palbo_adj_in_subgroup']['palbo_p']:.3g}  N={results['iter23_palbo_adj_in_subgroup']['n']}")

m = smf.ols(form, data=df[df['palbo_subgroup'] == 0]).fit()
results['iter23_palbo_adj_out_subgroup'] = {
    'palbo_coef': float(m.params['treatment_palbociclib']),
    'palbo_p': float(m.pvalues['treatment_palbociclib']),
    'r_squared': float(m.rsquared),
    'n': int(m.nobs),
}
print(f"Adjusted palbo effect outside ER+/HER2-/PIK3CA-: coef={results['iter23_palbo_adj_out_subgroup']['palbo_coef']:+.4f}  p={results['iter23_palbo_adj_out_subgroup']['palbo_p']:.3g}  N={results['iter23_palbo_adj_out_subgroup']['n']}")

print('\n=== ITER 24: Pembrolizumab in TNBC: refined sub-search ===')
# Pembro x TNBC interaction was significant (p=0.015). Look more carefully
# at pembro effect in TNBC stratified by other features.
df['ki67_high'] = (df['ki67_pct'] >= 20).astype(int)
def detail(treat, conds, label):
    mask = np.ones(len(df), dtype=bool)
    for c, v in conds.items():
        mask &= (df[c] == v)
    sub = df[mask]
    a = sub.loc[sub[treat] == 1, out]
    b = sub.loc[sub[treat] == 0, out]
    if len(a) < 5 or len(b) < 5:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {'label': label, 'diff': float(a.mean() - b.mean()),
            'n_tx': int(len(a)), 'n_ctrl': int(len(b)), 'p': float(p)}


pembro_in_tnbc = [
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'pik3ca_mutation': 0}, 'TNBC/PIK3CA-'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'pik3ca_mutation': 1}, 'TNBC/PIK3CA+'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'ki67_high': 1}, 'TNBC/ki67-high'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'ki67_high': 0}, 'TNBC/ki67-low'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'has_brain_mets': 0}, 'TNBC/no-brain-mets'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'has_brain_mets': 1}, 'TNBC/brain-mets'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'stage_iv': 0}, 'TNBC/non-stage-IV'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'stage_iv': 1}, 'TNBC/stage-IV'),
    detail('treatment_pembrolizumab', {'triple_neg': 1, 'brca_any': 1}, 'TNBC/BRCA-mut'),
]
results['iter24_pembro_tnbc_drill'] = pembro_in_tnbc
for r in pembro_in_tnbc:
    if r:
        print(f"  pembro in {r['label']:30s}: diff={r['diff']:+.3f} (n_tx={r['n_tx']}, n_ctrl={r['n_ctrl']}, p={r['p']:.3g})")

print('\n=== ITER 25: Final treatment heterogeneity tree-search via 3-way joint subgroups ===')
# Now do an exhaustive scan: test treatment effect in every combination of two
# binary modifiers (limit to those that showed interactions).
import itertools
top_modifiers = {
    'treatment_palbociclib': ['er_positive', 'her2_positive', 'pik3ca_mutation', 'her2_low', 'pr_positive', 'postmenopausal'],
    'treatment_pembrolizumab': ['triple_neg', 'er_positive', 'her2_positive', 'pik3ca_mutation', 'has_brain_mets'],
    'treatment_olaparib': ['brca_any', 'brca1_mutation', 'brca2_mutation', 'er_positive', 'her2_positive', 'triple_neg'],
    'treatment_trastuzumab': ['her2_positive', 'her2_low', 'er_positive'],
    'treatment_tamoxifen': ['er_positive', 'pr_positive', 'postmenopausal'],
    'treatment_sacituzumab_govitecan': ['triple_neg', 'her2_low', 'er_positive', 'her2_positive'],
}
best_subgroups = {}
for treat, mods in top_modifiers.items():
    rows = []
    for m1, m2 in itertools.combinations(mods, 2):
        for v1, v2 in itertools.product([0, 1], repeat=2):
            cond = {m1: v1, m2: v2}
            r = detail(treat, cond, f'{m1}={v1}/{m2}={v2}')
            if r and r['n_tx'] >= 30 and r['n_ctrl'] >= 30:
                rows.append(r)
    rows.sort(key=lambda x: -abs(x['diff']))
    best_subgroups[treat] = rows[:8]
    print(f"\n  {treat} top |diff| 2-way subgroups:")
    for r in rows[:8]:
        print(f"    {r['label']:50s} diff={r['diff']:+.3f}  p={r['p']:.3g} (n_tx={r['n_tx']})")
results['iter25_2way_subgroups'] = best_subgroups

with open('analysis_part4.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved analysis_part4.json')
