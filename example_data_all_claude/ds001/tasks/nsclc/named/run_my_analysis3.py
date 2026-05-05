"""Iterations 11-18: systematic heterogeneity search for each treatment."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
df['smoke_current'] = (df['smoking_status'] == 'current').astype(int)
df['smoke_never'] = (df['smoking_status'] == 'never').astype(int)
df['smoke_former'] = (df['smoking_status'] == 'former').astype(int)
df['histology_squamous'] = (df['histology'] == 'squamous').astype(int)
df['ecog_ge2'] = (df['ecog_ps'] >= 2).astype(int)
df['ecog_0'] = (df['ecog_ps'] == 0).astype(int)
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)
df['pdl1_50plus'] = (df['pdl1_tps'] >= 0.5).astype(int)
df['age_old'] = (df['age_years'] >= 70).astype(int)
df['ldh_high'] = (df['ldh_u_l'] >= df['ldh_u_l'].median()).astype(int)
df['albumin_low'] = (df['albumin_g_dl'] < 3.5).astype(int)
df['nlr_high'] = (df['nlr'] >= 5).astype(int)
df['weight_loss_high'] = (df['weight_loss_pct_6mo'] >= 5).astype(int)
df['crp_high'] = (df['crp_mg_l'] >= 10).astype(int)

with open('analysis_state.json') as f:
    results = json.load(f)


SCREEN_FEATURES = [
    'sex_female', 'stage_iv', 'has_brain_mets',
    'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation', 'brca2_mutation',
    'tmb_high', 'pdl1_high', 'smoke_current', 'smoke_never', 'smoke_former',
    'histology_squamous', 'ecog_ge2', 'ecog_0', 'age_old',
    'ldh_high', 'albumin_low', 'nlr_high', 'weight_loss_high', 'crp_high',
]

def screen_interactions(t):
    rows = []
    for f in SCREEN_FEATURES:
        m = smf.ols(f'pfs_months ~ {t} * {f}', data=df).fit()
        name = f'{t}:{f}'
        coef = float(m.params[name])
        p = float(m.pvalues[name])
        # stratified estimates
        d1 = df[df[f] == 1]; d0 = df[df[f] == 0]
        if len(d1) > 50 and len(d0) > 50:
            a1 = d1.loc[d1[t] == 1, 'pfs_months']; b1 = d1.loc[d1[t] == 0, 'pfs_months']
            a0 = d0.loc[d0[t] == 1, 'pfs_months']; b0 = d0.loc[d0[t] == 0, 'pfs_months']
            eff1 = a1.mean() - b1.mean() if len(a1) > 5 and len(b1) > 5 else np.nan
            eff0 = a0.mean() - b0.mean() if len(a0) > 5 and len(b0) > 5 else np.nan
        else:
            eff1 = eff0 = np.nan
        rows.append((f, coef, p, eff1, eff0))
    rows.sort(key=lambda r: r[2])
    return rows


# =========================================================
# Iter 11: pembrolizumab heterogeneity search
# =========================================================
print('\n=== Iter 11: pembrolizumab heterogeneity ===')
it11 = {'index': 11, 'proposed_hypotheses': [], 'analyses': []}
rows = screen_interactions('treatment_pembrolizumab')
for f, coef, p, eff1, eff0 in rows[:8]:
    print(f'  {f}: int={coef:+.3f}, p={p:.3g}; eff[1]={eff1:+.3f}, eff[0]={eff0:+.3f}')
hid = 'h11_1'
top = rows[0]
it11['proposed_hypotheses'].append({
    'id': hid,
    'text': f'Among the screened modifiers, the strongest interaction with treatment_pembrolizumab is on {top[0]}: pembrolizumab effect on pfs_months differs between {top[0]}=1 and {top[0]}=0.',
    'kind': 'novel',
})
it11['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "screen interaction p-values for treatment_pembrolizumab × each candidate modifier.",
    'result_summary': 'Top interaction modifiers: ' + '; '.join(f'{f}: int={c:+.3f}, p={p:.3g}' for f, c, p, _, _ in rows[:5]),
    'p_value': float(top[2]),
    'effect_estimate': float(top[1]),
    'significant': bool(top[2] < 0.05),
})
results['it11'] = it11

# =========================================================
# Iter 12: sotorasib heterogeneity search (we know kras_g12c, look for further modifiers)
# =========================================================
print('\n=== Iter 12: sotorasib heterogeneity (whole cohort) ===')
it12 = {'index': 12, 'proposed_hypotheses': [], 'analyses': []}
rows = screen_interactions('treatment_sotorasib')
for f, coef, p, eff1, eff0 in rows[:10]:
    print(f'  {f}: int={coef:+.3f}, p={p:.3g}; eff[1]={eff1:+.3f}, eff[0]={eff0:+.3f}')
top = rows[0]
hid = 'h12_1'
it12['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest interaction with treatment_sotorasib in the full cohort is on {top[0]} (interaction coef {top[1]:+.3f}); sotorasib effect differs between {top[0]}=1 and =0.',
    'kind': 'novel',
})
it12['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "interaction screen treatment_sotorasib × each modifier.",
    'result_summary': 'Top interaction modifiers: ' + '; '.join(f'{f}: int={c:+.3f}, p={p:.3g}' for f, c, p, _, _ in rows[:5]),
    'p_value': float(top[2]),
    'effect_estimate': float(top[1]),
    'significant': bool(top[2] < 0.05),
})
# WITHIN kras_g12c=1 sub-population: any further modifiers?
print('  -- Within kras_g12c=1 sub-cohort:')
sub = df[df['kras_g12c'] == 1].copy()
inner = []
for f in SCREEN_FEATURES:
    if f == 'kras_g12c':
        continue
    if sub[f].nunique() < 2:
        continue
    m = smf.ols(f'pfs_months ~ treatment_sotorasib * {f}', data=sub).fit()
    name = f'treatment_sotorasib:{f}'
    if name not in m.params:
        continue
    inner.append((f, float(m.params[name]), float(m.pvalues[name])))
inner.sort(key=lambda r: r[2])
for f, c, p in inner[:8]:
    print(f'    kras+ × {f}: int={c:+.3f}, p={p:.3g}')
hid = 'h12_2'
top = inner[0]
it12['proposed_hypotheses'].append({
    'id': hid,
    'text': f'Within kras_g12c=1 patients, sotorasib effect on pfs_months is further modified by {top[0]} (top interaction).',
    'kind': 'refined',
})
it12['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "within kras_g12c=1: smf.ols('pfs_months ~ treatment_sotorasib * f', data=sub).fit()",
    'result_summary': 'Top sub-cohort interactions in kras_g12c=1: ' + '; '.join(f'{f}: int={c:+.3f}, p={p:.3g}' for f, c, p in inner[:5]),
    'p_value': float(top[2]),
    'effect_estimate': float(top[1]),
    'significant': bool(top[2] < 0.05),
})
results['it12'] = it12

# =========================================================
# Iter 13: olaparib heterogeneity search
# =========================================================
print('\n=== Iter 13: olaparib heterogeneity ===')
it13 = {'index': 13, 'proposed_hypotheses': [], 'analyses': []}
rows = screen_interactions('treatment_olaparib')
for f, coef, p, eff1, eff0 in rows[:10]:
    print(f'  {f}: int={coef:+.3f}, p={p:.3g}; eff[1]={eff1:+.3f}, eff[0]={eff0:+.3f}')
top = rows[0]
hid = 'h13_1'
it13['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest interaction with treatment_olaparib is on {top[0]}: effect of olaparib differs between {top[0]}=1 and =0.',
    'kind': 'novel',
})
it13['analyses'].append({
    'hypothesis_ids': [hid],
    'code': 'interaction screen treatment_olaparib × each modifier.',
    'result_summary': 'Top: ' + '; '.join(f'{f}: int={c:+.3f}, p={p:.3g}' for f, c, p, _, _ in rows[:5]),
    'p_value': float(top[2]),
    'effect_estimate': float(top[1]),
    'significant': bool(top[2] < 0.05),
})
results['it13'] = it13

# =========================================================
# Iter 14: osimertinib heterogeneity search
# =========================================================
print('\n=== Iter 14: osimertinib heterogeneity ===')
it14 = {'index': 14, 'proposed_hypotheses': [], 'analyses': []}
rows = screen_interactions('treatment_osimertinib')
for f, coef, p, eff1, eff0 in rows[:10]:
    print(f'  {f}: int={coef:+.3f}, p={p:.3g}; eff[1]={eff1:+.3f}, eff[0]={eff0:+.3f}')
top = rows[0]
hid = 'h14_1'
it14['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest interaction with treatment_osimertinib is on {top[0]}: effect of osimertinib differs between {top[0]}=1 and =0.',
    'kind': 'novel',
})
it14['analyses'].append({
    'hypothesis_ids': [hid],
    'code': 'interaction screen treatment_osimertinib × each modifier.',
    'result_summary': 'Top: ' + '; '.join(f'{f}: int={c:+.3f}, p={p:.3g}' for f, c, p, _, _ in rows[:5]),
    'p_value': float(top[2]),
    'effect_estimate': float(top[1]),
    'significant': bool(top[2] < 0.05),
})
results['it14'] = it14

with open('analysis_state.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved iterations 11-14')
