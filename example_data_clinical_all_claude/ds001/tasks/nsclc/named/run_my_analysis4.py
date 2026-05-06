"""Iterations 15-20: deep dive into sotorasib subgroup, plus systematic 2- and 3-way subgroup search for the other treatments."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
df['smoke_current'] = (df['smoking_status'] == 'current').astype(int)
df['smoke_never'] = (df['smoking_status'] == 'never').astype(int)
df['smoke_former'] = (df['smoking_status'] == 'former').astype(int)
df['histology_squamous'] = (df['histology'] == 'squamous').astype(int)
df['ecog_ge2'] = (df['ecog_ps'] >= 2).astype(int)
df['ecog_0'] = (df['ecog_ps'] == 0).astype(int)
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)
df['age_old'] = (df['age_years'] >= 70).astype(int)
df['ldh_high'] = (df['ldh_u_l'] >= df['ldh_u_l'].median()).astype(int)
df['albumin_low'] = (df['albumin_g_dl'] < 3.5).astype(int)
df['nlr_high'] = (df['nlr'] >= 5).astype(int)
df['weight_loss_high'] = (df['weight_loss_pct_6mo'] >= 5).astype(int)
df['crp_high'] = (df['crp_mg_l'] >= 10).astype(int)

with open('analysis_state.json') as f:
    results = json.load(f)


def stratified(t, mask, df_=None):
    d = (df_ if df_ is not None else df)[mask]
    a = d.loc[d[t] == 1, 'pfs_months']; b = d.loc[d[t] == 0, 'pfs_months']
    if len(a) > 5 and len(b) > 5:
        tt, p = stats.ttest_ind(a, b, equal_var=False)
        return a.mean() - b.mean(), p, len(a), len(b), a.mean(), b.mean()
    return None


# =========================================================
# Iter 15: sotorasib effect within kras_g12c=1 by sex
# =========================================================
print('\n=== Iter 15: sotorasib effect in kras+ by sex ===')
it15 = {'index': 15, 'proposed_hypotheses': [], 'analyses': []}
# kras+ male
mask = (df['kras_g12c'] == 1) & (df['sex_female'] == 0)
res = stratified('treatment_sotorasib', mask)
print(f'kras+ male: sotorasib eff={res[0]:+.3f}, p={res[1]:.3g}, n={res[2]}/{res[3]}, on={res[4]:.2f}, off={res[5]:.2f}')
m_eff, m_p, m_na, m_nb = res[0], res[1], res[2], res[3]
hid = 'h15_1'
it15['proposed_hypotheses'].append({
    'id': hid,
    'text': 'In male, kras_g12c=1 patients (kras_g12c=1 AND sex_female=0), treatment_sotorasib increases pfs_months relative to no sotorasib.',
    'kind': 'novel',
})
it15['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "stats.ttest_ind on subset (kras_g12c==1 & sex_female==0)",
    'result_summary': f'In kras_g12c=1 & sex_female=0: sotorasib effect on pfs_months = {m_eff:+.3f} mo (p={m_p:.3g}, n_on={m_na}, n_off={m_nb}; mean on={res[4]:.2f}, off={res[5]:.2f}).',
    'p_value': float(m_p),
    'effect_estimate': float(m_eff),
    'significant': bool(m_p < 0.05),
})
# kras+ female
mask = (df['kras_g12c'] == 1) & (df['sex_female'] == 1)
res = stratified('treatment_sotorasib', mask)
print(f'kras+ female: sotorasib eff={res[0]:+.3f}, p={res[1]:.3g}, n={res[2]}/{res[3]}, on={res[4]:.2f}, off={res[5]:.2f}')
hid = 'h15_2'
it15['proposed_hypotheses'].append({
    'id': hid,
    'text': 'In female, kras_g12c=1 patients (kras_g12c=1 AND sex_female=1), treatment_sotorasib does NOT improve pfs_months (effect ≈ 0 or negative).',
    'kind': 'novel',
})
it15['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "stats.ttest_ind on subset (kras_g12c==1 & sex_female==1)",
    'result_summary': f'In kras_g12c=1 & sex_female=1: sotorasib effect on pfs_months = {res[0]:+.3f} mo (p={res[1]:.3g}, n_on={res[2]}, n_off={res[3]}).',
    'p_value': float(res[1]),
    'effect_estimate': float(res[0]),
    'significant': bool(res[1] < 0.05),
})
# kras+ × not alk_fusion / not brca2 (suppressors)
mask = (df['kras_g12c'] == 1) & (df['alk_fusion'] == 0)
res = stratified('treatment_sotorasib', mask)
print(f'kras+ alk-: sotorasib eff={res[0]:+.3f}, p={res[1]:.3g}, n={res[2]}/{res[3]}')
hid = 'h15_3'
it15['proposed_hypotheses'].append({
    'id': hid,
    'text': 'In kras_g12c=1 AND alk_fusion=0 patients, treatment_sotorasib improves pfs_months.',
    'kind': 'refined',
})
it15['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "stats.ttest_ind on subset (kras_g12c==1 & alk_fusion==0)",
    'result_summary': f'In kras_g12c=1 & alk_fusion=0: sotorasib effect = {res[0]:+.3f} mo (p={res[1]:.3g}, n={res[2]}/{res[3]}).',
    'p_value': float(res[1]),
    'effect_estimate': float(res[0]),
    'significant': bool(res[1] < 0.05),
})
# kras+ alk+ (sotorasib should be 0 or negative)
mask = (df['kras_g12c'] == 1) & (df['alk_fusion'] == 1)
res = stratified('treatment_sotorasib', mask)
print(f'kras+ alk+: sotorasib eff={res[0]:+.3f}, p={res[1]:.3g}, n={res[2]}/{res[3]}')
hid = 'h15_4'
it15['proposed_hypotheses'].append({
    'id': hid,
    'text': 'In kras_g12c=1 AND alk_fusion=1 patients, treatment_sotorasib does NOT improve pfs_months (alk fusion suppresses sotorasib benefit).',
    'kind': 'refined',
})
it15['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "stats.ttest_ind on (kras_g12c==1 & alk_fusion==1)",
    'result_summary': f'In kras_g12c=1 & alk_fusion=1 (n={res[2]}/{res[3]}): sotorasib effect = {res[0]:+.3f} mo (p={res[1]:.3g}).',
    'p_value': float(res[1]),
    'effect_estimate': float(res[0]),
    'significant': bool(res[1] < 0.05),
})
# Combined "ideal sotorasib" subgroup: kras+, male, alk-, brca2-
mask = (df['kras_g12c'] == 1) & (df['sex_female'] == 0) & (df['alk_fusion'] == 0) & (df['brca2_mutation'] == 0)
res = stratified('treatment_sotorasib', mask)
print(f'kras+ M alk- brca2-: sotorasib eff={res[0]:+.3f}, p={res[1]:.3g}, n={res[2]}/{res[3]}')
hid = 'h15_5'
it15['proposed_hypotheses'].append({
    'id': hid,
    'text': 'In kras_g12c=1 AND sex_female=0 AND alk_fusion=0 AND brca2_mutation=0 patients, treatment_sotorasib substantially improves pfs_months.',
    'kind': 'refined',
})
it15['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "stats.ttest_ind on (kras_g12c==1 & sex_female==0 & alk_fusion==0 & brca2_mutation==0)",
    'result_summary': f'Combined subgroup (n={res[2]}/{res[3]}): sotorasib effect = {res[0]:+.3f} mo (p={res[1]:.3g}; mean on={res[4]:.2f}, off={res[5]:.2f}).',
    'p_value': float(res[1]),
    'effect_estimate': float(res[0]),
    'significant': bool(res[1] < 0.05),
})
results['it15'] = it15

# =========================================================
# Iter 16: pembrolizumab — exhaustive 2-way subgroup search using
# strong main-effect features
# =========================================================
print('\n=== Iter 16: pembrolizumab 2-way subgroup search ===')
it16 = {'index': 16, 'proposed_hypotheses': [], 'analyses': []}
candidates = ['pdl1_high', 'tmb_high', 'stk11_mutation', 'kras_g12c',
              'egfr_mutation', 'alk_fusion', 'brca2_mutation',
              'sex_female', 'ecog_ge2', 'ecog_0', 'stage_iv', 'has_brain_mets',
              'smoke_current', 'smoke_never', 'smoke_former', 'histology_squamous',
              'age_old', 'ldh_high', 'albumin_low', 'nlr_high', 'weight_loss_high', 'crp_high']
results_grid = []
for f1, f2 in combinations(candidates, 2):
    for v1 in [0, 1]:
        for v2 in [0, 1]:
            mask = (df[f1] == v1) & (df[f2] == v2)
            if mask.sum() < 200:
                continue
            on = df.loc[mask & (df['treatment_pembrolizumab'] == 1), 'pfs_months']
            off = df.loc[mask & (df['treatment_pembrolizumab'] == 0), 'pfs_months']
            if len(on) < 30 or len(off) < 30:
                continue
            tt, p = stats.ttest_ind(on, off, equal_var=False)
            results_grid.append((f1, v1, f2, v2, on.mean() - off.mean(), p, len(on), len(off)))
results_grid.sort(key=lambda r: r[5])
print('Top pembrolizumab subgroups by p-value:')
for r in results_grid[:15]:
    print(f'  {r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}')
hid = 'h16_1'
top = results_grid[0]
it16['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest 2-way subgroup for treatment_pembrolizumab effect on pfs_months is {top[0]}={top[1]} AND {top[2]}={top[3]} (effect {top[4]:+.3f} mo).',
    'kind': 'novel',
})
it16['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "exhaustive 2-way subgroup t-tests for treatment_pembrolizumab effect",
    'result_summary': 'Top 5 subgroups: ' + '; '.join(
        f'{r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}'
        for r in results_grid[:5]
    ),
    'p_value': float(top[5]),
    'effect_estimate': float(top[4]),
    'significant': bool(top[5] < 0.05),
})
results['it16'] = it16

# =========================================================
# Iter 17: olaparib 2-way subgroup search
# =========================================================
print('\n=== Iter 17: olaparib 2-way subgroup search ===')
it17 = {'index': 17, 'proposed_hypotheses': [], 'analyses': []}
results_grid = []
for f1, f2 in combinations(candidates, 2):
    for v1 in [0, 1]:
        for v2 in [0, 1]:
            mask = (df[f1] == v1) & (df[f2] == v2)
            if mask.sum() < 200:
                continue
            on = df.loc[mask & (df['treatment_olaparib'] == 1), 'pfs_months']
            off = df.loc[mask & (df['treatment_olaparib'] == 0), 'pfs_months']
            if len(on) < 30 or len(off) < 30:
                continue
            tt, p = stats.ttest_ind(on, off, equal_var=False)
            results_grid.append((f1, v1, f2, v2, on.mean() - off.mean(), p, len(on), len(off)))
results_grid.sort(key=lambda r: r[5])
print('Top olaparib subgroups:')
for r in results_grid[:10]:
    print(f'  {r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}')
hid = 'h17_1'
top = results_grid[0]
it17['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest 2-way subgroup for treatment_olaparib effect is {top[0]}={top[1]} AND {top[2]}={top[3]} (effect {top[4]:+.3f} mo).',
    'kind': 'novel',
})
it17['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "exhaustive 2-way subgroup t-tests for treatment_olaparib",
    'result_summary': 'Top 5 subgroups: ' + '; '.join(
        f'{r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}'
        for r in results_grid[:5]
    ),
    'p_value': float(top[5]),
    'effect_estimate': float(top[4]),
    'significant': bool(top[5] < 0.05),
})
results['it17'] = it17

# =========================================================
# Iter 18: osimertinib 2-way subgroup search
# =========================================================
print('\n=== Iter 18: osimertinib 2-way subgroup search ===')
it18 = {'index': 18, 'proposed_hypotheses': [], 'analyses': []}
results_grid = []
for f1, f2 in combinations(candidates, 2):
    for v1 in [0, 1]:
        for v2 in [0, 1]:
            mask = (df[f1] == v1) & (df[f2] == v2)
            if mask.sum() < 200:
                continue
            on = df.loc[mask & (df['treatment_osimertinib'] == 1), 'pfs_months']
            off = df.loc[mask & (df['treatment_osimertinib'] == 0), 'pfs_months']
            if len(on) < 30 or len(off) < 30:
                continue
            tt, p = stats.ttest_ind(on, off, equal_var=False)
            results_grid.append((f1, v1, f2, v2, on.mean() - off.mean(), p, len(on), len(off)))
results_grid.sort(key=lambda r: r[5])
print('Top osimertinib subgroups:')
for r in results_grid[:10]:
    print(f'  {r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}')
hid = 'h18_1'
top = results_grid[0]
it18['proposed_hypotheses'].append({
    'id': hid,
    'text': f'The strongest 2-way subgroup for treatment_osimertinib effect is {top[0]}={top[1]} AND {top[2]}={top[3]} (effect {top[4]:+.3f} mo).',
    'kind': 'novel',
})
it18['analyses'].append({
    'hypothesis_ids': [hid],
    'code': "exhaustive 2-way subgroup t-tests for treatment_osimertinib",
    'result_summary': 'Top 5 subgroups: ' + '; '.join(
        f'{r[0]}={r[1]} & {r[2]}={r[3]}: eff={r[4]:+.3f}, p={r[5]:.3g}, n={r[6]}/{r[7]}'
        for r in results_grid[:5]
    ),
    'p_value': float(top[5]),
    'effect_estimate': float(top[4]),
    'significant': bool(top[5] < 0.05),
})
results['it18'] = it18

with open('analysis_state.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved iterations 15-18')
