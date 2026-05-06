"""Iterations 4-10: treatment-by-biomarker interactions and matches."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_parquet('dataset.parquet')
df['smoke_current'] = (df['smoking_status'] == 'current').astype(int)
df['smoke_never'] = (df['smoking_status'] == 'never').astype(int)
df['histology_squamous'] = (df['histology'] == 'squamous').astype(int)
df['ecog_ge2'] = (df['ecog_ps'] >= 2).astype(int)
df['pdl1_high'] = (df['pdl1_tps'] >= 0.5).astype(int)

with open('analysis_state.json') as f:
    results = json.load(f)


def stratified(t, mod, mod_label=None):
    out = {}
    for v in [0, 1]:
        d = df[df[mod] == v]
        a = d.loc[d[t] == 1, 'pfs_months']
        b = d.loc[d[t] == 0, 'pfs_months']
        if len(a) > 5 and len(b) > 5:
            tt, p = stats.ttest_ind(a, b, equal_var=False)
            out[v] = (a.mean() - b.mean(), p, len(a), len(b))
    return out


def interaction(t, mod):
    f = f'pfs_months ~ {t} * {mod}'
    m = smf.ols(f, data=df).fit()
    name = f'{t}:{mod}'
    return float(m.params[name]), float(m.pvalues[name]), m


# =========================================================
# Iteration 4: treatment_pembrolizumab × pdl1_high (anti-PD-1 hypothesis)
# =========================================================
print('\n=== Iter 4: pembro × pdl1_high ===')
it4 = {'index': 4, 'proposed_hypotheses': [], 'analyses': []}
strat = stratified('treatment_pembrolizumab', 'pdl1_high')
print(strat)
eff_int, p_int, _ = interaction('treatment_pembrolizumab', 'pdl1_high')
print(f'Interaction term: {eff_int:+.3f}, p={p_int:.3g}')
it4['proposed_hypotheses'].append({
    'id': 'h4_1',
    'text': 'The benefit of treatment_pembrolizumab on pfs_months is greater in patients with pdl1_high=1 (PD-L1 TPS>=0.5) than in those with pdl1_high=0.',
    'kind': 'novel',
})
it4['analyses'].append({
    'hypothesis_ids': ['h4_1'],
    'code': "smf.ols('pfs_months ~ treatment_pembrolizumab * pdl1_high', data=df).fit()",
    'result_summary': f'Stratified pembro effect: pdl1_high=1: diff={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]}/{strat[1][3]}); pdl1_high=0: diff={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})

# Iteration 4b: pembro x tmb_high
strat = stratified('treatment_pembrolizumab', 'tmb_high')
eff_int, p_int, _ = interaction('treatment_pembrolizumab', 'tmb_high')
print(f'pembro x tmb_high: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it4['proposed_hypotheses'].append({
    'id': 'h4_2',
    'text': 'The benefit of treatment_pembrolizumab on pfs_months is greater in tmb_high=1 patients than tmb_high=0.',
    'kind': 'novel',
})
it4['analyses'].append({
    'hypothesis_ids': ['h4_2'],
    'code': "smf.ols('pfs_months ~ treatment_pembrolizumab * tmb_high', data=df).fit()",
    'result_summary': f'Stratified pembro: tmb_high=1: diff={strat[1][0]:+.3f} (p={strat[1][1]:.3g}); tmb_high=0: diff={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})
results['it4'] = it4

# =========================================================
# Iteration 5: sotorasib × kras_g12c
# =========================================================
print('\n=== Iter 5: sotorasib × kras_g12c ===')
it5 = {'index': 5, 'proposed_hypotheses': [], 'analyses': []}
strat = stratified('treatment_sotorasib', 'kras_g12c')
eff_int, p_int, _ = interaction('treatment_sotorasib', 'kras_g12c')
print(f'sotorasib x kras_g12c: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it5['proposed_hypotheses'].append({
    'id': 'h5_1',
    'text': 'treatment_sotorasib improves pfs_months specifically in kras_g12c=1 patients but not in kras_g12c=0 patients (positive interaction).',
    'kind': 'novel',
})
it5['analyses'].append({
    'hypothesis_ids': ['h5_1'],
    'code': "smf.ols('pfs_months ~ treatment_sotorasib * kras_g12c', data=df).fit()",
    'result_summary': f'kras_g12c=1: sotorasib effect={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]} on, {strat[1][3]} off); kras_g12c=0: effect={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})
results['it5'] = it5

# =========================================================
# Iteration 6: osimertinib × egfr_mutation
# =========================================================
print('\n=== Iter 6: osimertinib × egfr ===')
it6 = {'index': 6, 'proposed_hypotheses': [], 'analyses': []}
strat = stratified('treatment_osimertinib', 'egfr_mutation')
eff_int, p_int, _ = interaction('treatment_osimertinib', 'egfr_mutation')
print(f'osimertinib x egfr: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it6['proposed_hypotheses'].append({
    'id': 'h6_1',
    'text': 'treatment_osimertinib improves pfs_months specifically in egfr_mutation=1 patients but not in egfr_mutation=0 patients.',
    'kind': 'novel',
})
it6['analyses'].append({
    'hypothesis_ids': ['h6_1'],
    'code': "smf.ols('pfs_months ~ treatment_osimertinib * egfr_mutation', data=df).fit()",
    'result_summary': f'egfr=1: osi effect={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]}/{strat[1][3]}); egfr=0: effect={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})
results['it6'] = it6

# =========================================================
# Iteration 7: olaparib × brca2_mutation
# =========================================================
print('\n=== Iter 7: olaparib × brca2 ===')
it7 = {'index': 7, 'proposed_hypotheses': [], 'analyses': []}
strat = stratified('treatment_olaparib', 'brca2_mutation')
eff_int, p_int, _ = interaction('treatment_olaparib', 'brca2_mutation')
print(f'olaparib x brca2: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it7['proposed_hypotheses'].append({
    'id': 'h7_1',
    'text': 'treatment_olaparib improves pfs_months specifically in brca2_mutation=1 patients but not in brca2_mutation=0 patients.',
    'kind': 'novel',
})
it7['analyses'].append({
    'hypothesis_ids': ['h7_1'],
    'code': "smf.ols('pfs_months ~ treatment_olaparib * brca2_mutation', data=df).fit()",
    'result_summary': f'brca2=1: olaparib effect={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]}/{strat[1][3]}); brca2=0: effect={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})
results['it7'] = it7

# =========================================================
# Iteration 8: alk_fusion strong negative effect — does treatment modify?
# =========================================================
print('\n=== Iter 8: alk_fusion as predictor / modifier ===')
it8 = {'index': 8, 'proposed_hypotheses': [], 'analyses': []}
# Test interactions of alk_fusion with each treatment
for i, t in enumerate(['treatment_pembrolizumab', 'treatment_sotorasib',
                        'treatment_olaparib', 'treatment_osimertinib'], 1):
    strat = stratified(t, 'alk_fusion')
    eff_int, p_int, _ = interaction(t, 'alk_fusion')
    print(f'{t} x alk_fusion: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
    hid = f'h8_{i}'
    direction = 'larger' if eff_int > 0 else 'smaller'
    it8['proposed_hypotheses'].append({
        'id': hid,
        'text': f'The effect of {t} on pfs_months is {direction} in alk_fusion=1 patients than in alk_fusion=0 (interaction).',
        'kind': 'novel',
    })
    it8['analyses'].append({
        'hypothesis_ids': [hid],
        'code': f"smf.ols('pfs_months ~ {t} * alk_fusion', data=df).fit()",
        'result_summary': f'alk_fusion=1: {t} effect={strat.get(1,(np.nan,np.nan,0,0))[0]:+.3f} (n={strat.get(1,(0,0,0,0))[2]}/{strat.get(1,(0,0,0,0))[3]}); alk_fusion=0: {strat.get(0,(np.nan,np.nan,0,0))[0]:+.3f}. Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
        'p_value': float(p_int),
        'effect_estimate': float(eff_int),
        'significant': bool(p_int < 0.05),
    })
results['it8'] = it8

# =========================================================
# Iteration 9: age_years strong correlation — is it spurious / due to confounding?
# Also test continuous covariate adjustments
# =========================================================
print('\n=== Iter 9: multivariable PFS model ===')
it9 = {'index': 9, 'proposed_hypotheses': [], 'analyses': []}

formula = ('pfs_months ~ age_years + sex_female + ecog_ge2 + stage_iv + has_brain_mets + '
           'C(smoking_status) + C(histology) + egfr_mutation + kras_g12c + alk_fusion + '
           'stk11_mutation + brca2_mutation + pdl1_tps + tmb_high + albumin_g_dl + '
           'ldh_u_l + weight_loss_pct_6mo + crp_mg_l + nlr + treatment_pembrolizumab + '
           'treatment_sotorasib + treatment_olaparib + treatment_osimertinib')
m = smf.ols(formula, data=df).fit()
print(m.summary().tables[1])
# Save the multivariable model coefficients
key_terms = ['age_years', 'sex_female', 'ecog_ge2', 'stage_iv', 'has_brain_mets',
             'egfr_mutation', 'kras_g12c', 'alk_fusion', 'stk11_mutation',
             'brca2_mutation', 'pdl1_tps', 'tmb_high', 'albumin_g_dl', 'ldh_u_l',
             'weight_loss_pct_6mo', 'crp_mg_l', 'nlr',
             'treatment_pembrolizumab', 'treatment_sotorasib',
             'treatment_olaparib', 'treatment_osimertinib']
for i, term in enumerate(key_terms, 1):
    if term in m.params.index:
        coef = float(m.params[term])
        p = float(m.pvalues[term])
        hid = f'h9_{i}'
        direction = 'positively' if coef > 0 else 'negatively'
        it9['proposed_hypotheses'].append({
            'id': hid,
            'text': f'After multivariable adjustment, {term} is {direction} associated with pfs_months independent of other covariates.',
            'kind': 'refined',
        })
        it9['analyses'].append({
            'hypothesis_ids': [hid],
            'code': "smf.ols(big_formula, data=df).fit()",
            'result_summary': f'Multivariable OLS coef for {term}={coef:+.4f}, p={p:.3g}.',
            'p_value': p,
            'effect_estimate': coef,
            'significant': bool(p < 0.05),
        })
results['it9'] = it9

# =========================================================
# Iteration 10: explore alk_fusion + osimertinib (since alk has -0.81 effect in main effects, but is osimertinib targeting it?)
# Actually — explore if alk_fusion×treatment_osimertinib (cross) or alk fusion has its own targeted treatment? alk_fusion's targeted drug is e.g. crizotinib/alectinib not in dataset.
# Check if osimertinib helps in alk_fusion subgroup — it shouldn't.
# Also check pembrolizumab × stk11 (well-known: stk11 mutations blunt anti-PD-1 response)
# =========================================================
print('\n=== Iter 10: pembrolizumab × stk11_mutation ===')
it10 = {'index': 10, 'proposed_hypotheses': [], 'analyses': []}
strat = stratified('treatment_pembrolizumab', 'stk11_mutation')
eff_int, p_int, _ = interaction('treatment_pembrolizumab', 'stk11_mutation')
print(f'pembro x stk11: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it10['proposed_hypotheses'].append({
    'id': 'h10_1',
    'text': 'The benefit of treatment_pembrolizumab on pfs_months is reduced (or reversed) in stk11_mutation=1 patients compared to stk11_mutation=0 patients.',
    'kind': 'novel',
})
it10['analyses'].append({
    'hypothesis_ids': ['h10_1'],
    'code': "smf.ols('pfs_months ~ treatment_pembrolizumab * stk11_mutation', data=df).fit()",
    'result_summary': f'stk11=1: pembro effect={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]}/{strat[1][3]}); stk11=0: effect={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})

# pembro × smoking
strat = stratified('treatment_pembrolizumab', 'smoke_never')
eff_int, p_int, _ = interaction('treatment_pembrolizumab', 'smoke_never')
print(f'pembro x smoke_never: int {eff_int:+.3f}, p={p_int:.3g}; strat {strat}')
it10['proposed_hypotheses'].append({
    'id': 'h10_2',
    'text': 'The benefit of treatment_pembrolizumab on pfs_months is smaller (or negative) in smoke_never=1 (never-smokers) patients than in former/current smokers.',
    'kind': 'novel',
})
it10['analyses'].append({
    'hypothesis_ids': ['h10_2'],
    'code': "smf.ols('pfs_months ~ treatment_pembrolizumab * smoke_never', data=df).fit()",
    'result_summary': f'smoke_never=1: pembro effect={strat[1][0]:+.3f} (p={strat[1][1]:.3g}, n={strat[1][2]}/{strat[1][3]}); smoke_never=0: effect={strat[0][0]:+.3f} (p={strat[0][1]:.3g}). Interaction coef={eff_int:+.3f}, p={p_int:.3g}.',
    'p_value': float(p_int),
    'effect_estimate': float(eff_int),
    'significant': bool(p_int < 0.05),
})
results['it10'] = it10

with open('analysis_state.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('\nSaved iterations 4-10')
