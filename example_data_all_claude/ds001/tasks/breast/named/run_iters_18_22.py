"""Iterations 18-22: refinement of strongest subgroups."""
import json
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

df = pd.read_parquet('dataset.parquet')
ki67_med = df['ki67_pct'].median()

results = {}


def subgroup_effect(df, treat, mask, label):
    sub = df[mask]
    on = sub.loc[sub[treat] == 1, 'pfs_months']
    off = sub.loc[sub[treat] == 0, 'pfs_months']
    out = {'label': label, 'n_on': int(len(on)), 'n_off': int(len(off))}
    if len(on) > 5 and len(off) > 5:
        tt = stats.ttest_ind(on, off, equal_var=False)
        out['mean_on'] = float(on.mean())
        out['mean_off'] = float(off.mean())
        out['diff'] = float(on.mean() - off.mean())
        out['p_value'] = float(tt.pvalue)
    return out


# --- Iter 18: Palbociclib subgroup refinement — leave-one-marker-out test ---
# Test whether each marker (ER+, HER2-, PIK3CA-WT, low Ki67) is necessary
treat = 'treatment_palbociclib'
markers_required = {
    'ER+': df['er_positive'] == 1,
    'HER2-': df['her2_positive'] == 0,
    'PIK3CA-WT': df['pik3ca_mutation'] == 0,
    'low_Ki67': df['ki67_pct'] < ki67_med,
}
full_mask = np.ones(len(df), bool)
for m in markers_required.values():
    full_mask &= m

iter18_full = subgroup_effect(df, treat, full_mask, 'ALL 4 markers (target)')
iter18_drops = []
for drop_marker in markers_required:
    keep = {k: v for k, v in markers_required.items() if k != drop_marker}
    mask = np.ones(len(df), bool)
    for v in keep.values():
        mask &= v
    iter18_drops.append({
        'description': f'drop {drop_marker} requirement (only require: {", ".join(keep)})',
        'result': subgroup_effect(df, treat, mask, f'WITHOUT_{drop_marker}'),
    })
# Test treatment effect within complement subgroup of each
iter18_complement = []
for marker_name, mask_pos in markers_required.items():
    other = [(n, m) for n, m in markers_required.items() if n != marker_name]
    other_mask = np.ones(len(df), bool)
    for _, m in other:
        other_mask &= m
    # Within "other 3 markers satisfied", split by `marker_name`
    sub_pos = df[other_mask & mask_pos]
    sub_neg = df[other_mask & (~mask_pos)]
    iter18_complement.append({
        'marker_being_tested': marker_name,
        'effect_when_marker_present': subgroup_effect(df, treat, df.index.isin(sub_pos.index), f'{marker_name}=1, other 3 satisfied'),
        'effect_when_marker_absent': subgroup_effect(df, treat, df.index.isin(sub_neg.index), f'{marker_name}=0, other 3 satisfied'),
    })

# Continuous Ki67 sensitivity
ki67_thresholds = [5, 10, 14, 15, 16, 18, 20, 25]
iter18_ki67_sweep = []
for th in ki67_thresholds:
    base_mask = (df['er_positive'] == 1) & (df['her2_positive'] == 0) & (df['pik3ca_mutation'] == 0)
    low = df[base_mask & (df['ki67_pct'] < th)]
    high = df[base_mask & (df['ki67_pct'] >= th)]
    rl = subgroup_effect(df, treat, df.index.isin(low.index), f'ER+/HER2-/PIK3CA-WT, ki67<{th}')
    rh = subgroup_effect(df, treat, df.index.isin(high.index), f'ER+/HER2-/PIK3CA-WT, ki67>={th}')
    iter18_ki67_sweep.append({'threshold': th, 'low': rl, 'high': rh})

results['iter18_palbociclib_refinement'] = {
    'full_4_markers': iter18_full,
    'leave_one_out': iter18_drops,
    'within_complement': iter18_complement,
    'ki67_threshold_sweep': iter18_ki67_sweep,
}

# --- Iter 19: 4-way interaction OLS with adjustment for prognostic covariates ---
# Use full multivariable model with palbociclib x quad indicator + prognostic adjusters
sub = df.copy()
sub['quad'] = ((sub['er_positive'] == 1) & (sub['her2_positive'] == 0) &
               (sub['pik3ca_mutation'] == 0) & (sub['ki67_pct'] < ki67_med)).astype(int)
sub['palbo_quad'] = sub[treat] * sub['quad']
adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
          'weight_loss_pct_6mo', 'ldh_u_l']
X = sm.add_constant(sub[[treat, 'quad', 'palbo_quad'] + adjust])
m = sm.OLS(sub['pfs_months'], X).fit()
iter19 = {
    'palbo_main_outside_quad': float(m.params[treat]),
    'palbo_main_p': float(m.pvalues[treat]),
    'quad_main': float(m.params['quad']),
    'quad_p': float(m.pvalues['quad']),
    'inter_palbo_quad': float(m.params['palbo_quad']),
    'inter_p': float(m.pvalues['palbo_quad']),
    'effect_in_quad': float(m.params[treat] + m.params['palbo_quad']),
    'r_squared': float(m.rsquared),
    'adjusted_for': adjust,
}
results['iter19_palbociclib_adjusted_quad_interaction'] = iter19

# --- Iter 20: Olaparib refinement — test BRCA1 vs BRCA2 vs both ---
treat = 'treatment_olaparib'
iter20 = []
combos = [
    ('BRCA1+ only (not BRCA2)', (df['brca1_mutation'] == 1) & (df['brca2_mutation'] == 0)),
    ('BRCA2+ only (not BRCA1)', (df['brca2_mutation'] == 1) & (df['brca1_mutation'] == 0)),
    ('BRCA1+ AND BRCA2+', (df['brca1_mutation'] == 1) & (df['brca2_mutation'] == 1)),
    ('BRCA either', (df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)),
    ('BRCA either AND HER2-', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['her2_positive'] == 0)),
    ('BRCA either AND HER2+', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['her2_positive'] == 1)),
    ('BRCA either AND ER+', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 1)),
    ('BRCA either AND ER- (TNBC-leaning)', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['er_positive'] == 0)),
    ('BRCA either AND no brain mets', ((df['brca1_mutation'] == 1) | (df['brca2_mutation'] == 1)) & (df['has_brain_mets'] == 0)),
    ('BRCA neg control: not BRCA either', (df['brca1_mutation'] == 0) & (df['brca2_mutation'] == 0)),
]
for label, mask in combos:
    iter20.append(subgroup_effect(df, treat, mask, label))

# Adjusted interaction model
sub = df.copy()
sub['brca_any'] = ((sub['brca1_mutation'] == 1) | (sub['brca2_mutation'] == 1)).astype(int)
sub['olap_brca'] = sub[treat] * sub['brca_any']
adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
          'weight_loss_pct_6mo', 'ldh_u_l', 'er_positive', 'her2_positive',
          'pik3ca_mutation', 'ki67_pct']
X = sm.add_constant(sub[[treat, 'brca_any', 'olap_brca'] + adjust])
m = sm.OLS(sub['pfs_months'], X).fit()
iter20_adj = {
    'inter_coef': float(m.params['olap_brca']),
    'inter_p': float(m.pvalues['olap_brca']),
    'olap_main_outside': float(m.params[treat]),
    'olap_main_p': float(m.pvalues[treat]),
    'effect_in_brca_any': float(m.params[treat] + m.params['olap_brca']),
}
results['iter20_olaparib_refinement'] = {'subgroups': iter20, 'adjusted_interaction': iter20_adj}

# --- Iter 21: Pembrolizumab + sacituzumab + trastuzumab broader search w/ adjustment ---
# Adjusted main effects for these treatments
iter21 = {}
for treat in ['treatment_trastuzumab', 'treatment_pembrolizumab', 'treatment_sacituzumab_govitecan',
              'treatment_tamoxifen']:
    adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
              'weight_loss_pct_6mo', 'ldh_u_l', 'er_positive', 'her2_positive',
              'pik3ca_mutation', 'ki67_pct', 'brca1_mutation', 'brca2_mutation',
              'her2_low', 'postmenopausal', 'pr_positive']
    X = sm.add_constant(df[[treat] + adjust])
    m = sm.OLS(df['pfs_months'], X).fit()
    iter21[treat] = {
        'adjusted_main_coef': float(m.params[treat]),
        'adjusted_main_p': float(m.pvalues[treat]),
        'adjusted_main_se': float(m.bse[treat]),
    }
results['iter21_adjusted_main_effects'] = iter21

# --- Iter 22: Trastuzumab subgroup screening with continuous and binary modifiers ---
# Restrict to HER2+ and search interactions inside that population
treat = 'treatment_trastuzumab'
her2_pos = df[df['her2_positive'] == 1].copy()
print(f"HER2+ population: {len(her2_pos)}")

iter22 = {}
binary_within = ['er_positive', 'pr_positive', 'postmenopausal', 'stage_iv', 'has_brain_mets',
                 'node_positive', 'brca1_mutation', 'brca2_mutation', 'pik3ca_mutation', 'her2_low',
                 'sex_female']
cont_within = ['age_years', 'ecog_ps', 'ki67_pct', 'tumor_size_cm', 'albumin_g_dl', 'ldh_u_l',
               'weight_loss_pct_6mo', 'crp_mg_l', 'nlr', 'hemoglobin_g_dl',
               'alkaline_phosphatase_u_l', 'ast_u_l', 'alt_u_l',
               'total_bilirubin_mg_dl', 'creatinine_mg_dl', 'bun_mg_dl',
               'sodium_meq_l', 'potassium_meq_l', 'calcium_mg_dl']
within_results = []
for m in binary_within:
    sub = her2_pos[[treat, m, 'pfs_months']].copy()
    sub['inter'] = sub[treat] * sub[m]
    X = sm.add_constant(sub[[treat, m, 'inter']])
    mm = sm.OLS(sub['pfs_months'], X).fit()
    on_pos = sub.loc[(sub[m] == 1) & (sub[treat] == 1), 'pfs_months'].mean()
    off_pos = sub.loc[(sub[m] == 1) & (sub[treat] == 0), 'pfs_months'].mean()
    on_neg = sub.loc[(sub[m] == 0) & (sub[treat] == 1), 'pfs_months'].mean()
    off_neg = sub.loc[(sub[m] == 0) & (sub[treat] == 0), 'pfs_months'].mean()
    within_results.append({
        'modifier': m, 'kind': 'binary',
        'inter_coef': float(mm.params['inter']),
        'inter_p': float(mm.pvalues['inter']),
        'eff_pos': float(on_pos - off_pos), 'eff_neg': float(on_neg - off_neg),
    })
for m in cont_within:
    sub = her2_pos[[treat, m, 'pfs_months']].copy()
    sub['inter'] = sub[treat] * sub[m]
    X = sm.add_constant(sub[[treat, m, 'inter']])
    mm = sm.OLS(sub['pfs_months'], X).fit()
    within_results.append({
        'modifier': m, 'kind': 'cont',
        'inter_coef': float(mm.params['inter']),
        'inter_p': float(mm.pvalues['inter']),
    })
within_results.sort(key=lambda r: r['inter_p'])
iter22['trastuzumab_within_HER2pos_top'] = within_results[:10]

# Adjusted main effect of trastuzumab restricted to HER2+
adjust = ['age_years', 'ecog_ps', 'stage_iv', 'has_brain_mets', 'albumin_g_dl',
          'weight_loss_pct_6mo', 'ldh_u_l', 'er_positive', 'pik3ca_mutation', 'ki67_pct',
          'postmenopausal', 'pr_positive', 'her2_low']
X = sm.add_constant(her2_pos[[treat] + adjust])
mm = sm.OLS(her2_pos['pfs_months'], X).fit()
iter22['trastuzumab_adjusted_in_HER2pos'] = {
    'coef': float(mm.params[treat]),
    'p': float(mm.pvalues[treat]),
    'se': float(mm.bse[treat]),
    'n': int(mm.nobs),
}
results['iter22_trastuzumab_within_her2pos'] = iter22

with open('iters_18_22_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Print results
print("\n=== Iter 18: palbociclib leave-one-out ===")
print("Full 4-marker:", iter18_full)
for d in iter18_drops:
    print(f"  {d['description']}: diff={d['result'].get('diff'):+.3f} p={d['result'].get('p_value'):.2e}, n_on={d['result']['n_on']}")
print("\nWithin-complement:")
for c in iter18_complement:
    a = c['effect_when_marker_present']
    b = c['effect_when_marker_absent']
    print(f"  {c['marker_being_tested']}: present diff={a.get('diff'):+.3f} (n={a['n_on']+a['n_off']}); absent diff={b.get('diff'):+.3f} (n={b['n_on']+b['n_off']})")

print("\nKi67 threshold sweep:")
for r in iter18_ki67_sweep:
    print(f"  th={r['threshold']}: low diff={r['low'].get('diff'):+.3f} (n_on={r['low']['n_on']}); high diff={r['high'].get('diff'):+.3f} (n_on={r['high']['n_on']})")

print("\n=== Iter 19: adjusted quad interaction ===")
print(json.dumps(iter19, indent=2))

print("\n=== Iter 20: olaparib refinement ===")
for r in iter20:
    if 'diff' in r:
        print(f"  {r['label']}: n_on={r['n_on']} n_off={r['n_off']} diff={r['diff']:+.3f} p={r['p_value']:.2e}")
print("Adjusted:", json.dumps(iter20_adj, indent=2))

print("\n=== Iter 21: adjusted main effects ===")
print(json.dumps(iter21, indent=2))

print("\n=== Iter 22: trastuzumab within HER2+ top interactions ===")
for r in iter22['trastuzumab_within_HER2pos_top']:
    print(f"  {r['modifier']} ({r['kind']}): inter_coef={r['inter_coef']:+.4f} p={r['inter_p']:.2e}")
print("Adjusted:", iter22['trastuzumab_adjusted_in_HER2pos'])
